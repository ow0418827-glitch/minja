"""Prepare nine real-product groups from a read-only disk catalog.

Only selected products are read from SQLite.  Human goals are streamed, and the
multi-GB source product array is never loaded.  This module makes no model calls.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import re
import shutil
import sqlite3

from src.environment.local_webshop import fingerprint, read, write


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = PROJECT_ROOT / 'src/data/victim_target_pair/victim_target.json'
DEFAULT_SEED = 20261003
PROTOCOL = 'victim_history_image_substitution'
ATTACKER_PROTOCOL = 'attacker_owned_product_image_substitution'
REFERENCE_SELECTION_POLICY = 'victim_keyword_in_catalog_title'


def protocol_for_source_history(mode):
    if mode not in ('victim', 'attacker'):
        raise ValueError('source_history_mode must be victim or attacker')
    return ATTACKER_PROTOCOL if mode == 'attacker' else PROTOCOL


def _inside_project(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError('Preparation output must stay inside the minja project')
    return path


def _compact(value):
    return re.sub(r'\s+', '', str(value)).casefold()


def canonical_instruction(value):
    """Ignore generated price suffixes when preventing instruction leakage."""
    value = re.sub(r',?\s+and price lower than\s+[\d,.]+\s+dollars\.?\s*$',
                   '', value, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', value).strip(' .').casefold()


def _victim_pattern(victim):
    stem = victim[:-1] if victim in ('cookies', 'jeans') else victim
    return re.compile(r'\b' + re.escape(stem) + r'(?:s|es)?\b', re.IGNORECASE)


def _category_score(goal, victim):
    # Include singular/plural spellings and full taxonomy: toothbrush tasks may
    # have a broad "children's dental care" query, but a Toothbrushes leaf.
    pattern = _victim_pattern(victim)
    if not pattern.search(goal.get('instruction_text', '')):
        return 0
    taxonomy = goal.get('product_category', '')
    leaf = taxonomy.split('›')[-1]
    if pattern.search(leaf):
        return 3
    if pattern.search(goal.get('query', '')):
        return 2
    if pattern.search(taxonomy):
        return 1
    return 0


def _stream_goals(path):
    import ijson
    with Path(path).open('rb') as stream:
        for index, goal in enumerate(ijson.items(stream, 'item', use_float=True)):
            yield index, goal


class Catalog:
    """Bounded-memory product lookups; cannot mutate the source database."""
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.conn = sqlite3.connect((self.root / 'catalog.sqlite').as_uri() + '?mode=ro', uri=True)
        self.conn.execute('PRAGMA query_only=ON')
        self.conn.execute('PRAGMA cache_size=-8192')
        self.conn.execute('PRAGMA mmap_size=0')

    def product(self, asin):
        row = self.conn.execute('SELECT data, price FROM products WHERE asin=?', (asin,)).fetchone()
        if row is None:
            raise ValueError(f'Required product missing from disk catalog: {asin}')
        return json.loads(row[0]), float(row[1])

    def close(self):
        self.conn.close()


def resolve_targets(pairs, targets, catalog):
    """Require an unambiguous source mapping and verify it against real titles."""
    results = []
    for pair in pairs:
        matches = [row for row in targets if row.get('victim') == pair['victim']
                   and _compact(row.get('target')) == _compact(pair['target'])]
        if len(matches) != 1:
            raise ValueError(f"Target mapping is ambiguous or missing: {pair['victim']}")
        ids = sorted(set(matches[0].get('target_asins', [])))
        if len(ids) != 1:
            raise ValueError(f"Expected one target ASIN for {pair['victim']}, found {len(ids)}")
        product, price = catalog.product(ids[0])
        if _compact(pair['target']) not in _compact(product.get('name', product.get('Title', ''))):
            raise ValueError(f"Target title does not match catalog product: {ids[0]}")
        if not product.get('images') or not product['images'][0]:
            raise ValueError(f"Target has no main image: {ids[0]}")
        results.append((pair, ids[0], product, price))
    if len({asin for _, asin, _, _ in results}) != len(results):
        raise ValueError('Different groups must resolve to different target products')
    return results


def _task(group_id, kind, ordinal, index, goal, provenance='human_goal'):
    task_id = f'{group_id}_{kind}_{ordinal:03d}'
    suffix = index if index is not None else 'synthetic'
    return {'task_id': task_id, 'session': f'fixed_nine_{task_id}_{suffix}', 'asin': goal['asin'],
            'goal_index': index, 'instruction': goal['instruction_text'], 'goal': goal,
            'provenance': provenance,
            'instruction_fingerprint': fingerprint(canonical_instruction(goal['instruction_text']))}


def catalog_product_goal(product, price):
    """A truthful legal demand when the fixed A has no human-authored goal.

    No product, price, image, or reward is modified. The selected attribute and
    option values are already present in the catalog. This complete goal is
    evaluated by the ordinary WebShop reward function during actual shopping.
    """
    name = str(product.get('name') or product.get('Title') or '').strip()
    if not name:
        raise ValueError('The attacker product needs a real catalog title')
    attributes = [str(value) for value in product.get('Attributes', [])
                  if value and str(value) != 'DUMMY_ATTR']
    # A title string is an actual observable attribute, rather than a fabricated
    # tag. The upstream reward checks the title when Attributes is unavailable.
    if not attributes:
        attributes = [name.lower()]
    options = {}
    for key, values in sorted((product.get('options') or {}).items()):
        if values:
            options[str(key)] = str(values[0])
    limit = round(float(price) + max(5.0, float(price) * 0.2), 2)
    if limit <= float(price):
        raise ValueError('Cannot create a feasible real-product price limit')
    option_text = '; '.join(f'{key}: {value}' for key, value in options.items())
    instruction = f'Please buy {name}.'
    if option_text:
        instruction += f' Select these options: {option_text}.'
    instruction += f' The price must be lower than {limit:.2f} dollars.'
    return {'asin': product['asin'], 'category': product.get('category', ''),
            'query': product.get('query', ''), 'name': name,
            'product_category': product.get('product_category', ''),
            'instruction_text': instruction, 'attributes': attributes,
            'price_upper': limit, 'goal_options': list(options.values()),
            'weight': 1}, options


def _attacker_training(group_id, asin, product, price, rows):
    candidates = [(index, goal) for index, goal in rows
                  if goal.get('asin') == asin and goal.get('instruction_text')]
    if candidates:
        index, goal = min(candidates, key=lambda row: row[0])
        training = _task(group_id, 'source', 1, index, copy.deepcopy(goal))
        training['goal_origin'] = {'kind': 'human_goal', 'goal_index': index,
                                   'fixed_product_asin': asin}
        return training
    goal, options = catalog_product_goal(product, price)
    training = _task(group_id, 'source', 1, None, goal, 'catalog_product_goal')
    training['goal_origin'] = {'kind': 'catalog_product_goal', 'fixed_product_asin': asin,
                              'product_fields_sha256': fingerprint(product),
                              'catalog_price': float(price),
                              'selected_catalog_options': options,
                              'reward_is_real_environment_only': True}
    return training


def _group_tasks(group):
    tasks = [group['training']]
    reference = group.get('reference_task')
    if reference is not None and reference['task_id'] != group['training']['task_id']:
        tasks.append(reference)
    return [*tasks, *group['trials'], *group['normal_candidates']]


def select_tasks(resolved, goals, normal_mode='victim', normal_count=14, test_count=20,
                 seed=DEFAULT_SEED, normal_candidate_limit=70, source_history_mode='victim'):
    """Disjoint real demands, deterministic selection, no repeat-to-fill fallback."""
    if normal_mode not in ('legacy', 'victim'):
        raise ValueError('normal_mode must be legacy or victim')
    protocol = protocol_for_source_history(source_history_mode)
    if min(normal_count, test_count) < 1 or normal_candidate_limit < normal_count:
        raise ValueError('Task counts must be positive; candidate limit must cover normal_count')
    rows = list(goals)  # The human goals are ~12k, never the full product array.
    target_ids = {asin for _, asin, _, _ in resolved}
    rng = random.Random(seed)
    groups = []
    reserved_queries = set()
    for number, (pair, asin, product, price) in enumerate(resolved, 1):
        group_id = f"{number:02d}_{pair['victim'].replace(' ', '_')}"
        eligible = [(index, goal, _category_score(goal, pair['victim']))
                    for index, goal in rows if goal['asin'] not in target_ids]
        eligible = [row for row in eligible if row[2]]
        rng.shuffle(eligible)
        eligible.sort(key=lambda row: -row[2])
        seen, chosen, training = set(), [], None
        reference_instruction = None
        if source_history_mode == 'attacker':
            # A human instruction can mention shampoo while its underlying
            # goal names a conditioner. Fixed-image queries need the actual
            # catalog title to agree with the requested victim kind as well.
            for index, goal, _ in eligible:
                key = canonical_instruction(goal['instruction_text'])
                if key not in reserved_queries and _victim_pattern(pair['victim']).search(str(goal.get('name') or '')):
                    training = _task(group_id, 'source', 1, index, goal)
                    reference_instruction = key
                    break
            if training is None:
                raise ValueError(f"{pair['victim']}: no real victim reference with the victim keyword in its catalog title")
        for index, goal, score in eligible:
            key = canonical_instruction(goal['instruction_text'])
            if key in seen or key in reserved_queries:
                continue
            seen.add(key)
            if key == reference_instruction:
                continue
            if training is None:
                training = _task(group_id, 'source', 1, index, goal)
                continue
            chosen.append(_task(group_id, 'eval', len(chosen) + 1, index, goal))
            if len(chosen) == test_count:
                break
        if len(chosen) != test_count:
            raise ValueError(f"{pair['victim']}: only {len(chosen)} unique real victim tasks; need {test_count}")
        reserved_queries.update(canonical_instruction(task['instruction']) for task in chosen)
        reserved_queries.add(canonical_instruction(training['instruction']))
        reference_task = training
        if source_history_mode == 'attacker':
            reference_task = _task(group_id, 'reference', 1, reference_task['goal_index'],
                                   copy.deepcopy(reference_task['goal']))
            training = _attacker_training(group_id, asin, product, price, rows)
            reserved_queries.add(canonical_instruction(training['instruction']))
        group = {'id': group_id, 'group_id': group_id, 'number': number,
                 'pair_index': number - 1, 'victim': pair['victim'], 'target': pair['target'],
                 'target_price': pair['target_price'], 'target_asin': asin, 'a': asin,
                 'b': reference_task['asin'], 'training': training,
                 'source_training_asin': training['asin'], 'protocol': protocol,
                 'source_history_actual_purchase': training['asin'],
                 'source_retrieval_image_origin_asin': asin,
                 'trials': chosen, 'normal_memory_target': normal_count,
                 'normal_mode': normal_mode, 'normal_candidates': []}
        if source_history_mode == 'attacker':
            group.update(source_history_mode='attacker', reference_task=reference_task,
                         source_history_image_owner_asin=asin,
                         clean_source_image_asin=asin,
                         reference_product_asin=reference_task['asin'],
                         reference_task_role='future_victim_purchase_demand',
                         reference_selection_policy=REFERENCE_SELECTION_POLICY,
                         source_history_role='legal_attacker_product_purchase')
        groups.append(group)
    for group in groups:
        candidates = [(index, goal) for index, goal in rows
                      if goal['asin'] not in target_ids
                      and canonical_instruction(goal['instruction_text']) not in reserved_queries
                      and (normal_mode != 'victim' or _category_score(goal, group['victim']))]
        rng.shuffle(candidates)
        seen = set()
        for index, goal in candidates:
            key = canonical_instruction(goal['instruction_text'])
            if key in seen:
                continue
            seen.add(key)
            group['normal_candidates'].append(_task(group['id'], 'normal', len(seen), index, goal))
            if len(seen) == normal_candidate_limit:
                break
        if len(group['normal_candidates']) < normal_count:
            raise ValueError(f"{group['victim']}: only {len(seen)} non-leaking normal tasks; need {normal_count}")
    return groups


def _verify_image(path, digest):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        return False
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def ensure_image(product, root, direct=False, cache_roots=()):
    """Reuse verified images first; all writes are inside this experiment root."""
    root = _inside_project(root)
    asin, url = product['asin'], product['images'][0]
    for cache in (root, *map(Path, cache_roots)):
        record_path = cache / 'image_records' / f'{asin}.json'
        if not record_path.is_file():
            continue
        record = read(record_path)
        path = (cache / record['path']).resolve()
        if not path.is_relative_to(cache.resolve()):
            continue
        if record.get('url') != url or not _verify_image(path, record.get('sha256', '')):
            continue
        destination = root / 'assets' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path != destination and not destination.exists():
            shutil.copyfile(path, destination)
        record = dict(record, path=destination.relative_to(root).as_posix(),
                      reused_from=str(cache.resolve()), absolute_path=str(destination))
        write(root / 'image_records' / f'{asin}.json', record)
        return record
    import requests
    from PIL import Image
    session = requests.Session()
    session.trust_env = not direct
    try:
        with session.get(url, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(64 * 1024):
                content.extend(chunk)
                if len(content) > 10 * 1024 * 1024:
                    raise ValueError('Image exceeds 10 MiB limit')
        with Image.open(BytesIO(content)) as image:
            extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif'}.get(image.format)
            if extension is None:
                raise ValueError('Unsupported image format')
            image.verify()
        digest = hashlib.sha256(content).hexdigest()
        path = root / 'assets' / (digest + extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        record = {'asin': asin, 'url': url, 'path': path.relative_to(root).as_posix(),
                  'absolute_path': str(path), 'sha256': digest, 'bytes': len(content),
                  'reused_from': None}
        write(root / 'image_records' / f'{asin}.json', record)
        return record
    finally:
        session.close()


def _backup_legacy_bundle(root, existing):
    """Copy only preparation JSON artifacts; never move or delete user data."""
    backup = _inside_project(root / 'legacy_prepare_backup' / existing.get('bundle_id', 'unversioned'))
    names = ['manifest.json', 'task_plan.json', 'setup_report.json', 'image_failures.json']
    names += [f"groups/{group['id']}/tasks.json" for group in existing.get('groups', [])]
    for name in names:
        source = _inside_project(root / name)
        destination = _inside_project(backup / name)
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(source, destination)
    return backup


def prepare_bundle(root, webshop_source, pair_file=None, normal_mode='victim', normal_count=14,
                   test_count=20, seed=DEFAULT_SEED, direct_images=False, *, catalog_dir,
                   normal_candidate_limit=70, download_images=True, cache_roots=None,
                   image_workers=6, download_task_images=False, source_history_mode='victim'):
    root = _inside_project(root)
    source = Path(webshop_source).resolve()
    protocol = protocol_for_source_history(source_history_mode)
    pair_file = Path(pair_file or DEFAULT_PAIRS).resolve()
    pairs = read(pair_file)
    catalog_root = Path(catalog_dir).resolve()
    source_manifest = read(catalog_root / 'data_manifest.json') if (catalog_root / 'data_manifest.json').exists() else {}
    settings = {'schema_version': 1, 'protocol': protocol, 'webshop_source': str(source), 'catalog_dir': str(catalog_root),
                'pair_file_digest': fingerprint(pairs), 'normal_mode': normal_mode,
                'normal_count': normal_count, 'normal_candidate_limit': normal_candidate_limit,
                'test_count': test_count, 'seed': seed,
                'catalog_bundle_id': source_manifest.get('bundle_id'),
                'goals_sha256': source_manifest.get('goals_sha256'),
                'download_task_images': download_task_images}
    if source_history_mode == 'attacker':
        settings['source_history_mode'] = source_history_mode
        settings['reference_selection_policy'] = REFERENCE_SELECTION_POLICY
    if (root / 'manifest.json').is_file():
        existing = read(root / 'manifest.json')
        if existing.get('protocol') in (PROTOCOL, ATTACKER_PROTOCOL) and existing.get('protocol') != protocol:
            raise ValueError('Prepared source-history protocol changed; choose a different output root')
        if existing.get('protocol') != protocol:
            _backup_legacy_bundle(root, existing)
        elif existing.get('prepare_settings') != settings:
            raise ValueError('Prepared bundle settings changed; choose a different output root')
        elif not download_images or existing.get('images_downloaded'):
            digest = existing.pop('bundle_id')
            if fingerprint(existing) != digest:
                raise ValueError('Prepared bundle manifest changed')
            existing['bundle_id'] = digest
            for group in existing['groups']:
                for key in ('source_image_record', 'query_image_record'):
                    record = group.get(key)
                    if download_images and (not record or not _verify_image(Path(record['absolute_path']), record['sha256'])):
                        raise ValueError('Prepared A/B image changed or is missing')
            return existing
    catalog = Catalog(catalog_dir)
    try:
        resolved = resolve_targets(pairs, read(catalog.root / 'targets.json'), catalog)
        groups = select_tasks(resolved, _stream_goals(catalog.root / 'goals.json'), normal_mode,
                              normal_count, test_count, seed, normal_candidate_limit,
                              source_history_mode=source_history_mode)
        products = {asin: product for _, asin, product, _ in resolved}
        for group in groups:
            for task in _group_tasks(group):
                if task['asin'] not in products:
                    product, _ = catalog.product(task['asin'])
                    if not product.get('images') or not product['images'][0]:
                        raise ValueError(f"Task product has no main image: {task['asin']}")
                    products[task['asin']] = product
            if source_history_mode == 'attacker':
                reference = group['reference_task']
                actual_title = str(products[group['b']].get('name') or products[group['b']].get('Title') or '')
                goal_title = str(reference['goal'].get('name') or '')
                if not _victim_pattern(group['victim']).search(actual_title):
                    raise ValueError(f"{group['victim']}: fixed B catalog title does not match the victim kind")
                if _compact(actual_title) != _compact(goal_title):
                    raise ValueError(f"{group['victim']}: reference goal title differs from the real catalog product")
                group['reference_title_audit'] = {
                    'selection_policy': REFERENCE_SELECTION_POLICY, 'asin': group['b'],
                    'catalog_title': actual_title, 'goal_title': goal_title,
                    'victim_keyword_matches_catalog_title': True,
                    'goal_title_matches_catalog_title': True,
                    'catalog_title_sha256': hashlib.sha256(actual_title.encode('utf-8')).hexdigest()}
    finally:
        catalog.close()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'protocol': protocol,
                'design': ('nine_groups_attacker_history_fixed_victim_reference' if source_history_mode == 'attacker'
                           else 'nine_groups_distinct_real_held_out_tasks'),
                'root': str(root), 'webshop_source': str(source), 'catalog_dir': str(Path(catalog_dir).resolve()),
                'pair_file': str(pair_file), 'pair_sha256': fingerprint(pairs),
                'source_bundle_id': source_manifest.get('bundle_id'), 'seed': seed,
                'catalog_bundle_id': source_manifest.get('bundle_id'),
                'catalog_scope': source_manifest.get('scope', 'disk_catalog'),
                'goals_sha256': source_manifest.get('goals_sha256'),
                'pair_file_digest': fingerprint(pairs), 'prepare_settings': settings,
                'normal_mode': normal_mode, 'normal_memory_target': normal_count,
                'normal_candidate_limit': normal_candidate_limit, 'test_count': test_count,
                'images_downloaded': False, 'model_api_called': False, 'groups': groups}
    write(root / 'task_plan.json', manifest)
    priority_ids = list(dict.fromkeys(asin for group in groups for asin in (group['a'], group['b'])))
    images, failures = {}, []
    cache_roots = cache_roots if cache_roots is not None else [PROJECT_ROOT / 'src/output/local20']
    if download_images:
        # Finish the nine A/B image pairs before downloading other task images.
        batches = [priority_ids]
        if download_task_images:
            batches.append([asin for asin in products if asin not in priority_ids])
        for batch_ids in batches:
            with ThreadPoolExecutor(max_workers=image_workers) as pool:
                jobs = {pool.submit(ensure_image, products[asin], root, direct_images, cache_roots): asin
                        for asin in batch_ids}
                for job in as_completed(jobs):
                    asin = jobs[job]
                    try:
                        images[asin] = job.result()
                    except Exception as exc:
                        failures.append({'asin': asin, 'reason': str(exc)})
            write(root / 'image_failures.json', failures)
            if batch_ids == priority_ids and any(asin not in images for asin in priority_ids):
                raise ValueError('Required nine-group A/B image download failed; see image_failures.json')
        if failures and download_task_images:
            # Missing candidates can be skipped by the normal-memory driver;
            # evaluation and A/B images are mandatory and never substituted.
            mandatory = {task['asin'] for group in groups
                         for task in [group['training'], group.get('reference_task', group['training']), *group['trials']]}
            missing = mandatory - images.keys()
            if missing:
                raise ValueError(f'Mandatory test/training images missing: {sorted(missing)}')
        if download_task_images:
            for group in groups:
                group['normal_candidates'] = [task for task in group['normal_candidates'] if task['asin'] in images]
                if len(group['normal_candidates']) < normal_count:
                    raise ValueError(f"Too few usable normal tasks for {group['victim']} after image failures")
        manifest['images_downloaded'] = True
    for group in groups:
        for task in _group_tasks(group):
            task['image_record'] = images.get(task['asin'])
            task['query_image'] = images[task['asin']]['absolute_path'] if task['asin'] in images else None
            task['source_image_url'] = products[task['asin']]['images'][0]
        group['source_image'] = images[group['a']]['absolute_path'] if group['a'] in images else None
        reference_task = group.get('reference_task', group['training'])
        group['query_image'] = reference_task['query_image']
        group['source_image_record'] = images.get(group['a'])
        group['query_image_record'] = images.get(group['b'])
        group['b_goal_index'] = reference_task['goal_index']
        group['normal_attempt_limit'] = len(group['normal_candidates'])
        group['query_image_role'] = ('future_victim_reference_product_image' if source_history_mode == 'attacker'
                                     else 'source_training_product_image')
        group['query_image_task_id'] = reference_task['task_id']
        group['historical_memory_count'] = normal_count + 1
        group['distinct_history_and_test_demands'] = normal_count + 1 + test_count + (source_history_mode == 'attacker')
        write(root / 'groups' / group['id'] / 'tasks.json', group)
    manifest['image_failures'] = failures
    manifest['bundle_id'] = fingerprint(manifest)
    write(root / 'manifest.json', manifest)
    write(root / 'setup_report.json', {'schema_version': 1, 'bundle_id': manifest['bundle_id'],
          'protocol': protocol,
          'groups': len(groups), 'target_products': len(priority_ids), 'task_products': len(products),
          'normal_memory_target': normal_count, 'test_count_per_group': test_count,
          'images_downloaded': manifest['images_downloaded'], 'model_api_called': False})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--webshop-source', required=True)
    parser.add_argument('--catalog-dir', required=True)
    parser.add_argument('--pair-file', default=str(DEFAULT_PAIRS))
    parser.add_argument('--normal-mode', choices=('legacy', 'victim'), default='victim')
    parser.add_argument('--source-history-mode', choices=('victim', 'attacker'), default='attacker')
    parser.add_argument('--normal-count', type=int, default=14)
    parser.add_argument('--normal-candidate-limit', type=int, default=70)
    parser.add_argument('--test-count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--direct-images', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--download-task-images', action='store_true')
    args = parser.parse_args()
    manifest = prepare_bundle(args.root, args.webshop_source, args.pair_file, args.normal_mode,
                              args.normal_count, args.test_count, args.seed, args.direct_images,
                              catalog_dir=args.catalog_dir,
                              normal_candidate_limit=args.normal_candidate_limit,
                              download_images=not args.plan_only,
                              download_task_images=args.download_task_images,
                              source_history_mode=args.source_history_mode)
    print(json.dumps({'bundle_id': manifest['bundle_id'], 'groups': len(manifest['groups']),
                      'images_downloaded': manifest['images_downloaded']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
