"""Prime a copied history bank and old-B queries without remote model calls."""
import hashlib
from pathlib import Path

import numpy as np

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver
from src.experiments.visual_memory_experiment import eligible
from src.vision.joint_encoder import make_joint_encoder
from src.vision.visual_retrieval import (
    VisualMemoryRetriever, canonical_digest, memory_index_text, resolve_image, sha256,
)


def verify_source_files(provenance):
    """Verify the copied-from run; never populate caches in the original run."""
    source = Path(provenance['source_root']).resolve()
    fingerprints = provenance.get('source_files_sha256')
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError('History reuse requires source file fingerprints')
    for name, expected in fingerprints.items():
        path = (source / str(name).replace('\\', '/')).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Source fingerprint path escapes its original root')
        if sha256(path) != expected:
            raise ValueError('Original source file changed: ' + str(name))
    return len(fingerprints)


def _entry_vector(retriever, bank, entry):
    # Reuse the exact cache scope created by rank, rather than an image-only
    # approximation or a different full-text encoder.
    identity = {
        'bank_sha256': sha256(retriever.memory_path),
        'entries_sha256': canonical_digest(bank),
        'model_sha256': canonical_digest(retriever.encoder.snapshot),
        'key_mode': retriever.key_mode,
        'index_text': retriever.index_text,
        'index_text_policy_sha256': canonical_digest(retriever.snapshot['index_text_policy']),
    }
    text = memory_index_text(entry, retriever.index_text)
    path = resolve_image(entry, retriever.memory_path)
    return retriever._encode(text, path, retriever.key_mode,
                             scope=canonical_digest(identity))[0]


def _vector_record(vector):
    values = vector.tolist()
    return {'vector': values, 'vector_sha256': canonical_digest(values),
            'dimension': len(values)}


def _by_id(bank):
    result = {entry['Id']: entry for entry in bank}
    if len(result) != 15 or len(bank) != 15:
        raise ValueError('Reused banks must retain exactly fifteen unique memories')
    return result


def _same_index_content(before, before_path, after, after_path, index_text):
    return (memory_index_text(before, index_text) == memory_index_text(after, index_text)
            and sha256(resolve_image(before, before_path))
            == sha256(resolve_image(after, after_path)))


def prime_retrieval(args, provenance):
    """Write offline vector/ranking evidence for twenty fixed old-B wordings."""
    if (getattr(args, 'reserve_tests', 0) != 0
            or args.query_specificity != 'vague'
            or args.memory_key_mode != 'joint' or args.memory_query_mode != 'joint'
            or args.index_text != 'full'):
        raise ValueError('Old-B comparison requires twenty vague full-joint queries without reserves')
    verified_count = verify_source_files(provenance)
    root, plan, groups = driver.load_plan(args)
    source_root = Path(provenance['source_root']).resolve()
    if source_root == root.resolve():
        raise ValueError('Retrieval preparation must use a separate destination root')
    source_groups = {group['id']: group for group in read(source_root / 'manifest.json')['groups']}
    vector_report = {'schema_version': 1, 'remote_model_calls': 0,
                     'old_retrieval_cache_reused': False, 'groups': []}
    report = {'schema_version': 1, 'source_root': str(source_root), 'root': str(root),
              'remote_model_calls': 0, 'old_retrieval_cache_reused': False,
              'source_files_verified': verified_count, 'groups': []}
    encoder = make_joint_encoder(
        kind=args.retrieval_encoder, optimization_config=args.optimization_config,
        device=args.device, gme_python=getattr(args, 'gme_python', None),
        gme_model=getattr(args, 'gme_model', None))
    try:
        vector_report['encoder'] = encoder.snapshot
        vector_report['encoder_sha256'] = canonical_digest(encoder.snapshot)
        report['encoder_sha256'] = vector_report['encoder_sha256']
        for prepared in groups:
            folder = root / 'groups' / prepared['id']
            group = driver.resolve_history_binding(args, root, plan, prepared, required=True)
            ready = driver.verify_banks(folder)
            source_group = source_groups[group['id']]
            query_sha = sha256(group['query_image'])
            if (group['b'] != source_group['b']
                    or query_sha != sha256(source_group['query_image'])):
                raise ValueError('Preparation query does not match the frozen old B image')
            evaluation_folder, _ = driver.evaluation_location(args, folder, save=True)
            tasks, tasks_sha = driver.evaluation_trials(
                args, group, folder, task_folder=evaluation_folder)
            if len(tasks) != 20 or any(task.get('query_image_sha256') != query_sha for task in tasks):
                raise ValueError('All twenty demands must retain the frozen old B query digest')
            source_id = ready['source_memory_id']
            source_folder = source_root / 'groups' / group['id']
            old_clean_path = source_folder / 'clean_memory.json'
            old_pixel_path = source_folder / 'memory.json'
            old_clean, old_pixel = read(old_clean_path), read(old_pixel_path)
            old_clean_ids, old_pixel_ids = _by_id(old_clean), _by_id(old_pixel)
            bank_paths = {'clean': folder / 'clean_memory.json', 'adversarial': folder / 'memory.json'}
            banks = {condition: read(path) for condition, path in bank_paths.items()}
            retrievers = {condition: VisualMemoryRetriever(
                encoder, path, key_mode='joint', query_mode='joint', index_text='full')
                for condition, path in bank_paths.items()}
            old_retriever = VisualMemoryRetriever(
                encoder, old_pixel_path, key_mode='joint', query_mode='joint',
                index_text='full', cache_dir=None)
            old_ranking = old_retriever.rank(eligible(old_pixel), tasks[0]['instruction'], group['query_image'])
            unchanged = []
            for condition, bank in banks.items():
                ids = _by_id(bank)
                if set(ids) != set(old_clean_ids) or set(ids) != set(old_pixel_ids):
                    raise ValueError('Copied memory identities changed')
                for memory_id in ids:
                    if memory_id == source_id:
                        continue
                    if (not _same_index_content(old_clean_ids[memory_id], old_clean_path,
                                                ids[memory_id], bank_paths[condition], 'full')
                            or not _same_index_content(old_pixel_ids[memory_id], old_pixel_path,
                                                      ids[memory_id], bank_paths[condition], 'full')):
                        raise ValueError('A normal memory text or retrieval image changed')
                if len(ids) - 1 != 14 or source_id not in ids:
                    raise ValueError('Fourteen normal memories and one fixed A memory are required')
                before = old_clean_ids[source_id]
                after = ids[source_id]
                if (memory_index_text(before) != memory_index_text(after)
                        or before.get('PrimaryImage', {}).get('SHA256')
                        != after.get('PrimaryImage', {}).get('SHA256')):
                    raise ValueError('Original A history text or purchased primary image changed')
                if condition == 'clean' and not _same_index_content(
                        before, old_clean_path, after, bank_paths[condition], 'full'):
                    raise ValueError('The reused clean A memory changed')
            trial_vectors, condition_rankings = [], {'clean': [], 'adversarial': []}
            for task in tasks:
                query_vectors = {}
                for condition, retriever in retrievers.items():
                    bank = eligible(banks[condition])
                    ranking = retriever.rank(bank, task['instruction'], group['query_image'])
                    if len(ranking) != 15:
                        raise ValueError('Every index must contain fifteen eligible memories')
                    vector, key = retriever._encode(task['instruction'], Path(group['query_image']), 'joint')
                    query_vectors[condition] = vector
                    selected = next(row for row in ranking if row['Id'] == source_id)
                    normal = max((row for row in ranking if row['Id'] != source_id),
                                 key=lambda row: row['score'])
                    condition_rankings[condition].append({
                        'task_id': task['task_id'], 'source_rank': selected['rank'],
                        'source_score': selected['score'], 'top_normal_id': normal['Id'],
                        'top_normal_score': normal['score'],
                        'leading_margin': selected['score'] - normal['score'],
                        'query_image_sha256': query_sha,
                    })
                if not np.allclose(query_vectors['clean'], query_vectors['adversarial'],
                                   rtol=1e-5, atol=1e-6):
                    raise ValueError('Clean and adversarial conditions encoded different old-B queries')
                trial_vectors.append({
                    'task_id': task['task_id'], 'instruction': task['instruction'],
                    'query_text_sha256': hashlib.sha256(task['instruction'].encode('utf-8')).hexdigest(),
                    'query_image_sha256': query_sha,
                    **_vector_record(query_vectors['clean']),
                    'condition_vector_sha256': {name: _vector_record(vector)['vector_sha256']
                                                for name, vector in query_vectors.items()},
                    'condition_vectors_equal': True,
                })
            a_vectors = {condition: _entry_vector(retriever, eligible(banks[condition]),
                                                   _by_id(banks[condition])[source_id])
                         for condition, retriever in retrievers.items()}
            a_vectors['old_pixel'] = _entry_vector(old_retriever, eligible(old_pixel), old_pixel_ids[source_id])
            for memory_id in old_clean_ids:
                if memory_id == source_id:
                    continue
                vectors = [_entry_vector(retrievers[name], eligible(banks[name]),
                                         _by_id(banks[name])[memory_id]) for name in bank_paths]
                if not np.allclose(vectors[0], vectors[1], rtol=1e-5, atol=1e-6):
                    raise ValueError('Normal memory vectors differ between new conditions')
                unchanged.append(memory_id)
            group_report = {
                'group': group['id'], 'reference_asin': group['b'], 'query_image_sha256': query_sha,
                'source_memory_id': source_id, 'tasks_sha256': tasks_sha, 'test_count': len(tasks),
                'unchanged_normal_memories': len(unchanged), 'primary_a_image_unchanged': True,
                'bank_vectors_per_condition': {name: len(banks[name]) for name in banks},
                'query_vectors_equal_across_conditions': True,
                'source_top1_counts': {name: sum(item['source_rank'] == 1 for item in rows)
                                       for name, rows in condition_rankings.items()},
                'rankings': condition_rankings,
                'old_pixel_first_source_rank': next(row['rank'] for row in old_ranking if row['Id'] == source_id),
                'source_vector_changed_from_old_pixel': not np.allclose(
                    a_vectors['adversarial'], a_vectors['old_pixel'], rtol=1e-5, atol=1e-6),
                'source_vector_changed_from_clean': not np.allclose(
                    a_vectors['adversarial'], a_vectors['clean'], rtol=1e-5, atol=1e-6),
                'index_cache_directory': str(folder / 'retrieval_index_cache'),
            }
            report['groups'].append(group_report)
            vector_report['groups'].append({
                'group': group['id'], 'query_image_sha256': query_sha, 'reference_asin': group['b'],
                'query_vectors': trial_vectors,
                'source_memory_vectors': {name: _vector_record(vector) for name, vector in a_vectors.items()},
            })
            print(group['id'] + '：旧乙图查询和两个十五条记忆索引已离线核验。', flush=True)
        verify_source_files(provenance)
        report['source_files_unchanged'] = True
        write(root / 'query_vector_preparation.json', vector_report)
        write(root / 'retrieval_index_preparation.json', report)
        return report
    finally:
        close = getattr(encoder, 'close', None)
        if callable(close):
            close()
