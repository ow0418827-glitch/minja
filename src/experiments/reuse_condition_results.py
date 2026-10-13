"""Audited inheritance of committed results when selecting one evaluation condition.

The original three-condition ledger is immutable. A separate evaluation variant
receives derived reports which explicitly retain their original protocol identity.
This module neither loads a model nor executes a shopping request.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from src.environment.local_webshop import read, write, fingerprint
from src.experiments import nine_group_experiment as driver
from src.experiments.deferred_529 import Deferred529Evaluation, _fingerprint, SCHEMA_VERSION, POLICY_VERSION


_NON_TREATMENT_FIELDS = {'evaluation_conditions', 'reuse_frozen_memory',
                         'source_run_settings_sha256', 'evaluation_code_sha256'}


def _semantics(protocol):
    return {key: value for key, value in protocol.items() if key not in _NON_TREATMENT_FIELDS}


def _file_entries(value):
    if isinstance(value, dict):
        return [(Path(path).resolve(), sha) for path, sha in value.items()]
    if isinstance(value, list):
        return [(Path(row['path']).resolve(), row['sha256']) for row in value]
    raise ValueError('Condition resume manifest must contain explicit file digests')


def _rows(manifest):
    rows = manifest.get('groups', [])
    if isinstance(rows, dict):
        return [dict(row, group=key) for key, row in rows.items()]
    return rows


def _read_only_ledger(path, trials):
    """Validate existing state without __init__, which publishes a ledger file."""
    state = read(path)
    identity = state['identity']
    conditions, orders = identity.get('conditions'), identity.get('condition_orders')
    if (identity.get('schema_version') != SCHEMA_VERSION or identity.get('policy_version') != POLICY_VERSION
            or not isinstance(conditions, list) or not 1 <= len(conditions) <= 3
            or len(set(conditions)) != len(conditions)
            or any(c not in ('no_memory', 'clean', 'adversarial') for c in conditions)
            or not isinstance(orders, list) or len(orders) != len(trials)
            or any(not isinstance(order, list) or len(order) != len(conditions)
                   or set(order) != set(conditions) for order in orders)
            or identity.get('primary_count') != len(trials)
            or identity.get('trials_sha256') != _fingerprint(trials)):
        raise ValueError('Source ledger belongs to different frozen trials')
    ledger = object.__new__(Deferred529Evaluation)
    ledger.path, ledger.trials = Path(path), copy.deepcopy(trials)
    ledger.identity, ledger.state = identity, state
    ledger.primary_count = len(trials)
    ledger.conditions = identity['conditions']
    ledger.condition_orders = identity['condition_orders']
    ledger._validate()
    return ledger


def _verify_frozen_memory(folder, row, ready):
    frozen = row.get('frozen_memory', row)
    required = {
        'memory_ready_sha256': folder / 'memory_ready.json',
        'history_binding_file_sha256': folder / 'history_binding.json',
        'memory_bank_sha256': folder / 'memory.json',
        'clean_memory_bank_sha256': folder / 'clean_memory.json',
        'attack_image_sha256': folder / 'attack' / 'adversarial.png',
    }
    for field, path in required.items():
        if frozen.get(field) != driver.digest(path):
            raise ValueError('Frozen condition-resume memory changed: ' + field)
    if (ready['attack_sha256'] != frozen['attack_image_sha256']
            or ready['bank_sha256']['adversarial'] != frozen['memory_bank_sha256']
            or ready['bank_sha256']['clean'] != frozen['clean_memory_bank_sha256']):
        raise ValueError('Condition-resume bank identity changed')


def _verify_encoder(root, folder, group_id, source_retrieval, ready, conditions):
    prepared = read(root / 'query_vector_preparation.json')
    encoder = source_retrieval['encoder']
    if (encoder != prepared['encoder']
            or source_retrieval['encoder_sha256'] != fingerprint(encoder)
            or prepared['encoder_sha256'] != _fingerprint(encoder)):
        raise ValueError('Condition-resume encoder differs from prepared vectors')
    indexes = read(root / 'retrieval_index_preparation.json')
    rows = [row for row in indexes['groups'] if row['group'] == group_id]
    if (len(rows) != 1 or rows[0]['retrieval_snapshot']['encoder'] != encoder
            or rows[0]['tasks_sha256'] != source_retrieval['tasks_sha256']):
        raise ValueError('Condition-resume retrieval preparation changed')
    protocol = source_retrieval['protocol']
    for condition in conditions:
        if condition == 'no_memory':
            continue
        expected = {'bank_sha256': ready['bank_sha256'][condition],
                    'model_sha256': _fingerprint(encoder),
                    'key_mode': protocol['memory_key_mode'],
                    'query_mode': protocol['memory_query_mode'],
                    'index_text': protocol['index_text'],
                    'key_image_weight': protocol.get('key_image_weight', 0.5),
                    'query_image_weight': protocol.get('query_image_weight', 0.5)}
        caches = [read(path).get('identity', {})
                  for path in (folder / 'retrieval_index_cache').glob('*.json')]
        if not any(all(item.get(key) == value for key, value in expected.items()) for item in caches):
            raise ValueError('Condition-resume bank has no matching prepared vector cache')


def _prepare_group(args, root, plan, group, row, verified_files, conditions):
    group = driver.resolve_history_binding(args, root, plan, group, required=True)
    folder = root / 'groups' / group['id']
    ready = driver.verify_banks(folder)
    _verify_frozen_memory(folder, row, ready)
    destination, template = driver.evaluation_location(args, folder, save=False)
    source = Path(row['source_evaluation_directory']).resolve()
    if not source.is_relative_to((folder / 'evaluations').resolve()) or source == destination.resolve():
        raise ValueError('Condition resume requires a separate source evaluation variant in the same group')
    if not source.exists():
        if row.get('status') != 'not_started':
            raise ValueError('Missing source evaluation must explicitly be marked not_started')
        return {'group': group, 'folder': folder, 'destination': destination,
                'source': source, 'imports': [], 'not_started': True}
    if row.get('status') == 'not_started':
        if (any((source / name).exists() for name in ('deferred_529_state.json', 'test_results.json'))
                or any((source / 'shopping').glob('*.json'))):
            raise ValueError('Source evaluation marked not_started contains prior shopping results')
        trials, tasks_sha = driver.evaluation_trials(args, group, folder, save=False, task_folder=destination)
        for name in ('evaluation_settings.json', 'targeted_tasks.json'):
            path = source / name
            if path.exists() and path.resolve() not in verified_files:
                raise ValueError('Missing frozen source file digest: ' + name)
        if (source / 'evaluation_settings.json').exists():
            settings = read(source / 'evaluation_settings.json')
            if (settings['template'] != template
                    or _semantics(settings['protocol']) != _semantics(driver.evaluation_protocol(args))):
                raise ValueError('Unstarted source template or query treatment changed')
        if (source / 'targeted_tasks.json').exists():
            saved_tasks = read(source / 'targeted_tasks.json')
            if saved_tasks['trials'] != trials or saved_tasks['tasks_sha256'] != tasks_sha:
                raise ValueError('Unstarted source frozen tasks changed')
        return {'group': group, 'folder': folder, 'destination': destination,
                'source': source, 'imports': [], 'not_started': True}
    required_names = ('evaluation_settings.json', 'retrieval_protocol.json', 'targeted_tasks.json',
                      'deferred_529_state.json', 'test_results.json')
    for name in required_names:
        if (source / name).resolve() not in verified_files:
            raise ValueError('Missing frozen source file digest: ' + name)
    source_settings = read(source / 'evaluation_settings.json')
    retrieval = read(source / 'retrieval_protocol.json')
    protocol = driver.evaluation_protocol(args)
    if source_settings['template'] != template or source_settings['protocol'] != retrieval['protocol']:
        raise ValueError('Source template or protocol disagrees with its saved retrieval metadata')
    if _semantics(retrieval['protocol']) != _semantics(protocol):
        raise ValueError('Condition-resume retrieval or query treatment changed')
    trials, tasks_sha = driver.evaluation_trials(args, group, folder, save=False, task_folder=destination)
    tasks = read(source / 'targeted_tasks.json')
    if (tasks['trials'] != trials or tasks['tasks_sha256'] != tasks_sha
            or tasks_sha != retrieval['tasks_sha256']):
        raise ValueError('Condition-resume frozen tasks changed')
    _verify_encoder(root, folder, group['id'], retrieval, ready, conditions)
    original = _read_only_ledger(source / 'deferred_529_state.json', trials)
    expected_bindings = {'history_binding_sha256': group['history_binding_sha256'],
                         'tasks_sha256': tasks_sha,
                         'retrieval_protocol_sha256': fingerprint(retrieval['protocol']),
                         'encoder_sha256': retrieval['encoder_sha256'],
                         'template_sha256': template['sha256'], 'template_level': template['selected_level']}
    if original.identity['bindings'] != expected_bindings:
        raise ValueError('Source ledger experiment bindings changed')
    formal = read(source / 'test_results.json')
    imports = []
    for index, slot in enumerate(original.state['slots']):
        item = trials[index]
        for condition in conditions:
            report = slot['results'].get(condition)
            if report is None or report.get('error') or report.get('deferred_after_529'):
                continue
            expected = dict(expected_bindings, query_instruction_sha256=fingerprint(item['instruction']),
                            query_image_sha256=driver.digest(group['query_image']))
            if any(report.get(key) != value for key, value in expected.items()):
                raise ValueError('Inherited report belongs to another query or protocol')
            if report not in formal.get(condition, []):
                raise ValueError('Committed source report is absent from its saved formal snapshot')
            trajectory_path = source / 'shopping' / f'{index + 1:02d}_{condition}.json'
            if trajectory_path.resolve() not in verified_files:
                raise ValueError('Inherited trajectory lacks its frozen source digest')
            trajectory = read(trajectory_path)
            attempts = [event for event in original.state['attempts']
                        if event['candidate_index'] == index + 1 and event['condition'] == condition
                        and event['report'] == report and not event['is_529']]
            if (slot['trajectories'].get(condition) != trajectory
                    or not any(event['trajectory'] == trajectory for event in attempts)):
                raise ValueError('Inherited trajectory differs from its committed ledger attempt')
            origin = {'source_evaluation_directory': str(source),
                      'source_report_sha256': fingerprint(report),
                      'source_trajectory': str(trajectory_path),
                      'source_trajectory_sha256': driver.digest(trajectory_path),
                      'source_protocol': retrieval['protocol'],
                      'source_evaluation_code_sha256': retrieval['protocol'].get('evaluation_code_sha256'),
                      'original_bindings': expected_bindings,
                      'migration': 'same_treatment_selected_condition_only_protocol_identity_migration'}
            derived = dict(copy.deepcopy(report), retrieval_protocol_sha256=fingerprint(protocol),
                           inherited_result=origin)
            imports.append({'report': derived, 'trajectory': trajectory, 'origin': origin,
                            'source_report': report, 'index': index + 1, 'condition': condition})
    bindings = dict(expected_bindings, retrieval_protocol_sha256=fingerprint(protocol))
    return {'group': group, 'folder': folder, 'destination': destination, 'source': source,
            'imports': imports, 'trials': trials, 'bindings': bindings, 'template': template,
            'retrieval': dict(retrieval, protocol=protocol), 'not_started': False}


def seed_selected_condition(args, source_manifest):
    """Create an auditable separate ledger, keeping every valid saved outcome.

    ``source_manifest`` binds source files and each group's five frozen-memory
    digests. Re-entry verifies imported reports in the existing destination and
    never replaces its later shopping results.
    """
    root, plan, groups = driver.load_plan(args)
    root = Path(root).resolve()
    if Path(source_manifest.get('root', root)).resolve() != root:
        raise ValueError('Condition-resume manifest belongs to another experiment root')
    conditions = tuple(driver.evaluation_conditions(args))
    if len(conditions) != 1 or getattr(args, 'reserve_tests', 0):
        raise ValueError('Condition inheritance supports one selected condition and no reserve tasks')
    entries = _file_entries(source_manifest.get('files', []))
    rows = _rows(source_manifest)
    for row in rows:
        entries += _file_entries(row.get('files', []))
    verified_files = {}
    for path, sha in entries:
        if not path.is_relative_to(root) or driver.digest(path) != sha:
            raise ValueError('Frozen condition-resume source file changed: ' + str(path))
        verified_files[path] = sha
    by_group = {row['group']: row for row in rows}
    if len(by_group) != len(rows) or set(by_group) != {group['id'] for group in groups}:
        raise ValueError('Condition resume manifest does not cover exactly the selected groups')
    # Finish all read-only checks before publishing a new evaluation variant.
    prepared = [_prepare_group(args, root, plan, group, by_group[group['id']], verified_files, conditions)
                for group in groups]
    audits = []
    implementation_sha256 = driver.digest(Path(__file__))
    for record in prepared:
        group, destination, imports = record['group'], record['destination'], record['imports']
        audit_path = destination / 'inherited_condition_results.json'
        audit = {'schema_version': 1, 'group': group['id'], 'conditions': list(conditions),
                 'implementation_sha256': implementation_sha256,
                 'source_manifest_sha256': fingerprint(source_manifest),
                 'source_evaluation_directory': str(record['source']),
                 'source_not_started': record['not_started'], 'inherited_count': len(imports),
                 'inherited_task_ids': [row['report']['task_id'] for row in imports],
                 'outcomes': [row['origin'] for row in imports],
                 'source_artifacts_rewritten': False, 'remote_policy_calls': 0}
        if audit_path.exists():
            if read(audit_path) != audit:
                raise ValueError('Previously inherited condition-results audit changed')
            if not record['not_started']:
                existing = _read_only_ledger(destination / 'deferred_529_state.json', record['trials'])
                if existing.identity['bindings'] != record['bindings'] or existing.conditions != list(conditions):
                    raise ValueError('Existing inherited ledger changed its experiment bindings')
                for item in imports:
                    slot = existing.state['slots'][item['index'] - 1]
                    if slot['results'].get(item['condition']) != item['report']:
                        raise ValueError('An inherited report was replaced in the destination')
            audits.append(audit)
            continue
        if (destination / 'deferred_529_state.json').exists():
            raise ValueError('Destination ledger exists without an audited inheritance record')
        driver.evaluation_location(args, record['folder'], save=True)
        if not record['not_started']:
            driver.evaluation_trials(args, group, record['folder'], task_folder=destination)
            write(destination / 'retrieval_protocol.json', record['retrieval'])
            ledger = Deferred529Evaluation(destination / 'deferred_529_state.json', record['trials'],
                conditions, record['bindings'], initial_results=[{'report': row['report'],
                'trajectory': row['trajectory']} for row in imports], primary_count=len(record['trials']),
                condition_orders=[list(conditions) for _ in record['trials']])
            for item in imports:
                canonical = destination / 'shopping' / f"{item['index']:02d}_{item['condition']}.json"
                canonical.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item['origin']['source_trajectory'], canonical)
                archive = destination / 'inherited_source' / f"{item['index']:02d}_{item['condition']}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item['origin']['source_trajectory'], archive.with_suffix('.trajectory.json'))
                write(archive.with_suffix('.report.json'), item['source_report'])
            driver._save_deferred_trajectories(ledger, destination)
            write(destination / 'test_results.json', ledger.final_results())
        write(audit_path, audit)
        audits.append(audit)
    result = {'schema_version': 1, 'conditions': list(conditions),
              'implementation_sha256': implementation_sha256,
              'source_manifest_sha256': fingerprint(source_manifest), 'groups': audits,
              'inherited_total': sum(row['inherited_count'] for row in audits),
              'remote_policy_calls': 0, 'source_artifacts_rewritten': False}
    return result
