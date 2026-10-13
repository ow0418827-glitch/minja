"""Reuse immutable real purchase histories for a separate frequency experiment.

This preparation module neither collects histories nor calls the policy model.
The original history documents retain their exact bytes and image references;
only the independent plan/bindings and the new attack artifacts are generated.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from src.environment.local_webshop import fingerprint, read, write
from src.experiments import nine_group_experiment as driver


PROVENANCE_FILE = 'history_reuse_provenance.json'
DEMAND_FIELDS = ('task_id', 'session', 'instruction', 'goal', 'asin',
                 'expected_asin', 'goal_index', 'query_image_sha256')
HISTORY_FILES = ('history_state.json', 'history_memory.json')
PROTECTED_GROUP_FILES = (*HISTORY_FILES, 'history_binding.json', 'clean_memory.json',
                         'memory.json', 'memory_ready.json', 'attack_ready.json')
IGNORED_CHANGED_FIELDS = {'RetrievalImage', 'ImageSubstitution'}


def _validate_options(args):
    expected = {'source_history_mode': 'attacker', 'history_product_mode': 'observed',
                'test_mode': 'targeted', 'query_specificity': 'vague',
                'retrieval_encoder': 'clip', 'memory_top_k': 1, 'memory_mode': 'full',
                'memory_key_mode': 'joint',
                'memory_query_mode': 'joint',
                'query_image_mode': 'reference', 'policy_query_image': True,
                'include_controls': True, 'reserve_tests': 0}
    for key, value in expected.items():
        if getattr(args, key, None) != value:
            raise ValueError('History reuse requires the original evaluation setting: ' + key)
    from src.experiments.visual_memory_experiment import retrieval_options
    retrieval_options(args)
    precomputed = (getattr(args, 'method', None) == 'scar' and getattr(args, 'attack_steps', None) == 0
                   and isinstance(getattr(args, 'precomputed_manifest', None), dict))
    if not precomputed and (getattr(args, 'method', None) not in ('frequency_momentum', 'ssa')
            or getattr(args, 'attack_steps', 0) < 1):
        raise ValueError('History reuse requires a positive frequency_momentum or ssa attack budget')
    if getattr(args, 'reference_overrides', None):
        raise ValueError('Old-reference history reuse cannot also replace the B product')


def _relocate(value, source, destination):
    if isinstance(value, dict):
        return {key: _relocate(item, source, destination) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate(item, source, destination) for item in value]
    if isinstance(value, str):
        for old, new in ((str(source), str(destination)),
                         (source.as_posix(), destination.as_posix())):
            candidate = value.casefold()
            prefix = old.casefold()
            if candidate == prefix or candidate.startswith(prefix + '/') or candidate.startswith(prefix + '\\'):
                return new + value[len(old):]
    return value


def _source_evaluation(source, group, *, override=None):
    matches = []
    required = {'test_mode': 'targeted', 'query_specificity': 'vague',
                'retrieval_encoder': 'clip', 'memory_top_k': 1, 'memory_mode': 'full',
                'memory_key_mode': 'joint',
                'memory_query_mode': 'joint', 'index_text': 'full',
                'query_image_mode': 'reference', 'policy_query_image': True}
    for path in sorted((source / 'groups' / group['id'] / 'evaluations').glob('*/evaluation_settings.json')):
        protocol = read(path).get('protocol', {})
        if any(protocol.get(key) != value for key, value in required.items()):
            continue
        if protocol.get('reserve_tests', 0) != 0:
            continue
        tasks_path = path.parent / 'targeted_tasks.json'
        if not tasks_path.is_file():
            continue
        tasks = read(tasks_path)
        if (tasks.get('fixed_asin') == group['b'] and tasks.get('test_count') == 20
                and len(tasks.get('trials', [])) == 20):
            matches.append(path.parent)
    if override is not None:
        selected = Path(override)
        evaluation_root = (source / 'groups' / group['id'] / 'evaluations').resolve()
        if (not selected.is_absolute() or not selected.resolve().is_relative_to(evaluation_root)
                or selected.resolve() not in {path.resolve() for path in matches}):
            raise ValueError('Source evaluation override must select a compliant variant inside the source group evaluations')
        return selected.resolve()
    if len(matches) != 1:
        raise ValueError('Expected exactly one original vague top-one twenty-demand evaluation: ' + group['id'])
    return matches[0]


def _source_fingerprints(source, groups, evaluations):
    paths = {source / name for name in ('manifest.json', 'task_plan.json', 'run_settings.json')}
    for name in ('assets', 'image_records', 'original_image_cache'):
        paths.update(path for path in (source / name).rglob('*') if path.is_file())
    for group in groups:
        folder = source / 'groups' / group['id']
        paths.update(folder / name for name in PROTECTED_GROUP_FILES)
        for name in ('memory_images', 'images'):
            paths.update(path for path in (folder / name).rglob('*') if path.is_file())
        paths.update(evaluations[group['id']] / name
                     for name in ('evaluation_settings.json', 'targeted_tasks.json'))
    return {path.relative_to(source).as_posix(): driver.digest(path) for path in sorted(paths)}


def _verify_source_files(provenance):
    source = Path(provenance['source_root']).resolve()
    for name, expected in provenance['source_files_sha256'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or driver.digest(path) != expected:
            raise ValueError('Original experiment file changed or disappeared: ' + name)


def _copy_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or driver.digest(source) != driver.digest(destination):
            raise ValueError('Existing cloned resource differs from its source: ' + str(destination))
        return
    shutil.copy2(source, destination)


def _write_exact(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing cloned plan or resource differs: ' + str(path))
        return
    write(path, value)


def _copy_resources(source, destination):
    for name in ('assets', 'image_records', 'original_image_cache'):
        for path in sorted((source / name).rglob('*')):
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            if name == 'image_records' and path.suffix.lower() == '.json':
                _write_exact(target, _relocate(read(path), source, destination))
            else:
                _copy_exact(path, target)


def _verify_demands(source_evaluation, resolved):
    from src.experiments.targeted_queries import make_targeted_trials, VAGUE_DEMAND_MODE
    original = read(Path(source_evaluation) / 'targeted_tasks.json')['trials']
    current = make_targeted_trials(resolved, count=20, demand_mode=VAGUE_DEMAND_MODE)
    if len(original) != 20 or any(
            {key: before.get(key) for key in DEMAND_FIELDS}
            != {key: after.get(key) for key in DEMAND_FIELDS}
            for before, after in zip(original, current)):
        raise ValueError('Generated old-reference demands differ from the original twenty demands')
    return fingerprint([{key: task.get(key) for key in DEMAND_FIELDS} for task in current])


def _seal(provenance):
    provenance['provenance_sha256'] = fingerprint({key: value for key, value in provenance.items()
                                                 if key != 'provenance_sha256'})
    return provenance


def _verify_seal(provenance):
    if provenance.get('provenance_sha256') != fingerprint(
            {key: value for key, value in provenance.items() if key != 'provenance_sha256'}):
        raise ValueError('History-reuse provenance was edited')


def prepare_reused_history(source_root, args):
    """Clone original histories and freeze new attack settings without model calls.

    A recognized interrupted clone can resume; an unrelated existing destination
    is never overwritten. Model optimization/finalization/evaluation are handled
    by the caller after this preparation has completed.
    """
    _validate_options(args)
    source = driver.workspace_output(source_root)
    destination = driver.workspace_output(args.root)
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('History reuse requires separate, non-nested experiment directories')
    source_args = copy.copy(args)
    source_args.root = source
    _, original_plan, original_groups = driver.load_plan(source_args)
    overrides = getattr(args, 'source_evaluation_overrides', {})
    if (not isinstance(overrides, dict)
            or set(overrides) - {group['id'] for group in original_groups}
            or any(not isinstance(value, (str, Path)) for value in overrides.values())):
        raise ValueError('Source evaluation overrides must be a mapping for the selected source groups')
    evaluations = {group['id']: _source_evaluation(source, group, override=overrides.get(group['id']))
                   for group in original_groups}
    original_bindings = {}
    original_ready = {}
    for group in original_groups:
        driver.verify_collected_history(source_args, source, original_plan, group)
        original_bindings[group['id']] = read(source / 'groups' / group['id'] / 'history_binding.json')
        original_ready[group['id']] = driver.verify_banks(source / 'groups' / group['id'])
    plan = _relocate(copy.deepcopy(original_plan), source, destination)
    plan['bundle_id'] = fingerprint({key: value for key, value in plan.items() if key != 'bundle_id'})
    # Compute image-dependent settings against existing source bytes first.
    # Every relocated image is copied exactly and rechecked below before freeze.
    settings = driver.execution_settings(args, original_plan)
    settings['plan_sha256'] = fingerprint(plan)
    expected = {'version': 1, 'source_root': str(source), 'destination_root': str(destination),
                'selected_groups': [group['id'] for group in original_groups],
                'source_files_sha256': _source_fingerprints(source, original_groups, evaluations),
                'new_plan_sha256': fingerprint(plan), 'new_execution_settings_sha256': fingerprint(settings),
                'same_twenty_demand_fields': list(DEMAND_FIELDS),
                'remote_model_calls_during_preparation': 0}
    if overrides:
        expected['source_evaluation_overrides'] = {group_id: str(evaluations[group_id])
                                                   for group_id in sorted(overrides)}
    provenance_path = destination / PROVENANCE_FILE
    if destination.exists():
        if not provenance_path.is_file():
            raise ValueError('Refusing to overwrite an unrecognized existing experiment directory')
        provenance = read(provenance_path)
        _verify_seal(provenance)
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError('History-reuse source, selected groups or frozen settings changed')
        _verify_source_files(provenance)
        if provenance.get('state') == 'history_prepared':
            verify_reused_artifacts(args, provenance, require_banks=False)
            driver.freeze_settings(args, destination, plan)
            return provenance
        if provenance.get('state') != 'cloning':
            raise ValueError('Unknown history-reuse preparation state')
    else:
        destination.mkdir(parents=True)
        provenance = _seal(dict(expected, state='cloning', groups=[]))
        write(provenance_path, provenance)
    _copy_resources(source, destination)
    for group in original_groups:
        old_folder = source / 'groups' / group['id']
        new_folder = destination / 'groups' / group['id']
        for path in sorted((old_folder / 'memory_images').rglob('*')):
            if path.is_file():
                _copy_exact(path, new_folder / path.relative_to(old_folder))
        for name in HISTORY_FILES:
            _copy_exact(old_folder / name, new_folder / name)
    _write_exact(destination / 'manifest.json', plan)
    _write_exact(destination / 'task_plan.json', _relocate(read(source / 'task_plan.json'), source, destination))
    _, checked_plan, selected_groups = driver.load_plan(args)
    evidence = []
    for group in selected_groups:
        folder = destination / 'groups' / group['id']
        _write_exact(folder / 'tasks.json', group)
        state = read(folder / 'history_state.json')
        binding = driver.save_history_binding(args, destination, checked_plan, group, state)
        original = original_bindings[group['id']]
        for key in ('all_history_sha256', 'source_history_sha256', 'source_image_sha256',
                    'reference_query_sha256', 'source_memory_id', 'actual_a', 'reference_b'):
            if binding[key] != original[key]:
                raise ValueError('Cloned history or old B reference changed: ' + key)
        resolved = driver.resolve_history_binding(args, destination, checked_plan, group, required=True)
        evidence.append({'group': group['id'], 'source_evaluation': str(evaluations[group['id']]),
                         'source_history_binding_sha256': original['binding_sha256'],
                         'new_history_binding_sha256': binding['binding_sha256'],
                         'source_memory_ready': original_ready[group['id']],
                         'all_history_sha256': binding['all_history_sha256'],
                         'source_history_sha256': binding['source_history_sha256'],
                         'same_twenty_demands_sha256': _verify_demands(evaluations[group['id']], resolved)})
    provenance.update(groups=evidence, state='history_prepared', history_files_byte_identical=True)
    _seal(provenance)
    verify_reused_artifacts(args, provenance, require_banks=False)
    driver.freeze_settings(args, destination, checked_plan)
    write(provenance_path, provenance)
    return provenance


def verify_reused_artifacts(args, provenance, require_banks=True, *, evaluation_only=False):
    """Verify immutable source/history and, after finalization, one-image substitution."""
    _validate_options(args)
    _verify_seal(provenance)
    _verify_source_files(provenance)
    source = Path(provenance['source_root']).resolve()
    root, plan, groups = driver.load_plan(args)
    settings_sha = (fingerprint(read(root / 'run_settings.json')) if evaluation_only
                    else fingerprint(driver.execution_settings(args, plan)))
    if (str(root) != provenance['destination_root']
            or [group['id'] for group in groups] != provenance['selected_groups']
            or fingerprint(plan) != provenance['new_plan_sha256']
            or settings_sha != provenance['new_execution_settings_sha256']):
        raise ValueError('Reused experiment plan or settings changed')
    if evaluation_only:
        # Retain the sealed preparation settings. The dedicated evaluation audit
        # permits only evaluation code/presentation changes and checks all banks.
        driver.verify_frozen_history_for_evaluation(args, root, plan, groups)
    saved = {item['group']: item for item in provenance['groups']}
    audits = []
    for group in groups:
        folder, old_folder = root / 'groups' / group['id'], source / 'groups' / group['id']
        for name in HISTORY_FILES:
            if driver.digest(folder / name) != driver.digest(old_folder / name):
                raise ValueError('Cloned history bytes changed: ' + group['id'] + '/' + name)
        driver.verify_collected_history(args, root, plan, group)
        resolved = driver.resolve_history_binding(args, root, plan, group, required=True)
        expected = saved[group['id']]
        if (resolved['history_binding_sha256'] != expected['new_history_binding_sha256']
                or _verify_demands(expected['source_evaluation'], resolved)
                != expected['same_twenty_demands_sha256']):
            raise ValueError('Cloned history binding or original demand identity changed')
        audit = {'group': group['id'], 'history_files_byte_identical': True,
                 'same_twenty_demands': True, 'reference_asin': group['b'],
                 'query_image_sha256': resolved['reference_query_sha256']}
        if require_banks:
            from src.experiments.visual_memory_experiment import pixel_audit
            ready = driver.verify_banks(folder)
            attack = read(folder / 'attack_ready.json')
            optimization = attack.get('metadata', {}).get('attack', {})
            if (attack['settings'] != driver.attack_settings(args, resolved)
                    or attack.get('history_binding_sha256') != resolved['history_binding_sha256']
                    or ready['history_binding_sha256'] != resolved['history_binding_sha256']
                    or driver.digest(attack['image']) != attack['sha256']
                    or attack['sha256'] != ready['attack_sha256']
                    or not isinstance(optimization.get('best_step'), int)
                    or not 0 <= optimization['best_step'] <= optimization.get('steps_completed', -1)):
                raise ValueError('Frequency attack metadata, image or history binding changed')
            from src.vision.attack_completion import validate_attack_completion
            try:
                validate_attack_completion(attack.get('metadata', {}), args.attack_steps, attack['settings'])
            except ValueError as error:
                raise ValueError('Frequency attack metadata is incomplete or invalid: ' + str(error)) from error
            pixels = pixel_audit(resolved['source_image'], attack['image'], args.epsilon_levels)
            pixels['within_budget'] = pixels['max_pixel_difference'] <= args.epsilon_levels
            if driver.digest(folder / 'clean_memory.json') != driver.digest(old_folder / 'clean_memory.json'):
                raise ValueError('Cloned clean memory differs from the original clean bank')
            original, current = read(old_folder / 'memory.json'), read(folder / 'memory.json')
            if len(original) != len(current) or len(current) != 15:
                raise ValueError('Reused memory bank must retain exactly fifteen histories')
            changed = []
            for before, after in zip(original, current):
                if ({key: value for key, value in before.items() if key not in IGNORED_CHANGED_FIELDS}
                        != {key: value for key, value in after.items() if key not in IGNORED_CHANGED_FIELDS}):
                    raise ValueError('Frequency substitution changed original historical facts')
                if before != after:
                    changed.append(after['Id'])
                if before['Id'] != ready['source_memory_id'] and before != after:
                    raise ValueError('Frequency substitution changed a normal memory')
            if changed != [ready['source_memory_id']]:
                raise ValueError('Exactly the designated A retrieval image must change')
            audit.update(clean_bank_byte_identical=True, unchanged_normal_memories=14,
                         changed_designated_memories=1, changed_memory_id=ready['source_memory_id'],
                         best_step=optimization['best_step'], steps_completed=optimization['steps_completed'],
                         decoded_pixel_audit=pixels)
        audits.append(audit)
    return {'source_files_unchanged': True, 'groups': audits}
