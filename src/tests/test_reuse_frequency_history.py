"""Real-image/history integrity and restart boundaries for frequency reuse."""
import copy
from pathlib import Path
import shutil

import pytest

from src.environment.local_webshop import fingerprint, read, write
from src.experiments import nine_group_experiment as driver
from src.experiments.reuse_frequency_history import (
    prepare_reused_history, verify_reused_artifacts,
)
from src.tests.test_attacker_history_driver import case, picture


REAL_EXECUTION_SETTINGS = driver.execution_settings
REAL_ATTACK_SETTINGS = driver.attack_settings
REAL_FREEZE_SETTINGS = driver.freeze_settings


@pytest.fixture
def reusable(case, monkeypatch):
    source = case.root
    assets = source / 'assets'
    assets.mkdir()
    for original, name, path_key, record_key in (
            (case.prepared_a, 'prepared_a.png', 'source_image', 'source_image_record'),
            (case.b, 'b.png', 'query_image', 'query_image_record')):
        new_path = assets / name
        shutil.copy2(original, new_path)
        case.group[path_key] = str(new_path)
        case.group[record_key]['absolute_path'] = str(new_path)
    case.plan['bundle_id'] = fingerprint({key: value for key, value in case.plan.items() if key != 'bundle_id'})
    write(source / 'manifest.json', case.plan)
    write(source / 'task_plan.json', case.plan)
    write(source / 'run_settings.json', {'original_method': 'legacy_pixel', 'original_steps': 1000})
    binding = driver.save_history_binding(case.args, source, case.plan, case.group, case.state)
    resolved = driver.resolve_history_binding(case.args, source, case.plan, case.group, required=True)
    old_attack = picture(case.folder / 'attack/adversarial.png', (202, 0, 0))
    old_ready = {'image': str(old_attack), 'sha256': driver.digest(old_attack),
                 'history_binding_sha256': binding['binding_sha256']}
    write(case.folder / 'attack_ready.json', old_ready)
    driver.materialize_banks(case.folder, case.state['normal'], case.state['designated'], old_ready, resolved)
    for directory in ('image_records', 'original_image_cache', 'retrieval_index_cache'):
        (source / directory).mkdir()
    write(source / 'image_records/B_shampoo.json', {'absolute_path': str(assets / 'b.png'),
          'sha256': driver.digest(assets / 'b.png'), 'asin': 'B_shampoo'})
    write(case.folder / 'retrieval_index_cache/stale.json', {'old_pixel_vectors': True})
    evaluation = case.folder / 'evaluations/old_top_one'
    from src.experiments.targeted_queries import make_targeted_trials, VAGUE_DEMAND_MODE
    trials = make_targeted_trials(resolved, count=20, demand_mode=VAGUE_DEMAND_MODE)
    protocol = {'test_mode': 'targeted', 'query_specificity': 'vague', 'memory_top_k': 1,
                'retrieval_encoder': 'clip', 'memory_mode': 'full',
                'memory_key_mode': 'joint', 'memory_query_mode': 'joint', 'index_text': 'full',
                'query_image_mode': 'reference', 'policy_query_image': True}
    write(evaluation / 'evaluation_settings.json', {'protocol': protocol})
    write(evaluation / 'targeted_tasks.json', {'fixed_asin': case.group['b'], 'test_count': 20, 'trials': trials})
    write(evaluation / 'summary.json', {'old_target_purchases': 19})
    args = copy.copy(case.args)
    args.root = source.parent / (source.name + '_frequency_reuse')
    args.group = ['7']
    for name, value in dict(method='frequency_momentum', attack_steps=500,
                            query_specificity='vague', memory_top_k=1,
                            retrieval_encoder='clip', memory_mode='full',
                            memory_key_mode='joint', memory_query_mode='joint', index_text='full',
                            query_image_mode='reference', reserve_tests=0, reference_overrides=None,
                            include_controls=True, policy_query_image=True).items():
        setattr(args, name, value)
    monkeypatch.setattr(driver, 'workspace_output', lambda path: Path(path).resolve())

    def load(options):
        root = Path(options.root).resolve()
        plan = read(root / 'manifest.json')
        return root, plan, [plan['groups'][0]]

    def settings(options, plan):
        return {'plan_sha256': fingerprint(plan), 'method': options.method,
                'steps': options.attack_steps, 'epsilon': options.epsilon_levels}

    def freeze(options, root, plan):
        current = settings(options, plan)
        path = root / 'run_settings.json'
        if path.exists() and read(path) != current:
            raise ValueError('Frozen test settings changed')
        write(path, current)
        return current

    monkeypatch.setattr(driver, 'load_plan', load)
    monkeypatch.setattr(driver, 'execution_settings', settings)
    monkeypatch.setattr(driver, 'freeze_settings', freeze)
    monkeypatch.setattr(driver, 'attack_settings', lambda options, group: {
        'source_sha256': driver.digest(group['source_image']),
        'target_sha256': driver.digest(group['query_image']),
        'steps': options.attack_steps, 'method': options.method, 'epsilon_levels': options.epsilon_levels})
    case.new_args, case.evaluation = args, evaluation
    return case


def materialize_frequency(reusable, color=(203, 0, 0)):
    args = reusable.new_args
    root, plan, groups = driver.load_plan(args)
    group = groups[0]
    folder = root / 'groups' / group['id']
    resolved = driver.resolve_history_binding(args, root, plan, group, required=True)
    image = picture(folder / 'attack/adversarial.png', color)
    ready = {'image': str(image), 'sha256': driver.digest(image),
             'settings': driver.attack_settings(args, resolved),
             'history_binding_sha256': resolved['history_binding_sha256'],
             'metadata': {'attack': {'steps_requested': 500, 'steps_completed': 500, 'best_step': 250}}}
    write(folder / 'attack_ready.json', ready)
    state = read(folder / 'history_state.json')
    driver.materialize_banks(folder, state['normal'], state['designated'], ready, resolved)
    return folder


def test_preparation_preserves_real_history_and_demand_without_old_attacks_or_vectors(reusable):
    before = {path: path.read_bytes() for path in reusable.root.rglob('*') if path.is_file()}
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    root = reusable.new_args.root
    folder = root / 'groups' / reusable.group['id']
    assert provenance['state'] == 'history_prepared'
    assert provenance['selected_groups'] == ['07_shampoo']
    assert provenance['groups'][0]['source_evaluation'] == str(reusable.evaluation)
    assert provenance['groups'][0]['source_memory_ready']['reference_asin'] == 'B_shampoo'
    assert provenance['groups'][0]['source_history_binding_sha256'] != provenance['groups'][0]['new_history_binding_sha256']
    for name in ('history_state.json', 'history_memory.json'):
        assert (folder / name).read_bytes() == (reusable.folder / name).read_bytes()
    assert not (folder / 'attack').exists()
    assert not (folder / 'retrieval_index_cache').exists()
    assert not (folder / 'evaluations').exists()
    assert read(root / 'run_settings.json')['steps'] == 500
    record = read(root / 'image_records/B_shampoo.json')
    assert record['absolute_path'] == str(root / 'assets/b.png')
    assert all(path.read_bytes() == original for path, original in before.items())
    assert verify_reused_artifacts(reusable.new_args, provenance, require_banks=False)['source_files_unchanged']


def test_ssa_reuses_identical_old_b_history_and_twenty_demands(reusable):
    reusable.new_args.method = 'ssa'
    reusable.new_args.frequency_axis_ratio = None
    reusable.new_args.frequency_momentum = None
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    folder = materialize_frequency(reusable)
    verification = verify_reused_artifacts(reusable.new_args, provenance)
    assert read(reusable.new_args.root / 'run_settings.json')['method'] == 'ssa'
    assert (folder / 'history_memory.json').read_bytes() == (reusable.folder / 'history_memory.json').read_bytes()
    assert verification['source_files_unchanged']
    assert verification['groups'][0]['same_twenty_demands']
    assert verification['groups'][0]['unchanged_normal_memories'] == 14


def test_finalized_frequency_changes_only_owned_retrieval_image_and_can_resume(reusable):
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    folder = materialize_frequency(reusable)
    verification = verify_reused_artifacts(reusable.new_args, provenance)
    assert verification['groups'][0]['unchanged_normal_memories'] == 14
    assert verification['groups'][0]['changed_designated_memories'] == 1
    assert verification['groups'][0]['steps_completed'] == 500
    assert verification['groups'][0]['decoded_pixel_audit']['within_budget']
    assert (folder / 'clean_memory.json').read_bytes() == (reusable.folder / 'clean_memory.json').read_bytes()
    before = {path: path.read_bytes() for path in folder.rglob('*') if path.is_file()}
    assert prepare_reused_history(reusable.root, reusable.new_args) == provenance
    assert all(path.read_bytes() == original for path, original in before.items())


def test_unknown_existing_destination_is_never_overwritten(reusable):
    root = reusable.new_args.root
    root.mkdir()
    sentinel = root / 'unrelated.txt'
    sentinel.write_text('keep this file', encoding='utf-8')
    with pytest.raises(ValueError, match='unrecognized'):
        prepare_reused_history(reusable.root, reusable.new_args)
    assert sentinel.read_text(encoding='utf-8') == 'keep this file'
    assert sorted(path.name for path in root.iterdir()) == ['unrelated.txt']


def test_refuses_changed_budget_on_resume(reusable):
    prepare_reused_history(reusable.root, reusable.new_args)
    reusable.new_args.epsilon_levels = 8
    with pytest.raises(ValueError, match='frozen settings changed'):
        prepare_reused_history(reusable.root, reusable.new_args)


@pytest.mark.parametrize('name', ['run_settings.json', 'groups/07_shampoo/history_memory.json',
                                 'groups/07_shampoo/evaluations/old_top_one/targeted_tasks.json'])
def test_source_protection_covers_settings_history_and_demands(reusable, name):
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    path = reusable.root / name
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Original experiment file changed'):
        verify_reused_artifacts(reusable.new_args, provenance, require_banks=False)


def test_rejects_demand_identity_change_before_copying_a_formal_experiment(reusable):
    path = reusable.evaluation / 'targeted_tasks.json'
    tasks = read(path)
    tasks['trials'][0]['goal']['asin'] = 'another-product'
    write(path, tasks)
    with pytest.raises(ValueError, match='original twenty demands'):
        prepare_reused_history(reusable.root, reusable.new_args)


def test_ambiguous_source_evaluation_is_rejected(reusable):
    shutil.copytree(reusable.evaluation, reusable.evaluation.with_name('another_top_one'))
    with pytest.raises(ValueError, match='exactly one'):
        prepare_reused_history(reusable.root, reusable.new_args)
    assert not reusable.new_args.root.exists()


def test_explicit_compliant_source_evaluation_resolves_ambiguity_and_is_frozen(reusable):
    other = reusable.evaluation.with_name('another_top_one')
    shutil.copytree(reusable.evaluation, other)
    before = {path: path.read_bytes() for path in reusable.root.rglob('*') if path.is_file()}
    reusable.new_args.source_evaluation_overrides = {'07_shampoo': str(reusable.evaluation.resolve())}
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    assert provenance['source_evaluation_overrides'] == {'07_shampoo': str(reusable.evaluation.resolve())}
    assert provenance['groups'][0]['source_evaluation'] == str(reusable.evaluation.resolve())
    assert 'groups/07_shampoo/evaluations/old_top_one/targeted_tasks.json' in provenance['source_files_sha256']
    assert all(path.read_bytes() == previous for path, previous in before.items())
    assert prepare_reused_history(reusable.root, reusable.new_args) == provenance
    reusable.new_args.source_evaluation_overrides = {'07_shampoo': str(other.resolve())}
    with pytest.raises(ValueError, match='source, selected groups or frozen settings changed'):
        prepare_reused_history(reusable.root, reusable.new_args)


@pytest.mark.parametrize('invalid', ['relative', 'outside_group', 'noncompliant', 'unknown_group', 'wrong_type', 'wrong_value'])
def test_source_evaluation_override_rejects_unsafe_or_noncompliant_selection_before_copy(reusable, invalid):
    selected = reusable.evaluation.resolve()
    if invalid == 'relative':
        selected = Path('evaluations/old_top_one')
    elif invalid == 'outside_group':
        selected = reusable.root / 'groups/other_group/evaluations/copied_variant'
        shutil.copytree(reusable.evaluation, selected)
    elif invalid == 'noncompliant':
        selected = reusable.evaluation.with_name('wrong_encoder')
        shutil.copytree(reusable.evaluation, selected)
        settings = read(selected / 'evaluation_settings.json')
        settings['protocol']['retrieval_encoder'] = 'gme'
        write(selected / 'evaluation_settings.json', settings)
    if invalid == 'unknown_group':
        reusable.new_args.source_evaluation_overrides = {'01_camera': str(selected)}
    elif invalid == 'wrong_type':
        reusable.new_args.source_evaluation_overrides = [str(selected)]
    elif invalid == 'wrong_value':
        reusable.new_args.source_evaluation_overrides = {'07_shampoo': None}
    else:
        reusable.new_args.source_evaluation_overrides = {'07_shampoo': str(selected)}
    with pytest.raises(ValueError, match='override'):
        prepare_reused_history(reusable.root, reusable.new_args)
    assert not reusable.new_args.root.exists()


def test_saved_frequency_steps_must_be_complete(reusable):
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    folder = materialize_frequency(reusable)
    ready = read(folder / 'attack_ready.json')
    ready['metadata']['attack']['steps_completed'] = 499
    write(folder / 'attack_ready.json', ready)
    with pytest.raises(ValueError, match='metadata'):
        verify_reused_artifacts(reusable.new_args, provenance)


def test_pixel_budget_is_checked_from_actual_saved_image(reusable):
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    materialize_frequency(reusable, color=(220, 0, 0))
    with pytest.raises(ValueError, match='pixel budget'):
        verify_reused_artifacts(reusable.new_args, provenance)


def test_changed_normal_memory_is_rejected_even_if_ready_digest_is_updated(reusable):
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    folder = materialize_frequency(reusable)
    bank = read(folder / 'memory.json')
    bank[0]['Actions'].append('think[changed normal history]')
    write(folder / 'memory.json', bank)
    ready = read(folder / 'memory_ready.json')
    ready['bank_sha256']['adversarial'] = driver.digest(folder / 'memory.json')
    write(folder / 'memory_ready.json', ready)
    with pytest.raises(ValueError, match='historical content'):
        verify_reused_artifacts(reusable.new_args, provenance)


def test_old_reference_image_bytes_remain_bound_on_resume(reusable):
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    picture(reusable.new_args.root / 'assets/b.png', (9, 9, 9))
    with pytest.raises(ValueError, match='digest'):
        verify_reused_artifacts(reusable.new_args, provenance, require_banks=False)


def test_recognized_partial_clone_resumes_without_overwriting_retained_files(reusable, monkeypatch):
    original_save = driver.save_history_binding
    calls = []

    def interrupt_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('simulated interruption after cloning')
        return original_save(*args, **kwargs)

    monkeypatch.setattr(driver, 'save_history_binding', interrupt_once)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        prepare_reused_history(reusable.root, reusable.new_args)
    history = reusable.new_args.root / 'groups/07_shampoo/history_memory.json'
    retained = history.read_bytes()
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    assert provenance['state'] == 'history_prepared' and history.read_bytes() == retained


def test_source_and_destination_must_be_separate(reusable):
    reusable.new_args.root = reusable.root / 'nested'
    with pytest.raises(ValueError, match='non-nested'):
        prepare_reused_history(reusable.root, reusable.new_args)


def test_edited_provenance_cannot_redirect_the_verified_source(reusable):
    provenance = prepare_reused_history(reusable.root, reusable.new_args)
    provenance['source_root'] = str(reusable.new_args.root)
    with pytest.raises(ValueError, match='provenance was edited'):
        verify_reused_artifacts(reusable.new_args, provenance, require_banks=False)


def test_rejects_a_different_real_retrieval_protocol(reusable):
    reusable.new_args.retrieval_encoder = 'gme'
    with pytest.raises(ValueError, match='retrieval_encoder'):
        prepare_reused_history(reusable.root, reusable.new_args)
    assert not reusable.new_args.root.exists()


def test_real_execution_settings_can_digest_images_before_destination_exists(reusable, monkeypatch):
    """The real freezer reads planned source/target files, rather than fake hashes."""
    args = reusable.new_args
    model_config = reusable.root / 'fixture_model.json'
    optimization = reusable.root / 'fixture_optimization.json'
    write(model_config, {'model': {'provider': 'minimax', 'model': 'fixture-model',
          'base_url': 'https://example.org/v1', 'api_key': 'not-a-real-key'}})
    write(optimization, {'frequency': {'initial_axis_ratio': 0.25}})
    args.llm_config = model_config
    args.optimization_config = optimization
    for name, value in dict(num_steps=20, memory_prompt_budget=128000,
                            policy_max_tokens=2048, history_max_attempts=3,
                            normal_retries=2, frequency_axis_ratio=0.25,
                            frequency_momentum=1.0).items():
        setattr(args, name, value)
    monkeypatch.setattr(driver, 'execution_settings', REAL_EXECUTION_SETTINGS)
    monkeypatch.setattr(driver, 'attack_settings', REAL_ATTACK_SETTINGS)
    monkeypatch.setattr(driver, 'freeze_settings', REAL_FREEZE_SETTINGS)
    assert not args.root.exists()
    provenance = prepare_reused_history(reusable.root, args)
    plan = read(args.root / 'manifest.json')
    actual = REAL_EXECUTION_SETTINGS(args, plan)
    assert provenance['new_execution_settings_sha256'] == fingerprint(actual)
    assert read(args.root / 'run_settings.json') == actual
    assert actual['attack_method'] == 'frequency_momentum' and actual['attack_steps'] == 500
    assert verify_reused_artifacts(args, provenance, require_banks=False)['source_files_unchanged']
