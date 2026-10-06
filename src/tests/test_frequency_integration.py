"""Optional frequency selection preserves frozen experiment contracts."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.environment.local_webshop import read, write
from src.experiments import local_experiment as local
from src.experiments import nine_group_experiment as nine
from src.experiments import visual_memory_experiment as visual
from src.tests.test_nine_group_experiment import group, raster
from src.tests.test_visual_retrieval import Encoder, PAIR, experiment
from src.vision import frequency_attack


FREQUENCY_KEYS = {'frequency_axis_ratio', 'frequency_momentum', 'frequency_implementation_sha256'}
LEGACY_ATTACK_KEYS = {'source_sha256', 'target_sha256', 'steps', 'method', 'epsilon_levels',
                      'optimization_config_sha256'}


@pytest.fixture
def attack_fixture(tmp_path, monkeypatch):
    config = tmp_path / 'optimization.json'
    write(config, {'frequency': {'initial_axis_ratio': 0.3}})
    selected = group()
    selected['source_image'] = str(raster(tmp_path / 'a.png', (100, 0, 0)))
    selected['query_image'] = str(raster(tmp_path / 'b.png', (0, 0, 200)))
    args = SimpleNamespace(method='frequency_momentum', attack_steps=2, epsilon_levels=16,
        optimization_config=config, frequency_axis_ratio=None, frequency_momentum=None, device='cpu')
    monkeypatch.setattr(nine, 'load_plan', lambda _: (tmp_path, {}, [selected]))
    monkeypatch.setattr(nine, 'resolve_history_binding', lambda *unused, **kwargs: selected)
    calls = []

    def optimize(source, target, output, **kwargs):
        calls.append((source, target, output, kwargs))
        return Path(source), {'attack': {'steps_requested': 2, 'steps_completed': 2}}

    monkeypatch.setattr(frequency_attack, 'generate_attack', optimize)
    return args, selected, tmp_path / 'groups' / selected['id'], calls


@pytest.mark.parametrize('method', ['legacy_pixel', 'fair_pixel', 'frequency'])
def test_old_attack_settings_and_calls_have_no_new_parameters(attack_fixture, method):
    args, selected, folder, calls = attack_fixture
    args.method = method
    settings = nine.attack_settings(args, selected)
    assert set(settings) == LEGACY_ATTACK_KEYS
    nine.attack(args)
    assert calls[0][3] == {'optimization_config': args.optimization_config, 'method': method,
                          'steps': 2, 'epsilon_levels': 16, 'device': 'cpu'}
    assert read(folder / 'attack_ready.json')['settings'] == settings
    nine.attack(args)
    assert len(calls) == 1


def test_frequency_settings_freeze_effective_defaults_and_propagate(attack_fixture):
    args, selected, folder, calls = attack_fixture
    settings = nine.attack_settings(args, selected)
    assert set(settings) == LEGACY_ATTACK_KEYS | FREQUENCY_KEYS
    assert settings['frequency_axis_ratio'] == 0.3
    assert settings['frequency_momentum'] == 1.0
    assert settings['frequency_implementation_sha256'] == nine.digest(frequency_attack.__file__)
    nine.attack(args)
    assert calls[0][3]['frequency_axis_ratio'] is None
    assert calls[0][3]['frequency_momentum'] is None
    assert read(folder / 'attack_ready.json')['settings'] == settings
    nine.attack(args)
    assert len(calls) == 1


@pytest.mark.parametrize('field,value', [('frequency_axis_ratio', 0.5), ('frequency_momentum', 0.5),
                                        ('method', 'frequency')])
def test_frozen_attack_refuses_frequency_or_method_changes(attack_fixture, field, value):
    args, _, folder, calls = attack_fixture
    nine.attack(args)
    before = (folder / 'attack_ready.json').read_bytes()
    setattr(args, field, value)
    with pytest.raises(ValueError, match='Frozen attack changed'):
        nine.attack(args)
    assert len(calls) == 1
    assert (folder / 'attack_ready.json').read_bytes() == before


def test_frozen_attack_refuses_changed_frequency_implementation(attack_fixture, monkeypatch):
    args, _, folder, calls = attack_fixture
    nine.attack(args)
    original = frequency_attack.frequency_settings
    monkeypatch.setattr(frequency_attack, 'frequency_settings', lambda *a, **kw:
        {**original(*a, **kw), 'frequency_implementation_sha256': 'changed-implementation'})
    with pytest.raises(ValueError, match='Frozen attack changed'):
        nine.attack(args)
    assert len(calls) == 1


def test_execution_settings_only_add_frequency_fields_for_new_method(attack_fixture, monkeypatch):
    args, selected, _, _ = attack_fixture
    args.llm_config = 'unused'
    args.num_steps, args.memory_top_k, args.memory_mode = 20, 1, 'full'
    args.memory_prompt_budget, args.policy_max_tokens = 128000, 2048
    args.history_max_attempts, args.normal_retries, args.catalog = 3, 2, Path('.')
    monkeypatch.setattr(nine, 'load_model_config', lambda _: SimpleNamespace(public=lambda: {'model': 'mock'}))
    plan = {'protocol': nine.VICTIM_PROTOCOL, 'groups': [selected]}
    current = nine.execution_settings(args, plan)
    assert all(current[key] == nine.attack_settings(args, selected)[key] for key in FREQUENCY_KEYS)
    args.method = 'legacy_pixel'
    old = nine.execution_settings(args, plan)
    assert set(current) == set(old) | FREQUENCY_KEYS
    assert current['code'] == old['code']


@pytest.mark.parametrize('method', ['legacy_pixel', 'frequency_momentum'])
def test_single_pair_prepare_passes_selected_method_and_parameters(tmp_path, monkeypatch, experiment, method):
    paths, original_output, _, _, _ = experiment
    config = tmp_path / 'pair.json'
    write(config, PAIR)
    args = SimpleNamespace(pairing_config=config, source_asin=None, target_asin=None,
        memory=original_output / 'clean_memory.json', memory_id='memory-a', output=tmp_path / 'prepared',
        top_k=1, epsilon_levels=16, source_image=paths['a'], target_image=paths['b'],
        query_image=None, adversarial_image=None, instruction='Buy product B', key_mode='image',
        query_mode='image', optimization_config=None, dual_rag_root=tmp_path,
        method=method, steps=2, device='cpu', frequency_axis_ratio=0.5 if method == 'frequency_momentum' else None,
        frequency_momentum=0.7 if method == 'frequency_momentum' else None)
    calls = []
    monkeypatch.setattr(visual, 'generate_attack', lambda *a, **kw:
        (calls.append(kw) or paths['adv'], {'attack': {'steps_completed': 2}}))
    monkeypatch.setattr(visual, 'encoder_from_args', lambda _: Encoder())
    visual.prepare(args)
    assert calls[0]['method'] == method
    manifest = read(args.output / 'manifest.json')
    if method == 'frequency_momentum':
        assert calls[0]['frequency_axis_ratio'] == 0.5 and calls[0]['frequency_momentum'] == 0.7
        assert set(manifest['frequency_settings']) == FREQUENCY_KEYS
    else:
        assert not (FREQUENCY_KEYS & calls[0].keys())
        assert 'frequency_settings' not in manifest


@pytest.mark.parametrize('method', ['legacy_pixel', 'frequency_momentum'])
def test_local_prepare_subprocess_receives_selected_frequency_options(tmp_path, monkeypatch, method):
    run = tmp_path / 'run'
    tasks = {'query_image': 'b.png', 'trials': [{'instruction': 'Buy B'}]}
    args = SimpleNamespace(run_name='run', llm_config='unused', num_steps=20, attack_steps=2,
        memory_mode='full', trials=20, method=method,
        frequency_axis_ratio=0.5 if method == 'frequency_momentum' else None,
        frequency_momentum=0.7 if method == 'frequency_momentum' else None)
    monkeypatch.setattr(local, 'context', lambda _: (tmp_path, run, {}, tasks))
    monkeypatch.setattr(local, 'load_pairing', lambda _: {})
    monkeypatch.setattr(local, 'verify_memory', lambda *unused: {'memory_id': 'source'})
    calls = []
    monkeypatch.setattr(local.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    local.run_all(args)
    assert len(calls) == 3 and calls[1][3:5] == ['src.experiments.visual_memory_experiment', 'prepare']
    for command in calls:
        if method == 'frequency_momentum':
            assert command[command.index('--method') + 1] == method
            assert command[command.index('--frequency-axis-ratio') + 1] == '0.5'
            assert command[command.index('--frequency-momentum') + 1] == '0.7'
        else:
            assert not any(option in command for option in ('--method', '--frequency-axis-ratio', '--frequency-momentum'))


@pytest.fixture
def local_context_fixture(tmp_path, monkeypatch):
    write(tmp_path / 'local_config.json', {'port': 3000})
    write(tmp_path / 'tasks.json', {'bundle_id': 'bundle'})
    monkeypatch.setattr(local.requests, 'get', lambda *unused, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {'bundle_id': 'bundle'}))
    monkeypatch.setattr(local, 'load_model_config', lambda _: SimpleNamespace(
        model='mock', base_url='https://example.invalid', public=lambda: {'model': 'mock'}))
    args = SimpleNamespace(root=tmp_path, run_name='source', stage='train', llm_config='unused',
        num_steps=20, attack_steps=2, method='frequency_momentum', frequency_axis_ratio=0.5,
        frequency_momentum=0.7, trials=20, memory_mode='full')
    return args


def test_local_defaults_preserve_original_settings_structure(local_context_fixture):
    args = local_context_fixture
    args.method, args.frequency_axis_ratio, args.frequency_momentum = 'legacy_pixel', None, None
    _, run, _, _ = local.context(args)
    settings = read(run / 'run_settings.json')
    assert 'attack_method' not in settings and not (FREQUENCY_KEYS & settings.keys())


@pytest.mark.parametrize('field,value', [('method', 'frequency'), ('frequency_axis_ratio', 0.3),
                                        ('frequency_momentum', 0.2), ('unchanged', None)])
def test_local_evaluation_matches_frozen_frequency_settings(local_context_fixture, monkeypatch, field, value):
    args = local_context_fixture
    _, source, _, _ = local.context(args)
    source_settings = read(source / 'run_settings.json')
    assert source_settings['attack_method'] == 'frequency_momentum'
    assert FREQUENCY_KEYS <= source_settings.keys()
    write(source / 'visual/manifest.json', {'attack': {'attack': {'steps_completed': 2}}})
    write(source / 'memory_ready.json', {'normal_count': 14})
    protected = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
    monkeypatch.setattr(local, 'verify_memory', lambda *unused: None)
    monkeypatch.setattr(local, 'load_pairing', lambda *unused: {})
    args.stage, args.run_name, args.source_run = 'evaluate', 'evaluation', 'source'
    if field == 'unchanged':
        _, run, _, _ = local.context(args)
        assert args.experiment == str(source / 'visual')
        assert read(run / 'run_settings.json')['reuses_frozen_attack']
    else:
        setattr(args, field, value)
        with pytest.raises(ValueError, match='frozen source attack'):
            local.context(args)
    assert all(path.read_bytes() == body for path, body in protected.items())
