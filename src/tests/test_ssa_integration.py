"""SSA options reach generation and remain part of immutable experiment identity."""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from src.environment.local_webshop import read, write
from src.experiments import local_experiment as local
from src.experiments import nine_group_experiment as nine
from src.experiments import visual_memory_experiment as visual
from src.paths import PROJECT
from src.tests.test_frequency_integration import (
    LEGACY_ATTACK_KEYS, attack_fixture, local_context_fixture)
from src.tests.test_visual_retrieval import Encoder, PAIR, experiment
from src.vision import ssa_attack


SSA_ARGUMENTS = {'ssa_samples': 4, 'ssa_rho': 0.4, 'ssa_sigma': 12.0, 'ssa_momentum': 0.6}


@pytest.fixture
def ssa_fixture(attack_fixture):
    args, selected, folder, calls = attack_fixture
    args.method = 'ssa'
    return args, selected, folder, calls


def test_ssa_default_settings_are_frozen_and_propagated(ssa_fixture):
    args, selected, folder, calls = ssa_fixture
    expected = ssa_attack.ssa_settings(args.optimization_config)
    settings = nine.attack_settings(args, selected)
    assert set(settings) == LEGACY_ATTACK_KEYS | expected.keys()
    assert {name: settings[name] for name in expected} == expected
    assert settings['ssa_samples'] == 20
    assert settings['ssa_rho'] == 0.5
    assert settings['ssa_sigma'] == 16.0
    assert settings['ssa_momentum'] == 1.0
    assert settings['ssa_implementation_sha256'] == nine.digest(ssa_attack.__file__)
    nine.attack(args)
    assert {name: calls[0][3][name] for name in SSA_ARGUMENTS} == dict.fromkeys(SSA_ARGUMENTS)
    assert read(folder / 'attack_ready.json')['settings'] == settings
    nine.attack(args)
    assert len(calls) == 1


def test_ssa_config_defaults_and_explicit_options_have_distinct_identity(ssa_fixture):
    args, selected, _, calls = ssa_fixture
    write(args.optimization_config, {'ssa': {'samples': 3, 'rho': 0.3, 'sigma': 10, 'momentum': 0.7}})
    configured = nine.attack_settings(args, selected)
    assert configured['ssa_samples'] == 3 and configured['ssa_sigma'] == 10
    for key, value in SSA_ARGUMENTS.items():
        setattr(args, key, value)
    explicit = nine.attack_settings(args, selected)
    assert {key: explicit[key] for key in SSA_ARGUMENTS} == SSA_ARGUMENTS
    nine.attack(args)
    assert {key: calls[0][3][key] for key in SSA_ARGUMENTS} == SSA_ARGUMENTS


@pytest.mark.parametrize('field,value', list(SSA_ARGUMENTS.items()))
def test_ssa_frozen_attack_rejects_parameter_changes(ssa_fixture, field, value):
    args, _, folder, calls = ssa_fixture
    nine.attack(args)
    protected = (folder / 'attack_ready.json').read_bytes()
    setattr(args, field, value)
    with pytest.raises(ValueError, match='Frozen attack changed'):
        nine.attack(args)
    assert len(calls) == 1
    assert (folder / 'attack_ready.json').read_bytes() == protected


def test_ssa_frozen_attack_rejects_implementation_changes(ssa_fixture, monkeypatch):
    args, _, _, calls = ssa_fixture
    nine.attack(args)
    original = ssa_attack.ssa_settings
    monkeypatch.setattr(ssa_attack, 'ssa_settings', lambda *a, **kw:
        {**original(*a, **kw), 'ssa_implementation_sha256': 'changed-implementation'})
    with pytest.raises(ValueError, match='Frozen attack changed'):
        nine.attack(args)
    assert len(calls) == 1


def test_ssa_execution_settings_freeze_same_effective_attack_settings(ssa_fixture, monkeypatch):
    args, selected, _, _ = ssa_fixture
    args.llm_config = 'unused'
    args.num_steps, args.memory_top_k, args.memory_mode = 20, 1, 'full'
    args.memory_prompt_budget, args.policy_max_tokens = 128000, 2048
    args.history_max_attempts, args.normal_retries, args.catalog = 3, 2, Path('.')
    monkeypatch.setattr(nine, 'load_model_config', lambda _: SimpleNamespace(public=lambda: {'model': 'mock'}))
    plan = {'protocol': nine.ATTACKER_PROTOCOL, 'groups': [selected]}
    current = nine.execution_settings(args, plan)
    keys = ssa_attack.ssa_settings(args.optimization_config).keys()
    assert all(current[key] == nine.attack_settings(args, selected)[key] for key in keys)
    args.method = 'legacy_pixel'
    legacy = nine.execution_settings(args, plan)
    assert set(current) == set(legacy) | keys
    assert current['code'] == legacy['code']


@pytest.mark.parametrize('method', ['legacy_pixel', 'fair_pixel', 'frequency', 'frequency_momentum'])
@pytest.mark.parametrize('field', list(SSA_ARGUMENTS))
def test_ssa_parameters_cannot_be_ignored_by_other_methods(method, field):
    args = SimpleNamespace(method=method, **{field: SSA_ARGUMENTS[field]})
    with pytest.raises(ValueError, match='SSA parameters require'):
        visual.ssa_options(args)


def test_single_pair_prepare_forwards_and_freezes_ssa(tmp_path, monkeypatch, experiment):
    paths, original_output, _, _, _ = experiment
    config = tmp_path / 'pair.json'
    write(config, PAIR)
    args = SimpleNamespace(pairing_config=config, source_asin=None, target_asin=None,
        memory=original_output / 'clean_memory.json', memory_id='memory-a', output=tmp_path / 'prepared',
        top_k=1, epsilon_levels=16, source_image=paths['a'], target_image=paths['b'],
        query_image=None, adversarial_image=None, instruction='Buy product B', key_mode='image',
        query_mode='image', optimization_config=None, dual_rag_root=tmp_path,
        method='ssa', steps=2, device='cpu', **SSA_ARGUMENTS)
    calls = []
    monkeypatch.setattr(visual, 'generate_attack', lambda *a, **kw:
        (calls.append(kw) or paths['adv'], {'attack': {'steps_completed': 2}}))
    monkeypatch.setattr(visual, 'encoder_from_args', lambda _: Encoder())
    visual.prepare(args)
    assert calls[0]['method'] == 'ssa'
    assert {key: calls[0][key] for key in SSA_ARGUMENTS} == SSA_ARGUMENTS
    manifest = read(args.output / 'manifest.json')
    assert manifest['ssa_settings'] == ssa_attack.ssa_settings(root=tmp_path, **SSA_ARGUMENTS)
    assert 'frequency_settings' not in manifest


def test_imported_image_rejects_ignored_ssa_parameters(tmp_path):
    args = SimpleNamespace(method='ssa', optimization_config=None, dual_rag_root=tmp_path,
                           adversarial_image='import.png', **SSA_ARGUMENTS)
    with pytest.raises(ValueError, match='cannot be used with --adversarial-image'):
        visual.prepare(args)


def test_local_subprocesses_all_receive_ssa_options(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    tasks = {'query_image': 'b.png', 'trials': [{'instruction': 'Buy B'}]}
    args = SimpleNamespace(run_name='run', llm_config='unused', num_steps=20, attack_steps=2,
        memory_mode='full', trials=20, method='ssa', **SSA_ARGUMENTS)
    monkeypatch.setattr(local, 'context', lambda _: (tmp_path, run, {}, tasks))
    monkeypatch.setattr(local, 'load_pairing', lambda _: {})
    monkeypatch.setattr(local, 'verify_memory', lambda *unused: {'memory_id': 'source'})
    calls = []
    monkeypatch.setattr(local.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    local.run_all(args)
    assert len(calls) == 3
    for command in calls:
        assert command[command.index('--method') + 1] == 'ssa'
        for key, value in SSA_ARGUMENTS.items():
            option = '--' + key.replace('_', '-')
            assert command[command.index(option) + 1] == str(value)
        assert '--frequency-axis-ratio' not in command


@pytest.mark.parametrize('field,value', list(SSA_ARGUMENTS.items()) + [('unchanged', None)])
def test_local_evaluation_validates_frozen_ssa_settings(local_context_fixture, monkeypatch, field, value):
    args = local_context_fixture
    args.method, args.frequency_axis_ratio, args.frequency_momentum = 'ssa', None, None
    _, source, _, _ = local.context(args)
    settings = read(source / 'run_settings.json')
    assert ssa_attack.ssa_settings().keys() <= settings.keys()
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


def powershell_command(script, *args):
    shell = shutil.which('pwsh') or shutil.which('powershell')
    if shell is None:
        pytest.skip('PowerShell is not installed')
    return [shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script), *args]


@pytest.mark.parametrize('script_name,stage', [('run_nine_groups.ps1', 'Attack'), ('run_local20.ps1', 'Run')])
def test_powershell_public_wrappers_forward_ssa_options(tmp_path, script_name, stage):
    script = tmp_path / script_name
    script.write_text((PROJECT / script_name).read_text(encoding='utf-8'), encoding='utf-8')
    mock_python = tmp_path / 'mock_python.ps1'
    mock_python.write_text("Write-Output ('__CAPTURE__' + (ConvertTo-Json -InputObject @($args) -Compress))\n"
                           "$global:LASTEXITCODE = 0\n", encoding='utf-8')
    if script_name == 'run_local20.ps1':
        write(tmp_path / 'src/output/local20/setup_report.json', {})
    command = powershell_command(script, '-Stage', stage, '-Python', str(mock_python), '-Method', 'ssa',
        '-SsaSamples', '4', '-SsaRho', '0.4', '-SsaSigma', '12', '-SsaMomentum', '0.6')
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
    capture = next(line.removeprefix('__CAPTURE__') for line in result.stdout.splitlines()
                   if line.startswith('__CAPTURE__'))
    forwarded = json.loads(capture)
    assert forwarded[forwarded.index('--method') + 1] == 'ssa'
    for key, value in SSA_ARGUMENTS.items():
        option = '--' + key.replace('_', '-')
        assert float(forwarded[forwarded.index(option) + 1]) == value
    assert '--frequency-axis-ratio' not in forwarded


@pytest.mark.parametrize('method', ['frequency_momentum', 'ssa'])
def test_identity_wrapper_preserves_default_and_isolates_new_method(tmp_path, method):
    script = tmp_path / 'run_identity_frequency_126.ps1'
    script.write_text((PROJECT / script.name).read_text(encoding='utf-8'), encoding='utf-8')
    mock = tmp_path / 'run_nine_groups.ps1'
    mock.write_text("param([string]$Stage, [string]$Root, [string]$ReferenceOverrides, [string[]]$Group, "
        "[string]$Method, [int]$AttackSteps, [Nullable[double]]$FrequencyAxisRatio, "
        "[Nullable[double]]$FrequencyMomentum, [int]$EpsilonLevels, [string]$SourceHistoryMode, "
        "[string]$HistoryProductMode, [string]$RetrievalEncoder, [string]$MemoryKeyMode, "
        "[string]$MemoryQueryMode, [string]$MemoryMode, [int]$MemoryTopK, [string]$IndexText, "
        "[string]$TestMode, [string]$QuerySpecificity, [string]$QueryImageMode, "
        "[switch]$PolicyQueryImage, [switch]$IncludeControls, [int]$ReserveTests, "
        "[int]$NumSteps, [string]$Device, [int]$Port)\n"
        "Write-Output ('__CAPTURE__' + (ConvertTo-Json -InputObject $PSBoundParameters -Compress))\n",
        encoding='utf-8')
    options = [] if method == 'frequency_momentum' else ['-Method', method]
    result = subprocess.run(powershell_command(script, '-Stage', 'Prepare', *options),
                            capture_output=True, text=True, encoding='utf-8', check=True)
    capture = next(line.removeprefix('__CAPTURE__') for line in result.stdout.splitlines()
                   if line.startswith('__CAPTURE__'))
    forwarded = json.loads(capture)
    assert forwarded['Method'] == method
    expected_root = 'nine_groups_identity_ssa_126' if method == 'ssa' else 'nine_groups_identity_frequency_126'
    assert Path(forwarded['Root']).name == expected_root
    if method == 'frequency_momentum':
        assert forwarded['FrequencyAxisRatio'] == 0.25 and forwarded['FrequencyMomentum'] == 1.0
    else:
        assert 'FrequencyAxisRatio' not in forwarded and 'FrequencyMomentum' not in forwarded
