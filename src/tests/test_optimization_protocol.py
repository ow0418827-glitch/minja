"""Optional retrieval/SSA treatments are propagated and frozen separately."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from src.environment.local_webshop import read, write
from src.experiments import local_experiment as local
from src.experiments import nine_group_experiment as nine
from src.experiments import visual_memory_experiment as visual
from src.tests.test_frequency_integration import attack_fixture
from src.tests.test_ssa_integration import powershell_command
from src.tests.test_visual_retrieval import Encoder, PAIR, experiment
from src.paths import PROJECT


def test_ssa_json_and_file_have_identical_effective_options(tmp_path):
    options = {'early_stop': True, 'min_steps': 200, 'min_delta': 0.002}
    path = tmp_path / 'ssa.json'
    write(path, options)
    inline = visual.ssa_options(SimpleNamespace(method='ssa', ssa_options=json.dumps(options)))
    file = visual.ssa_options(SimpleNamespace(method='ssa', ssa_options=str(path)))
    assert inline == file and inline['ssa_options'] == options


@pytest.mark.parametrize('value', ['[1]', '{broken}', {'not': 'for-other-methods'}])
def test_ssa_structured_options_cannot_be_silently_ignored(value):
    with pytest.raises(ValueError):
        visual.ssa_options(SimpleNamespace(method='frequency_momentum', ssa_options=value))


def test_extended_ssa_options_reach_attack_and_freeze(attack_fixture):
    args, group, folder, calls = attack_fixture
    args.method, args.ssa_options = 'ssa', {'early_stop': True, 'min_steps': 200}
    settings = nine.attack_settings(args, group)
    assert settings['ssa_early_stop'] is True and settings['ssa_min_steps'] == 200
    nine.attack(args)
    assert calls[0][3]['ssa_options'] == args.ssa_options
    protected = (folder / 'attack_ready.json').read_bytes()
    args.ssa_options = {'early_stop': True, 'min_steps': 250}
    with pytest.raises(ValueError, match='Frozen attack changed'):
        nine.attack(args)
    assert (folder / 'attack_ready.json').read_bytes() == protected


def test_retrieval_modes_and_weights_have_separate_evaluation_protocols():
    args = SimpleNamespace(index_text='full')
    original = nine.evaluation_protocol(args)
    assert 'key_image_weight' not in original and 'query_image_weight' not in original
    args.key_image_weight = 0.8
    weighted = nine.evaluation_protocol(args)
    assert weighted['key_image_weight'] == 0.8 and weighted != original
    args.index_text = 'semantic'
    short = nine.evaluation_protocol(args)
    assert short['index_text'] == 'semantic' and short != weighted


@pytest.mark.parametrize('value', [-0.1, 1.1, float('nan'), True])
def test_invalid_retrieval_weight_rejected_before_model_loading(value):
    with pytest.raises(ValueError, match='between 0 and 1'):
        visual.retrieval_options(SimpleNamespace(key_image_weight=value))


def test_local_subprocesses_forward_structured_ssa_and_retrieval(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    args = SimpleNamespace(run_name='run', llm_config='unused', num_steps=20, attack_steps=500,
        memory_mode='full', trials=20, method='ssa', ssa_options={'early_stop': True},
        index_text='semantic', key_image_weight=0.8, query_image_weight=0.5,
        key_mode='joint', query_mode='joint')
    monkeypatch.setattr(local, 'context', lambda _: (tmp_path, run, {},
        {'query_image': 'b.png', 'trials': [{'instruction': 'Buy attached item'}]}))
    monkeypatch.setattr(local, 'load_pairing', lambda _: {})
    monkeypatch.setattr(local, 'verify_memory', lambda *unused: {'memory_id': 'source'})
    calls = []
    monkeypatch.setattr(local.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    local.run_all(args)
    assert len(calls) == 3
    for command in calls:
        assert json.loads(command[command.index('--ssa-options') + 1]) == {'early_stop': True}
        assert command[command.index('--index-text') + 1] == 'semantic'
        assert command[command.index('--key-image-weight') + 1] == '0.8'
        assert command[command.index('--key-mode') + 1] == 'joint'
        assert command[command.index('--query-mode') + 1] == 'joint'


def test_frozen_early_stop_is_reusable_but_truncation_is_rejected(attack_fixture, monkeypatch):
    from src.vision import frequency_attack
    args, group, folder, calls = attack_fixture
    args.method, args.attack_steps = 'ssa', 500
    args.ssa_options = {'early_stop': True, 'min_steps': 200}
    settings = nine.attack_settings(args, group)
    checks = [{'completed_steps': t, 'best_score': 0.799 + count * 0.001,
               'baseline_best_score': 0.799 + (count - 1) * 0.001,
               'improvement': 0.001, 'low_gain_windows': count, 'triggered': count == 2}
              for t, count in [(200, 1), (250, 2)]]
    metadata = {'method': 'ssa', 'status': 'success',
                'attack': {'steps_requested': 500, 'steps_completed': 250, 'best_step': 240,
                          'stop_reason': 'early_stop_plateau'},
                'ssa': {'settings': settings, 'early_stop_checks': checks}}
    monkeypatch.setattr(frequency_attack, 'generate_attack', lambda *a, **kw:
                        (Path(group['source_image']), metadata))
    nine.attack(args)
    nine.attack(args)
    ready = read(folder / 'attack_ready.json')
    assert ready['metadata']['attack']['steps_completed'] == 250
    ready['metadata']['attack']['stop_reason'] = 'interrupted'
    write(folder / 'attack_ready.json', ready)
    with pytest.raises(ValueError):
        nine.attack(args)


def test_evaluation_reuses_legitimate_plateau_completion(tmp_path, monkeypatch):
    from src.tests.test_frozen_memory_reevaluation import reuse_fixture
    from src.vision.ssa_attack import ssa_settings
    args, plan, group, current, folder = reuse_fixture(tmp_path, monkeypatch)
    ready = read(folder / 'attack_ready.json')
    ready['settings'].update(ssa_settings(ssa_options={'early_stop': True}))
    monkeypatch.setattr(nine, 'attack_settings', lambda *unused: ready['settings'])
    ready['metadata'].update(method='ssa', status='success', ssa={
        'settings': ready['settings'], 'early_stop_checks': [
            {'completed_steps': 200, 'best_score': 0.8, 'baseline_best_score': 0.799,
             'improvement': 0.001, 'low_gain_windows': 1, 'triggered': False},
            {'completed_steps': 250, 'best_score': 0.801, 'baseline_best_score': 0.8,
             'improvement': 0.001, 'low_gain_windows': 2, 'triggered': True}]})
    ready['metadata']['attack'].update(steps_completed=250, best_step=240,
                                     stop_reason='early_stop_plateau')
    write(folder / 'attack_ready.json', ready)
    protected = (folder / 'attack_ready.json').read_bytes()
    evidence = nine.verify_frozen_history_for_evaluation(args, tmp_path, plan, [group])
    assert evidence['verified_groups'][0]['attack_steps_completed'] == 250
    assert (folder / 'attack_ready.json').read_bytes() == protected


@pytest.mark.parametrize('script_name,stage', [('run_nine_groups.ps1', 'Attack'), ('run_local20.ps1', 'Run')])
def test_powershell_forwards_config_path_and_retrieval_weights(tmp_path, script_name, stage):
    script = tmp_path / script_name
    script.write_text((PROJECT / script_name).read_text(encoding='utf-8'), encoding='utf-8')
    mock = tmp_path / 'mock_python.ps1'
    mock.write_text("Write-Output ('__CAPTURE__' + (ConvertTo-Json -InputObject @($args) -Compress))\n"
                    "$global:LASTEXITCODE = 0\n", encoding='utf-8')
    if script_name == 'run_local20.ps1':
        write(tmp_path / 'src/output/local20/setup_report.json', {})
    config = tmp_path / 'options with spaces.json'
    write(config, {'early_stop': True})
    modes = ['-KeyMode', 'joint', '-QueryMode', 'joint'] if script_name == 'run_local20.ps1' else []
    result = subprocess.run(powershell_command(script, '-Stage', stage, '-Python', str(mock),
        '-Method', 'ssa', '-SsaOptions', str(config), '-IndexText', 'semantic',
        '-KeyImageWeight', '0.8', '-QueryImageWeight', '0.6', *modes),
        capture_output=True, text=True, encoding='utf-8', check=True)
    forwarded = json.loads(next(line.removeprefix('__CAPTURE__') for line in result.stdout.splitlines()
                                if line.startswith('__CAPTURE__')))
    assert forwarded[forwarded.index('--ssa-options') + 1] == str(config)
    assert forwarded[forwarded.index('--index-text') + 1] == 'semantic'
    assert float(forwarded[forwarded.index('--key-image-weight') + 1]) == 0.8
    assert float(forwarded[forwarded.index('--query-image-weight') + 1]) == 0.6
    if modes:
        assert forwarded[forwarded.index('--key-mode') + 1] == 'joint'
        assert forwarded[forwarded.index('--query-mode') + 1] == 'joint'


def test_old_manifest_accepts_added_fusion_description_but_rejects_model_change(experiment, monkeypatch):
    paths, output, clean, attacked, audit = experiment
    encoder = Encoder()
    manifest = {'schema_version': 2, 'pairing': PAIR, 'memory_id': 'memory-a',
        'source_asin': PAIR['purchase_product']['asin'],
        'target_asin': PAIR['requested_product']['asin'], 'pixel_audit': audit,
        'bank_sha256': {name: visual.sha256(output / f'{name}_memory.json')
                        for name in ('clean', 'adversarial')},
        'query_image': visual.copy_image(paths['b'], output), 'encoder': dict(encoder.snapshot)}
    write(output / 'manifest.json', manifest)
    encoder.snapshot = {**encoder.snapshot, 'weighted_fusion': {'default_image_weight': 0.5}}
    monkeypatch.setattr(visual, 'encoder_from_args', lambda _: encoder)
    args = SimpleNamespace(experiment=output)
    visual.load_experiment(args)
    encoder.snapshot['model'] = 'different-model'
    with pytest.raises(ValueError, match='Encoder differs'):
        visual.load_experiment(args)


def test_single_pair_weighted_prepare_rank_and_shop_use_frozen_text_policy(experiment, tmp_path, monkeypatch):
    from src.tests.test_joint_retrieval import fake_clip
    from src.agent import minja
    paths, original, _, _, _ = experiment
    config = tmp_path / 'pair.json'
    write(config, PAIR)
    args = SimpleNamespace(pairing_config=config, source_asin=None, target_asin=None,
        memory=original / 'clean_memory.json', memory_id='memory-a', output=tmp_path / 'weighted',
        top_k=1, epsilon_levels=16, source_image=paths['a'], target_image=paths['b'],
        query_image=None, adversarial_image=paths['adv'], instruction='Buy attached item',
        key_mode='joint', query_mode='joint', optimization_config=None, dual_rag_root=tmp_path,
        surrogate=2, method='legacy_pixel', steps=2, device='cpu',
        index_text='instruction', key_image_weight=0.9, query_image_weight=0.8)
    monkeypatch.setattr(visual, 'make_encoder', lambda *unused: fake_clip()[0].base)
    visual.prepare(args)
    manifest = read(args.output / 'manifest.json')
    assert manifest['retrieval_options'] == {'index_text': 'instruction',
                                             'key_image_weight': 0.9, 'query_image_weight': 0.8}
    report = read(args.output / 'retrieval_report.json')
    assert all(row['index_text'] == 'instruction' and row['text_characters'] == len('Buy product B')
               for row in report['adversarial']['ranking'])
    load_args = SimpleNamespace(experiment=args.output, dual_rag_root=tmp_path,
                               optimization_config=None, surrogate=2, device='cpu')
    loaded = visual.load_experiment(load_args)
    assert loaded[-1].supports_image_weight
    visual.rank(load_args)
    captures = []
    retriever_class = visual.VisualMemoryRetriever
    def capture(*a, **kw):
        captures.append(kw)
        return retriever_class(*a, **kw)
    monkeypatch.setattr(visual, 'VisualMemoryRetriever', capture)
    monkeypatch.setattr(minja, 'generate_embeddings', lambda entries, encoder: (entries, {}))
    monkeypatch.setattr(minja, 'webshopEnv', lambda *unused: object())
    def episode(**kw):
        if kw['memory']:
            kw['visual_retriever'].rank(kw['memory'], 'Buy attached item', kw['query_image'])
        return None, {'Completed': True, 'Reward': 1, 'FinalASIN': 'PRODUCTBBB',
                      'RetrievalTrace': [], 'PolicyTrace': []}, 0
    monkeypatch.setattr(minja, 'webshop_run_rap', episode)
    shop_args = SimpleNamespace(output=tmp_path / 'shopping', text_only_policy=True, num_steps=2,
        candidates=1, query_image=None, instruction=None, webshop_url='http://localhost',
        session='fixture', provider='fixture', model_name='fixture', emb_model='fixture', quiet=True)
    visual.shop(shop_args, runtime={'experiment': loaded, 'client': object(), 'text_encoder': object()})
    assert captures and all(kw == manifest['retrieval_options'] for kw in captures)


@pytest.mark.parametrize('mode', ['full', 'instruction', 'semantic'])
def test_single_pair_index_mode_reaches_retriever(mode):
    options = visual.retrieval_options(SimpleNamespace(index_text=mode))
    assert options.get('index_text', 'full') == mode


@pytest.mark.parametrize('side', ['key', 'query'])
def test_nondefault_text_fusion_weight_requires_joint_mode(side):
    args = SimpleNamespace(**{side + '_mode': 'image', side + '_image_weight': 0.9})
    with pytest.raises(ValueError, match='requires joint'):
        visual.retrieval_options(args)


@pytest.mark.parametrize('already_finalized', [False, True])
@pytest.mark.parametrize('invalid_metadata', [None, {'attack': {'steps_completed': 1}}])
def test_ssa_finalization_rejects_missing_or_incomplete_completion(attack_fixture, monkeypatch,
                                                                 already_finalized, invalid_metadata):
    args, group, folder, _ = attack_fixture
    args.method = 'ssa'
    group['history_binding_sha256'] = 'binding'
    monkeypatch.setattr(nine, 'freeze_settings', lambda *unused: {})
    image = Path(group['source_image'])
    ready = {'settings': nine.attack_settings(args, group), 'image': str(image),
             'sha256': nine.digest(image), 'history_binding_sha256': 'binding'}
    if invalid_metadata is not None:
        ready['metadata'] = invalid_metadata
    write(folder / 'attack_ready.json', ready)
    if already_finalized:
        write(folder / 'memory_ready.json', {'fixture': True})
        monkeypatch.setattr(nine, 'verify_banks', lambda *unused:
            {'history_binding_sha256': 'binding', 'attack_sha256': ready['sha256']})
    with pytest.raises(ValueError):
        nine.finalize(args)
    if already_finalized:
        ready['metadata'] = {'attack': {'steps_requested': 2, 'steps_completed': 2}}
        write(folder / 'attack_ready.json', ready)
        nine.finalize(args)
