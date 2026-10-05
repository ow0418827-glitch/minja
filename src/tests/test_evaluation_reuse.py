from types import SimpleNamespace

import pytest

from src.experiments import local_experiment as driver
from src.environment.local_webshop import read, write


def test_evaluation_uses_source_manifest_and_never_rewrites_source(tmp_path, monkeypatch):
    write(tmp_path / 'local_config.json', {'port': 3000})
    write(tmp_path / 'tasks.json', {'bundle_id': 'bundle'})
    source = tmp_path / 'runs/source'
    write(source / 'run_settings.json', {'bundle_id': 'bundle'})
    write(source / 'memory_ready.json', {'normal_count': 27})
    write(source / 'visual/manifest.json', {'attack': {'attack': {'steps_completed': 1000}}})
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    monkeypatch.setattr(driver, 'verify_memory', lambda *a: None)
    monkeypatch.setattr(driver, 'load_pairing', lambda *a: {})
    monkeypatch.setattr(driver, 'load_model_config', lambda *a: SimpleNamespace(model='M3', base_url='https://example.invalid',
        public=lambda: {'provider': 'minimax', 'model': 'M3', 'base_url': 'https://example.invalid'}))
    monkeypatch.setattr(driver.requests, 'get', lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {'bundle_id': 'bundle'}))
    args = SimpleNamespace(root=tmp_path, run_name='new_eval', stage='evaluate', source_run='source',
        llm_config='unused', num_steps=20, attack_steps=1000, trials=3, memory_mode='full')
    _, run, _, _ = driver.context(args)
    assert args.experiment == str(source / 'visual')
    settings = read(run / 'run_settings.json')
    assert settings['execution_version'] == 4
    assert settings['normal_memory_count'] == 27 and settings['trials'] == 3
    assert all(p.read_bytes() == value for p, value in before.items())
    write(run / 'run_settings.json', {**settings, 'execution_version': 2})
    with pytest.raises(ValueError, match='Settings changed'):
        driver.context(args)
    write(run / 'run_settings.json', settings)
    args.include_text_baseline = True
    with pytest.raises(ValueError, match='Settings changed'):
        driver.context(args)
    args.run_name = 'four_groups'
    _, text_run, _, _ = driver.context(args)
    assert read(text_run / 'run_settings.json')['conditions_per_trial'] == 4
    args.text_top_k = 1
    with pytest.raises(ValueError, match='Settings changed'):
        driver.context(args)
    args.include_text_baseline = False
    args.run_name = 'new_eval'
    args.memory_mode = 'fragment'
    with pytest.raises(ValueError, match='Settings changed'):
        driver.context(args)
    args.run_name = 'source'
    with pytest.raises(ValueError, match='new run-name'):
        driver.context(args)


def test_no_completed_pairs_has_no_rate():
    rows = {c: {'status': 'ok', 'completed': False, 'bought_a': False, 'bought_b': False,
                'source_memory_used': False, 'termination_reason': 'step_limit'}
            for c in ['no_memory', 'clean', 'adversarial']}
    rows['paired'] = {'both_purchases_completed': False, 'new_source_memory_use': False,
                      'new_wrong_purchase': None, 'clean_correct_to_a': None}
    result = driver.aggregate({'1': rows}, 3)
    assert result['paired']['valid_completed_pairs'] == 0
    assert result['paired']['new_wrong_purchase_rate'] is None
    assert result['conditions']['clean']['incomplete'] == 1
