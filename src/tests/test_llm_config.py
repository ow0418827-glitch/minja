import json
from types import SimpleNamespace

import pytest

from src.configuration.llm_config import load_model_config
from src.environment.local_webshop import read, write


def configuration(tmp_path, key='private-test-key'):
    path = tmp_path / 'model.json'
    write(path, {'model': {'provider': 'minimax', 'model': 'MiniMax-M3',
                          'base_url': 'https://example.invalid/v1', 'api_key': key}})
    return path


def test_file_credentials_override_environment_and_stay_private(tmp_path, monkeypatch):
    import openai
    monkeypatch.setenv('MINIMAX_API_KEY', 'wrong-environment-key')
    path = configuration(tmp_path)
    config = load_model_config(path)
    captured = {}
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: captured.update(kw))
    config.client()
    assert captured['api_key'] == 'private-test-key'
    assert captured['base_url'] == 'https://example.invalid/v1'
    assert 'private-test-key' not in repr(config)
    assert 'private-test-key' not in json.dumps(config.public())
    configuration(tmp_path, '')
    with pytest.raises(ValueError, match='model.api_key'):
        load_model_config(path)


def test_context_records_actual_model_and_blocks_mixed_resume(tmp_path, monkeypatch):
    from src.experiments import local_experiment as driver
    path = configuration(tmp_path)
    write(tmp_path / 'local_config.json', {'port': 3000})
    write(tmp_path / 'tasks.json', {'bundle_id': 'expected'})
    monkeypatch.setattr(driver.requests, 'get', lambda *a, **kw:
        SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'bundle_id': 'expected'}))
    args = SimpleNamespace(root=tmp_path, run_name='test', llm_config=path, num_steps=20, attack_steps=1000)
    _, run, _, _ = driver.context(args)
    assert args.model_name == 'MiniMax-M3'
    assert read(run / 'run_settings.json')['base_url'] == 'https://example.invalid/v1'
    assert 'private-test-key' not in (run / 'run_settings.json').read_text(encoding='utf-8')
    changed = read(path)
    changed['model']['model'] = 'other-model'
    write(path, changed)
    with pytest.raises(ValueError, match='Settings changed'):
        driver.context(args)
