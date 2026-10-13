"""Run-level config identity must not require an unselected group's imported image."""
from types import SimpleNamespace

import pytest

from src.experiments import nine_group_experiment as driver
from src.vision.dual_rag_bridge import DEFAULT_DUAL_ROOT


def settings_args(tmp_path):
    return SimpleNamespace(llm_config='unused', optimization_config=None, attack_steps=0,
        method='scar', epsilon_levels=16, device='cpu', num_steps=20, memory_top_k=1,
        memory_mode='full', memory_prompt_budget=128000, policy_max_tokens=2048,
        history_max_attempts=3, normal_retries=2, catalog=tmp_path,
        precomputed_manifest={'groups': {'07_shampoo': {'frozen_import': True}}})


@pytest.mark.parametrize('explicit_config', [False, True])
def test_single_group_scar_run_hashes_config_without_requesting_group_one_import(tmp_path, monkeypatch, explicit_config):
    monkeypatch.setattr(driver, 'load_model_config', lambda *unused: SimpleNamespace(public=lambda: {}))
    monkeypatch.setattr(driver, 'attack_settings', lambda *unused: pytest.fail('Run-level hashing must not inspect group attack records'))
    args = settings_args(tmp_path)
    if explicit_config:
        args.optimization_config = tmp_path / 'config.json'
        args.optimization_config.write_text('{"seed": 7}\n', encoding='utf-8')
    plan = {'protocol': driver.ATTACKER_PROTOCOL, 'groups': [{'id': '01_camera'}, {'id': '07_shampoo'}]}
    settings = driver.execution_settings(args, plan)
    path = args.optimization_config or DEFAULT_DUAL_ROOT / 'configs/optimization.json'
    assert settings['attack_optimization_sha256'] == driver.digest(path)
    assert settings['precomputed_manifest_sha256'] == driver.fingerprint(args.precomputed_manifest)
    assert set(args.precomputed_manifest['groups']) == {'07_shampoo'}


def test_legacy_run_optimization_identity_matches_existing_attack_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'load_model_config', lambda *unused: SimpleNamespace(public=lambda: {}))
    args = settings_args(tmp_path)
    args.method, args.attack_steps = 'legacy_pixel', 1000
    source, target = tmp_path / 'source.bin', tmp_path / 'target.bin'
    source.write_bytes(b'original-source')
    target.write_bytes(b'original-target')
    group = {'id': '01_camera', 'source_image': str(source), 'query_image': str(target)}
    plan = {'protocol': driver.ATTACKER_PROTOCOL, 'groups': [group]}
    assert driver.execution_settings(args, plan)['attack_optimization_sha256'] == driver.attack_settings(args, group)['optimization_config_sha256']
