"""A transport repair preserves bound purchases and refuses downstream mixing."""
import copy

import pytest

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver
from src.tests.test_attacker_history_driver import case, bind, picture


def repair_settings(case, monkeypatch):
    bind(case)
    names = sorted(driver.BOUND_HISTORY_REPAIR_CODE)
    old = {'attack_steps': 1000, 'plan_sha256': driver.fingerprint(case.plan),
           'code': {name: 'original' for name in names}}
    new = dict(old, code={name: 'repaired' for name in names})
    write(case.root / 'run_settings.json', old)
    monkeypatch.setattr(driver, 'execution_settings', lambda *args: copy.deepcopy(new))
    case.args.allow_code_update_resume = True
    case.args.allow_bound_history_repair_resume = True
    return old, new


def original_history_bytes(case):
    return {name: (case.folder / name).read_bytes() for name in (
        'history_binding.json', 'history_state.json', 'history_memory.json')}


def test_bound_repair_verifies_every_image_and_retains_exact_history_files(case, monkeypatch):
    old, new = repair_settings(case, monkeypatch)
    before = original_history_bytes(case)
    assert driver.freeze_settings(case.args, case.root, case.plan) == new
    assert original_history_bytes(case) == before
    audit = read(case.root / 'code_update_resume.json')
    assert len(audit) == 1
    assert len(audit[0]['verified_history_bindings'][0]['images']) == 15
    assert audit[0]['verified_history_bindings'][0]['binding_file_sha256'] == driver.digest(case.folder / 'history_binding.json')
    assert read(case.root / 'run_settings_revisions' / (driver.fingerprint(old) + '.json')) == old
    driver.freeze_settings(case.args, case.root, case.plan)
    assert len(read(case.root / 'code_update_resume.json')) == 1


@pytest.mark.parametrize('fault', ['normal_image', 'history_file', 'binding', 'model_code', 'attack'])
def test_bound_repair_refuses_corrupt_assets_unrelated_code_or_started_attack(case, monkeypatch, fault):
    old, new = repair_settings(case, monkeypatch)
    if fault == 'normal_image':
        picture(case.normal_b, (1, 2, 3))
    elif fault == 'history_file':
        write(case.folder / 'history_memory.json', [])
    elif fault == 'binding':
        payload = read(case.folder / 'history_binding.json')
        payload['actual_a'] = 'different-product'
        write(case.folder / 'history_binding.json', payload)
    elif fault == 'model_code':
        old['code']['src/agent/minja.py'] = 'original'
        new['code']['src/agent/minja.py'] = 'repaired'
        write(case.root / 'run_settings.json', old)
    else:
        write(case.folder / 'attack_ready.json', {})
    with pytest.raises(ValueError):
        driver.freeze_settings(case.args, case.root, case.plan)
    assert read(case.root / 'run_settings.json') == old
    assert not (case.root / 'code_update_resume.json').exists()


def test_bound_repair_requires_its_explicit_scope_switch(case, monkeypatch):
    old, _ = repair_settings(case, monkeypatch)
    case.args.allow_bound_history_repair_resume = False
    with pytest.raises(ValueError, match='history freeze'):
        driver.freeze_settings(case.args, case.root, case.plan)
    assert read(case.root / 'run_settings.json') == old


def test_bound_repair_accepts_native_windows_frozen_code_paths(case, monkeypatch):
    old, new = repair_settings(case, monkeypatch)
    old['code'] = {name.replace('/', '\\'): value for name, value in old['code'].items()}
    new['code'] = {name.replace('/', '\\'): value for name, value in new['code'].items()}
    write(case.root / 'run_settings.json', old)
    driver.freeze_settings(case.args, case.root, case.plan)
    audit = read(case.root / 'code_update_resume.json')[0]
    assert set(audit['changed_code_files']) == driver.BOUND_HISTORY_REPAIR_CODE


def test_collect_skips_verified_completed_group_without_rewriting_or_model_calls(case, monkeypatch):
    bind(case)
    before = original_history_bytes(case)
    monkeypatch.setattr(driver, 'load_plan', lambda *args: (case.root, case.plan, [case.group]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *args: {})
    monkeypatch.setattr(driver, 'server_health', lambda *args: {})
    monkeypatch.setattr(driver, 'runtime', lambda *args: {})
    monkeypatch.setattr(driver, 'execute_episode', lambda *args: pytest.fail('Completed history was recollected'))
    monkeypatch.setattr(driver, 'save_history_binding', lambda *args: pytest.fail('Completed binding was rewritten'))
    driver.collect(case.args)
    assert original_history_bytes(case) == before
