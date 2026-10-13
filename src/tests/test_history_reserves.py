"""Frozen spare history demands stay separate from test reserves and the A history."""
import copy
from types import SimpleNamespace

import pytest

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver
from src.tests.test_nine_group_experiment import group as example_group, successful_row


def setup_collection(tmp_path, monkeypatch, *, reserve_count=5, normal_retries=2):
    group = example_group()
    folder = tmp_path / 'groups' / group['id']
    args = SimpleNamespace(history_reserve_count=reserve_count, normal_retries=normal_retries,
                           history_max_attempts=2, history_product_mode='observed')
    monkeypatch.setattr(driver, 'load_plan', lambda *unused: (tmp_path, {'groups': [group]}, [group]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *unused: {})
    monkeypatch.setattr(driver, 'server_health', lambda *unused: {})
    def persist(row, path, strict):
        assert strict
        row['PrimaryImage'] = {'ASIN': row['FinalASIN'], 'LocalPath': 'fixture.png'}
    monkeypatch.setattr(driver, 'runtime', lambda *unused: {
        'agent': SimpleNamespace(persist_memory_images=persist)})
    monkeypatch.setattr(driver, 'history_product_audit', lambda args, group, row, targets:
                        {'accepted': True} if row['Completed'] else None)
    bound = []
    monkeypatch.setattr(driver, 'save_history_binding', lambda args, root, plan, group, state:
                        bound.append(copy.deepcopy(state)))
    return args, group, folder, bound


def episode_row(task, *, accepted=True):
    row = successful_row(task['session'])
    row['Instruction'] = task['instruction']
    row['Completed'] = accepted
    row['EnvironmentReward'] = 1 if accepted else 0
    return row


def test_single_slot_plan_skips_used_ids_in_original_order_and_freezes_five_reserves():
    group = example_group()
    excluded = [task['task_id'] for task in group['normal_candidates'][:15]]
    plan = driver.history_candidate_plan(group, primary_count=1, reserve_count=5, exclude_task_ids=excluded)
    assert plan['primary_task_ids'] == ['normal_15']
    assert plan['reserve_task_ids'] == [f'normal_{index}' for index in range(16, 21)]
    assert [entry['candidate_index'] for entry in plan['candidates']] == list(range(15, 21))
    assert plan['plan_sha256'] == driver.fingerprint({key: value for key, value in plan.items()
                                                   if key != 'plan_sha256'})
    group['normal_candidates'][15]['instruction'] = 'new instruction'
    assert plan['candidates'][0]['task']['instruction'] != 'new instruction'


@pytest.mark.parametrize('failure', ['duplicate', 'overlap', 'short'])
def test_frozen_history_pool_rejects_invalid_or_leaking_demands(failure):
    group = example_group()
    if failure == 'duplicate':
        group['normal_candidates'][14] = copy.deepcopy(group['normal_candidates'][0])
    elif failure == 'overlap':
        group['normal_candidates'][14]['instruction'] = group['trials'][0]['instruction']
    else:
        group['normal_candidates'] = group['normal_candidates'][:18]
    with pytest.raises(ValueError, match='independent|unique'):
        driver.history_candidate_plan(group, reserve_count=5)


def test_all_primary_successes_preserved_and_reserves_never_called(tmp_path, monkeypatch):
    args, group, folder, bound = setup_collection(tmp_path, monkeypatch)
    calls = []
    def execute(options, root, selected, task, *unused):
        calls.append(task['task_id'])
        return '', episode_row(task), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.collect(args)
    state = read(folder / 'history_state.json')
    assert calls == [f'normal_{index}' for index in range(14)] + [group['training']['task_id']]
    assert len(state['normal']) == 14 and state['designated'] is not None and len(bound) == 1
    assert state['normal_collection_audit']['reserve_used'] == 0
    assert state['normal_collection_audit']['status'] == 'completed'
    original = [row['Id'] for row in state['normal']]
    driver.collect(args)
    assert [row['Id'] for row in read(folder / 'history_state.json')['normal']] == original
    assert len(calls) == 15


def test_only_exhausted_primary_activates_first_successful_reserve(tmp_path, monkeypatch):
    args, group, folder, bound = setup_collection(tmp_path, monkeypatch)
    calls = []
    def execute(options, root, selected, task, *unused):
        calls.append(task['task_id'])
        return '', episode_row(task, accepted=task['task_id'] != 'normal_0'), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.collect(args)
    state = read(folder / 'history_state.json')
    assert calls[:2] == ['normal_0', 'normal_0']
    assert calls[2:15] == [f'normal_{index}' for index in range(1, 14)]
    assert calls[15:] == ['normal_14', group['training']['task_id']]
    audit = state['normal_collection_audit']
    assert audit['reserve_used'] == 1 and audit['reserve_used_task_ids'] == ['normal_14']
    assert [item['task_id'] for item in audit['trigger_failures']] == ['normal_0']
    assert audit['trigger_failures'][0]['attempt_count'] == 2
    assert not audit['reserve_exhausted'] and audit['status'] == 'completed'
    assert len(state['normal']) == 14 and len(bound) == 1
    assert state['designated']['Instruction'] == group['training']['instruction']


def test_primary_retry_success_retained_without_spare(tmp_path, monkeypatch):
    args, _, folder, _ = setup_collection(tmp_path, monkeypatch)
    calls = []
    def execute(options, root, selected, task, *unused):
        calls.append(task['task_id'])
        accepted = task['task_id'] != 'normal_0' or calls.count('normal_0') == 2
        return '', episode_row(task, accepted=accepted), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.collect(args)
    state = read(folder / 'history_state.json')
    assert len(state['normal']) == 14 and state['normal_collection_audit']['reserve_used'] == 0
    assert state['normal_collection_audit']['trigger_failures'] == []
    assert calls.count('normal_0') == 2 and 'normal_14' not in calls


def test_reserves_exhaust_without_using_the_rest_of_candidate_pool(tmp_path, monkeypatch):
    args, group, folder, bound = setup_collection(tmp_path, monkeypatch)
    calls = []
    def execute(options, root, selected, task, *unused):
        calls.append(task['task_id'])
        return '', episode_row(task, accepted=False), False
    monkeypatch.setattr(driver, 'execute_episode', execute)
    with pytest.raises(RuntimeError, match='14 real successful'):
        driver.collect(args)
    state = read(folder / 'history_state.json')
    assert len(calls) == 38 and set(calls) == {f'normal_{index}' for index in range(19)}
    assert state['normal'] == [] and state['designated'] is None and bound == []
    assert state['normal_collection_audit']['reserve_used'] == 5
    assert state['normal_collection_audit']['reserve_exhausted']
    assert state['normal_collection_audit']['status'] == 'reserve_exhausted'
    with pytest.raises(RuntimeError, match='14 real successful'):
        driver.collect(args)
    assert len(calls) == 38
    assert group['training']['task_id'] not in calls


def test_interrupted_reserve_consumes_attempt_and_resumes_without_repeat(tmp_path, monkeypatch):
    args, _, folder, _ = setup_collection(tmp_path, monkeypatch)
    sessions, interrupted = [], [False]
    def execute(options, root, selected, task, *unused):
        sessions.append(task['session'])
        if task['task_id'] == 'normal_14' and not interrupted[0]:
            interrupted[0] = True
            raise KeyboardInterrupt()
        return '', episode_row(task, accepted=task['task_id'] != 'normal_0'), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    with pytest.raises(KeyboardInterrupt):
        driver.collect(args)
    interrupted_session = sessions[-1]
    preserved = [row['Id'] for row in read(folder / 'history_state.json')['normal']]
    driver.collect(args)
    state = read(folder / 'history_state.json')
    assert sessions.count(interrupted_session) == 1
    assert [row['Id'] for row in state['normal'][:13]] == preserved
    assert len(state['normal']) == 14 and state['normal_collection_audit']['reserve_used'] == 1
    assert next(item for item in state['attempts'] if item['session'] == interrupted_session)['status'] == 'interrupted'


def test_recoverable_technical_failure_exhausts_primary_budget_before_reserve(tmp_path, monkeypatch):
    from src.tests.test_collection_recovery import RecoverablePolicyFailure
    args, _, folder, _ = setup_collection(tmp_path, monkeypatch)
    calls = []
    def execute(options, root, selected, task, *unused):
        calls.append(task['task_id'])
        if task['task_id'] == 'normal_0':
            raise RecoverablePolicyFailure()
        return '', episode_row(task), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.collect(args)
    state = read(folder / 'history_state.json')
    failure = state['normal_collection_audit']['trigger_failures'][0]
    assert failure['attempt_count'] == 2 and failure['attempt_statuses'] == ['technical_error'] * 2
    assert state['normal_collection_audit']['reserve_used_task_ids'] == ['normal_14']
    assert len(list((folder / 'collection_errors').glob('*.json'))) == 2
    assert len(state['normal']) == 14


def test_failed_designated_a_uses_same_training_task_and_never_normal_reserves(tmp_path, monkeypatch):
    args, group, folder, bound = setup_collection(tmp_path, monkeypatch)
    group['protocol'] = driver.ATTACKER_PROTOCOL
    monkeypatch.setattr(driver, 'designated_history_audit', lambda *unused: None)
    calls = []
    def execute(options, root, selected, task, *unused):
        calls.append(task['task_id'])
        return '', episode_row(task, accepted=task['task_id'] != group['training']['task_id']), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    with pytest.raises(RuntimeError, match='designated source purchase failed'):
        driver.collect(args)
    state = read(folder / 'history_state.json')
    assert calls == [f'normal_{index}' for index in range(14)] + [group['training']['task_id']] * 2
    assert state['normal_collection_audit']['reserve_used'] == 0
    assert len(state['normal']) == 14 and state['designated'] is None and bound == []


def test_frozen_plan_tampering_stops_resume_before_more_calls(tmp_path, monkeypatch):
    args, group, folder, _ = setup_collection(tmp_path, monkeypatch)
    candidate_plan = driver.history_candidate_plan(group, reserve_count=5)
    candidate_plan['reserve_task_ids'].reverse()
    write(folder / 'history_state.json', {'normal': [], 'designated': None, 'cursor': 0, 'attempts': [],
                                        'normal_collection_plan': candidate_plan})
    monkeypatch.setattr(driver, 'execute_episode', lambda *unused: pytest.fail('Unexpected model call'))
    with pytest.raises(ValueError, match='plan changed'):
        driver.collect(args)


def test_reserve_audit_rejects_use_before_primary_retry_budget_is_exhausted():
    plan = driver.history_candidate_plan(example_group(), primary_count=1, reserve_count=5)
    attempts = [{'kind': 'normal', 'candidate_index': 0, 'task_id': 'normal_0',
                 'accepted': False, 'status': 'rejected'},
                {'kind': 'normal', 'candidate_index': 1, 'task_id': 'normal_1',
                 'accepted': False, 'status': 'running'}]
    with pytest.raises(ValueError, match='exhausted primary'):
        driver.history_reserve_audit(plan, attempts, normal_retries=2)


@pytest.mark.parametrize('invalid', [-1, 6, True, 1.5])
def test_reserve_count_bounds(invalid):
    with pytest.raises(ValueError, match='zero and five'):
        driver.history_reserve_count(SimpleNamespace(history_reserve_count=invalid))


def test_only_nonzero_history_reserve_changes_execution_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'load_model_config', lambda *unused: SimpleNamespace(public=lambda: {}))
    monkeypatch.setattr(driver, 'attack_settings', lambda *unused: {'optimization_config_sha256': None})
    args = SimpleNamespace(llm_config='unused', attack_steps=1000, method='legacy_pixel', epsilon_levels=16,
                           device='cpu', num_steps=20, memory_top_k=1, memory_mode='full',
                           memory_prompt_budget=128000, policy_max_tokens=2048,
                           history_max_attempts=3, normal_retries=2, catalog=tmp_path)
    plan = {'protocol': driver.ATTACKER_PROTOCOL, 'groups': [example_group()]}
    original = driver.execution_settings(args, plan)
    args.history_reserve_count = 0
    assert driver.execution_settings(args, plan) == original
    args.history_reserve_count = 5
    explicit = driver.execution_settings(args, plan)
    assert explicit['history_reserve_count'] == 5
    assert explicit['normal_history_candidate_policy'] == 'fourteen_primary_then_frozen_reserves_v1'
    assert {key: value for key, value in explicit.items() if key not in
            ('history_reserve_count', 'normal_history_candidate_policy')} == original
