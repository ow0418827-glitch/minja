"""Offline driver checks for bounded original-demand overload retries and resume."""
from collections import Counter
import copy
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import InternalServerError
import pytest
import requests

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver
from src.experiments.deferred_529 import POLICY_VERSION
from src.tests.test_reserve_driver import condition_from, setup_run, shopping_row


REAL_STATUS = driver.status


def setup_deferred(tmp_path, monkeypatch, *, reserve=0):
    group, args, folder, _, original = setup_run(tmp_path, monkeypatch)
    args.reserve_tests = reserve
    location, _ = driver.evaluation_location(args, folder)
    return group, args, folder, location, original


def overload(task):
    response = httpx.Response(529, request=httpx.Request('POST', 'https://example.invalid/v1/chat/completions'))
    error = InternalServerError('Error code: 529 - 当前服务集群负载较高 PRIVATE_TEST_KEY',
        response=response, body={'error': {'type': 'overloaded_error', 'http_code': '529'}})
    error.partial_memory = {'Id': task['session'], 'Instruction': task['instruction'],
        'Completed': False, 'FinalASIN': None, 'EnvironmentReward': None,
        'Actions': ['Action: search[original demand]'], 'TerminationReason': 'policy_call_error',
        'PolicyTrace': [{'Step': 2, 'CallAttempts': 3, 'Executed': False}], 'RetrievalTrace': []}
    return error


def fake_episode(group, calls, failure, *, completed=True):
    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        condition = condition_from(task)
        calls.append((task['task_id'], condition))
        failure(task, condition, calls)
        return '', shopping_row(group, task, condition, completed=completed), completed
    return execute


@pytest.mark.parametrize('reserve', [0, 5])
def test_529_defers_original_demand_preserves_valid_failures_and_never_uses_spare(
        tmp_path, monkeypatch, capsys, reserve):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch, reserve=reserve)
    calls = []
    def fail_once(task, condition, history):
        if task['variant_id'] == 1 and condition == 'no_memory' and history.count((task['task_id'], condition)) == 1:
            raise overload(task)
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, fail_once, completed=False))
    driver.evaluate(args)
    results, summary = read(location / 'test_results.json'), read(location / 'summary.json')
    assert len(calls) == 61
    assert calls[0] == (results['no_memory'][0]['task_id'], 'no_memory')
    assert calls[1][0].endswith('_002')  # Skip remaining conditions of the overloaded original demand.
    assert calls[1][1] == 'clean'  # Preserve the existing rotating condition order.
    assert calls[58][0].endswith('_001')  # Retry only after the other 19 demands.
    assert all(len(rows) == 20 and all(not row.get('error') for row in rows) for rows in results.values())
    assert all(not row['completed'] for rows in results.values() for row in rows)
    assert all(row['candidate_role'] == 'primary' for rows in results.values() for row in rows)
    assert summary['deferred_overload_events_total'] == summary['recovered_overload_events_total'] == 1
    assert summary['deferred_529_audit']['status'] == 'completed'
    assert all(summary[condition]['technical_errors'] == 0 for condition in results)
    if reserve:
        assert summary['replacement_audit']['reserve_used'] == 0
        assert summary['replacement_audit']['technical_errors_total'] == 0
    archives = sorted((location / 'shopping_attempts/01_no_memory').glob('*.json'))
    assert len(archives) == 2
    assert read(archives[0])['TerminationReason'] == 'policy_call_error'
    assert read(archives[1])['TerminationReason'] == 'step_limit'
    output = capsys.readouterr().out
    assert '集群负载较高' in output and 'PRIVATE_TEST_KEY' not in output
    assert not any('PRIVATE_TEST_KEY' in path.read_text(encoding='utf-8') for path in location.rglob('*.json'))
    driver.evaluate(args)
    assert len(calls) == 61


@pytest.mark.parametrize('reserve', [0, 5])
def test_three_different_initial_529_demands_stop_immediately_and_explicit_resume_keeps_audit(
        tmp_path, monkeypatch, capsys, reserve):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch, reserve=reserve)
    calls = []
    def first_three(task, condition, history):
        if task['variant_id'] <= 3:
            raise overload(task)
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, first_three))
    with pytest.raises(driver.Deferred529Halt, match='连续三个'):
        driver.evaluate(args)
    assert len(calls) == 3 and len({task for task, _ in calls}) == 3
    summary = read(location / 'summary.json')
    assert summary['deferred_529_audit']['status'] == 'halted'
    assert summary['deferred_529_audit']['halt_reason'] == 'three_consecutive_primary_529'
    assert summary['deferred_529_audit']['rounds_attempted'] == 0
    assert summary['deferred_overload_events_total'] == 3
    output = capsys.readouterr().out
    assert '连续三个' in output and '未完成需求' in output and 'deferred_529_state.json' in output
    assert '集群负载较高' in output and 'PRIVATE_TEST_KEY' not in output
    failed = {path: path.read_bytes() for path in (location / 'shopping_attempts').rglob('*.json')}
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, lambda *unused: None))
    driver.evaluate(args)
    assert len(calls) == 63
    assert all(path.read_bytes() == previous for path, previous in failed.items())
    summary = read(location / 'summary.json')
    assert summary['deferred_529_audit']['status'] == 'completed'
    assert summary['recovered_overload_events_total'] == 3
    assert all(summary[condition]['technical_errors'] == 0 for condition in ('no_memory', 'clean', 'adversarial'))
    if reserve:
        assert summary['replacement_audit']['reserve_used'] == 0


def test_two_rounds_retry_only_pending_conditions_then_resume_commits_only_one_missing_result(
        tmp_path, monkeypatch, capsys):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch)
    calls = []
    def persistent(task, condition, history):
        if task['variant_id'] == 1 and condition == 'no_memory':
            raise overload(task)
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, persistent))
    with pytest.raises(driver.Deferred529Halt, match='两轮'):
        driver.evaluate(args)
    assert len(calls) == 62
    counts = Counter(calls)
    first_id = next(task for task, _ in calls if task.endswith('_001'))
    assert counts[first_id, 'no_memory'] == 3
    assert counts[first_id, 'clean'] == counts[first_id, 'adversarial'] == 1
    results, summary = read(location / 'test_results.json'), read(location / 'summary.json')
    assert all(len(rows) == 20 for rows in results.values())
    assert summary['adversarial']['valid_tests'] == 20 and summary['clean']['valid_tests'] == 20
    assert summary['no_memory']['valid_tests'] == 19
    assert summary['adversarial']['target_purchase_rate'] is None
    assert summary['deferred_529_audit']['rounds_attempted'] == 2
    assert summary['deferred_529_audit']['pending_slots'] == [1]
    original = {row['task_id']: copy.deepcopy(row) for row in results['adversarial']}
    output = capsys.readouterr().out
    assert '延期补跑第 2/2 轮' in output and '集群负载较高' in output
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, lambda *unused: None))
    driver.evaluate(args)
    assert len(calls) == 63 and calls[-1] == (first_id, 'no_memory')
    resumed = read(location / 'test_results.json')
    assert {row['task_id']: row for row in resumed['adversarial']} == original
    assert read(location / 'summary.json')['recovered_overload_events_total'] == 3


def test_each_pending_demand_has_at_most_six_deferred_condition_attempts_per_command(tmp_path, monkeypatch):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch)
    calls = []
    def all_three(task, condition, history):
        if task['variant_id'] == 1:
            raise overload(task)
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, all_three))
    with pytest.raises(driver.Deferred529Halt):
        driver.evaluate(args)
    state = read(location / 'deferred_529_state.json')
    deferred = [attempt for attempt in state['attempts'] if attempt['phase'] == 'deferred']
    assert len(deferred) == 6
    assert Counter(attempt['round'] for attempt in deferred) == {1: 3, 2: 3}
    assert {attempt['condition'] for attempt in deferred} == {'no_memory', 'clean', 'adversarial'}
    assert len(calls) == 64
    assert len(list((location / 'shopping_attempts/01_no_memory').glob('*.json'))) == 3


def test_non529_reserve_error_retains_old_replacement_but_reserve_529_keeps_same_replacement_demand(
        tmp_path, monkeypatch):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch, reserve=5)
    calls = []
    def failures(task, condition, history):
        if task['variant_id'] == 1 and condition == 'adversarial':
            raise requests.ConnectionError('temporary connection failure')
        if task['variant_id'] == 21 and condition == 'adversarial' and history.count((task['task_id'], condition)) == 1:
            raise overload(task)
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, failures))
    driver.evaluate(args)
    summary, results = read(location / 'summary.json'), read(location / 'test_results.json')
    assert len(calls) == 64
    assert summary['replacement_audit']['reserve_used'] == 1
    assert summary['replacement_audit']['technical_errors_total'] == 1
    assert summary['deferred_overload_events_total'] == summary['recovered_overload_events_total'] == 1
    assert all(rows[0]['task_id'].endswith('_021') and rows[0]['evaluation_slot'] == 1
               for rows in results.values())
    assert not any(task.endswith('_022') for task, _ in calls)
    driver.evaluate(args)
    assert len(calls) == 64


def temporary_server_error(task):
    response = httpx.Response(500, request=httpx.Request('POST', 'https://example.invalid/v1/chat/completions'))
    error = InternalServerError('Error code: 500 - temporary server failure PRIVATE_TEST_KEY',
                                response=response, body={'error': {'type': 'server_error'}})
    error.partial_memory = overload(task).partial_memory
    return error


@pytest.mark.parametrize('reserve', [0, 5])
def test_529_then_500_and_timeout_use_only_six_original_condition_retries_and_resume(
        tmp_path, monkeypatch, capsys, reserve):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch, reserve=reserve)
    calls = []
    def mixed_errors(task, condition, history):
        if task['variant_id'] != 1:
            return
        if condition == 'no_memory' and history.count((task['task_id'], condition)) == 1:
            raise overload(task)
        if condition == 'clean':
            error = requests.Timeout('temporary timeout PRIVATE_TEST_KEY')
            error.partial_memory = overload(task).partial_memory
            raise error
        raise temporary_server_error(task)
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, mixed_errors))
    with pytest.raises(driver.Deferred529Halt, match='两轮'):
        driver.evaluate(args)
    state = read(location / 'deferred_529_state.json')
    deferred = [attempt for attempt in state['attempts'] if attempt['phase'] == 'deferred']
    assert len(calls) == 64 and len(deferred) == 6
    assert Counter(attempt['round'] for attempt in deferred) == {1: 3, 2: 3}
    assert all(not attempt['is_529'] and attempt['report']['deferred_after_529']
               for attempt in deferred)
    assert all(task.endswith(tuple(f'_{number:03d}' for number in range(1, 21)))
               for task, _ in calls)
    summary = read(location / 'summary.json')
    assert summary['deferred_529_audit']['status'] == 'halted'
    assert summary['deferred_overload_events_total'] == 1
    assert summary['recovered_overload_events_total'] == 0
    assert not (location / 'candidate_results.json').exists()
    output = capsys.readouterr().out
    assert '曾收到 529' in output and '当前 no_memory 出现 HTTP 500' in output
    assert '当前 clean 出现 Timeout' in output and 'PRIVATE_TEST_KEY' not in output
    previous = read(location / 'test_results.json')
    normal_rows = {condition: rows[1:] for condition, rows in previous.items()}
    archived = {path: path.read_bytes() for path in (location / 'shopping_attempts').rglob('*.json')}
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, lambda *unused: None))
    driver.evaluate(args)
    assert len(calls) == 67 and all(task.endswith('_001') for task, _ in calls[-3:])
    resumed = read(location / 'test_results.json')
    assert {condition: rows[1:] for condition, rows in resumed.items()} == normal_rows
    assert all(path.read_bytes() == original for path, original in archived.items())
    summary = read(location / 'summary.json')
    assert summary['deferred_529_audit']['status'] == 'completed'
    assert summary['recovered_overload_events_total'] == 1
    if reserve:
        assert summary['replacement_audit']['reserve_used'] == 0
    driver.evaluate(args)
    assert len(calls) == 67


@pytest.mark.parametrize('reserve', [0, 5])
def test_recovered_529_condition_still_pins_other_500_condition_and_preserves_success_on_resume(
        tmp_path, monkeypatch, reserve):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch, reserve=reserve)
    calls = []
    def mixed_errors(task, condition, history):
        if task['variant_id'] != 1:
            return
        if condition == 'no_memory' and history.count((task['task_id'], condition)) == 1:
            raise overload(task)
        if condition == 'clean':
            if history.count((task['task_id'], condition)) == 1:
                raise temporary_server_error(task)
            error = requests.Timeout('temporary timeout')
            error.partial_memory = overload(task).partial_memory
            raise error
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, mixed_errors, completed=False))
    with pytest.raises(driver.Deferred529Halt, match='两轮'):
        driver.evaluate(args)
    results = read(location / 'test_results.json')
    first_id = results['no_memory'][0]['task_id']
    assert len(calls) == 62
    assert Counter(condition for task, condition in calls if task == first_id) == {
        'no_memory': 2, 'clean': 2, 'adversarial': 1}
    assert results['clean'][0]['error'] == 'Timeout'
    assert results['clean'][0]['deferred_after_529'] is True
    assert not results['no_memory'][0].get('error') and not results['adversarial'][0].get('error')
    assert not results['adversarial'][0]['completed']  # Valid shopping failures remain committed.
    summary = read(location / 'summary.json')
    assert summary['deferred_529_audit']['pending_slots'] == [1]
    assert summary['recovered_overload_events_total'] == 1
    assert not (location / 'candidate_results.json').exists()
    kept = {condition: copy.deepcopy(rows) for condition, rows in results.items() if condition != 'clean'}
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, lambda *unused: None))
    driver.evaluate(args)
    assert len(calls) == 63 and calls[-1] == (first_id, 'clean')
    resumed = read(location / 'test_results.json')
    assert {condition: rows for condition, rows in resumed.items() if condition != 'clean'} == kept
    assert all(rows[0]['task_id'] == first_id for rows in resumed.values())
    if reserve:
        assert read(location / 'summary.json')['replacement_audit']['reserve_used'] == 0


@pytest.mark.parametrize('fatal_condition,expected_calls', [('no_memory', 1), ('adversarial', 3)])
def test_no_reserve_fatal_stops_immediately_and_status_and_resume_keep_blocking(
        tmp_path, monkeypatch, fatal_condition, expected_calls):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch)
    calls = []
    def fatal(task, condition, history):
        if condition == fatal_condition:
            raise ValueError('frozen identity changed')
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, fatal))
    with pytest.raises(ValueError, match='frozen identity'):
        driver.evaluate(args)
    assert len(calls) == expected_calls
    assert read(location / 'summary.json')['deferred_529_audit']['status'] == 'fatal'
    monkeypatch.setattr(driver, 'load_plan', lambda *unused: (
        tmp_path, {'protocol': driver.VICTIM_PROTOCOL, 'catalog_scope': 'fixture'}, [copy.deepcopy(group)]))
    assert REAL_STATUS(args)['status'] == 'stopped_with_fatal_error'
    with pytest.raises(RuntimeError, match='Fatal evaluation error is saved'):
        driver.evaluate(args)
    assert len(calls) == expected_calls


def test_reserve_child_fatal_stops_before_other_conditions_and_blocks_resume(tmp_path, monkeypatch):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch, reserve=5)
    calls = []
    def failures(task, condition, history):
        if task['variant_id'] == 1 and condition == 'adversarial':
            raise requests.ConnectionError('temporary connection failure')
        if task['variant_id'] == 21:
            raise ValueError('frozen identity changed')
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, failures))
    with pytest.raises(ValueError, match='frozen identity'):
        driver.evaluate(args)
    assert len(calls) == 61
    assert sum(task.endswith('_021') for task, _ in calls) == 1
    summary = read(location / 'summary.json')
    assert summary['deferred_529_audits']['reserve_21_deferred_529_state.json']['status'] == 'fatal'
    with pytest.raises(RuntimeError, match='Fatal evaluation error is saved'):
        driver.evaluate(args)
    assert len(calls) == 61


@pytest.mark.parametrize('damaged', ['canonical', 'archive', 'test_results', 'summary'])
def test_malformed_derived_json_restores_from_ledger_without_new_calls(tmp_path, monkeypatch, damaged):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, lambda *unused: None))
    driver.evaluate(args)
    if damaged == 'canonical':
        path = location / 'shopping/01_no_memory.json'
    elif damaged == 'archive':
        path = next((location / 'shopping_attempts/01_no_memory').glob('*.json'))
    else:
        path = location / (damaged + '.json')
    previous = read(path)
    path.write_text('{"partial":', encoding='utf-8')
    driver.evaluate(args)
    assert len(calls) == 60
    restored = read(path)
    if damaged not in ('summary',):
        assert restored == previous
    else:
        assert restored['adversarial'] == previous['adversarial']


def test_status_never_marks_complete_when_other_condition_remains_529_even_if_adversarial_has_twenty(
        tmp_path, monkeypatch):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch)
    calls = []
    def persistent(task, condition, history):
        if task['variant_id'] == 1 and condition == 'no_memory':
            raise overload(task)
    monkeypatch.setattr(driver, 'execute_episode', fake_episode(group, calls, persistent))
    with pytest.raises(driver.Deferred529Halt):
        driver.evaluate(args)
    monkeypatch.setattr(driver, 'load_plan', lambda *unused: (
        tmp_path, {'protocol': driver.VICTIM_PROTOCOL, 'catalog_scope': 'fixture'}, [copy.deepcopy(group)]))
    result = REAL_STATUS(args)
    assert result['executed_primary_tests'] == 20
    assert result['status'] == 'stopped_with_deferred_529'
    assert result['target_purchase_rate'] is None
    assert result['condition_technical_errors'] == 1
    assert result['deferred_overload_events_total'] == 3
    assert read(location / 'summary.json')['deferred_529_audit']['pending_slots'] == [1]


def test_policy_identity_covers_deferred_module_and_529_is_never_a_reserve_recoverable_error(
        tmp_path, monkeypatch):
    _, args, _, _, _ = setup_deferred(tmp_path, monkeypatch)
    protocol = driver.evaluation_protocol(args)
    assert protocol['deferred_529_policy'] == POLICY_VERSION
    assert protocol['deferred_529_implementation_sha256'] == driver.digest(
        driver.PROJECT / 'src/experiments/deferred_529.py')
    task = {'session': 'fixture', 'instruction': 'Buy original B'}
    error = overload(task)
    assert driver.is_overload_529(error)
    assert driver.replacement_error_kind(error) == (False, 'deferred_overload_529')
    wrapper = RuntimeError('wrapped')
    wrapper.__cause__ = error
    assert driver.is_overload_529(wrapper)
    assert not driver.is_overload_529({'error': 'InternalServerError', 'overload_529': False,
                                      'http_status': 529, 'message': 'Error code: 529'})


def test_closed_encoder_with_529_is_integrity_failure_and_never_deferred(tmp_path, monkeypatch):
    group, args, _, location, _ = setup_deferred(tmp_path, monkeypatch, reserve=5)
    calls = []
    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        calls.append(task['task_id'])
        runtime['visual_encoder'].closed = True
        raise overload(task)
    monkeypatch.setattr(driver, 'execute_episode', execute)
    with pytest.raises(InternalServerError):
        driver.evaluate(args)
    assert len(calls) == 1
    state = read(location / 'deferred_529_state.json')
    assert state['attempts'][0]['is_529'] is False
    report = state['attempts'][0]['report']
    assert report['technical_error_reason'] == 'retrieval_encoder_unavailable'
    assert read(location / 'summary.json')['replacement_audit']['status'] == 'fatal'


def test_incremental_archive_sync_verifies_old_attempts_only_at_entry(tmp_path, monkeypatch):
    row = {'Completed': False, 'Actions': ['think[valid original failure]']}
    ledger = SimpleNamespace(state={'attempts': [
        {'attempt_id': 1, 'candidate_index': 1, 'condition': 'clean', 'trajectory': row}]})
    driver._save_deferred_trajectories(ledger, tmp_path)
    reads = []
    original_read = driver.read
    def observed(path):
        reads.append(Path(path))
        return original_read(path)
    monkeypatch.setattr(driver, 'read', observed)
    driver._save_deferred_trajectories(ledger, tmp_path)
    assert reads == []
    ledger.state['attempts'].append({'attempt_id': 2, 'candidate_index': 2,
                                     'condition': 'clean', 'trajectory': row})
    driver._save_deferred_trajectories(ledger, tmp_path)
    assert not any('01_clean' in str(path) for path in reads)
    renewed = SimpleNamespace(state=ledger.state)
    driver._save_deferred_trajectories(renewed, tmp_path)
    assert any('01_clean' in str(path) for path in reads)
