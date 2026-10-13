"""Treatment selection uses independent identities and reuses saved single runs."""
from types import SimpleNamespace

import pytest

from src.environment.local_webshop import read
from src.experiments import nine_group_experiment as driver
from src.tests.test_nine_group_experiment import evaluation_fixture, successful_row
from src.tests.test_reserve_driver import setup_run, shopping_row, condition_from


@pytest.mark.parametrize('choice,controls,expected', [
    (None, False, ('adversarial',)),
    (None, True, ('no_memory', 'clean', 'adversarial')),
    ('all', False, ('no_memory', 'clean', 'adversarial')),
    ('all', True, ('no_memory', 'clean', 'adversarial')),
    ('adversarial', True, ('adversarial',)),
    ('clean', True, ('clean',)),
    ('no_memory', True, ('no_memory',)),
])
def test_condition_selection_preserves_legacy_and_overrides_controls(choice, controls, expected):
    assert driver.evaluation_conditions(SimpleNamespace(condition=choice, include_controls=controls)) == expected


def test_invalid_condition_rejected_before_execution():
    with pytest.raises(ValueError, match='condition'):
        driver.evaluation_conditions(SimpleNamespace(condition='unknown'))


def test_explicit_treatments_have_distinct_identities_without_changing_legacy_default():
    args = SimpleNamespace(include_controls=True)
    original = driver.evaluation_protocol(args)
    args.condition = None
    assert driver.evaluation_protocol(args) == original
    protocols = []
    for choice in ('all', 'adversarial', 'clean', 'no_memory'):
        args.condition = choice
        protocol = driver.evaluation_protocol(args)
        assert protocol['evaluation_conditions'] == list(driver.evaluation_conditions(args))
        protocols.append(driver.fingerprint(protocol))
    assert len(set(protocols)) == 4
    assert driver.fingerprint(original) not in protocols


@pytest.mark.parametrize('choice,expected', [
    ('adversarial', ('adversarial',)),
    ('clean', ('clean',)),
    ('no_memory', ('no_memory',)),
    ('all', ('no_memory', 'clean', 'adversarial')),
])
def test_selected_conditions_execute_and_resume_with_correct_status(tmp_path, monkeypatch, choice, expected):
    group, folder, args = evaluation_fixture(tmp_path, monkeypatch)
    args.condition = choice
    if choice == 'all':
        args.include_controls = False
    calls = []

    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        condition = condition_from(task)
        calls.append((task['task_id'], condition))
        assert condition in expected
        assert bool(bank) == (condition != 'no_memory')
        row = successful_row(task['session'], group['a'] if condition == 'adversarial' else group['b'])
        row['EnvironmentReward'] = 0 if condition == 'adversarial' else 1
        row['RetrievalTrace'] = [{'SelectedIds': ['source']}] if bank else []
        return '', row, []

    monkeypatch.setattr(driver, 'execute_episode', execute)
    result = driver.evaluate(args)
    location, _ = driver.evaluation_location(args, folder)
    reports = read(location / 'test_results.json')
    summary = read(location / 'summary.json')
    assert set(reports) == set(expected)
    assert len(calls) == len(set(calls)) == 20 * len(expected)
    assert all(len(rows) == 20 for rows in reports.values())
    assert result['status'] == 'completed'
    assert result['primary_condition'] == ('adversarial' if 'adversarial' in expected else choice)
    assert result['planned_primary_tests'] == result['executed_primary_tests'] == 20
    assert result['planned_condition_tests'] == result['executed_condition_tests'] == 20 * len(expected)
    assert result['target_purchase_rate'] == (1 if 'adversarial' in expected else 0)
    if choice != 'all':
        assert 'target_purchase_increment' not in summary
        assert 'correct_to_target_error_pairs' not in summary
    saved = {path: path.read_bytes() for path in (location / 'shopping').glob('*.json')}
    driver.evaluate(args)
    assert len(calls) == 20 * len(expected)
    assert all(path.read_bytes() == content for path, content in saved.items())


@pytest.mark.parametrize('choice', ['clean', 'no_memory'])
def test_single_controls_with_reserves_use_selected_primary_tasks(tmp_path, monkeypatch, choice):
    group, args, folder, _, _ = setup_run(tmp_path, monkeypatch)
    args.condition = choice
    calls = []

    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        condition = condition_from(task)
        calls.append(condition)
        return '', shopping_row(group, task, condition), True

    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.evaluate(args)
    location, _ = driver.evaluation_location(args, folder)
    summary = read(location / 'summary.json')
    assert calls == [choice] * 20
    assert summary[choice]['executed_tests'] == 20
    assert summary[choice]['status'] == 'completed'
    assert len(summary['test_design']['selected_task_ids']) == 20
    assert summary['replacement_audit']['reserve_used'] == 0
    assert 'adversarial' not in summary
    assert driver.evaluation_protocol(args)['include_controls'] is False


@pytest.mark.parametrize('choice', ['adversarial', 'clean', 'no_memory'])
def test_single_condition_overload_recovers_same_request_without_other_treatments(tmp_path, monkeypatch, choice):
    from src.tests.test_deferred_529_driver import setup_deferred, overload
    group, args, folder, _, _ = setup_deferred(tmp_path, monkeypatch)
    args.condition = choice
    calls = []

    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        condition = condition_from(task)
        calls.append((task['task_id'], condition))
        if task['variant_id'] == 1 and calls.count((task['task_id'], condition)) == 1:
            raise overload(task)
        return '', shopping_row(group, task, condition), True

    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.evaluate(args)
    location, _ = driver.evaluation_location(args, folder)
    summary = read(location / 'summary.json')
    assert len(calls) == 21 and {condition for _, condition in calls} == {choice}
    assert calls[0] == calls[-1]
    assert summary['deferred_529_audit']['status'] == 'completed'
    assert summary[choice]['executed_tests'] == 20
    assert summary[choice]['technical_errors'] == 0
    assert summary['recovered_overload_events_total'] == 1
