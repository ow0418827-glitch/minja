"""Reserve-demand integration: pairing, audit, interrupted runs and finite budgets."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver
from src.tests.test_targeted_joint_driver import fixed_group, args as targeted_args


def setup_run(tmp_path, monkeypatch):
    from src.vision import visual_retrieval
    group, args = fixed_group(tmp_path), targeted_args()
    args.query_specificity = 'vague'
    args.memory_top_k = 1
    args.reserve_tests = 5
    folder = tmp_path / 'groups' / group['id']
    for name in ('memory.json', 'clean_memory.json'):
        write(folder / name, [{'Id': f'history_{i}'} for i in range(15)])
    original = {name: (folder / name).read_bytes() for name in ('memory.json', 'clean_memory.json')}
    monkeypatch.setattr(driver, 'load_plan', lambda *unused: (tmp_path, {}, [copy.deepcopy(group)]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *unused: {})
    monkeypatch.setattr(driver, 'server_health', lambda *unused: {})
    monkeypatch.setattr(driver, 'resolve_history_binding', lambda *unused, **kwargs: copy.deepcopy(group))
    monkeypatch.setattr(driver, 'verify_banks', lambda *unused: {
        'source_memory_id': 'history_14', 'history_binding_sha256': group['history_binding_sha256']})
    monkeypatch.setattr(driver, 'runtime', lambda *unused, **kwargs: {
        'visual_encoder': SimpleNamespace(snapshot={'kind': 'reserve-fixture'}),
        'model': SimpleNamespace(api_key='PRIVATE_TEST_KEY')})
    monkeypatch.setattr(visual_retrieval, 'VisualMemoryRetriever', lambda *unused, **kwargs: object())
    monkeypatch.setattr(driver, 'status', lambda *unused: {})
    location, _ = driver.evaluation_location(args, folder)
    return group, args, folder, location, original


def shopping_row(group, task, condition, *, completed=True):
    return {'Id': task['session'], 'Instruction': task['instruction'],
            'Completed': completed, 'FinalASIN': group['a'] if condition == 'adversarial' else group['b'],
            'EnvironmentReward': 0, 'Actions': ['Action: search[camera]'],
            'TerminationReason': 'purchase_completed' if completed else 'step_limit'}


def condition_from(task):
    return next(name for name in ('no_memory', 'clean', 'adversarial') if name in task['session'])


def test_normal_failures_keep_twenty_pairs_without_using_spares_and_resume_without_calls(tmp_path, monkeypatch):
    group, args, folder, location, original = setup_run(tmp_path, monkeypatch)
    calls = []
    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        condition = condition_from(task)
        calls.append((task['task_id'], condition))
        assert kwargs['query_image'] == group['query_image']
        assert len(bank) == (0 if condition == 'no_memory' else 15)
        return '', shopping_row(group, task, condition, completed=False), False
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.evaluate(args)
    summary = read(location / 'summary.json')
    assert len(calls) == 60
    assert summary['replacement_audit']['reserve_used'] == 0
    assert summary['adversarial']['valid_tests'] == 20
    assert summary['adversarial']['target_purchase_rate'] == 0
    assert summary['test_design']['candidate_wordings'] == 25
    assert summary['test_design']['wordings'] == 20
    trajectories = {p: p.read_bytes() for p in (location / 'shopping').glob('*.json')}
    driver.evaluate(args)
    assert len(calls) == 60
    assert all(p.read_bytes() == body for p, body in trajectories.items())
    assert all((folder / name).read_bytes() == body for name, body in original.items())


def test_one_technical_branch_replaces_entire_pair_and_preserves_original_other_branches(tmp_path, monkeypatch):
    group, args, folder, location, original = setup_run(tmp_path, monkeypatch)
    calls = []
    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        condition = condition_from(task)
        calls.append((task['task_id'], condition))
        if task['variant_id'] == 1 and condition == 'adversarial':
            raise requests.exceptions.SSLError('connection EOF PRIVATE_TEST_KEY')
        return '', shopping_row(group, task, condition), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.evaluate(args)
    result = read(location / 'test_results.json')
    summary = read(location / 'summary.json')
    assert len(calls) == 63
    assert all(len(rows) == 20 for rows in result.values())
    assert len({tuple(x['task_id'] for x in rows) for rows in result.values()}) == 1
    assert result['adversarial'][0]['task_id'].endswith('_021')
    assert result['adversarial'][0]['evaluation_slot'] == 1
    assert result['adversarial'][0]['replaces_task_id'].endswith('_001')
    assert summary['adversarial']['target_purchase_rate'] == 1
    assert summary['replacement_audit']['technical_errors_total'] == 1
    assert summary['replacement_audit']['reserve_used'] == 1
    ledger = read(location / 'candidate_results.json')
    first = ledger['candidates'][0]
    assert first['status'] == 'excluded_technical'
    assert set(first['results']) == {'no_memory', 'clean', 'adversarial'}
    assert first['results']['clean']['completed']
    assert (location / 'shopping' / '01_clean.json').exists()
    assert (location / 'shopping' / '21_clean.json').exists()
    assert 'PRIVATE_TEST_KEY' not in (location / 'technical_errors.json').read_text(encoding='utf-8')
    assert read(location / 'technical_errors.json')['total'] == 1
    driver.evaluate(args)
    assert len(calls) == 63
    assert all((folder / name).read_bytes() == body for name, body in original.items())


def test_interruption_resumes_only_uncommitted_conditions(tmp_path, monkeypatch):
    group, args, _, location, _ = setup_run(tmp_path, monkeypatch)
    calls, interrupt = [], [True]
    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        condition = condition_from(task)
        if task['variant_id'] == 1 and condition == 'clean' and interrupt[0]:
            interrupt[0] = False
            raise KeyboardInterrupt()
        calls.append((task['task_id'], condition))
        return '', shopping_row(group, task, condition), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    with pytest.raises(KeyboardInterrupt):
        driver.evaluate(args)
    first = (location / 'shopping' / '01_no_memory.json').read_bytes()
    assert len(calls) == 1
    driver.evaluate(args)
    assert len(calls) == len(set(calls)) == 60
    assert (location / 'shopping' / '01_no_memory.json').read_bytes() == first
    assert read(location / 'summary.json')['replacement_audit']['reserve_used'] == 0


def test_committed_trajectory_is_restored_without_repeating_model_call(tmp_path, monkeypatch):
    group, args, _, location, _ = setup_run(tmp_path, monkeypatch)
    calls, fail_once = [], [True]
    actual_write = driver.write
    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        condition = condition_from(task)
        calls.append((task['task_id'], condition))
        return '', shopping_row(group, task, condition), True
    def interrupted_write(path, content):
        if Path(path).name == '01_no_memory.json' and fail_once[0]:
            fail_once[0] = False
            raise OSError('Interrupted after the committed result')
        return actual_write(path, content)
    monkeypatch.setattr(driver, 'execute_episode', execute)
    monkeypatch.setattr(driver, 'write', interrupted_write)
    with pytest.raises(OSError, match='committed'):
        driver.evaluate(args)
    assert len(calls) == 1
    assert not (location / 'shopping' / '01_no_memory.json').exists()
    driver.evaluate(args)
    assert len(calls) == len(set(calls)) == 60
    assert (location / 'shopping' / '01_no_memory.json').exists()
    assert read(location / 'summary.json')['replacement_audit']['reserve_used'] == 0


def test_saved_shopping_trajectory_cannot_silently_change_on_resume(tmp_path, monkeypatch):
    group, args, _, location, _ = setup_run(tmp_path, monkeypatch)
    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        return '', shopping_row(group, task, condition_from(task)), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.evaluate(args)
    path = location / 'shopping' / '01_no_memory.json'
    row = read(path)
    row['FinalASIN'] = 'tampered'
    write(path, row)
    with pytest.raises(ValueError, match='trajectory'):
        driver.evaluate(args)


def test_five_spares_are_finite_and_unfilled_rate_stays_unknown(tmp_path, monkeypatch):
    group, args, _, location, _ = setup_run(tmp_path, monkeypatch)
    def execute(options, root, selected, task, bank, path, runtime, **kwargs):
        condition = condition_from(task)
        if task['variant_id'] <= 6:
            raise requests.ConnectionError('temporary image connection failure')
        return '', shopping_row(group, task, condition), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.evaluate(args)
    summary = read(location / 'summary.json')
    assert summary['replacement_audit']['status'] == 'reserve_exhausted'
    assert summary['replacement_audit']['reserve_used'] == 5
    assert summary['replacement_audit']['accepted_pairs'] == 19
    assert summary['replacement_audit']['technical_errors_total'] == 6
    assert summary['adversarial']['planned_tests'] == 20
    assert summary['adversarial']['executed_tests'] == 19
    assert summary['adversarial']['target_purchase_rate'] is None


def test_integrity_failure_is_logged_and_stops_without_burning_reserves(tmp_path, monkeypatch):
    _, args, _, location, _ = setup_run(tmp_path, monkeypatch)
    calls = []
    def execute(*unused, **kwargs):
        calls.append(True)
        raise ValueError('Frozen image digest mismatch')
    monkeypatch.setattr(driver, 'execute_episode', execute)
    with pytest.raises(ValueError, match='digest'):
        driver.evaluate(args)
    audit = read(location / 'summary.json')['replacement_audit']
    assert audit['status'] == 'fatal'
    assert audit['reserve_used'] == 0
    assert audit['fatal_errors'] == 1
    with pytest.raises(RuntimeError, match='Fatal'):
        driver.evaluate(args)
    assert len(calls) == 1


def test_query_image_lost_during_episode_is_logged_without_rereading_it(tmp_path, monkeypatch):
    group, args, _, location, _ = setup_run(tmp_path, monkeypatch)
    def execute(*unused, **kwargs):
        Path(group['query_image']).unlink()
        raise FileNotFoundError('Reference image was lost')
    monkeypatch.setattr(driver, 'execute_episode', execute)
    with pytest.raises(FileNotFoundError):
        driver.evaluate(args)
    errors = read(location / 'technical_errors.json')
    report = errors['events'][0]['report']
    assert report['query_image_sha256'] == group['reference_query_sha256']
    assert report['error'] == 'FileNotFoundError'
    assert report['technical_recoverable'] is False
    assert errors['state'] == 'fatal'


def test_closed_encoder_timeout_is_fatal_without_using_spares(tmp_path, monkeypatch):
    _, args, _, location, _ = setup_run(tmp_path, monkeypatch)
    encoder = SimpleNamespace(snapshot={'kind': 'fixture'}, closed=False)
    monkeypatch.setattr(driver, 'runtime', lambda *unused, **kwargs: {
        'visual_encoder': encoder, 'model': SimpleNamespace(api_key='PRIVATE_TEST_KEY')})
    def execute(*unused, **kwargs):
        encoder.closed = True
        raise TimeoutError('Encoding worker timed out and closed')
    monkeypatch.setattr(driver, 'execute_episode', execute)
    with pytest.raises(TimeoutError):
        driver.evaluate(args)
    summary = read(location / 'summary.json')
    assert summary['replacement_audit']['reserve_used'] == 0
    assert summary['replacement_audit']['status'] == 'fatal'
    report = read(location / 'technical_errors.json')['events'][0]['report']
    assert report['technical_error_reason'] == 'retrieval_encoder_unavailable'


def test_condition_set_has_independent_reserve_protocol_directory(tmp_path):
    args = targeted_args()
    args.reserve_tests = 5
    with_controls, _ = driver.evaluation_location(args, tmp_path)
    args.include_controls = False
    without_controls, _ = driver.evaluation_location(args, tmp_path)
    assert with_controls != without_controls


def test_reserve_protocol_is_separate_from_legacy_twenty_queries(tmp_path):
    group, args = fixed_group(tmp_path), targeted_args()
    args.query_specificity = 'vague'
    old_location, _ = driver.evaluation_location(args, tmp_path)
    old, old_digest = driver.evaluation_trials(args, group, tmp_path)
    protected = (old_location / 'targeted_tasks.json').read_bytes()
    args.reserve_tests = 5
    new_location, _ = driver.evaluation_location(args, tmp_path)
    tasks, new_digest = driver.evaluation_trials(args, group, tmp_path)
    assert new_location != old_location and new_digest != old_digest
    assert len(old) == 20 and len(tasks) == 25
    assert [x['instruction'] for x in tasks[:20]] == [x['instruction'] for x in old]
    assert (old_location / 'targeted_tasks.json').read_bytes() == protected
    assert read(new_location / 'targeted_tasks.json')['reserve_count'] == 5


@pytest.mark.parametrize('error,replaceable', [
    (requests.Timeout(), True), (requests.ConnectionError(), True),
    (requests.exceptions.SSLError('UNEXPECTED_EOF_WHILE_READING'), True),
    (requests.exceptions.SSLError('CERTIFICATE_VERIFY_FAILED'), False),
    (ValueError('image digest mismatch'), False), (OSError('disk full'), False),
])
def test_only_known_transient_faults_can_replace_a_task(error, replaceable):
    assert driver.replacement_error_kind(error)[0] is replaceable


@pytest.mark.parametrize('status,replaceable', [(404, True), (503, True), (401, False), (403, False)])
def test_http_failure_classification(status, replaceable):
    response = requests.Response()
    response.status_code = status
    assert driver.replacement_error_kind(requests.HTTPError(response=response))[0] is replaceable
