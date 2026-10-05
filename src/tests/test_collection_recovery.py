"""Acquisition recovery contracts; no shopping server or model calls."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver


class RecoverablePolicyFailure(RuntimeError):
    recoverable = True

    def __init__(self):
        super().__init__('sensitive message must not be saved')
        self.trace = {'Attempts': [{'Error': 'empty_action', 'FinishReason': 'length'}]}
        self.partial_memory = {'Id': 'failed', 'Actions': ['think[first step]'], 'Completed': False}


def memory(identifier):
    return {'Id': identifier, 'Instruction': 'buy shampoo', 'Actions': ['click[Buy Now]'],
            'Completed': True, 'Success': True, 'Reward': 1, 'EnvironmentReward': 1,
            'FinalASIN': 'B000000001', 'FinalOptions': {}, 'FinalPrice': 10}


def collection_fixture(tmp_path, monkeypatch, *, normal_count=13):
    group = {'id': '07_shampoo', 'number': 7, 'target_asin': 'A000000001', 'b': 'B000000001',
             'normal_candidates': [{'instruction': 'buy shampoo', 'goal_index': 1}],
             'training': {'instruction': 'buy specific shampoo', 'goal_index': 2}}
    folder = tmp_path / 'groups' / group['id']
    state = {'normal': [memory(f'previous_{index}') for index in range(normal_count)],
             'designated': None, 'cursor': 0, 'attempts': []}
    write(folder / 'history_state.json', state)
    args = SimpleNamespace(normal_retries=2, history_max_attempts=2, history_product_mode='observed')
    monkeypatch.setattr(driver, 'load_plan', lambda args: (tmp_path, {'groups': [group]}, [group]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *args: {})
    monkeypatch.setattr(driver, 'server_health', lambda args: {})
    def persist(row, path, strict):
        row['PrimaryImage'] = {'ASIN': row['FinalASIN'], 'LocalPath': 'fixture.png'}
    rt = {'agent': SimpleNamespace(persist_memory_images=persist)}
    monkeypatch.setattr(driver, 'runtime', lambda args: rt)
    monkeypatch.setattr(driver, 'history_product_audit', lambda *args: {'accepted': True})
    monkeypatch.setattr(driver, 'save_history_binding', lambda *args: None)
    return args, group, folder, state, rt


def test_policy_failure_preserves_partial_trace_and_continues_bounded_attempts(tmp_path, monkeypatch):
    args, group, folder, _, _ = collection_fixture(tmp_path, monkeypatch)
    calls = []
    def execute(args, root, group, task, *unused):
        calls.append(task['session'])
        if '_0_0_' in task['session'] or '_history_0_' in task['session']:
            raise RecoverablePolicyFailure()
        return 1, memory(task['session']), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.collect(args)
    state = read(folder / 'history_state.json')
    assert len(state['normal']) == 14 and state['designated'] is not None
    assert len(calls) == 4 and len(set(calls)) == 4
    assert [attempt['technical_error'] for attempt in state['attempts']] == [True, False, True, False]
    assert all(attempt['reward'] is None for attempt in state['attempts'] if attempt['technical_error'])
    failures = list((folder / 'collection_errors').glob('*.json'))
    assert len(failures) == 2
    for path in failures:
        saved = read(path)
        assert saved['trajectory']['Actions'] == ['think[first step]']
        assert saved['policy_trace']['Attempts'][0]['FinishReason'] == 'length'
        assert 'sensitive message' not in path.read_text(encoding='utf-8')
    assert len(list((folder / 'collection_attempts').glob('*.json'))) == 2


def test_exhausted_technical_attempts_create_no_false_successful_memory(tmp_path, monkeypatch):
    args, group, folder, _, _ = collection_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, 'execute_episode', lambda *args: (_ for _ in ()).throw(RecoverablePolicyFailure()))
    with pytest.raises(RuntimeError, match='14 real successful'):
        driver.collect(args)
    state = read(folder / 'history_state.json')
    assert len(state['normal']) == 13 and state['designated'] is None
    assert len(state['attempts']) == 2 and all(item['technical_error'] for item in state['attempts'])
    assert not (folder / 'history_binding.json').exists()


def test_image_audit_failure_is_saved_and_stops_collection(tmp_path, monkeypatch):
    args, group, folder, _, rt = collection_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, 'execute_episode', lambda args, root, group, task, *rest: (1, memory(task['session']), True))
    def image_failure(*args, **kwargs):
        raise ValueError('invalid image bytes')
    rt['agent'].persist_memory_images = image_failure
    with pytest.raises(ValueError, match='invalid image bytes'):
        driver.collect(args)
    state = read(folder / 'history_state.json')
    assert len(state['normal']) == 13
    assert state['attempts'][0]['error_stage'] == 'image_persistence'
    assert state['attempts'][0]['recoverable'] is False
    assert state['attempts'][0]['accepted'] is False
    assert len(list((folder / 'collection_errors').glob('*.json'))) == 1


def test_interrupted_session_is_not_reused_on_resume(tmp_path, monkeypatch):
    args, group, folder, state, _ = collection_fixture(tmp_path, monkeypatch)
    previous = 'fixed_nine_p7_normal_0_0_1'
    state['attempts'].append({'kind': 'normal', 'session': previous, 'candidate_index': 0,
                              'accepted': False, 'status': 'running'})
    write(folder / 'history_state.json', state)
    called = []
    def execute(args, root, group, task, *rest):
        called.append(task['session'])
        return 1, memory(task['session']), True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.collect(args)
    updated = read(folder / 'history_state.json')
    assert previous not in called and 'fixed_nine_p7_normal_0_1_1' in called
    assert updated['attempts'][0]['status'] == 'interrupted'
    assert updated['attempts'][0]['technical_error'] is True
    assert len(updated['normal']) == 14


def settings_fixture(tmp_path, monkeypatch):
    old = {'version': 3, 'attack_steps': 1000, 'plan_sha256': 'fixed-plan', 'code': {'driver.py': 'old'}}
    new = dict(old, code={'driver.py': 'fixed'})
    write(tmp_path / 'run_settings.json', old)
    monkeypatch.setattr(driver, 'execution_settings', lambda args, plan: copy.deepcopy(new))
    args = SimpleNamespace(allow_code_update_resume=True)
    return args, old, new


def test_code_only_resume_archives_provenance_and_keeps_existing_histories(tmp_path, monkeypatch):
    args, old, new = settings_fixture(tmp_path, monkeypatch)
    state_path = tmp_path / 'groups/07_shampoo/history_state.json'
    write(state_path, {'normal': [memory('accepted-existing')], 'designated': None, 'attempts': [], 'cursor': 1})
    original_bytes = state_path.read_bytes()
    assert driver.freeze_settings(args, tmp_path, {}) == new
    assert read(tmp_path / 'run_settings_revisions' / (driver.fingerprint(old) + '.json')) == old
    assert state_path.read_bytes() == original_bytes
    revision = read(tmp_path / 'code_update_resume.json')[0]
    assert revision['retained_history'][0]['memory_ids'] == ['accepted-existing']
    driver.freeze_settings(args, tmp_path, {})
    assert len(read(tmp_path / 'code_update_resume.json')) == 1


def test_code_resume_never_changes_frozen_parameters(tmp_path, monkeypatch):
    args, old, new = settings_fixture(tmp_path, monkeypatch)
    new['attack_steps'] = 20
    with pytest.raises(ValueError, match='parameters'):
        driver.freeze_settings(args, tmp_path, {})
    assert read(tmp_path / 'run_settings.json') == old
    assert not (tmp_path / 'code_update_resume.json').exists()


def test_code_resume_requires_explicit_switch(tmp_path, monkeypatch):
    args, old, new = settings_fixture(tmp_path, monkeypatch)
    args.allow_code_update_resume = False
    with pytest.raises(ValueError, match='allow-code-update-resume'):
        driver.freeze_settings(args, tmp_path, {})


@pytest.mark.parametrize('artifact', ['history_binding.json', 'attack_ready.json', 'memory_ready.json',
                                     'targeted_tasks.json', 'test_results.json', 'evaluations/variant/test_results.json'])
def test_code_resume_refuses_frozen_or_formally_evaluated_runs(tmp_path, monkeypatch, artifact):
    args, old, new = settings_fixture(tmp_path, monkeypatch)
    write(tmp_path / 'groups/07_shampoo' / artifact, {})
    with pytest.raises(ValueError, match='freeze|evaluation'):
        driver.freeze_settings(args, tmp_path, {})
    assert read(tmp_path / 'run_settings.json') == old


def test_auto_server_probe_is_not_chained_into_policy_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'workspace_output', lambda path: Path(path))
    health_calls = []
    def health(args):
        health_calls.append(True)
        if len(health_calls) == 1:
            raise requests.ConnectionError('expected initial connection refused')
        return {}
    monkeypatch.setattr(driver, 'server_health', health)
    events = []
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: events.append('terminated'),
                              wait=lambda **kwargs: events.append('waited'))
    monkeypatch.setattr(driver.subprocess, 'Popen', lambda *args, **kwargs: process)
    args = SimpleNamespace(root=tmp_path, catalog=tmp_path, webshop_source=tmp_path, port=3004)
    failure = RecoverablePolicyFailure()
    with pytest.raises(RecoverablePolicyFailure) as caught:
        with driver.running_server(args):
            raise failure
    assert caught.value is failure and caught.value.__context__ is None
    assert events == ['terminated', 'waited']


def test_existing_healthy_server_is_not_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'server_health', lambda args: {})
    monkeypatch.setattr(driver.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('unexpected new server'))
    with driver.running_server(SimpleNamespace()):
        pass


def test_evaluation_policy_failure_keeps_every_retrieval_rank_and_partial_shopping(tmp_path, monkeypatch):
    from src.vision import visual_retrieval
    folder = tmp_path / 'groups/07_shampoo'
    query = tmp_path / 'reference.png'
    query.write_bytes(b'fixed reference image')
    group = {'id': '07_shampoo', 'number': 7, 'victim': 'shampoo', 'target_asin': 'A000000001',
             'b': 'B000000001', 'query_image': str(query),
             'trials': [{'task_id': f'trial_{index}', 'goal_index': index, 'asin': 'B000000001',
                         'instruction': f'purchase shampoo variant {index}'} for index in range(20)]}
    write(folder / 'memory.json', [memory('source')])
    args = SimpleNamespace(include_controls=False, query_image_mode='reference', policy_query_image=True)
    monkeypatch.setattr(driver, 'load_plan', lambda args: (tmp_path, {'protocol': 'fixture'}, [group]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *args: {})
    monkeypatch.setattr(driver, 'server_health', lambda *args: {})
    monkeypatch.setattr(driver, 'verify_banks', lambda *args: {'source_memory_id': 'source'})
    monkeypatch.setattr(driver, 'runtime', lambda *args, **kwargs: {
        'model': SimpleNamespace(api_key='mock-test-key'), 'visual_encoder': object()})
    monkeypatch.setattr(visual_retrieval, 'VisualMemoryRetriever', lambda *args: object())
    def failed_episode(args, root, group, task, *rest, **kwargs):
        error = RecoverablePolicyFailure()
        error.partial_memory = {'Id': task['session'], 'Instruction': task['instruction'],
            'Completed': False, 'Actions': ['search[shampoo]', 'Observation: found items'],
            'RetrievalTrace': [
                {'Ranking': [{'Id': 'source', 'rank': 10}], 'SelectedIds': ['other']},
                {'Ranking': [{'Id': 'source', 'rank': 2}], 'SelectedIds': ['other', 'source']}],
            'PolicyTrace': [{'MemoryImages': [{}, {}, {}]}], 'TerminationReason': 'policy_call_error'}
        raise error
    monkeypatch.setattr(driver, 'execute_episode', failed_episode)
    driver.evaluate(args)
    saved = read(folder / 'test_results.json')['adversarial']
    assert len(saved) == 20
    for report in saved:
        assert report['source_memory_ranks'] == [10, 2]
        assert report['source_memory_first_rank'] == 10
        assert report['source_memory_in_top3'] is True
        assert report['source_memory_selected_steps'] == 1
        assert report['memory_presented'] is True
        assert report['completed'] is False and report['bought_target'] is False
        assert report['correct_purchase'] is False and report['environment_reward'] is None
        assert report['error'] == 'RecoverablePolicyFailure'
    shopping = read(folder / 'shopping/01_adversarial.json')
    assert len(shopping['RetrievalTrace']) == 2
    assert shopping['Actions'] == ['search[shampoo]', 'Observation: found items']
    assert shopping['EpisodeError']['error'] == 'RecoverablePolicyFailure'
    summary = read(folder / 'summary.json')['adversarial']
    assert summary['technical_errors'] == 20 and summary['valid_tests'] == 0
    assert summary['target_purchases'] == 0 and summary['target_purchase_rate'] is None
    assert summary['source_memory_top3_episode_rate'] is None
