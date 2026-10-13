from types import SimpleNamespace

import pytest

from src.environment.local_webshop import read, write
from src.experiments import replace_normal_memory_experiment as replacement


@pytest.fixture
def collection(tmp_path, monkeypatch):
    group = {'id': replacement.GROUP, 'number': 6, 'target_asin': 'attack-product',
             'training': {'task_id': 'training', 'instruction': 'original attacker'}, 'trials': [],
             'normal_candidates': [{'task_id': f'candidate-{i}', 'instruction': f'normal demand {i}',
                                    'goal_index': 9000 + i, 'goal': {}} for i in range(6)]}
    candidate_plan = replacement.driver.history_candidate_plan(group, primary_count=1, reserve_count=5)
    normals = [{'Id': f'old-{i}', 'FinalASIN': f'product-{i}'} for i in range(13)]
    state = {'normal': normals, 'designated': {'Id': 'attack'}, 'attempts': [], 'cursor': 0,
             'normal_collection_plan': candidate_plan}
    folder = tmp_path / 'groups' / replacement.GROUP
    write(folder / 'history_state.json', state)
    args = SimpleNamespace(root=tmp_path, normal_retries=2, history_reserve_count=5,
        replacement_policy={'candidate_plan': candidate_plan, 'removed_position': 9,
                            'retained_normal_ids': [row['Id'] for row in normals],
                            'excluded_replacement_asins': ['removed-product']})
    monkeypatch.setattr(replacement.driver, 'load_plan', lambda a: (tmp_path, {'groups': [group]}, [group]))
    monkeypatch.setattr(replacement.driver, 'runtime', lambda a: {})
    monkeypatch.setattr(replacement.driver, 'load_auxiliary', lambda p: ('neutral', {}))
    bindings = []
    monkeypatch.setattr(replacement.driver, 'save_history_binding', lambda *a: bindings.append(a[-1]))
    calls = []

    def install(decisions, purchased='new-product'):
        iterator = iter(decisions)

        def attempt(a, root, g, task, current, state_path, memory_path, rt, excluded, *, kind, candidate_index):
            accepted = next(iterator)
            calls.append((candidate_index, task['session'], set(excluded)))
            current['attempts'].append({'kind': 'normal', 'task_id': task['task_id'],
                'candidate_index': candidate_index, 'accepted': accepted,
                'status': 'accepted' if accepted else 'rejected'})
            if accepted:
                current['normal'].append({'Id': 'new-memory', 'FinalASIN': purchased})
            current['normal_collection_audit'] = replacement.driver.history_reserve_audit(
                candidate_plan, current['attempts'], normal_retries=2)
            write(state_path, current)
            return accepted

        monkeypatch.setattr(replacement.driver, 'collect_history_attempt', attempt)

    return args, folder, normals, calls, bindings, install


def test_first_valid_purchase_preserves_thirteen_memories_and_position(collection):
    args, folder, original, calls, bindings, install = collection
    install([True])
    replacement.collect_replacement(args)
    state = read(folder / 'history_state.json')
    assert state['normal'][9]['Id'] == 'new-memory'
    assert [row for row in state['normal'] if row['Id'] != 'new-memory'] == original
    assert len(calls) == len(bindings) == 1
    assert calls[0][2] == {'attack-product', 'removed-product'}
    assert state['normal_collection_audit']['reserve_used'] == 0


def test_primary_failure_uses_first_reserve_without_rank_filter(collection):
    args, folder, _, calls, _, install = collection
    install([False, False, True])
    replacement.collect_replacement(args)
    assert [call[0] for call in calls] == [0, 0, 1]
    audit = read(folder / 'history_state.json')['normal_collection_audit']
    assert audit['status'] == 'completed' and audit['reserve_used'] == 1
    assert audit['trigger_failures'][0]['task_id'] == 'candidate-0'


def test_all_five_reserves_exhausted_preserves_partial_state(collection):
    args, folder, original, calls, bindings, install = collection
    install([False] * 12)
    with pytest.raises(RuntimeError, match='exhausted'):
        replacement.collect_replacement(args)
    assert len(calls) == 12 and bindings == []
    assert read(folder / 'history_state.json')['normal'] == original
    assert read(args.root / 'replacement_failure.json')['reserve_exhausted']


def test_replacement_cannot_reintroduce_removed_product(collection):
    args, _, _, _, bindings, install = collection
    install([True], purchased='removed-product')
    with pytest.raises(ValueError, match='exclusion'):
        replacement.collect_replacement(args)
    assert bindings == []


def test_bound_replacement_resume_does_not_recollect(collection, monkeypatch):
    args, folder, _, calls, _, install = collection
    write(folder / 'history_binding.json', {})
    audits = []
    monkeypatch.setattr(replacement.driver, 'verify_collected_history', lambda *a: audits.append(True))
    install([])
    replacement.collect_replacement(args)
    assert audits == [True] and calls == []
