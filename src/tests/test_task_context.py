"""Offline regressions for losing the live task during history truncation."""
import pytest

from src.agent import minja


def test_trim_old_pairs_preserves_complete_task_and_latest_page():
    task = 'Buy a sensitive-teeth toothbrush under $30; see http://example.invalid/item.'
    latest = ['Action: click[ITEM]', 'Observation: START\n' + 'product details ' * 600 + '\nEND']
    actions = ['Action: search[old]', 'Observation: obsolete page'] + latest
    prompt = minja.build_shopping_prompt('EXAMPLE', task, actions, max_chars=6400)
    assert task in prompt
    assert '\n'.join(latest) in prompt
    assert 'obsolete page' not in prompt
    assert 'search[old]' not in prompt
    assert prompt.endswith('\n\nAction:')
    # Oversized current observations are intact, not silently sliced.
    assert len(prompt) > 6400


def test_examples_larger_than_budget_do_not_remove_task_or_corrupt_history():
    prompt = minja.build_shopping_prompt('example ' * 1000, 'Buy B', [
        'Action: search[old]', 'Observation: old',
        'Action: click[ITEM]', 'Observation: latest'], max_chars=6400)
    assert 'Instruction: Buy B' in prompt
    assert 'Action: click[ITEM]\nObservation: latest' in prompt
    assert 'Observation: old' not in prompt


@pytest.mark.parametrize('mode', ['no_memory', 'full', 'fragment'])
def test_every_policy_call_preserves_task_and_complete_recent_observation(monkeypatch, mode):
    task = 'Buy a manual toothbrush for sensitive teeth under 30 dollars.'
    class Environment:
        def __init__(self):
            self.steps = 0
        def step(self, idx, action):
            self.steps += 1
            self.latest = f'PAGE-{self.steps}-START\n' + 'product details ' * 200 + f'\nPAGE-{self.steps}-END'
            return self.latest, 0, False, {'instruction': task}
        def action_guidance(self, idx):
            return 'Allowed actions: search or click.'
    env = Environment()
    def policy(prompt, *args, **kwargs):
        assert 'Instruction: ' + task in prompt
        assert prompt.index('EXAMPLE-DEODORANT') < prompt.index('End of examples.')
        assert prompt.index('End of examples.') < prompt.index('Instruction: ' + task)
        if env.steps > 1:
            assert env.latest in prompt
        return 'search[toothbrush]'
    monkeypatch.setattr(minja, 'llm', policy)
    monkeypatch.setattr(minja, 'generate_examples', lambda *a, **kw: ('EXAMPLE-DEODORANT', ''))
    _, row, _ = minja.webshop_run_rap(
        'session', '', '', 'EXAMPLE-DEODORANT', env, None, 'fake', 'minimax',
        None, [] if mode == 'no_memory' else [{'Id': 'example'}], {}, 20,
        honest_reward=True, to_print=False,
        memory_mode='fragment' if mode == 'fragment' else 'full')
    assert len(row['PolicyTrace']) == 20
    assert row['TerminationReason'] == 'step_limit'
    # Trimming affects only the policy prompt, not the saved full trajectory.
    assert 'PAGE-2-START' in '\n'.join(row['Actions'])
    final = row['PolicyTrace'][-1]['Prompt']
    assert ('PAGE-2-START' in final) == (mode != 'no_memory')


def test_short_context_keeps_all_pairs_and_respects_budget():
    actions = ['Action: search[B]', 'Observation: results',
               'Action: click[B]', 'Observation: detail']
    prompt = minja.build_shopping_prompt('example', 'Buy B', actions, max_chars=6400)
    assert all(item in prompt for item in actions)
    assert len(prompt) <= 6400
