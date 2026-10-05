from types import SimpleNamespace
import numpy as np
import pytest

from src.agent import minja


@pytest.mark.parametrize('text,expected', [
    ('Action: search[toothbrush]', 'search[toothbrush]'),
    ('\nthink[check options]\n', 'think[check options]'),
    ('Let me search for you.', '__invalid_action_format__'),
    ('search[a]\nclick[b]', '__invalid_action_format__'),
])
def test_parser_requires_one_action(text, expected):
    assert minja.parse_action(text) == expected


def page(**kw):
    kind = kw['page_type']
    info = {'instruction': 'Buy B'}
    if kind == 'search':
        info.update(asins=['PRODUCTBBB'], has_next=kw.get('page_num', 1) < 2)
    if kind == 'end':
        info['reward'] = 1.0
    return kind, info


def run(monkeypatch, sequence):
    monkeypatch.setattr(minja, 'webshop_text', page)
    calls = []
    def policy(prompt, *a, **kw):
        calls.append(prompt)
        return sequence[len(calls)-1]
    monkeypatch.setattr(minja, 'llm', policy)
    result = minja.webshop_run_rap('session', '', '', 'shop', minja.webshopEnv(),
        None, 'fake', 'minimax', None, [], {}, len(sequence), honest_reward=True, to_print=False)
    return result, calls


def test_last_budgeted_action_is_executed(monkeypatch):
    (_, row, _), calls = run(monkeypatch, ['search[B]', 'click[PRODUCTBBB]', 'click[Buy Now]'])
    assert len(calls) == 3
    assert row['Completed'] and row['FinalASIN'] == 'PRODUCTBBB'
    assert all(t['Executed'] for t in row['PolicyTrace'])
    assert row['TerminationReason'] == 'purchase_completed'


def test_invalid_search_explains_recovery_and_state_is_preserved(monkeypatch):
    (_, row, _), calls = run(monkeypatch, ['search[B]', 'search[new]', 'click[Back to Search]', 'search[new]'])
    assert 'First click[Back to Search]' in calls[2]
    assert sum(t['InvalidAction'] for t in row['PolicyTrace']) == 1
    assert len(row['PolicyTrace']) == 4
    assert row['TerminationReason'] == 'step_limit'


def test_next_and_previous_follow_page_bounds(monkeypatch):
    monkeypatch.setattr(minja, 'webshop_text', page)
    env = minja.webshopEnv()
    env.step('s', 'reset'); env.step('s', 'search[B]')
    env.step('s', 'click[Next >]')
    assert env.sessions['s']['page_num'] == 2
    with pytest.raises(AssertionError, match='no next'):
        env.step('s', 'click[Next >]')
    env.step('s', 'click[< Prev]')
    assert env.sessions['s']['page_num'] == 1


def test_raw_response_and_parse_failure_are_both_logged():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Here is my advice.'), finish_reason='stop')]))))
    trace = {}
    assert minja.llm('task', 'fake', client, 'minimax', trace=trace) == '__invalid_action_format__'
    assert trace['RawResponse'] == 'Here is my advice.'
    assert trace['FormatValid'] is False


def test_embedding_alignment_and_fragment_preserve_initial_reason():
    seen = []
    class Encoder:
        def encode(self, texts):
            seen.append(texts)
            return np.array([[1., 0.] if 'think[' in t else [0., 1.] for t in texts], dtype=np.float32)
    memory = [{'Id': 'm', 'Instruction': 'Buy B', 'Success': True, 'Reward': 1,
               'Actions': ['Action: think[bridge reason]', 'Observation: OK.', 'Action: search[A]']}]
    encoder = Encoder()
    kept, embeddings = minja.generate_embeddings(memory, encoder)
    assert seen[-1][0] == memory[0]['Actions'][0]
    actions = ['Action: reset', 'Observation: home', 'Action: think[bridge reason]', 'Observation: OK.']
    prompt, _ = minja.generate_examples({'instruction': 'Buy B'}, actions, kept, embeddings,
        encoder, act_len=10, use_act_obs=True, k=1)
    assert 'Action: think[bridge reason]' in prompt
