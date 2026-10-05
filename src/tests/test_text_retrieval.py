import numpy as np
import pytest

from src.retrieval.text_retrieval import TextMemoryRetriever


class Encoder:
    def encode(self, texts):
        assert texts == ['Buy B']
        return np.array([[1., 0.]])


def test_cosine_ranking_ignores_images_and_preserves_embedding_alignment():
    memory = [{'Id': str(i), 'RetrievalImage': {'LocalPath': 'does-not-exist'}} for i in range(4)]
    embeddings = {'Instruction': np.array([[0., 1.], [2., 0.], [.8, .6], [-1., 0.]]),
                  'Actions': ['a0', 'a1', 'a2', 'a3'], 'Reward': [1, 2, 3, 4]}
    kept, vectors, ranked = TextMemoryRetriever(Encoder()).shortlist(memory, embeddings, 'Buy B', 3)
    assert [r['Id'] for r in ranked] == ['1', '2', '0', '3']
    assert [r['Id'] for r in kept] == ['1', '2', '0']
    assert vectors['Actions'] == ['a1', 'a2', 'a0']
    assert vectors['Reward'] == [2, 3, 1]
    np.testing.assert_allclose(vectors['RetrievalScores'], [1., .8, 0.])
    assert 'RetrievalScores' not in embeddings
    kept, _, _ = TextMemoryRetriever(Encoder()).shortlist(memory, embeddings, 'Buy B', 1)
    assert [r['Id'] for r in kept] == ['1']


def test_small_bank_and_invalid_vectors():
    r = TextMemoryRetriever(Encoder())
    assert r.shortlist([], {}, 'Buy B', 3) == ([], {}, [])
    with pytest.raises(ValueError, match='positive'):
        r.shortlist([], {}, 'Buy B', 0)
    with pytest.raises(ValueError, match='Invalid'):
        r.shortlist([{'Id': 'x'}], {'Instruction': np.array([[0., 0.]])}, 'Buy B', 3)
    kept, _, _ = r.shortlist([{'Id': 'x'}], {'Instruction': np.array([[1., 0.]])}, 'Buy B', 3)
    assert len(kept) == 1


def test_trace_distinguishes_shortlist_from_prompt_budget(monkeypatch):
    from src.agent import minja
    memory = [{'Id': 'oversized', 'Instruction': 'Buy B', 'Actions': ['Action: think[' + 'x' * 20000 + ']']}]
    embeddings = {'Instruction': np.array([[1., 0.]])}
    class Env:
        def step(self, idx, action):
            return 'page', 0, False, {'instruction': 'Buy B'}
    monkeypatch.setattr(minja, 'llm', lambda *a, **kw: 'search[B]')
    _, row, _ = minja.webshop_run_rap('s', '', '', 'example', Env(), None, 'fake', 'minimax',
        Encoder(), memory, embeddings, 1, honest_reward=True, to_print=False, memory_mode='full',
        text_retriever=TextMemoryRetriever(Encoder()), retrieval_top_k=3)
    trace = row['RetrievalTrace'][0]
    assert trace['CandidateIds'] == ['oversized']
    assert trace['SelectedIds'] == []
    assert trace['Method'] == 'text'
    assert 'Instruction: Buy B' in row['PolicyTrace'][0]['Prompt']
