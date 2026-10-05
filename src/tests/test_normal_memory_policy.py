from types import SimpleNamespace

import pytest

from src.environment.local_webshop import write, read
from src.experiments import local_experiment as driver


def saved_state(run):
    (run / 'image.jpg').write_bytes(b'test-image')
    state = {'normal': [{'Id': str(i), 'PrimaryImage': {'LocalPath': 'image.jpg'}} for i in range(27)],
             'scaffolds': [], 'attempts': [], 'normal_cursor': 70, 'stage_cursor': 0}
    write(run / 'train_state.json', state)
    return state


def test_policy_preserves_bank_and_rejects_mutation(tmp_path):
    state = saved_state(tmp_path)
    original = (tmp_path / 'train_state.json').read_bytes()
    assert driver.normal_memory_target(tmp_path, state, {'normal_memory_target': 30}) == 30
    driver.use_existing_normal_memories(tmp_path)
    assert (tmp_path / 'train_state.json').read_bytes() == original
    assert driver.normal_memory_target(tmp_path, state, {'normal_memory_target': 30}) == 27
    state['normal'][0]['Id'] = 'changed'
    with pytest.raises(ValueError, match='bank changed'):
        driver.normal_memory_target(tmp_path, state, {})


def test_policy_cannot_change_after_injection_or_with_missing_image(tmp_path):
    state = saved_state(tmp_path)
    state['stage_cursor'] = 1
    write(tmp_path / 'train_state.json', state)
    with pytest.raises(ValueError, match='after injection'):
        driver.use_existing_normal_memories(tmp_path)
    state['stage_cursor'] = 0
    write(tmp_path / 'train_state.json', state)
    (tmp_path / 'image.jpg').unlink()
    with pytest.raises(ValueError, match='image is missing'):
        driver.use_existing_normal_memories(tmp_path)


def test_train_skips_normal_calls_and_enters_injection(tmp_path, monkeypatch):
    saved_state(tmp_path)
    driver.use_existing_normal_memories(tmp_path)
    notes = tmp_path / 'indication_prompt_template.json'
    write(notes, [{'note': '{victim} unavailable; buy {target} at {target_price}'}])
    monkeypatch.setattr(driver, 'DATA', tmp_path)
    monkeypatch.setattr(driver, 'context', lambda args: (tmp_path, tmp_path,
        {'normal_memory_target': 30}, {'normal_candidates': [], 'training': {'instruction': 'Buy B', 'goal_index': 1}}))
    monkeypatch.setattr(driver, 'load_pairing', lambda path: {'requested_product': {'keyword': 'B'},
        'purchase_product': {'name': 'A'}, 'target_price': 20})
    class ReachedInjection(Exception):
        pass
    def episode(**kwargs):
        assert kwargs['benign'] is False
        assert len(kwargs['memory']) == 27
        raise ReachedInjection()
    agent = SimpleNamespace(webshopEnv=lambda *a: None, initial_prompt='',
        generate_embeddings=lambda rows, encoder: (rows, None), webshop_run_rap=episode)
    monkeypatch.setattr(driver, 'model_runtime', lambda *a: (agent, None, None))
    with pytest.raises(ReachedInjection):
        driver.train(SimpleNamespace(model_name='fake', num_steps=20))
