import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiments.local_experiment import aggregate, batch, context
from src.environment.local_webshop import write, read


def report():
    result = {}
    for condition in ('no_memory', 'clean', 'adversarial'):
        a = condition == 'adversarial'
        result[condition] = {'status': 'ok', 'completed': True, 'bought_a': a,
                             'bought_b': not a, 'source_memory_used': a}
    result['paired'] = {'both_purchases_completed': True, 'new_wrong_purchase': True,
                         'clean_correct_to_a': True, 'new_source_memory_use': True}
    return result


def test_aggregate_separates_errors_and_conditional_purchase():
    one, two = report(), report()
    two['adversarial'] = {'status': 'error'}
    del two['paired']
    summary = aggregate({'1': one, '2': two})
    assert summary['reported_trials'] == 2
    assert summary['planned_trials'] == 20
    assert summary['conditions']['adversarial']['errors'] == 1
    assert summary['conditions']['adversarial']['buy_a_given_memory_used'] == 1
    assert summary['conditions']['clean']['buy_a_given_memory_used'] is None
    assert summary['paired']['both_completed'] == 1


@pytest.mark.parametrize('trial_count', [3, 20])
@pytest.mark.parametrize('include_text', [False, True, 'only'])
def test_batch_runs_sixty_conditions_and_resumes_completed_trials(tmp_path, monkeypatch, trial_count, include_text):
    from src.experiments import local_experiment as driver
    from src.experiments import visual_memory_experiment as visual
    root, run = tmp_path, tmp_path / 'run'
    run.mkdir()
    trials = [{'trial': i + 1, 'session': f'fixed_eval_{i}_0', 'instruction': 'Buy B'} for i in range(20)]
    tasks = {'query_image': 'b.png', 'trials': trials}
    monkeypatch.setattr(driver, 'context', lambda args: (root, run, {'port': 3000}, tasks))
    monkeypatch.setattr(driver, 'model_runtime', lambda *args: (None, None, None))
    monkeypatch.setattr(visual, 'load_experiment', lambda args: ('experiment',))
    orders = []
    def fake_shop(args, runtime):
        orders.append(args.condition_order)
        result = report()
        if include_text:
            result['text'] = dict(result['clean'])
        if include_text == 'only':
            result = {'text': result['text']}
        write(Path(args.output) / 'shopping_report.json', result)
        return result
    monkeypatch.setattr(visual, 'shop', fake_shop)
    args = SimpleNamespace(model_name='fake', api_base='https://example.invalid/v1', num_steps=5,
                           trials=trial_count, include_text_baseline=bool(include_text),
                           text_baseline_only=include_text == 'only')
    batch(args)
    assert len(orders) == trial_count
    count = 1 if include_text == 'only' else (4 if include_text else 3)
    assert sum(len(order) for order in orders) == trial_count * count
    assert len({tuple(order) for order in orders}) == min(trial_count, count)
    batch(args)
    assert len(orders) == trial_count
    if include_text != 'only':
        assert read(run / 'summary.json')['conditions']['adversarial']['bought_a'] == trial_count
    else:
        assert set(read(run / 'summary.json')['conditions']) == {'text'}
    assert read(run / 'summary.json')['planned_trials'] == trial_count
    assert ('text' in read(run / 'summary.json')['conditions']) == bool(include_text)
    altered = run / 'shopping/trial_01/shopping_report.json'
    write(altered, {'changed': True})
    with pytest.raises(ValueError, match='report changed'):
        batch(args)


def test_context_refuses_wrong_shop_before_creating_run(tmp_path, monkeypatch):
    from src.experiments import local_experiment as driver
    write(tmp_path / 'local_config.json', {'port': 3000})
    write(tmp_path / 'tasks.json', {'bundle_id': 'expected'})
    monkeypatch.setattr(driver.requests, 'get', lambda *a, **kw:
        SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'bundle_id': 'wrong'}))
    args = SimpleNamespace(root=tmp_path, run_name='test', model_name='test', num_steps=5, attack_steps=2)
    with pytest.raises(ValueError, match='not the prepared'):
        context(args)
    assert not (tmp_path / 'runs').exists()
