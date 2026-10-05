"""Frozen-history reuse, isolated query variants and evaluation-only continuation."""
import copy
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver
from src.tests.test_targeted_joint_driver import fixed_group, args as targeted_args


def reuse_fixture(tmp_path, monkeypatch):
    group = {'id': '07_fixture', 'history_binding_sha256': 'binding'}
    plan = {'protocol': driver.ATTACKER_PROTOCOL, 'groups': [group]}
    args = SimpleNamespace(memory_top_k=1, attack_steps=1000, reuse_frozen_memory=True)
    original = {'model': {'model': 'original'}, 'plan_sha256': 'plan', 'attack_steps': 1000,
        'epsilon_levels': 16, 'memory_top_k': 3,
        'code': {'src\\experiments\\nine_group_experiment.py': 'old-driver',
                 'src\\experiments\\targeted_queries.py': 'old-queries',
                 'src\\agent\\minja.py': 'original-agent'}}
    current = copy.deepcopy(original)
    current['memory_top_k'] = 1
    current['code'].update({'src\\experiments\\nine_group_experiment.py': 'new-driver',
                            'src\\experiments\\targeted_queries.py': 'new-queries',
                            'src\\experiments\\purchase_stage_metrics.py': 'new-metrics',
                            'src\\experiments\\reserve_evaluation.py': 'new-reserve-queue'})
    write(tmp_path / 'run_settings.json', original)
    folder = tmp_path / 'groups' / group['id']
    folder.mkdir(parents=True)
    image = folder / 'attack.png'
    Image.new('RGB', (4, 4), (10, 20, 30)).save(image)
    settings = {'steps': 1000, 'epsilon_levels': 16}
    ready = {'source_memory_id': 'source', 'history_binding_sha256': 'binding',
             'attack_sha256': driver.digest(image), 'bank_sha256': {'clean': 'clean', 'adversarial': 'adv'}}
    write(folder / 'memory_ready.json', ready)
    attack = {'image': str(image), 'sha256': driver.digest(image), 'settings': settings,
              'history_binding_sha256': 'binding',
              'metadata': {'attack': {'steps_requested': 1000, 'steps_completed': 1000}}}
    write(folder / 'attack_ready.json', attack)
    monkeypatch.setattr(driver, 'execution_settings', lambda *unused: current)
    monkeypatch.setattr(driver, 'resolve_history_binding', lambda *unused, **kwargs: group)
    monkeypatch.setattr(driver, 'verify_banks', lambda *unused: ready)
    monkeypatch.setattr(driver, 'attack_settings', lambda *unused: settings)
    return args, plan, group, current, folder


def test_reuse_allows_top_one_and_evaluation_code_without_rewriting_original(tmp_path, monkeypatch):
    args, plan, group, current, folder = reuse_fixture(tmp_path, monkeypatch)
    protected = [tmp_path / 'run_settings.json', folder / 'memory_ready.json', folder / 'attack_ready.json']
    original = {path: path.read_bytes() for path in protected}
    context = driver.verify_frozen_history_for_evaluation(args, tmp_path, plan, [group])
    assert context['original_memory_top_k'] == 3 and context['evaluation_memory_top_k'] == 1
    assert context['verified_groups'][0]['attack_steps_completed'] == 1000
    assert any('purchase_stage_metrics.py' in path for path in context['evaluation_code'])
    assert any('reserve_evaluation.py' in path for path in context['evaluation_code'])
    assert all(path.read_bytes() == raw for path, raw in original.items())


@pytest.mark.parametrize('key,value', [('model', {'model': 'different'}),
                                     ('attack_steps', 999), ('epsilon_levels', 32),
                                     ('plan_sha256', 'different-plan')])
def test_reuse_refuses_changes_to_frozen_experiment(tmp_path, monkeypatch, key, value):
    args, plan, group, current, _ = reuse_fixture(tmp_path, monkeypatch)
    current[key] = value
    with pytest.raises(ValueError, match='historical parameters'):
        driver.verify_frozen_history_for_evaluation(args, tmp_path, plan, [group])


def test_reuse_refuses_changes_to_shopping_agent_code(tmp_path, monkeypatch):
    args, plan, group, current, _ = reuse_fixture(tmp_path, monkeypatch)
    current['code']['src\\agent\\minja.py'] = 'different'
    with pytest.raises(ValueError, match='outside evaluation code'):
        driver.verify_frozen_history_for_evaluation(args, tmp_path, plan, [group])


@pytest.mark.parametrize('kind', ['steps', 'image', 'settings', 'binding'])
def test_reuse_refuses_incomplete_or_changed_attack(tmp_path, monkeypatch, kind):
    args, plan, group, current, folder = reuse_fixture(tmp_path, monkeypatch)
    path = folder / 'attack_ready.json'
    ready = read(path)
    if kind == 'steps':
        ready['metadata']['attack']['steps_completed'] = 999
    elif kind == 'image':
        Image.new('RGB', (4, 4), (200, 200, 200)).save(ready['image'])
    elif kind == 'settings':
        ready['settings']['epsilon_levels'] = 32
    else:
        ready['history_binding_sha256'] = 'different'
    write(path, ready)
    with pytest.raises(ValueError, match='reuse'):
        driver.verify_frozen_history_for_evaluation(args, tmp_path, plan, [group])


def test_vague_queries_are_frozen_separately_from_existing_strict_tasks(tmp_path):
    group, args = fixed_group(tmp_path), targeted_args()
    strict, strict_sha = driver.evaluation_trials(args, group, tmp_path)
    original = (tmp_path / 'targeted_tasks.json').read_bytes()
    args.query_specificity = 'vague'
    args.memory_top_k = 1
    vague, vague_sha = driver.evaluation_trials(args, group, tmp_path)
    location, _ = driver.evaluation_location(args, tmp_path)
    assert len(vague) == 20 and strict_sha != vague_sha
    assert {task['asin'] for task in vague} == {group['b']}
    assert all(task['original_requirements'] not in task['instruction'] for task in vague)
    assert all(task['query_image_sha256'] == group['reference_query_sha256'] for task in vague)
    assert (location / 'targeted_tasks.json').is_file()
    assert (tmp_path / 'targeted_tasks.json').read_bytes() == original
    assert driver.evaluation_trials(args, group, tmp_path) == (vague, vague_sha)
    changed = read(location / 'targeted_tasks.json')
    changed['trials'][0]['instruction'] = 'changed'
    write(location / 'targeted_tasks.json', changed)
    with pytest.raises(ValueError, match='Frozen targeted'):
        driver.evaluation_trials(args, group, tmp_path)


def test_query_specificity_and_top_k_have_separate_variants(tmp_path):
    args = targeted_args()
    strict, _ = driver.evaluation_location(args, tmp_path)
    args.query_specificity = 'vague'
    vague_three, _ = driver.evaluation_location(args, tmp_path)
    args.memory_top_k = 1
    vague_one, _ = driver.evaluation_location(args, tmp_path)
    assert len({strict, vague_three, vague_one}) == 3


def test_vague_correctness_keeps_actual_reward_separate_from_exact_b(tmp_path):
    group = dict(fixed_group(tmp_path), test_mode='targeted', query_specificity='vague')
    row = {'Completed': True, 'FinalASIN': group['b'], 'EnvironmentReward': 0,
           'Actions': [f'Action: click[{group["b"]}]', 'Action: click[Buy Now]']}
    report = driver.outcome(row, group, 'source')
    assert report['correct_purchase'] and report['bought_reference_product']
    assert report['environment_reward'] == 0 and not report['original_requirements_satisfied']
    row.update(FinalASIN=group['a'])
    report = driver.outcome(row, group, 'source')
    assert report['bought_target'] and not report['correct_purchase']


def test_reuse_evaluation_only_runs_new_tasks_and_resumes_them(tmp_path, monkeypatch):
    from src.vision import visual_retrieval
    group, args = fixed_group(tmp_path), targeted_args()
    args.query_specificity, args.memory_top_k, args.reuse_frozen_memory = 'vague', 1, True
    folder = tmp_path / 'groups' / group['id']
    for name in ('memory.json', 'clean_memory.json'):
        write(folder / name, [{'Id': f'history_{i}'} for i in range(15)])
    write(tmp_path / 'summary.json', {'original_completed_experiment': True})
    write(folder / 'targeted_tasks.json', {'old_strict_tasks': True})
    protected = {path: path.read_bytes() for path in (tmp_path / 'summary.json', folder / 'targeted_tasks.json',
                                                       folder / 'memory.json', folder / 'clean_memory.json')}
    context = {'source_run_settings_sha256': 'settings', 'evaluation_code': {'driver': 'version'}}
    def verify(*unused):
        args.frozen_memory_reuse = context
        return context
    def forbidden(*unused, **kwargs):
        raise AssertionError('Re-evaluation must not freeze, collect, optimize or materialize')
    monkeypatch.setattr(driver, 'load_plan', lambda *unused: (
        tmp_path, {'protocol': driver.VICTIM_PROTOCOL}, [group]))
    monkeypatch.setattr(driver, 'workspace_output', lambda path: Path(path))
    monkeypatch.setattr(driver, 'verify_frozen_history_for_evaluation', verify)
    for name in ('freeze_settings', 'collect', 'attack', 'materialize_banks'):
        monkeypatch.setattr(driver, name, forbidden)
    monkeypatch.setattr(driver, 'server_health', lambda *unused: {})
    monkeypatch.setattr(driver, 'resolve_history_binding', lambda *unused, **kwargs: copy.deepcopy(group))
    monkeypatch.setattr(driver, 'verify_banks', lambda *unused: {
        'source_memory_id': 'history_0', 'history_binding_sha256': group['history_binding_sha256']})
    encoder = SimpleNamespace(snapshot={'model': 'test'}, close=lambda: None)
    monkeypatch.setattr(driver, 'runtime', lambda *unused, **kwargs: {
        'visual_encoder': encoder, 'model': SimpleNamespace(api_key='unused')})
    monkeypatch.setattr(visual_retrieval, 'VisualMemoryRetriever', lambda *unused, **kwargs: object())
    calls = []
    def execute(options, root, selected, task, bank, bank_path, runtime, **kwargs):
        calls.append((task['task_id'], task['session'], len(bank)))
        assert options.memory_top_k == 1
        assert task['original_requirements'] not in task['instruction']
        assert task['goal']['instruction_text'] == task['instruction']
        assert kwargs['query_image'] == group['query_image']
        return '', {'Id': task['session'], 'Instruction': task['instruction'], 'Completed': True,
                    'FinalASIN': group['b'], 'EnvironmentReward': 0,
                    'Actions': ['Action: search[camera]', f'Action: click[{group["b"]}]',
                                'Action: click[Buy Now]'], 'RetrievalTrace': []}, []
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.evaluate(args)
    assert len(calls) == 60 and {count for _, _, count in calls} == {0, 15}
    location, _ = driver.evaluation_location(args, folder)
    summary = read(location / 'summary.json')
    assert summary['adversarial']['correct_purchases'] == 20
    assert summary['adversarial']['original_requirements_satisfied'] == 0
    assert summary['test_design']['query_specificity'] == 'vague'
    assert summary['adversarial']['purchase_stages']['episodes'] == 20
    shopping = read(location / 'shopping' / '01_adversarial.json')
    assert 'PurchaseStageMetrics' in shopping
    assert shopping['EvaluationTaskAudit']['original_requirements_visibility'] == 'audit_only'
    driver.evaluate(args)
    assert len(calls) == 60
    assert all(path.read_bytes() == raw for path, raw in protected.items())


def test_vague_status_preserves_original_root_summary(tmp_path, monkeypatch):
    group, args = fixed_group(tmp_path), targeted_args()
    args.query_specificity = 'vague'
    write(tmp_path / 'summary.json', {'old_completed': True})
    original = (tmp_path / 'summary.json').read_bytes()
    monkeypatch.setattr(driver, 'load_plan', lambda *unused: (
        tmp_path, {'protocol': driver.VICTIM_PROTOCOL}, [group]))
    driver.status(args)
    assert (tmp_path / 'summary.json').read_bytes() == original
    assert len(list((tmp_path / 'evaluation_summaries').glob('*.json'))) == 1
