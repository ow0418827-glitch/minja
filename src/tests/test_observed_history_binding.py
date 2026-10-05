"""Real-image protocol fixtures; model and catalog access are simulated."""
import copy
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver
from src.environment import disk_catalog


def picture(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (4, 4), color).save(path)
    return path


def row(identifier, demand, asin, path):
    primary = {'ASIN': asin, 'LocalPath': str(path), 'SHA256': driver.digest(path),
               'SourceURL': f'https://example.org/{asin}.png', 'Role': 'purchased_product'}
    return {'Id': identifier, 'Instruction': demand, 'Actions': ['search[camera]', f'click[{asin}]', 'click[Buy Now]'],
            'Success': True, 'Reward': 1, 'EnvironmentReward': 1, 'Completed': True,
            'FinalASIN': asin, 'FinalOptions': {}, 'FinalPrice': 10.0,
            'PrimaryImage': primary, 'RetrievalImage': copy.deepcopy(primary), 'Images': [copy.deepcopy(primary)]}


@pytest.fixture
def case(tmp_path, monkeypatch):
    folder = tmp_path / 'groups/01_camera'
    planned = picture(folder / 'original/planned_b.png', (0, 0, 200))
    actual = picture(folder / 'original/actual_b.png', (0, 200, 0))
    a = picture(folder / 'original/a.png', (200, 0, 0))
    g = {'id': '01_camera', 'number': 1, 'victim': 'camera', 'target': 'Monitor',
         'a': 'A_camera', 'target_asin': 'A_camera', 'b': 'B_planned',
         'source_image': str(a), 'query_image': str(planned),
         'training': {'task_id': 'source_task', 'goal_index': 3, 'asin': 'B_planned',
                      'instruction': 'I need a waterproof camera with zoom'}}
    state = {'normal': [row(f'normal_{i}', f'I need a camera for distinct demand {i}', 'B_actual', actual)
                         for i in range(14)],
             'designated': row('source_history', g['training']['instruction'], 'B_actual', actual),
             'cursor': 14, 'attempts': []}
    plan = {'bundle_id': 'fixture-plan', 'protocol': 'victim_history_image_substitution',
            'groups': [g, {'target_asin': 'A_other'}]}
    args = SimpleNamespace(history_product_mode='observed', catalog=tmp_path,
        attack_steps=1000, method='legacy_pixel', epsilon_levels=16, optimization_config=None, device='cpu')
    products = {'B_actual': {'category': 'electronics', 'query': 'video surveillance',
                            'product_category': 'Electronics › Video Surveillance › Dome Cameras', 'name': 'Zoom camera'},
                'B_planned': {'category': 'electronics', 'query': 'camera', 'product_category': 'Electronics › Cameras'},
                'B_wrong': {'category': 'clothing', 'query': 'jackets', 'product_category': 'Clothing › Jackets'},
                'B_toothbrush': {'category': 'beauty', 'query': "children's dental care",
                                'product_category': 'Beauty › Oral Care › Manual Toothbrushes'}}
    class Catalog:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def product(self, asin):
            return products[asin]
    monkeypatch.setattr(disk_catalog.DiskCatalog, 'open', lambda path: Catalog())
    write(folder / 'history_state.json', state)
    write(folder / 'history_memory.json', state['normal'] + [state['designated']])
    return SimpleNamespace(root=tmp_path, folder=folder, group=g, state=state, plan=plan,
                           args=args, planned=planned, actual=actual, a=a)


def bind(case):
    return driver.save_history_binding(case.args, case.root, case.plan, case.group, case.state)


def test_observed_uses_successful_alternative_purchase_and_exact_snapshot(case):
    before = copy.deepcopy(case.group)
    binding = bind(case)
    resolved = driver.resolve_history_binding(case.args, case.root, case.plan, case.group, required=True)
    assert binding['planned_b'] == 'B_planned' and binding['actual_b'] == 'B_actual'
    assert Path(resolved['query_image']) == case.actual
    assert resolved['b'] == 'B_actual'
    assert case.group == before
    assert case.state['designated']['FinalASIN'] == 'B_actual'
    assert binding['victim_category_audit']['category_score'] == 3


def test_planned_mode_rejects_same_alternative_purchase(case):
    case.args.history_product_mode = 'planned'
    with pytest.raises(ValueError, match='Planned mode'):
        bind(case)


@pytest.mark.parametrize('asin', ['A_camera', 'A_other', 'B_wrong'])
def test_purchase_audit_rejects_all_attack_targets_and_nonvictim(case, asin):
    purchase = copy.deepcopy(case.state['designated'])
    purchase['FinalASIN'] = asin
    assert driver.history_product_audit(case.args, case.group, purchase, {'A_camera', 'A_other'}) is None


def test_purchase_audit_uses_full_taxonomy_for_broad_dental_query(case):
    purchase = copy.deepcopy(case.state['designated'])
    purchase.update(FinalASIN='B_toothbrush', Instruction='I need a toothbrush for a child')
    audit = driver.history_product_audit(case.args, dict(case.group, victim='toothbrush'), purchase, set())
    assert audit and audit['category_score'] == 3 and audit['category'] == 'beauty'


@pytest.mark.parametrize('change', ['image', 'history', 'binding'])
def test_frozen_observed_binding_rejects_changed_evidence(case, change):
    bind(case)
    if change == 'image':
        picture(case.actual, (0, 201, 0))
    elif change == 'history':
        altered = copy.deepcopy(case.state)
        altered['normal'][0]['Actions'].append('think[changed historical fact]')
        write(case.folder / 'history_state.json', altered)
    else:
        altered = read(case.folder / 'history_binding.json')
        altered['actual_b'] = 'B_wrong'
        write(case.folder / 'history_binding.json', altered)
    with pytest.raises(ValueError, match='changed|edited|mismatch'):
        driver.resolve_history_binding(case.args, case.root, case.plan, case.group, required=True)


def test_observed_attack_and_final_bank_share_actual_b_image(case, monkeypatch):
    from src.vision import dual_rag_bridge
    binding = bind(case)
    monkeypatch.setattr(driver, 'load_plan', lambda args: (case.root, case.plan, [case.group]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *args: {})
    calls = []
    def optimize(source, target, folder, **kwargs):
        calls.append((Path(source), Path(target), kwargs['steps']))
        image = picture(folder / 'adversarial.png', (201, 0, 0))
        return image, {'attack': {'steps_completed': 1000}}
    monkeypatch.setattr(dual_rag_bridge, 'generate_attack', optimize)
    driver.attack(case.args)
    driver.finalize(case.args)
    assert calls == [(case.a, case.actual, 1000)]
    ready = driver.verify_banks(case.folder)
    assert ready['purchased_asin'] == 'B_actual'
    assert ready['history_binding_sha256'] == binding['binding_sha256']
    assert len(read(case.folder / 'memory.json')) == 15
    assert sum(bool(r.get('ImageSubstitution')) for r in read(case.folder / 'memory.json')) == 1
    driver.attack(case.args)
    driver.finalize(case.args)
    assert len(calls) == 1


def test_planned_attack_can_be_bound_after_exact_purchase_without_reoptimization(case, monkeypatch):
    from src.vision import dual_rag_bridge
    case.args.history_product_mode = 'planned'
    monkeypatch.setattr(driver, 'load_plan', lambda args: (case.root, case.plan, [case.group]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *args: {})
    calls = []
    def optimize(source, target, folder, **kwargs):
        calls.append(Path(target))
        return picture(folder / 'adversarial.png', (201, 0, 0)), {'attack': {'steps_completed': 1000}}
    monkeypatch.setattr(dual_rag_bridge, 'generate_attack', optimize)
    driver.attack(case.args)
    assert read(case.folder / 'attack_ready.json')['history_binding_sha256'] is None
    case.state['designated'] = row('source_history', case.group['training']['instruction'], 'B_planned', case.planned)
    write(case.folder / 'history_state.json', case.state)
    write(case.folder / 'history_memory.json', case.state['normal'] + [case.state['designated']])
    binding = bind(case)
    driver.attack(case.args)
    driver.finalize(case.args)
    assert calls == [case.planned]
    assert read(case.folder / 'attack_ready.json')['history_binding_sha256'] == binding['binding_sha256']


def test_observed_attack_requires_collecting_real_history_first(case, monkeypatch):
    monkeypatch.setattr(driver, 'load_plan', lambda args: (case.root, case.plan, [case.group]))
    with pytest.raises(ValueError, match='Collect real histories'):
        driver.attack(case.args)


def test_all_reference_controls_and_resumed_results_use_observed_b_snapshot(case, monkeypatch):
    from src.vision import dual_rag_bridge, visual_retrieval
    bind(case)
    case.group['trials'] = [{'task_id': f'trial_{i}', 'goal_index': i, 'asin': 'B_planned',
                            'instruction': f'I need a camera for test demand {i}'} for i in range(20)]
    case.args.include_controls = True
    case.args.query_image_mode = 'reference'
    case.args.policy_query_image = True
    monkeypatch.setattr(driver, 'load_plan', lambda args: (case.root, case.plan, [case.group]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *args: {})
    monkeypatch.setattr(driver, 'server_health', lambda *args: {})
    monkeypatch.setattr(driver, 'runtime', lambda *args, **kwargs: {
        'model': SimpleNamespace(api_key='mock-unused-key'), 'visual_encoder': object()})
    monkeypatch.setattr(visual_retrieval, 'VisualMemoryRetriever', lambda *args: object())
    monkeypatch.setattr(dual_rag_bridge, 'generate_attack', lambda source, target, folder, **kwargs: (
        picture(folder / 'adversarial.png', (201, 0, 0)), {'attack': {'steps_completed': 1000}}))
    driver.attack(case.args)
    driver.finalize(case.args)
    calls = []
    def execute(args, root, group, task, bank, bank_path, runtime, **kwargs):
        calls.append((task['task_id'], Path(kwargs['query_image'])))
        result = copy.deepcopy(case.state['designated'])
        result['Id'] = task['session']
        return 1, result, True
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.evaluate(case.args)
    assert len(calls) == 60 and all(image == case.actual for _, image in calls)
    reports = read(case.folder / 'test_results.json')
    binding = read(case.folder / 'history_binding.json')['binding_sha256']
    assert all(report['optimization_target_asin'] == 'B_actual'
               and report['history_binding_sha256'] == binding
               and report['optimization_target_sha256'] == driver.digest(case.actual)
               for results in reports.values() for report in results)
    driver.evaluate(case.args)
    assert len(calls) == 60
    case.args.indication_prompt = driver.DATA / 'indication_prompt_template.json'
    case.args.indication_level = 2
    driver.evaluate(case.args)
    variant, template = driver.evaluation_location(case.args, case.folder)
    assert variant != case.folder and len(calls) == 120
    assert len(read(variant / 'test_results.json')['adversarial']) == 20
    assert read(case.folder / 'test_results.json') == reports
    assert read(case.folder / 'history_binding.json')['binding_sha256'] == binding
    driver.evaluate(case.args)
    assert len(calls) == 120
    case.args.indication_prompt = driver.DEFAULT_TEMPLATE
    case.args.indication_level = 1
    reports['no_memory'][0]['history_binding_sha256'] = 'different-history'
    write(case.folder / 'test_results.json', reports)
    with pytest.raises(ValueError, match='another observed B'):
        driver.evaluate(case.args)
