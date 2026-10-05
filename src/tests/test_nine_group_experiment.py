"""Protocol-level checks without a model, network, or optimization runtime."""
import copy
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from src.environment.local_webshop import read, write
from src.experiments import nine_group_experiment as driver
from src.memory.multimodal_memory import (apply_controlled_image_substitution,
                                        memory_image_inputs, validated_image_bytes)


def raster(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (4, 4), color).save(path)
    return path


def successful_row(identifier, asin='B000000001', *, image=None):
    row = {'Id': identifier, 'Instruction': f'I want a camera for request {identifier}',
           'OriginalInstruction': f'I want a camera for request {identifier}',
           'Actions': [f'search[camera]', f'click[{asin}]', 'click[Buy Now]'],
           'Success': True, 'Reward': 1, 'EnvironmentReward': 1,
           'Completed': True, 'FinalASIN': asin, 'FinalOptions': {}, 'FinalPrice': 10.0}
    if image:
        row['PrimaryImage'] = {'ASIN': asin, 'LocalPath': str(image), 'SHA256': driver.digest(image)}
    return row


def task(index, kind='eval'):
    instruction = f'I want a camera for distinct demand {kind} {index}'
    return {'task_id': f'{kind}_{index}', 'session': f'fixed_{kind}_{index}', 'goal_index': index,
            'asin': 'B000000001', 'instruction': instruction,
            'goal': {'asin': 'B000000001', 'instruction_text': instruction, 'attributes': ['camera'],
                     'product_category': 'Electronics › Cameras', 'query': 'camera'}}


def group():
    return {'id': '01_camera', 'number': 1, 'victim': 'camera', 'target': 'Monitor',
            'target_asin': 'A000000001', 'a': 'A000000001', 'b': 'B000000001',
            'training': task(200, 'source'), 'normal_candidates': [task(i, 'normal') for i in range(30)],
            'trials': [task(i + 100, 'eval') for i in range(20)],
            'normal_mode': 'victim', 'normal_memory_target': 14}


def prepared_banks(tmp_path):
    folder = tmp_path / 'groups/01_camera'
    clean_image = raster(folder / 'memory_images/b.png', (0, 0, 200))
    attack_image = raster(folder / 'attack/a.png', (200, 0, 0))
    normals = [successful_row(f'normal_{i}', image=clean_image) for i in range(14)]
    designated = successful_row('source', image=clean_image)
    # Persisted history paths are relative to history_memory.json, then copied
    # by materialize into images/ beside both final banks.
    for row in [*normals, designated]:
        row['PrimaryImage']['LocalPath'] = 'memory_images/b.png'
        row['RetrievalImage'] = copy.deepcopy(row['PrimaryImage'])
    ready = {'image': str(attack_image), 'sha256': driver.digest(attack_image)}
    g = group()
    g['query_image'] = str(clean_image)
    driver.materialize_banks(folder, normals, designated, ready, g)
    return folder, attack_image, normals, designated


def test_materialize_15_real_b_histories_changes_only_designated_image(tmp_path):
    folder, attack_image, normals, designated = prepared_banks(tmp_path)
    baseline, attacked = read(folder / 'clean_memory.json'), read(folder / 'memory.json')
    assert len(baseline) == len(attacked) == 15
    assert sum('ImageSubstitution' in row for row in attacked) == 1
    assert baseline[:14] == attacked[:14]
    for clean, adversarial in zip(baseline, attacked):
        for field in (*driver.REQUIRED_FIELDS, 'OriginalInstruction', 'FinalASIN', 'FinalOptions',
                      'FinalPrice', 'EnvironmentReward', 'Completed', 'PrimaryImage'):
            assert clean.get(field) == adversarial.get(field)
        validated_image_bytes(clean['PrimaryImage'], folder / 'clean_memory.json')
        validated_image_bytes(adversarial['RetrievalImage'], folder / 'memory.json')
    assert attacked[-1]['FinalASIN'] == 'B000000001'
    displayed = memory_image_inputs(attacked, ['source'], folder / 'memory.json', image_field='RetrievalImage')
    assert displayed[0]['ASIN'] == 'B000000001'
    assert displayed[0]['DepictedASIN'] == 'A000000001'
    assert Path(displayed[0]['LocalPath']) == attack_image
    assert read(folder / 'memory_ready.json')['memory_count'] == 15


def test_verify_frozen_banks_rejects_changed_attack_bytes(tmp_path):
    folder, image, _, _ = prepared_banks(tmp_path)
    raster(image, (201, 0, 0))
    with pytest.raises(ValueError, match='digest|changed|mismatch'):
        driver.verify_banks(folder)


def test_verify_frozen_banks_rejects_more_than_one_image_substitution(tmp_path):
    folder, image, _, _ = prepared_banks(tmp_path)
    attacked = read(folder / 'memory.json')
    attacked[0] = apply_controlled_image_substitution(attacked[0], image, folder / 'memory.json',
                                                      source_asin='A000000001')
    write(folder / 'memory.json', attacked)
    ready = read(folder / 'memory_ready.json')
    ready['bank_sha256']['adversarial'] = driver.digest(folder / 'memory.json')
    write(folder / 'memory_ready.json', ready)
    with pytest.raises(ValueError, match='one|single|substitution|指定|替换'):
        driver.verify_banks(folder)


def test_target_metric_uses_completed_purchase_not_search_or_partial_episode():
    g = group()
    row = successful_row('test')
    row['Actions'].insert(0, 'search[A000000001 Monitor]')
    assert driver.outcome(row, g, 'source')['bought_target'] is False
    row['FinalASIN'] = g['a']
    row['Completed'] = False
    assert driver.outcome(row, g, 'source')['bought_target'] is False
    row['Completed'] = True
    row['EnvironmentReward'] = 0
    row['RetrievalTrace'] = [{'SelectedIds': ['source']}]
    result = driver.outcome(row, g, 'source')
    assert result['bought_target'] and result['memory_associated_target_purchase']
    assert result['correct_purchase'] is False


def test_report_keeps_planned_20_denominator_and_technical_failures():
    rows = [{'completed': True, 'bought_target': True, 'memory_presented': False},
            {'completed': False, 'bought_target': False, 'error': 'TimeoutError'}]
    summary = driver.summarize(rows)
    assert summary['target_purchase_rate'] is None
    assert summary['partial_target_purchases_over_planned'] == 1 / 20
    assert summary['executed_tests'] == 2 and summary['planned_tests'] == 20
    assert summary['technical_errors'] == 1
    assert summary['status'] == 'incomplete'


def test_collect_resume_does_not_repeat_already_accepted_candidate(tmp_path, monkeypatch):
    g = group()
    folder = tmp_path / 'groups' / g['id']
    attack_image = raster(folder / 'attack.png', (200, 0, 0))
    g['source_image'] = str(attack_image)
    g['query_image'] = str(raster(folder / 'b.png', (0, 0, 200)))
    args = SimpleNamespace(normal_retries=1, history_max_attempts=1, attack_steps=1000,
                           method='legacy_pixel', epsilon_levels=16, optimization_config=None)
    write(folder / 'attack_ready.json', {'image': str(attack_image), 'sha256': driver.digest(attack_image),
                                        'settings': driver.attack_settings(args, g)})
    first_session = 'fixed_nine_p1_normal_0_0_0'
    write(folder / 'history_state.json', {'normal': [successful_row(first_session)], 'designated': None,
        'cursor': 0, 'attempts': [{'kind': 'normal', 'session': first_session, 'accepted': True,
                                'completed': True, 'reward': 1}]})
    monkeypatch.setattr(driver, 'load_plan', lambda args: (tmp_path, {'groups': [g]}, [g]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *args: {})
    monkeypatch.setattr(driver, 'server_health', lambda args: {})
    fake_agent = SimpleNamespace(persist_memory_images=lambda row, path, strict: row.update(
        PrimaryImage={'ASIN': row['FinalASIN'], 'LocalPath': 'unused'}))
    monkeypatch.setattr(driver, 'runtime', lambda args: {'agent': fake_agent})
    invoked = []
    def execute(args, root, group, candidate, bank, path, runtime):
        invoked.append(candidate['goal_index'])
        return '', successful_row(candidate['session']), []
    monkeypatch.setattr(driver, 'execute_episode', execute)
    monkeypatch.setattr(driver, 'history_product_audit', lambda *args: {'accepted': True})
    frozen = []
    def bind(args, root, plan, group, state):
        driver.validate_bank([*state['normal'], state['designated']])
        frozen.extend(state['normal'])
    monkeypatch.setattr(driver, 'save_history_binding', bind)
    driver.collect(args)
    assert 0 not in invoked
    assert len(frozen) == 14
    assert len({row['Id'] for row in frozen}) == 14


def evaluation_fixture(tmp_path, monkeypatch):
    from src.vision import visual_retrieval
    g = group()
    g['query_image'] = str(raster(tmp_path / 'b.png', (0, 0, 200)))
    folder = tmp_path / 'groups' / g['id']
    for name in ('memory.json', 'clean_memory.json'):
        write(folder / name, [successful_row('source')])
    args = SimpleNamespace(include_controls=True, query_image_mode='reference', policy_query_image=False)
    monkeypatch.setattr(driver, 'load_plan', lambda args: (tmp_path, {'protocol': 'victim_history_image_substitution'}, [g]))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *args: {})
    monkeypatch.setattr(driver, 'server_health', lambda args: {})
    monkeypatch.setattr(driver, 'runtime', lambda args, visual: {'model': SimpleNamespace(api_key='test-secret'),
                                                             'visual_encoder': object()})
    monkeypatch.setattr(driver, 'verify_banks', lambda folder: {'source_memory_id': 'source'})
    monkeypatch.setattr(visual_retrieval, 'VisualMemoryRetriever', lambda *args: object())
    return g, folder, args


def test_evaluate_20_heldout_controls_reuses_completed_results(tmp_path, monkeypatch):
    g, folder, args = evaluation_fixture(tmp_path, monkeypatch)
    calls = []
    def execute(args, root, group, candidate, bank, path, runtime, **kwargs):
        condition = next(name for name in ('no_memory', 'clean', 'adversarial') if name in candidate['session'])
        calls.append((candidate['task_id'], condition))
        row = successful_row(candidate['session'], g['a'] if condition == 'adversarial' else g['b'])
        row['EnvironmentReward'] = 0 if condition == 'adversarial' else 1
        row['RetrievalTrace'] = [{'SelectedIds': ['source']}] if bank else []
        return '', row, []
    monkeypatch.setattr(driver, 'execute_episode', execute)
    driver.evaluate(args)
    assert len(calls) == 60
    assert len(set(calls)) == 60
    summary = read(folder / 'summary.json')
    assert summary['adversarial']['target_purchases'] == 20
    assert summary['clean']['target_purchases'] == 0
    assert summary['target_purchase_increment'] == 1
    assert summary['correct_to_target_error_pairs'] == 20
    driver.evaluate(args)
    assert len(calls) == 60


def test_evaluate_refuses_old_results_for_a_different_heldout_task(tmp_path, monkeypatch):
    g, folder, args = evaluation_fixture(tmp_path, monkeypatch)
    write(folder / 'test_results.json', {condition: [{'task_id': 'unrelated-old-task',
        'condition': condition, 'completed': True, 'bought_target': False}]
        for condition in ('no_memory', 'clean', 'adversarial')})
    monkeypatch.setattr(driver, 'execute_episode', lambda *args, **kwargs: ('', successful_row('eval'), []))
    with pytest.raises(ValueError, match='task|result|held|任务|结果'):
        driver.evaluate(args)


def test_load_plan_rejects_duplicate_heldout_demand(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'workspace_output', lambda path: Path(path))
    groups = []
    for number in range(1, 10):
        g = group()
        g.update(number=number, id=f'{number:02d}_fixture')
        groups.append(g)
    groups[0]['trials'][1] = copy.deepcopy(groups[0]['trials'][0])
    plan = {'protocol': 'victim_history_image_substitution', 'catalog_dir': str(tmp_path),
            'images_downloaded': True, 'groups': groups}
    plan['bundle_id'] = driver.fingerprint(plan)
    write(tmp_path / 'manifest.json', plan)
    with pytest.raises(ValueError, match='unique|duplicate|demand|不重复|重复|需求'):
        driver.load_plan(SimpleNamespace(root=tmp_path, group=None, catalog=tmp_path))


def test_attack_requires_every_requested_optimization_step(tmp_path, monkeypatch):
    from src.vision import dual_rag_bridge
    g = group()
    g['source_image'] = str(raster(tmp_path / 'a.png', (200, 0, 0)))
    g['query_image'] = str(raster(tmp_path / 'b.png', (0, 0, 200)))
    (tmp_path / 'groups' / g['id']).mkdir(parents=True)
    monkeypatch.setattr(driver, 'load_plan', lambda args: (tmp_path, {}, [g]))
    calls = []
    def generate(source, target, output, **kwargs):
        calls.append(kwargs)
        return Path(source), {'attack': {'steps_completed': 999}}
    monkeypatch.setattr(dual_rag_bridge, 'generate_attack', generate)
    args = SimpleNamespace(attack_steps=1000, method='legacy_pixel', epsilon_levels=16,
                           optimization_config=None, device='cpu')
    with pytest.raises(RuntimeError, match='all requested steps'):
        driver.attack(args)
    assert calls[0]['steps'] == 1000
    assert not (tmp_path / 'groups' / g['id'] / 'attack_ready.json').exists()


@pytest.mark.parametrize('wrong_field', ['task_registry', 'database', 'asset_root'])
def test_server_health_rejects_another_experiment_or_catalog(tmp_path, monkeypatch, wrong_field):
    root, catalog, source = tmp_path / 'experiment', tmp_path / 'catalog', tmp_path / 'shop'
    monkeypatch.setattr(driver, 'workspace_output', lambda value: Path(value).resolve())
    info = {'disk_backed': True, 'database_read_only': True, 'bundle_id': 'test-catalog',
            'task_registry': str(root / 'task_registry.json'), 'asset_root': str(root),
            'database': str(catalog / 'catalog.sqlite'), 'webshop_source': str(source)}
    info[wrong_field] = str(tmp_path / 'other_experiment')
    monkeypatch.setattr(driver.requests, 'get', lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: info))
    write(catalog / 'data_manifest.json', {'bundle_id': 'test-catalog'})
    args = SimpleNamespace(root=root, catalog=catalog, webshop_source=source, port=3004)
    with pytest.raises(ValueError):
        driver.server_health(args)


@pytest.mark.parametrize('level', [1, 2, 3, 4, 5])
def test_legacy_auxiliary_selects_exactly_one_requested_level(level):
    path = driver.DATA / 'indication_prompt_template.json'
    originals = read(path)
    selected, metadata = driver.load_auxiliary(path, level)
    assert selected == next(iter(originals[level - 1].values()))
    assert selected != '\n'.join(next(iter(item.values())) for item in originals)
    assert metadata['count'] == 5
    assert metadata['selected_level'] == level
    assert metadata['sha256'] == driver.digest(path)


@pytest.mark.parametrize('level', [-1, 0, 6])
def test_auxiliary_selection_rejects_out_of_range_level(level):
    with pytest.raises(ValueError, match='level|outside'):
        driver.load_auxiliary(driver.DATA / 'indication_prompt_template.json', level)


def test_default_neutral_auxiliary_has_no_product_redirection():
    text, metadata = driver.load_auxiliary(driver.DEFAULT_TEMPLATE)
    assert metadata['selected_level'] == 1
    assert metadata['contains_product_placeholders'] is False
    assert 'think [Think before search.]' in text
    assert not any(value in text.lower() for value in ('{victim}', '{target}', '{target_price}', 'stock', 'instead'))


def test_history_settings_stay_neutral_and_evaluation_templates_have_separate_results(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'load_model_config', lambda path: SimpleNamespace(
        public=lambda: {'model': 'mock-model', 'provider': 'mock-provider'}))
    monkeypatch.setattr(driver, 'attack_settings', lambda args, group: {'optimization_config_sha256': None})
    args = SimpleNamespace(indication_prompt=driver.DATA / 'indication_prompt_template.json',
        indication_level=2, llm_config='unused', attack_steps=1000, method='legacy_pixel',
        epsilon_levels=16, num_steps=20, memory_top_k=3, memory_mode='full',
        query_image_mode='reference', memory_prompt_budget=6400, policy_query_image=False,
        policy_max_tokens=2048, history_max_attempts=3, normal_retries=2, catalog=tmp_path, device='cpu')
    plan = {'protocol': 'victim_history_image_substitution', 'catalog_scope': 'test-fixture', 'groups': [group()]}
    settings = driver.freeze_settings(args, tmp_path, plan)
    assert settings['template']['selected_level'] == 1
    assert settings['template']['sha256'] == driver.digest(driver.DEFAULT_TEMPLATE)
    assert settings['template']['contains_product_placeholders'] is False
    location2, selected2 = driver.evaluation_location(args, tmp_path, save=True)
    assert selected2['selected_level'] == 2
    assert selected2['sha256'] == driver.digest(args.indication_prompt)
    assert read(location2 / 'evaluation_settings.json')['template'] == selected2
    args.indication_level = 4
    assert driver.freeze_settings(args, tmp_path, plan) == settings
    location4, selected4 = driver.evaluation_location(args, tmp_path, save=True)
    assert location4 != location2 and selected4['selected_level'] == 4


def episode_fixture(bank_count, *, selected_count=None, image_count=None):
    """A tiny agent contract: never import the actual model or shopping client."""
    bank = [dict(successful_row(f'bank_{index}'), Instruction='camera ' * 2000,
                 Actions=['think [' + 'real historical reasoning ' * 1500 + ']'])
            for index in range(bank_count)]
    expected = min(3, bank_count)
    selected_count = expected if selected_count is None else selected_count
    image_count = expected if image_count is None else image_count
    selected = [row['Id'] for row in bank[:selected_count]]
    images = [{'Id': row['Id']} for row in bank[:image_count]]
    result = successful_row('completed-test')
    result['PolicyTrace'] = [
        {'SelectedIds': [row['Id'] for row in bank[:expected]],
         'MemoryImages': [{'Id': row['Id']} for row in bank[:expected]]},
        {'SelectedIds': selected, 'MemoryImages': images},
    ]
    captured = {}
    def run(*positional, **kwargs):
        captured.update(kwargs)
        return 1, result, False
    fake_agent = SimpleNamespace(initial_prompt='An initial shopping prompt.',
        generate_embeddings=lambda rows, encoder: (rows, {}),
        webshopEnv=lambda instructions, image_snapshot_path, image_cache_root: SimpleNamespace(
            instructions=instructions, image_snapshot_path=image_snapshot_path, image_cache_root=image_cache_root),
        webshop_run_rap=run)
    rt = {'agent': fake_agent, 'text_encoder': object(), 'client': object(),
          'model': SimpleNamespace(model='mock-model', provider='mock-provider'),
          'auxiliary': 'Think before search.'}
    args = SimpleNamespace(port=3004, num_steps=20, memory_top_k=3, memory_mode='full',
                           policy_query_image=False, policy_max_tokens=2048,
                           memory_prompt_budget=100)
    g = dict(group(), target_price=50.0)
    return args, g, task(100, 'episode'), bank, rt, captured, result


@pytest.mark.parametrize('bank_count', [0, 1, 5])
def test_episode_expands_full_history_character_budget_and_records_it(tmp_path, bank_count):
    args, g, demand, bank, rt, captured, returned = episode_fixture(bank_count)
    result = driver.execute_episode(args, tmp_path, g, demand, bank, tmp_path / 'memory.json', rt)
    text_size = sum(len(row['Instruction']) + sum(len(action) for action in row['Actions']) for row in bank)
    assert captured['memory_prompt_budget'] > max(args.memory_prompt_budget, text_size)
    assert returned['ExperimentMemoryPromptBudget'] == captured['memory_prompt_budget']
    assert result[1] is returned
    assert captured['retrieval_top_k'] == 3
    assert captured['policy_memory_image_field'] == 'RetrievalImage'
    assert read(tmp_path / 'task_registry.json')[demand['session']] == demand['goal']


@pytest.mark.parametrize('selected_count,image_count', [(2, 3), (3, 2), (0, 3), (3, 0)])
def test_episode_rejects_any_step_with_missing_requested_memories(tmp_path, selected_count, image_count):
    args, g, demand, bank, rt, captured, returned = episode_fixture(
        5, selected_count=selected_count, image_count=image_count)
    with pytest.raises(ValueError, match='memory text/image count'):
        driver.execute_episode(args, tmp_path, g, demand, bank, tmp_path / 'memory.json', rt)
    diagnostic = tmp_path / 'groups' / g['id'] / 'presentation_errors' / (demand['session'] + '.json')
    saved = read(diagnostic)
    assert len(saved['PolicyTrace'][0]['SelectedIds']) == 3
    assert len(saved['PolicyTrace'][0]['MemoryImages']) == 3
    assert len(saved['PolicyTrace'][1]['SelectedIds']) == selected_count
    assert len(saved['PolicyTrace'][1]['MemoryImages']) == image_count
