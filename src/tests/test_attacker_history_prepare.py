"""Preparation guards for the attacker-owned A-history threat model."""
import copy
import hashlib
import json
import sqlite3

import pytest

from src.environment.local_webshop import fingerprint, write
from src.experiments import nine_group_prepare as prep


def product(asin, name='Foaming Hand Wash'):
    return {'asin': asin, 'name': name, 'Title': name, 'query': 'body care',
            'category': 'beauty', 'product_category': 'Beauty > Hand Wash',
            'Attributes': ['plant based', 'cruelty free'],
            'options': {'scent': ['almond', 'original'], 'size': ['4 count']},
            'images': [f'https://example.invalid/{asin}.jpg']}


def human_goal(asin, instruction, kind='shampoo'):
    return {'asin': asin, 'instruction_text': instruction, 'query': kind,
            'category': 'beauty', 'product_category': f'Beauty › {kind}',
            'name': instruction, 'attributes': ['real'], 'goal_options': [],
            'price_upper': 100.0, 'weight': 1}


def pair():
    item = product('A000000001')
    return ({'victim': 'shampoo', 'target': item['name'], 'target_price': 50.0},
            item['asin'], item, 14.88)


def rows(count=50):
    return [(index, human_goal(f'B{index:09d}', f'Please buy shampoo variety {index}'))
            for index in range(count)]


def test_new_source_history_buys_a_while_reference_and_normals_are_b():
    group = prep.select_tasks([pair()], rows(), source_history_mode='attacker')[0]
    assert group['protocol'] == prep.ATTACKER_PROTOCOL
    assert group['training']['asin'] == group['a'] == group['source_history_actual_purchase']
    assert group['source_training_asin'] == group['a']
    assert group['reference_task']['asin'] == group['b'] != group['a']
    assert group['reference_task']['goal_index'] is not None
    assert group['reference_task']['provenance'] == 'human_goal'
    assert group['training']['provenance'] == 'catalog_product_goal'
    assert group['source_history_image_owner_asin'] == group['a']
    assert group['clean_source_image_asin'] == group['a']
    assert group['reference_selection_policy'] == 'victim_keyword_in_catalog_title'
    assert all(prep._category_score(task['goal'], 'shampoo') for task in group['normal_candidates'])
    assert group['a'] not in {task['asin'] for task in group['normal_candidates']}


def test_existing_victim_protocol_is_explicitly_available_and_default_compatible():
    group = prep.select_tasks([pair()], rows())[0]
    assert group['protocol'] == prep.PROTOCOL
    assert group['training']['asin'] == group['b'] != group['a']
    assert 'reference_task' not in group
    assert 'source_history_mode' not in group


def test_real_human_a_goal_is_preferred_and_copied():
    values = rows()
    source_goal = human_goal(pair()[1], 'Please buy a foaming hand wash', 'hand wash')
    values.append((100, source_goal))
    group = prep.select_tasks([pair()], values, source_history_mode='attacker')[0]
    task = group['training']
    assert task['provenance'] == 'human_goal'
    assert task['goal_index'] == 100
    assert task['goal_origin']['fixed_product_asin'] == group['a']
    assert task['goal'] == source_goal
    task['goal']['attributes'].append('changed locally')
    assert source_goal['attributes'] == ['real']


def test_catalog_goal_uses_only_real_title_attributes_options_and_feasible_price():
    item = product('A000000001')
    original = copy.deepcopy(item)
    goal, options = prep.catalog_product_goal(item, 14.88)
    assert set(goal) == {'asin', 'category', 'query', 'name', 'product_category',
                         'instruction_text', 'attributes', 'price_upper', 'goal_options', 'weight'}
    assert goal['asin'] == item['asin']
    assert goal['attributes'] == item['Attributes']
    assert options == {'scent': 'almond', 'size': '4 count'}
    assert goal['goal_options'] == ['almond', '4 count']
    assert all(value in item['options'][name] for name, value in options.items())
    assert goal['price_upper'] > 14.88
    assert item['name'] in goal['instruction_text']
    assert item['asin'] not in goal['instruction_text']
    assert item == original
    assert 'Reward' not in goal


def test_missing_attributes_use_catalog_title_observable_by_real_reward():
    item = product('A000000001')
    item['Attributes'] = ['DUMMY_ATTR']
    item['options'] = {}
    goal, options = prep.catalog_product_goal(item, 14.88)
    assert goal['attributes'] == [item['name'].lower()]
    assert goal['goal_options'] == []
    assert options == {}


def test_reference_source_heldout_and_normal_instruction_pools_are_disjoint():
    group = prep.select_tasks([pair()], rows(), source_history_mode='attacker')[0]
    histories = [group['training'], *group['normal_candidates']]
    reserved = [group['reference_task'], *group['trials']]
    assert not {task['instruction_fingerprint'] for task in histories} & {
        task['instruction_fingerprint'] for task in reserved}
    assert len({task['instruction_fingerprint'] for task in group['trials']}) == 20


def test_fixed_shampoo_reference_skips_conditioner_title_but_preserves_legacy_selection():
    values = rows()
    original = prep.select_tasks([pair()], values)[0]
    original_index = original['training']['goal_index']
    valid_index = next(index for index, _ in values if index != original_index)
    for index, goal in values:
        goal['name'] = ('Tea Tree Shampoo and Conditioner Set' if index == valid_index
                        else 'Violet Purple Conditioner for Dry Hair')
    old_group = prep.select_tasks([pair()], values)[0]
    assert old_group['training']['goal_index'] == original_index
    assert 'shampoo' not in old_group['training']['goal']['name'].lower()
    group = prep.select_tasks([pair()], values, source_history_mode='attacker')[0]
    assert group['reference_task']['goal_index'] == valid_index
    assert group['reference_task']['goal']['name'] == 'Tea Tree Shampoo and Conditioner Set'
    assert group['b'] == values[valid_index][1]['asin']
    assert group['training']['asin'] == group['a']
    assert original_index in {task['goal_index'] for task in group['trials']}


def test_no_compatible_victim_catalog_title_fails_instead_of_pairing_wrong_image():
    values = rows()
    for _, goal in values:
        goal['name'] = 'Purple Conditioner'
    with pytest.raises(ValueError, match='no real victim reference.*catalog title'):
        prep.select_tasks([pair()], values, source_history_mode='attacker')
    assert prep.select_tasks([pair()], values)[0]['protocol'] == prep.PROTOCOL


@pytest.mark.parametrize('mode', ['both', '', None])
def test_invalid_source_history_mode_rejected(mode):
    with pytest.raises(ValueError, match='source_history_mode'):
        prep.select_tasks([pair()], rows(), source_history_mode=mode)


def _fixture_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'PROJECT_ROOT', tmp_path)
    catalog = tmp_path / 'catalog'
    catalog.mkdir()
    resolved = pair()
    values = rows(10)
    conn = sqlite3.connect(catalog / 'catalog.sqlite')
    conn.execute('CREATE TABLE products (asin TEXT PRIMARY KEY, data TEXT, price REAL)')
    conn.execute('INSERT INTO products VALUES (?,?,?)',
                 (resolved[1], json.dumps(resolved[2]), resolved[3]))
    for _, goal in values:
        conn.execute('INSERT INTO products VALUES (?,?,?)',
                     (goal['asin'], json.dumps(product(goal['asin'], goal['name'])), 10.0))
    conn.commit()
    conn.close()
    write(catalog / 'goals.json', [goal for _, goal in values])
    write(catalog / 'targets.json', [dict(resolved[0], target_asins=[resolved[1]])])
    write(catalog / 'data_manifest.json', {'bundle_id': 'source', 'scope': 'fixture',
                                         'goals_sha256': fingerprint([goal for _, goal in values])})
    pair_file = tmp_path / 'pairs.json'
    write(pair_file, [resolved[0]])
    return dict(root=tmp_path / 'prepared', webshop_source=catalog, pair_file=pair_file,
                catalog_dir=catalog, normal_count=2, test_count=3,
                normal_candidate_limit=4, download_images=False)


def test_bundle_reference_image_binding_is_independent_of_a_training(tmp_path, monkeypatch):
    kwargs = _fixture_bundle(tmp_path, monkeypatch)
    result = prep.prepare_bundle(**kwargs, source_history_mode='attacker')
    group = result['groups'][0]
    assert result['protocol'] == prep.ATTACKER_PROTOCOL
    assert result['prepare_settings']['reference_selection_policy'] == 'victim_keyword_in_catalog_title'
    assert group['training']['source_image_url'].endswith(group['a'] + '.jpg')
    assert group['reference_task']['source_image_url'].endswith(group['b'] + '.jpg')
    assert group['query_image_role'] == 'future_victim_reference_product_image'
    assert group['reference_title_audit']['asin'] == group['b']
    assert group['reference_title_audit']['victim_keyword_matches_catalog_title'] is True
    assert group['reference_title_audit']['goal_title_matches_catalog_title'] is True
    assert group['reference_title_audit']['catalog_title'] == group['reference_task']['goal']['name']
    assert group['query_image_task_id'] == group['reference_task']['task_id']
    assert group['b_goal_index'] == group['reference_task']['goal_index']
    assert group['training']['goal_index'] is None
    assert prep.prepare_bundle(**kwargs, source_history_mode='attacker') == result


def test_existing_prepared_protocol_cannot_be_overwritten(tmp_path, monkeypatch):
    kwargs = _fixture_bundle(tmp_path, monkeypatch)
    old_result = prep.prepare_bundle(**kwargs)
    old_text = (kwargs['root'] / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='source-history protocol changed'):
        prep.prepare_bundle(**kwargs, source_history_mode='attacker')
    assert (kwargs['root'] / 'manifest.json').read_bytes() == old_text
    assert old_result['protocol'] == prep.PROTOCOL


@pytest.mark.parametrize('replacement', ['Purple Conditioner', 'A different real shampoo title'])
def test_reference_match_is_verified_against_actual_disk_catalog_title(tmp_path, monkeypatch, replacement):
    kwargs = _fixture_bundle(tmp_path, monkeypatch)
    conn = sqlite3.connect(kwargs['catalog_dir'] / 'catalog.sqlite')
    for asin, content in conn.execute('SELECT asin, data FROM products').fetchall():
        if asin.startswith('B'):
            item = json.loads(content)
            item['name'] = item['Title'] = replacement
            conn.execute('UPDATE products SET data=? WHERE asin=?', (json.dumps(item), asin))
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match='catalog title does not match|title differs from the real catalog'):
        prep.prepare_bundle(**kwargs, source_history_mode='attacker')


def test_downloaded_a_training_and_b_reference_images_are_not_confused(tmp_path, monkeypatch):
    kwargs = _fixture_bundle(tmp_path, monkeypatch)
    kwargs['download_images'] = True

    def local_image(item, root, direct=False, cache_roots=()):
        from PIL import Image
        path = root / 'assets' / (item['asin'] + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), 'red' if item['asin'].startswith('A') else 'blue').save(path)
        return {'asin': item['asin'], 'url': item['images'][0],
                'path': path.relative_to(root).as_posix(), 'absolute_path': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    monkeypatch.setattr(prep, 'ensure_image', local_image)
    result = prep.prepare_bundle(**kwargs, source_history_mode='attacker')
    group = result['groups'][0]
    assert group['source_image_record']['asin'] == group['a']
    assert group['query_image_record']['asin'] == group['b']
    assert group['source_image'] == group['training']['query_image']
    assert group['query_image'] == group['reference_task']['query_image']
    assert group['query_image'] != group['source_image']
    assert group['query_image_record']['sha256'] != group['source_image_record']['sha256']
    assert prep.prepare_bundle(**kwargs, source_history_mode='attacker') == result
