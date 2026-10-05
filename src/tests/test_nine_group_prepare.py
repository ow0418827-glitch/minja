import json
from pathlib import Path
import sqlite3

import pytest

from src.experiments import nine_group_prepare as prep
from src.environment.local_webshop import fingerprint, write


def product(asin, name='Real target', attrs=None):
    return {'asin': asin, 'name': name, 'Title': name, 'query': 'target',
            'category': 'fixture', 'product_category': 'Fixture › Target',
            'Attributes': attrs or [], 'options': {'color': ['blue', 'white']},
            'images': [f'https://example.invalid/{asin}.jpg']}


def goal(asin, instruction, category='camera'):
    return {'asin': asin, 'instruction_text': instruction, 'query': category,
            'category': 'fixture', 'product_category': f'Fixture › {category}',
            'name': asin, 'attributes': ['real'], 'goal_options': [],
            'price_upper': 100.0, 'weight': 1}


def resolved_pair(victim='camera', asin='A000000001'):
    item = product(asin)
    return ({'victim': victim, 'target': item['name'], 'target_price': 50.0}, asin, item, 10.0)


def rows(victim='camera', count=30, start=0):
    return [(start + i, goal(f'B{i:09d}', f'I want {victim} model {i}', victim)) for i in range(count)]


def test_price_only_variants_do_not_fill_distinct_test_tasks():
    values = [(i, goal(f'B{i:09d}', f'I want a camera, and price lower than {i+20}.00 dollars'))
              for i in range(25)]
    with pytest.raises(ValueError, match='only 0 unique'):
        prep.select_tasks([resolved_pair()], values, normal_count=1, test_count=20)


def test_real_victim_histories_are_unique_and_disjoint_from_tests():
    values = rows(count=50) + [(100 + i, goal(f'C{i:09d}', f'I want ordinary item {i}', 'chair')) for i in range(30)]
    groups = prep.select_tasks([resolved_pair()], values, normal_count=14, test_count=20,
                               normal_candidate_limit=30)
    group = groups[0]
    assert len(group['trials']) == 20
    assert len({prep.canonical_instruction(task['instruction']) for task in group['trials']}) == 20
    history = [group['training'], *group['normal_candidates']]
    assert not {task['instruction_fingerprint'] for task in group['trials']} & {task['instruction_fingerprint'] for task in history}
    assert group['a'] not in {task['asin'] for task in group['normal_candidates']}
    assert group['training']['provenance'] == 'human_goal'
    assert group['training']['goal_index'] is not None
    assert group['training']['asin'] == group['b'] != group['a']
    assert group['protocol'] == 'victim_history_image_substitution'
    assert all(prep._category_score(task['goal'], 'camera') for task in history)
    assert all(task['session'].endswith('_' + str(task['goal_index'])) for task in group['trials'])
    assert groups == prep.select_tasks([resolved_pair()], values, normal_count=14, test_count=20,
                                       normal_candidate_limit=30)


def test_victim_normals_remain_disjoint_and_fail_when_insufficient():
    values = rows(count=40)
    group = prep.select_tasks([resolved_pair()], values, normal_mode='victim', normal_count=14,
                              test_count=20)[0]
    assert len(group['normal_candidates']) == 19
    assert all(prep._category_score(task['goal'], 'camera') for task in group['normal_candidates'])
    with pytest.raises(ValueError, match='non-leaking normal tasks'):
        prep.select_tasks([resolved_pair()], rows(count=30), normal_mode='victim',
                          normal_count=14, test_count=20)


def test_category_filter_excludes_mentions_in_unrelated_product():
    chair = goal('B000000001', 'I want a salon chair with shampoo equipment', 'salon chairs')
    brush = goal('B000000002', 'I want a toothbrush', "children's dental care")
    brush['product_category'] = 'Beauty › Oral Care › Manual Toothbrushes'
    assert prep._category_score(chair, 'shampoo') == 0
    assert prep._category_score(brush, 'toothbrush') == 3


def test_different_real_demands_for_same_product_can_supply_35_tasks():
    values = [(i, goal('B000000001', f'I want a camera with distinct real option {i}')) for i in range(40)]
    group = prep.select_tasks([resolved_pair()], values)[0]
    history = [group['training'], *group['normal_candidates'][:14]]
    assert len({task['instruction_fingerprint'] for task in [*history, *group['trials']]}) == 35
    assert {task['asin'] for task in [*history, *group['trials']]} == {'B000000001'}


def test_ambiguous_target_ids_are_rejected():
    pair = resolved_pair()[0]
    class FakeCatalog:
        def product(self, asin):
            return product(asin), 10.0
    with pytest.raises(ValueError, match='Expected one target'):
        prep.resolve_targets([pair], [dict(pair, target_asins=['A000000001', 'A000000002'])], FakeCatalog())


def test_sqlite_source_is_readonly_and_prepare_reuses_matching_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'PROJECT_ROOT', tmp_path)
    catalog = tmp_path / 'source'
    catalog.mkdir()
    pair, a, target, price = resolved_pair()
    conn = sqlite3.connect(catalog / 'catalog.sqlite')
    conn.execute('CREATE TABLE products (asin TEXT PRIMARY KEY, data TEXT, price REAL)')
    conn.execute('INSERT INTO products VALUES (?,?,?)', (a, json.dumps(target), price))
    tasks = rows(count=8)
    for _, task in tasks:
        conn.execute('INSERT INTO products VALUES (?,?,?)',
                     (task['asin'], json.dumps(product(task['asin'], 'Real camera')), 10.0))
    conn.commit()
    conn.close()
    write(catalog / 'goals.json', [task for _, task in tasks])
    write(catalog / 'targets.json', [dict(pair, target_asins=[a])])
    write(catalog / 'data_manifest.json', {'bundle_id': 'fixture-source', 'scope': 'fixture',
                                        'goals_sha256': fingerprint([task for _, task in tasks])})
    pair_file = tmp_path / 'pairs.json'
    write(pair_file, [pair])
    read_only = prep.Catalog(catalog)
    with pytest.raises(sqlite3.OperationalError, match='readonly'):
        read_only.conn.execute('DELETE FROM products')
    read_only.close()
    output = tmp_path / 'prepared'
    kwargs = dict(root=output, webshop_source=catalog, pair_file=pair_file,
                  catalog_dir=catalog, normal_count=2, test_count=3,
                  normal_candidate_limit=4, download_images=False)
    result = prep.prepare_bundle(**kwargs)
    assert result['catalog_scope'] == 'fixture'
    assert result['catalog_bundle_id'] == 'fixture-source'
    assert result['groups'][0]['trials'][0]['source_image_url'].startswith('https://')
    assert prep.prepare_bundle(**kwargs) == result
    with pytest.raises(ValueError, match='settings changed'):
        prep.prepare_bundle(**dict(kwargs, seed=123))
    altered = dict(result, test_count=30)
    write(output / 'manifest.json', altered)
    with pytest.raises(ValueError, match='manifest changed'):
        prep.prepare_bundle(**kwargs)


def test_legacy_prepare_json_is_backed_up_without_moving_images(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'PROJECT_ROOT', tmp_path)
    original = {'bundle_id': 'old', 'groups': [{'id': '01_camera'}]}
    write(tmp_path / 'manifest.json', original)
    write(tmp_path / 'groups/01_camera/tasks.json', {'training': 'old-A'})
    image = tmp_path / 'assets/a.png'
    image.parent.mkdir()
    image.write_bytes(b'unchanged')
    backup = prep._backup_legacy_bundle(tmp_path, original)
    assert json.loads((backup / 'manifest.json').read_text(encoding='utf-8')) == original
    assert (tmp_path / 'manifest.json').is_file()
    assert image.read_bytes() == b'unchanged'
    assert json.loads((backup / 'groups/01_camera/tasks.json').read_text(encoding='utf-8'))['training'] == 'old-A'


def test_output_cannot_write_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'PROJECT_ROOT', tmp_path / 'project')
    with pytest.raises(ValueError, match='inside the minja project'):
        prep._inside_project(tmp_path / 'outside')


def test_neutral_template_is_single_action_hint_without_target_redirection():
    path = Path(prep.__file__).parents[1] / 'data/indication_prompt_template_neutral.json'
    values = json.loads(path.read_text(encoding='utf-8'))
    assert len(values) == 1
    text = values[0]['note1']
    assert 'think, search, think' in text
    assert 'think [Think before search.]' in text
    assert not any(marker in text.lower() for marker in ('stock', '{victim}', '{target}', 'instead'))
