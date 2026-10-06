"""Reviewed B substitutions preserve real product truth and frozen inputs."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from src.environment.local_webshop import fingerprint, read, write
from src.experiments import nine_group_prepare as prep


@pytest.fixture
def reviewed_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'PROJECT_ROOT', tmp_path)
    catalog = tmp_path / 'catalog'
    catalog.mkdir()

    def product(asin, title):
        return {'asin': asin, 'name': title, 'Title': title, 'query': 'camera',
                'category': 'fixture', 'product_category': 'Fixture › Cameras',
                'Attributes': ['real catalog feature'], 'options': {'color': ['blue', 'white']},
                'images': [f'https://example.invalid/{asin}_main.jpg',
                           f'https://example.invalid/{asin}_detail.jpg']}

    a = product('A000000001', 'Legal attacker target')
    b = product('NEWB000001', 'Reviewed Brand camera MODEL1')
    b['option_to_image'] = {'blue': f'https://example.invalid/{b["asin"]}_blue.jpg'}
    products = [a, b]
    goals = []
    for index in range(50):
        item = product(f'H{index:09d}', f'Human camera model {index}')
        products.append(item)
        goals.append({'asin': item['asin'], 'instruction_text': f'I want a camera model {index}',
                      'name': item['name'], 'query': 'camera', 'category': 'fixture',
                      'product_category': 'Fixture › Cameras', 'attributes': ['real catalog feature'],
                      'goal_options': [], 'price_upper': 100.0, 'weight': 1})
    connection = sqlite3.connect(catalog / 'catalog.sqlite')
    connection.execute('CREATE TABLE products (asin TEXT PRIMARY KEY, data TEXT, price REAL)')
    for item in products:
        connection.execute('INSERT INTO products VALUES (?,?,?)', (item['asin'], json.dumps(item), 10.0))
    connection.commit()
    connection.close()
    pair = {'victim': 'camera', 'target': a['name'], 'target_price': 10.0}
    pair_file = tmp_path / 'pairs.json'
    write(pair_file, [pair])
    write(catalog / 'targets.json', [dict(pair, target_asins=[a['asin']])])
    write(catalog / 'goals.json', goals)
    write(catalog / 'data_manifest.json', {'bundle_id': 'fixture-source',
                                         'goals_sha256': fingerprint(goals)})
    reviewed = tmp_path / 'reviewed.png'
    Image.new('RGB', (4, 3), 'blue').save(reviewed)
    digest = hashlib.sha256(reviewed.read_bytes()).hexdigest()
    config = tmp_path / 'reference_overrides.json'
    payload = {'groups': {'1': {'asin': b['asin'], 'image_path': reviewed.name,
                               'catalog_image_index': 1, 'sha256': digest}}}
    write(config, payload)
    image_requests = []

    def cached_image(item, root, direct=False, cache_roots=()):
        image_requests.append((item['asin'], item['images'][0]))
        path = Path(root) / 'assets' / f'{item["asin"]}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (3, 3), 'red').save(path)
        return {'asin': item['asin'], 'url': item['images'][0],
                'path': path.relative_to(root).as_posix(), 'absolute_path': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

    monkeypatch.setattr(prep, 'ensure_image', cached_image)
    return {'a': a, 'b': b, 'goals': goals, 'catalog': catalog, 'config': config,
            'payload': payload, 'reviewed': reviewed, 'digest': digest,
            'image_requests': image_requests,
            'kwargs': {'root': tmp_path / 'prepared', 'webshop_source': catalog,
                       'pair_file': pair_file, 'catalog_dir': catalog,
                       'source_history_mode': 'attacker', 'normal_count': 14,
                       'test_count': 20, 'normal_candidate_limit': 20,
                       'reference_overrides': config}}


def test_reference_override_uses_catalog_truth_and_preserves_source_a(reviewed_catalog):
    fixture = reviewed_catalog
    database = fixture['catalog'] / 'catalog.sqlite'
    original_database = hashlib.sha256(database.read_bytes()).hexdigest()
    original_reviewed = fixture['reviewed'].read_bytes()
    baseline = prep.prepare_bundle(**dict(fixture['kwargs'],
                                         root=fixture['kwargs']['root'].parent / 'baseline',
                                         reference_overrides=None))['groups'][0]
    manifest = prep.prepare_bundle(**fixture['kwargs'])
    group = manifest['groups'][0]
    reference = group['reference_task']
    expected_goal, expected_options = prep.catalog_product_goal(fixture['b'], 10.0)
    assert reference['goal'] == expected_goal
    assert reference['goal_index'] is None and group['b_goal_index'] is None
    assert reference['provenance'] == 'catalog_product_goal'
    assert reference['goal_origin']['product_fields_sha256'] == fingerprint(fixture['b'])
    assert reference['goal_origin']['reference_goal_sha256'] == fingerprint(reference['goal'])
    assert reference['goal_origin']['selected_catalog_options'] == expected_options
    assert reference['goal_origin']['fixed_product_asin'] == fixture['b']['asin']
    assert reference['goal_origin']['reward_is_real_environment_only'] is True
    assert group['b'] == group['reference_product_asin'] == reference['goal']['asin']
    for key in ('goal', 'goal_index', 'provenance', 'goal_origin', 'instruction', 'instruction_fingerprint'):
        assert group['training'][key] == baseline['training'][key]
    assert group['a'] == group['source_history_actual_purchase'] == fixture['a']['asin']
    assert group['source_image_record']['url'] == fixture['a']['images'][0]
    assert group['source_image_record']['sha256'] == baseline['source_image_record']['sha256']
    record = group['query_image_record']
    assert record == reference['image_record']
    assert record['url'] == reference['source_image_url'] == fixture['b']['images'][1]
    assert record['source_url'] == record['url']
    assert record['sha256'] == fixture['digest']
    assert Path(record['absolute_path']).is_relative_to(fixture['kwargs']['root'])
    assert Path(record['absolute_path']).read_bytes() == original_reviewed
    assert group['reference_title_audit']['catalog_title'] == fixture['b']['Title']
    assert group['reference_title_audit']['goal_title_matches_catalog_title'] is True
    assert len(group['trials']) == 20 and len(group['normal_candidates']) >= 14
    reference_key = reference['instruction_fingerprint']
    assert reference_key not in {task['instruction_fingerprint'] for task in
                                [group['training'], *group['trials'], *group['normal_candidates']]}
    assert fixture['b']['asin'] not in {task['asin'] for task in
                                      [*group['trials'], *group['normal_candidates']]}
    assert fixture['b']['asin'] not in {asin for asin, _ in fixture['image_requests']}
    assert hashlib.sha256(database.read_bytes()).hexdigest() == original_database
    assert fixture['reviewed'].read_bytes() == original_reviewed


def test_reference_override_rejects_image_url_owned_by_another_product(reviewed_catalog):
    fixture = reviewed_catalog
    payload = copy.deepcopy(fixture['payload'])
    entry = payload['groups']['1']
    entry.pop('catalog_image_index')
    entry['image_url'] = fixture['a']['images'][0]
    write(fixture['config'], payload)
    with pytest.raises(ValueError, match='does not belong to catalog product'):
        prep.prepare_bundle(**fixture['kwargs'])
    assert not fixture['kwargs']['root'].exists()
    assert not fixture['image_requests']


def test_reference_override_rejects_sha256_mismatch_before_writes(reviewed_catalog):
    fixture = reviewed_catalog
    payload = copy.deepcopy(fixture['payload'])
    payload['groups']['1']['sha256'] = '0' * 64
    write(fixture['config'], payload)
    with pytest.raises(ValueError, match='sha256 mismatch'):
        prep.prepare_bundle(**fixture['kwargs'])
    assert not fixture['kwargs']['root'].exists()
    assert not fixture['image_requests']


def test_reference_override_resume_freezes_configuration_and_original_image(reviewed_catalog):
    fixture = reviewed_catalog
    manifest = prep.prepare_bundle(**fixture['kwargs'])
    frozen = manifest['prepare_settings']['reference_overrides']
    assert frozen['content_sha256'] == fingerprint(fixture['payload'])
    assert frozen['content'] == fixture['payload']
    assert frozen['image_sha256'] == {'1': fixture['digest']}
    assert prep.prepare_bundle(**fixture['kwargs']) == manifest
    payload = copy.deepcopy(fixture['payload'])
    payload['groups']['1']['catalog_image_index'] = 0
    write(fixture['config'], payload)
    with pytest.raises(ValueError, match='settings changed'):
        prep.prepare_bundle(**fixture['kwargs'])
    write(fixture['config'], fixture['payload'])
    Image.new('RGB', (4, 3), 'green').save(fixture['reviewed'])
    with pytest.raises(ValueError, match='sha256 mismatch'):
        prep.prepare_bundle(**fixture['kwargs'])


def test_reference_override_accepts_real_option_image_and_requires_attacker_mode(reviewed_catalog):
    fixture = reviewed_catalog
    payload = copy.deepcopy(fixture['payload'])
    entry = payload['groups']['1']
    entry.pop('catalog_image_index')
    entry['image_url'] = fixture['b']['option_to_image']['blue']
    write(fixture['config'], payload)
    with pytest.raises(ValueError, match='require source_history_mode=attacker'):
        prep.prepare_bundle(**dict(fixture['kwargs'], source_history_mode='victim'))
    manifest = prep.prepare_bundle(**fixture['kwargs'])
    assert manifest['groups'][0]['query_image_record']['url'] == entry['image_url']


def test_reference_override_goal_options_match_selected_real_variant(reviewed_catalog):
    fixture = reviewed_catalog
    payload = copy.deepcopy(fixture['payload'])
    payload['groups']['1']['goal_options'] = {'color': 'white'}
    write(fixture['config'], payload)
    manifest = prep.prepare_bundle(**fixture['kwargs'])
    reference = manifest['groups'][0]['reference_task']
    expected, options = prep.catalog_product_goal(fixture['b'], 10.0, {'color': 'white'})
    assert reference['goal'] == expected
    assert reference['goal']['goal_options'] == ['white']
    assert 'color: white' in reference['instruction']
    assert reference['goal_origin']['selected_catalog_options'] == options == {'color': 'white'}
    assert reference['goal_origin']['reference_goal_sha256'] == fingerprint(expected)


def test_reference_override_rejects_invented_goal_options(reviewed_catalog):
    fixture = reviewed_catalog
    payload = copy.deepcopy(fixture['payload'])
    payload['groups']['1']['goal_options'] = {'color': 'invented color'}
    write(fixture['config'], payload)
    with pytest.raises(ValueError, match='absent from the real catalog'):
        prep.prepare_bundle(**fixture['kwargs'])
    assert not fixture['kwargs']['root'].exists()


def test_reference_override_cannot_reduce_held_out_trial_count(reviewed_catalog):
    fixture = reviewed_catalog
    catalog = prep.Catalog(fixture['catalog'])
    try:
        pair = read(fixture['kwargs']['pair_file'])[0]
        resolved = prep.resolve_targets([pair], read(fixture['catalog'] / 'targets.json'), catalog)
        groups = prep.select_tasks(resolved, enumerate(fixture['goals']), normal_count=14,
                                   test_count=20, normal_candidate_limit=20,
                                   source_history_mode='attacker')
        collision = groups[0]['trials'][0]['asin']
        entry = dict(fixture['payload']['groups']['1'], asin=collision)
        with pytest.raises(ValueError, match='fewer than 20 disjoint trials'):
            prep._apply_reference_overrides(groups, {1: entry}, catalog, 14, 20)
    finally:
        catalog.close()
