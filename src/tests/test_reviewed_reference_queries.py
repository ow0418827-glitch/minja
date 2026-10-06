"""Catalog B references must remain truthful while image-only queries stay neutral."""
import copy

import pytest

from src.environment.local_webshop import fingerprint
from src.experiments.targeted_queries import make_targeted_trials, VAGUE_DEMAND_MODE
from src.tests.test_attacker_targeted_queries import attacker_group


def reviewed_group(group):
    group = copy.deepcopy(group)
    reference = group['reference_task']
    reference.update(goal_index=None, provenance='catalog_product_goal')
    reference['goal']['name'] = 'Catalog Shampoo Set'
    group['query_image_record']['url'] = 'https://example.test/native-shampoo.jpg'
    reference['goal_origin'] = {
        'kind': 'catalog_product_goal', 'fixed_product_asin': group['b'],
        'product_fields_sha256': 'a' * 64,
        'reference_goal_sha256': fingerprint(reference['goal']),
        'reward_is_real_environment_only': True,
        'reviewed_reference_image_sha256': group['query_image_record']['sha256'],
        'reviewed_reference_image_url': group['query_image_record']['url']}
    group['reference_selection_policy'] = 'reviewed_catalog_product_image'
    group['reference_title_audit'] = {
        'asin': group['b'], 'catalog_title': reference['goal']['name'],
        'victim_keyword_matches_catalog_title': True, 'goal_title_matches_catalog_title': True}
    return group


def test_catalog_reference_has_no_fake_human_index_and_keeps_a_history(attacker_group):
    group = reviewed_group(attacker_group)
    before = copy.deepcopy(group)
    trials = make_targeted_trials(group, count=25, demand_mode=VAGUE_DEMAND_MODE)
    assert len(trials) == len({t['instruction'] for t in trials}) == 25
    assert all(t['goal_index'] is None and t['expected_asin'] == group['b'] for t in trials)
    assert all(group['reference_task']['instruction'] not in t['instruction'] for t in trials)
    assert all(group['b'] not in t['instruction'] and group['a'] not in t['instruction'] for t in trials)
    assert group == before
    assert group['training'] == attacker_group['training']


@pytest.mark.parametrize('failure', ['no_provenance', 'wrong_product', 'changed_goal',
                                    'changed_image_origin', 'unreviewed', 'wrong_title'])
def test_missing_or_changed_catalog_evidence_is_rejected(attacker_group, failure):
    group = reviewed_group(attacker_group)
    reference = group['reference_task']
    if failure == 'no_provenance':
        reference['provenance'] = 'human_goal'
    elif failure == 'wrong_product':
        reference['goal_origin']['fixed_product_asin'] = group['a']
    elif failure == 'changed_goal':
        reference['goal']['goal_options'].append('invented option')
    elif failure == 'changed_image_origin':
        reference['goal_origin']['reviewed_reference_image_sha256'] = 'b' * 64
    elif failure == 'unreviewed':
        group['reference_selection_policy'] = 'victim_keyword_in_catalog_title'
    else:
        group['reference_title_audit']['catalog_title'] = 'Another product'
    with pytest.raises(ValueError, match='real goal index'):
        make_targeted_trials(group, demand_mode=VAGUE_DEMAND_MODE)


def test_an_unindexed_human_reference_is_still_rejected(attacker_group):
    attacker_group['reference_task']['goal_index'] = None
    with pytest.raises(ValueError, match='real goal index'):
        make_targeted_trials(attacker_group, demand_mode=VAGUE_DEMAND_MODE)


def test_reviewed_reference_works_with_real_resolved_history_image_record(attacker_group):
    group = reviewed_group(attacker_group)
    prepared = group['query_image_record']
    group['query_image_record'] = {
        'ASIN': prepared['asin'], 'LocalPath': prepared['absolute_path'],
        'SHA256': prepared['sha256'], 'SourceURL': prepared['url'], 'Role': 'victim_reference'}
    trials = make_targeted_trials(group, demand_mode=VAGUE_DEMAND_MODE)
    assert len(trials) == 20
    assert all(t['goal_index'] is None and t['expected_asin'] == group['b'] for t in trials)
    group['query_image_record']['SourceURL'] = 'https://example.test/another-product.png'
    with pytest.raises(ValueError, match='real goal index'):
        make_targeted_trials(group, demand_mode=VAGUE_DEMAND_MODE)
