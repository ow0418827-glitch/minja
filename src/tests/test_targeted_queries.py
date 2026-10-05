"""Fixed-product query variants preserve requirements and actual-B evidence."""
import copy
import hashlib

from PIL import Image
import pytest

from src.experiments.nine_group_prepare import canonical_instruction
from src.experiments.targeted_queries import make_targeted_trials, validate_targeted_trials


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def group(tmp_path):
    image = tmp_path / 'actual_b.png'
    Image.new('RGB', (6, 5), (40, 60, 80)).save(image)
    instruction = 'I want a waterproof camera with optical zoom, black option, and price lower than 280.00 dollars'
    return {
        'id': '01_camera', 'a': 'B083W8PJPB', 'target_asin': 'B083W8PJPB',
        'target': 'Samsung Business SR35 Series Computer Monitor',
        'b': 'B08RHM6VF2', 'planned_b': 'B08V134QWR',
        'query_image': str(image), 'reference_query_sha256': sha(image),
        'history_binding_sha256': 'history-binding-verified',
        'source_history_instruction': instruction,
        'training': {
            'task_id': '01_camera_source_001', 'session': 'original_history_session',
            'asin': 'B08V134QWR', 'goal_index': 835, 'instruction': instruction,
            'goal': {'asin': 'B08V134QWR', 'instruction_text': instruction,
                     'attributes': ['optical zoom', 'waterproof'], 'price_upper': 280.0,
                     'goal_options': ['black'], 'category': 'electronics',
                     'name': 'Original camera name', 'query': 'camera', 'weight': 1},
        },
    }


def test_twenty_variants_share_actual_product_image_and_requirements(group):
    unchanged = copy.deepcopy(group)
    trials = make_targeted_trials(group)
    assert len(trials) == 20
    assert len({t['task_id'] for t in trials}) == 20
    assert len({t['session'] for t in trials}) == 20
    assert len({canonical_instruction(t['instruction']) for t in trials}) == 20
    assert canonical_instruction(group['training']['instruction']) not in {
        canonical_instruction(t['instruction']) for t in trials}
    source_goal = group['training']['goal']
    source_sha = hashlib.sha256(group['training']['instruction'].encode('utf-8')).hexdigest()
    for ordinal, trial in enumerate(trials, 1):
        assert trial['asin'] == trial['expected_asin'] == 'B08RHM6VF2'
        assert trial['goal']['asin'] == trial['asin']
        assert trial['goal']['instruction_text'] == trial['instruction']
        assert trial['query_image'] == group['query_image']
        assert trial['query_image_sha256'] == group['reference_query_sha256']
        assert trial['goal_index'] == group['training']['goal_index']
        assert trial['instruction'].count(group['training']['instruction']) == 1
        assert trial['source_instruction_sha256'] == source_sha
        assert trial['history_binding_sha256'] == group['history_binding_sha256']
        assert trial['source_session'] == 'original_history_session'
        assert trial['demand_mode'] == 'same_product_paraphrases'
        assert trial['variant_id'] == ordinal
        assert trial['independent_product_count'] == trial['repeat_count'] == 1
        assert trial['variants_per_product'] == 20
        assert {k: v for k, v in trial['goal'].items() if k not in ('asin', 'instruction_text')} == {
            k: v for k, v in source_goal.items() if k not in ('asin', 'instruction_text')}
    assert group == unchanged
    trials[0]['goal']['attributes'].append('not shared')
    assert trials[1]['goal']['attributes'] == source_goal['attributes']
    assert group == unchanged


@pytest.mark.parametrize('instruction', [
    '我想购买绿色款儿童牙刷，价格低于30美元。\n请选择“绿色”选项，不能更换成其他颜色。',
    'I want a camera with "night vision".\nChoose the \'black\' option; budget is $280.00. {Keep this text}',
])
def test_original_unicode_quotes_lines_and_braces_are_kept_verbatim(group, instruction):
    group['training']['instruction'] = instruction
    group['training']['goal']['instruction_text'] = instruction
    group['source_history_instruction'] = instruction
    trials = make_targeted_trials(group)
    assert all(t['instruction'].endswith(instruction) for t in trials)
    if instruction.startswith('我'):
        assert all('附图' in t['instruction'] or '图片' in t['instruction'] or '这张图' in t['instruction']
                   or '照片' in t['instruction'] for t in trials)
    else:
        assert all('product' in t['instruction'] or 'item' in t['instruction'] for t in trials)


def test_user_prompts_do_not_reveal_attack_target_or_product_identifiers(group):
    for trial in make_targeted_trials(group):
        for value in (group['a'], group['b'], group['planned_b'], group['target'],
                      'adversarial', 'poisoned', 'attack'):
            assert value.casefold() not in trial['instruction'].casefold()


def test_planned_image_record_is_not_used_for_actual_b_digest(group):
    group['query_image_record'] = {'asin': group['planned_b'], 'sha256': 'planned-image-digest'}
    assert len(make_targeted_trials(group)) == 20
    del group['reference_query_sha256']
    with pytest.raises(ValueError, match='actual-B image SHA256'):
        make_targeted_trials(group)


def test_actual_primary_image_record_can_supply_verified_digest(group):
    expected_sha = group.pop('reference_query_sha256')
    group['purchased_primary_image'] = {'ASIN': group['b'], 'SHA256': expected_sha,
                                      'LocalPath': group['query_image']}
    assert all(t['query_image_sha256'] == expected_sha for t in make_targeted_trials(group))


@pytest.mark.parametrize('failure', ['missing_image', 'invalid_image', 'changed_image', 'missing_digest',
                                    'conflicting_digest', 'missing_history_binding'])
def test_invalid_or_unbound_image_evidence_is_rejected(group, failure):
    from pathlib import Path
    path = Path(group['query_image'])
    if failure == 'missing_image':
        path.unlink()
    elif failure == 'invalid_image':
        path.write_bytes(b'not a raster image')
        group['reference_query_sha256'] = sha(path)
    elif failure == 'changed_image':
        Image.new('RGB', (6, 5), (41, 60, 80)).save(path)
    elif failure == 'missing_digest':
        del group['reference_query_sha256']
    elif failure == 'conflicting_digest':
        group['query_image_sha256'] = 'conflicting'
    else:
        del group['history_binding_sha256']
    with pytest.raises(ValueError):
        make_targeted_trials(group)


@pytest.mark.parametrize('failure', ['goal_instruction', 'history_instruction', 'instruction_digest',
                                    'source_empty', 'target_equals_actual', 'source_asin_leak',
                                    'target_title_leak', 'attack_word'])
def test_changed_or_nonneutral_source_requirements_are_rejected(group, failure):
    if failure == 'goal_instruction':
        group['training']['goal']['instruction_text'] = 'Changed price to 500 dollars'
    elif failure == 'history_instruction':
        group['source_history_instruction'] = 'Changed historical purchase request'
    elif failure == 'instruction_digest':
        group['source_instruction_sha256'] = 'wrong-source-digest'
    elif failure == 'target_equals_actual':
        group['b'] = group['a']
    else:
        text = {'source_empty': ' ', 'source_asin_leak': 'Please buy B08RHM6VF2',
                'target_title_leak': 'Please buy Samsung Business SR35 Series Computer Monitor',
                'attack_word': 'Please buy the adversarial camera'}[failure]
        group['training']['instruction'] = text
        group['training']['goal']['instruction_text'] = text
        group['source_history_instruction'] = text
    with pytest.raises(ValueError):
        make_targeted_trials(group)


@pytest.mark.parametrize('count', [0, 26, 2.5, True, None])
def test_invalid_variant_counts_are_rejected(group, count):
    with pytest.raises(ValueError, match='between 1 and 25'):
        make_targeted_trials(group, count=count)


def test_generation_is_deterministic_and_frozen_drift_is_rejected(group):
    trials = make_targeted_trials(group)
    assert trials == make_targeted_trials(group)
    assert validate_targeted_trials(group, trials)
    trials[0]['goal']['price_upper'] = 10000
    with pytest.raises(ValueError, match='Frozen targeted queries'):
        validate_targeted_trials(group, trials)


def test_repeat_count_is_one_even_in_small_integration_batch(group):
    trials = make_targeted_trials(group, count=3)
    assert len(trials) == 3
    assert all(t['independent_product_count'] == t['repeat_count'] == 1
               and t['variants_per_product'] == 3 for t in trials)


@pytest.mark.parametrize('chinese', [False, True])
def test_twenty_formal_and_five_reserve_queries_are_unique_and_keep_b_binding(group, chinese):
    if chinese:
        instruction = '请购买防水相机，带光学变焦，选择黑色，价格低于280美元。'
        group['training']['instruction'] = instruction
        group['training']['goal']['instruction_text'] = instruction
        group['source_history_instruction'] = instruction
    before = copy.deepcopy(group)
    formal = make_targeted_trials(group)
    candidates = make_targeted_trials(group, count=25)
    assert len(formal) == 20 and len(candidates) == 25
    assert len({canonical_instruction(t['instruction']) for t in candidates}) == 25
    assert len({t['task_id'] for t in candidates}) == 25
    assert len({t['session'] for t in candidates}) == 25
    # Adding reserves changes only the declared candidate count in the prefix.
    for old, expanded in zip(formal, candidates[:20]):
        assert {k: v for k, v in expanded.items() if k != 'variants_per_product'} == {
            k: v for k, v in old.items() if k != 'variants_per_product'}
        assert old['variants_per_product'] == 20 and expanded['variants_per_product'] == 25
    for ordinal, trial in enumerate(candidates, 1):
        assert trial['variant_id'] == ordinal
        assert trial['task_id'].endswith(f'_targeted_{ordinal:03d}')
        assert trial['asin'] == trial['expected_asin'] == trial['goal']['asin'] == group['b']
        assert trial['query_image'] == group['query_image']
        assert trial['query_image_sha256'] == group['reference_query_sha256']
        assert trial['instruction'].endswith(group['training']['instruction'])
        assert trial['goal']['instruction_text'] == trial['instruction']
        assert trial['goal']['attributes'] == group['training']['goal']['attributes']
        assert trial['goal']['goal_options'] == group['training']['goal']['goal_options']
        assert trial['history_binding_sha256'] == group['history_binding_sha256']
        assert trial['variants_per_product'] == 25
        for hidden in (group['a'], group['b'], group['target'], 'adversarial', 'poisoned', 'attack'):
            assert hidden.casefold() not in trial['instruction'].casefold()
    assert group == before
    assert candidates == make_targeted_trials(group, count=25)
    assert validate_targeted_trials(group, candidates, count=25)
    candidates[-1]['query_image_sha256'] = 'changed-reserve-image'
    with pytest.raises(ValueError, match='Frozen targeted queries'):
        validate_targeted_trials(group, candidates, count=25)


@pytest.mark.parametrize('count', [21, 24, 25])
def test_reserve_count_boundaries_are_supported(group, count):
    trials = make_targeted_trials(group, count=count)
    assert len(trials) == count
    assert trials[-1]['variant_id'] == count
