"""Root status and variant isolation for twenty formal tests plus five reserves."""
import copy
from types import SimpleNamespace

import pytest

from src.environment.local_webshop import fingerprint, read, write
from src.experiments import nine_group_experiment as driver
from src.tests.test_targeted_joint_driver import args as targeted_args, fixed_group


def setup_status(tmp_path, monkeypatch, *, specificity='strict', group_count=1):
    base, args = fixed_group(tmp_path), targeted_args()
    args.reserve_tests = 5
    args.query_specificity = specificity
    groups = []
    for index in range(group_count):
        group = copy.deepcopy(base)
        group['id'] = f'{index + 1:02d}_camera'
        groups.append(group)
    plan = {'protocol': driver.VICTIM_PROTOCOL, 'catalog_scope': 'status-fixture'}
    monkeypatch.setattr(driver, 'load_plan', lambda *unused: (tmp_path, plan, groups))
    write(tmp_path / 'summary.json', {'existing_experiment': '原始二十条需求汇总', 'status': 'completed'})
    original = (tmp_path / 'summary.json').read_bytes()
    return args, groups, original


def save_variant(root, args, group, *, executed=20, targets=8, audit_status='complete',
                 errors_total=0, reserves_used=0):
    folder = root / 'groups' / group['id']
    location, _ = driver.evaluation_location(args, folder)
    summary = {'adversarial': {
        'planned_tests': 20, 'executed_tests': executed, 'valid_tests': executed,
        'target_purchases': targets, 'technical_errors': 0,
        'target_purchase_rate': targets / 20 if executed == 20 else None},
        'replacement_audit': {'status': audit_status, 'primary_count': 20,
            'candidate_pool_size': 25, 'reserve_count': 5,
            'technical_errors_total': errors_total, 'reserve_used': reserves_used},
        'test_design': {'wordings': 20, 'candidate_wordings': 25, 'reserve_wordings': 5}}
    write(location / 'summary.json', summary)
    return folder, location


def saved_root_summary(root, args):
    return root / 'evaluation_summaries' / (fingerprint(driver.evaluation_protocol(args)) + '.json')


@pytest.mark.parametrize('specificity', ['strict', 'vague'])
@pytest.mark.parametrize('executed,targets,audit_status,errors_total,reserves_used,expected_status', [
    (20, 8, 'complete', 0, 0, 'completed'),
    (20, 8, 'complete', 2, 2, 'completed_with_replacements'),
    (19, 8, 'reserve_exhausted', 6, 5, 'reserve_exhausted'),
    (3, 1, 'fatal', 1, 0, 'stopped_with_fatal_error'),
    (19, 8, 'incomplete', 1, 0, 'incomplete'),
])
def test_root_reserve_status_uses_twenty_formal_tests_and_separate_error_audit(
        tmp_path, monkeypatch, specificity, executed, targets, audit_status,
        errors_total, reserves_used, expected_status):
    args, groups, original = setup_status(tmp_path, monkeypatch, specificity=specificity)
    _, location = save_variant(tmp_path, args, groups[0], executed=executed, targets=targets,
        audit_status=audit_status, errors_total=errors_total, reserves_used=reserves_used)
    group_summary = (location / 'summary.json').read_bytes()

    result = driver.status(args)

    assert result['status'] == expected_status
    assert result['planned_primary_tests'] == 20
    assert result['executed_primary_tests'] == executed
    assert result['target_purchases'] == targets
    assert result['technical_errors'] == 0
    assert result['technical_errors_total'] == errors_total
    assert result['reserve_demands_used'] == reserves_used
    assert result['target_purchase_rate'] == (targets / 20 if executed == 20 else None)
    assert result['evaluation_protocol']['reserve_tests'] == 5
    assert result['evaluation_protocol']['primary_tests'] == 20
    assert result['evaluation_protocol']['query_specificity'] == specificity
    assert result['groups'][0]['evaluation_directory'] == str(location)
    assert read(saved_root_summary(tmp_path, args)) == result
    assert (tmp_path / 'summary.json').read_bytes() == original
    assert (location / 'summary.json').read_bytes() == group_summary


@pytest.mark.parametrize('specificity', ['strict', 'vague'])
def test_missing_group_summary_remains_incomplete_with_unknown_rate(tmp_path, monkeypatch, specificity):
    args, _, original = setup_status(tmp_path, monkeypatch, specificity=specificity)

    result = driver.status(args)

    assert result['status'] == 'incomplete'
    assert result['groups'][0]['tests'] == {}
    assert result['planned_primary_tests'] == 20
    assert result['executed_primary_tests'] == 0
    assert result['technical_errors_total'] == result['reserve_demands_used'] == 0
    assert result['target_purchase_rate'] is None
    assert (tmp_path / 'summary.json').read_bytes() == original


@pytest.mark.parametrize('specificity', ['strict', 'vague'])
def test_targeted_tasks_ready_checks_reserve_variant_instead_of_legacy_folder(
        tmp_path, monkeypatch, specificity):
    args, groups, original = setup_status(tmp_path, monkeypatch, specificity=specificity)
    folder, location = save_variant(tmp_path, args, groups[0])
    write(folder / 'targeted_tasks.json', {'legacy_tasks': '原二十条需求'})
    legacy = (folder / 'targeted_tasks.json').read_bytes()

    assert driver.status(args)['groups'][0]['targeted_tasks_ready'] is False
    write(location / 'targeted_tasks.json', {'candidate_count': 25, 'test_count': 20})
    result = driver.status(args)

    assert result['groups'][0]['targeted_tasks_ready'] is True
    assert result['groups'][0]['evaluation_directory'] == str(location)
    assert (folder / 'targeted_tasks.json').read_bytes() == legacy
    assert (tmp_path / 'summary.json').read_bytes() == original


def test_root_aggregates_multiple_groups_without_including_reserves_in_denominator(tmp_path, monkeypatch):
    args, groups, original = setup_status(tmp_path, monkeypatch, group_count=2)
    save_variant(tmp_path, args, groups[0], targets=8, errors_total=2, reserves_used=2)
    save_variant(tmp_path, args, groups[1], targets=12, errors_total=1, reserves_used=1)

    result = driver.status(args)

    assert result['status'] == 'completed_with_replacements'
    assert result['planned_primary_tests'] == result['executed_primary_tests'] == 40
    assert result['target_purchases'] == 20
    assert result['target_purchase_rate'] == 0.5
    assert result['technical_errors_total'] == result['reserve_demands_used'] == 3
    assert result['technical_errors'] == 0
    assert (tmp_path / 'summary.json').read_bytes() == original


@pytest.mark.parametrize('fatal_first', [False, True])
def test_fatal_group_takes_priority_over_reserve_exhaustion(tmp_path, monkeypatch, fatal_first):
    args, groups, _ = setup_status(tmp_path, monkeypatch, group_count=2)
    exhausted, fatal = groups if not fatal_first else groups[::-1]
    save_variant(tmp_path, args, exhausted, executed=19, audit_status='reserve_exhausted',
                 errors_total=6, reserves_used=5)
    save_variant(tmp_path, args, fatal, executed=0, targets=0, audit_status='fatal', errors_total=1)

    result = driver.status(args)

    assert result['status'] == 'stopped_with_fatal_error'
    assert result['planned_primary_tests'] == 40
    assert result['executed_primary_tests'] == 19
    assert result['technical_errors_total'] == 7
    assert result['reserve_demands_used'] == 5
    assert result['target_purchase_rate'] is None


def test_strict_and_vague_root_summaries_have_independent_variant_paths(tmp_path, monkeypatch):
    args, groups, original = setup_status(tmp_path, monkeypatch)
    _, strict_location = save_variant(tmp_path, args, groups[0], targets=8)
    strict_result = driver.status(args)
    strict_path = saved_root_summary(tmp_path, args)
    strict_summary = strict_path.read_bytes()

    args.query_specificity = 'vague'
    _, vague_location = save_variant(tmp_path, args, groups[0], targets=12,
                                     errors_total=1, reserves_used=1)
    vague_result = driver.status(args)
    vague_path = saved_root_summary(tmp_path, args)

    assert strict_location != vague_location and strict_path != vague_path
    assert strict_result['target_purchase_rate'] == 0.4
    assert vague_result['target_purchase_rate'] == 0.6
    assert strict_result['status'] == 'completed'
    assert vague_result['status'] == 'completed_with_replacements'
    assert strict_path.read_bytes() == strict_summary
    assert read(vague_path) == vague_result
    assert (tmp_path / 'summary.json').read_bytes() == original


@pytest.mark.parametrize('failure', [ValueError('Frozen image digest mismatch'), KeyboardInterrupt()])
@pytest.mark.parametrize('status_update_fails', [False, True])
def test_interrupted_evaluation_refreshes_status_and_preserves_original_error(
        tmp_path, monkeypatch, failure, status_update_fails):
    args, groups, _ = setup_status(tmp_path, monkeypatch)
    events = []
    encoder = SimpleNamespace(close=lambda: events.append('close'))
    monkeypatch.setattr(driver, 'freeze_settings', lambda *unused: {})
    monkeypatch.setattr(driver, 'server_health', lambda *unused: {})
    monkeypatch.setattr(driver, 'runtime', lambda *unused, **kwargs: {'visual_encoder': encoder})

    def fail_evaluation(options, root, plan, selected_groups, runtime):
        assert options is args and selected_groups is groups
        events.append('evaluate')
        raise failure

    def refresh_status(options):
        assert options is args
        events.append('status')
        if status_update_fails:
            raise OSError('Fixture cannot write root summary')
        return {'status': 'stopped_with_fatal_error'}

    monkeypatch.setattr(driver, 'evaluate_groups', fail_evaluation)
    monkeypatch.setattr(driver, 'status', refresh_status)

    with pytest.raises(type(failure)) as caught:
        driver.evaluate(args)

    assert caught.value is failure
    assert events == ['evaluate', 'status', 'close']
