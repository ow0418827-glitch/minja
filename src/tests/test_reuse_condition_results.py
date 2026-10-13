import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.environment.local_webshop import read, write, fingerprint
from src.experiments import reuse_condition_results as reuse
from src.experiments.deferred_529 import Deferred529Evaluation, _fingerprint


@pytest.fixture
def inheritance(tmp_path, monkeypatch):
    root = tmp_path
    folder = root / 'groups' / '01_camera'
    source, destination = folder / 'evaluations' / 'old', folder / 'evaluations' / 'new'
    source.mkdir(parents=True)
    query = folder / 'query.jpg'
    query.write_bytes(b'fixed-query-image')
    for name in ('memory_ready.json', 'history_binding.json', 'memory.json', 'clean_memory.json'):
        write(folder / name, {'frozen': name})
    image = folder / 'attack' / 'adversarial.png'
    image.parent.mkdir()
    image.write_bytes(b'fixed-selected-image')
    sha = reuse.driver.digest
    ready = {'attack_sha256': sha(image), 'bank_sha256': {
        'adversarial': sha(folder / 'memory.json'), 'clean': sha(folder / 'clean_memory.json')}}
    group = {'id': '01_camera', 'query_image': str(query), 'history_binding_sha256': 'history-binding'}
    trials = [{'task_id': f'task-{index}', 'instruction': f'vague-request-{index}'} for index in range(20)]
    template = {'sha256': 'template-sha', 'selected_level': 1}
    protocol = {'test_mode': 'targeted', 'memory_key_mode': 'joint', 'memory_query_mode': 'joint',
                'index_text': 'full', 'query_specificity': 'vague', 'evaluation_code_sha256': 'old-code'}
    new_protocol = dict(protocol, evaluation_code_sha256='new-selector-code',
                        evaluation_conditions=['adversarial'])
    encoder = {'provider': 'clip', 'immutable_weights': 'weight-sha'}
    tasks_sha = fingerprint(trials)
    bindings = {'history_binding_sha256': group['history_binding_sha256'], 'tasks_sha256': tasks_sha,
                'retrieval_protocol_sha256': fingerprint(protocol), 'encoder_sha256': fingerprint(encoder),
                'template_sha256': template['sha256'], 'template_level': 1}
    reports, trajectories, wrapped = [], [], []
    for index in range(2):
        report = dict(bindings, task_id=trials[index]['task_id'], condition='adversarial',
                      query_instruction_sha256=fingerprint(trials[index]['instruction']),
                      query_image_sha256=sha(query), completed=index == 0,
                      candidate_index=index + 1, evaluation_slot=index + 1,
                      candidate_role='primary', replaces_task_id=None,
                      termination_reason='purchase_completed' if index == 0 else 'step_limit')
        trajectory = {'Instruction': trials[index]['instruction'], 'Actions': [f'action-{index}']}
        reports.append(report)
        trajectories.append(trajectory)
        wrapped.append({'report': report, 'trajectory': trajectory})
        write(source / 'shopping' / f'{index + 1:02d}_adversarial.json', trajectory)
    old = Deferred529Evaluation(source / 'deferred_529_state.json', trials,
        ['no_memory', 'clean', 'adversarial'], bindings, initial_results=wrapped)
    write(source / 'test_results.json', old.final_results())
    write(source / 'targeted_tasks.json', {'trials': trials, 'tasks_sha256': tasks_sha})
    write(source / 'evaluation_settings.json', {'template': template, 'protocol': protocol})
    write(source / 'retrieval_protocol.json', {'protocol': protocol, 'encoder': encoder,
          'encoder_sha256': fingerprint(encoder), 'tasks_sha256': tasks_sha})
    write(root / 'query_vector_preparation.json', {'encoder': encoder, 'encoder_sha256': _fingerprint(encoder)})
    write(root / 'retrieval_index_preparation.json', {'groups': [{'group': group['id'],
          'retrieval_snapshot': {'encoder': encoder}, 'tasks_sha256': tasks_sha}]})
    write(folder / 'retrieval_index_cache' / 'index.json', {'identity': {
          'bank_sha256': ready['bank_sha256']['adversarial'], 'model_sha256': _fingerprint(encoder),
          'key_mode': 'joint', 'query_mode': 'joint', 'index_text': 'full',
          'key_image_weight': 0.5, 'query_image_weight': 0.5}})
    frozen = {'memory_ready_sha256': sha(folder / 'memory_ready.json'),
              'history_binding_file_sha256': sha(folder / 'history_binding.json'),
              'memory_bank_sha256': sha(folder / 'memory.json'),
              'clean_memory_bank_sha256': sha(folder / 'clean_memory.json'),
              'attack_image_sha256': sha(image)}
    files = [{'path': str(path.resolve()), 'sha256': sha(path)} for path in source.rglob('*.json')]
    manifest = {'root': str(root), 'groups': [{'group': group['id'],
                'source_evaluation_directory': str(source), 'files': files, 'frozen_memory': frozen}]}
    args = SimpleNamespace(root=root, condition='adversarial', reserve_tests=0)
    monkeypatch.setattr(reuse.driver, 'load_plan', lambda _: (root, {}, [group]))
    monkeypatch.setattr(reuse.driver, 'resolve_history_binding', lambda a, r, p, g, **k: g)
    monkeypatch.setattr(reuse.driver, 'verify_banks', lambda _: ready)
    monkeypatch.setattr(reuse.driver, 'evaluation_protocol', lambda _: new_protocol)

    def location(args, parent, save=False):
        if save:
            write(destination / 'evaluation_settings.json', {'template': template, 'protocol': new_protocol})
        return destination, template

    def evaluation_trials(args, group, parent, save=True, task_folder=None):
        if save:
            write(destination / 'targeted_tasks.json', {'trials': trials, 'tasks_sha256': tasks_sha})
        return trials, tasks_sha

    monkeypatch.setattr(reuse.driver, 'evaluation_location', location)
    monkeypatch.setattr(reuse.driver, 'evaluation_trials', evaluation_trials)
    return args, manifest, source, destination, reports, new_protocol


def test_inherit_purchased_and_step_limit_without_touching_source(inheritance):
    args, manifest, source, destination, reports, protocol = inheritance
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    audit = reuse.seed_selected_condition(args, manifest)
    assert audit['inherited_total'] == 2
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    final = read(destination / 'test_results.json')['adversarial']
    assert [r['completed'] for r in final] == [True, False]
    assert all(r['retrieval_protocol_sha256'] == fingerprint(protocol) for r in final)
    assert final[1]['inherited_result']['source_report_sha256'] == fingerprint(reports[1])
    assert (destination / 'shopping' / '02_adversarial.json').read_bytes() == (source / 'shopping' / '02_adversarial.json').read_bytes()
    state = read(destination / 'deferred_529_state.json')
    assert state['identity']['conditions'] == ['adversarial']
    assert all(a['imported'] and a['phase'] == 'imported' for a in state['attempts'])


def test_reentry_verifies_without_replacing_later_results(inheritance):
    args, manifest, source, destination, _, _ = inheritance
    first = reuse.seed_selected_condition(args, manifest)
    before = {p: p.read_bytes() for p in destination.rglob('*') if p.is_file()}
    second = reuse.seed_selected_condition(args, manifest)
    assert second == first
    assert before == {p: p.read_bytes() for p in destination.rglob('*') if p.is_file()}


def test_reject_source_digest_change_before_publishing(inheritance):
    args, manifest, source, destination, _, _ = inheritance
    write(source / 'shopping' / '01_adversarial.json', {'changed': True})
    with pytest.raises(ValueError, match='source file changed'):
        reuse.seed_selected_condition(args, manifest)
    assert not destination.exists()


def test_reject_changed_treatment_even_if_source_digests_match(inheritance, monkeypatch):
    args, manifest, source, destination, _, protocol = inheritance
    monkeypatch.setattr(reuse.driver, 'evaluation_protocol', lambda _: dict(protocol, index_text='semantic'))
    with pytest.raises(ValueError, match='treatment changed'):
        reuse.seed_selected_condition(args, manifest)
    assert not destination.exists()


def test_reject_changed_memory_budget_image(inheritance):
    args, manifest, source, destination, _, _ = inheritance
    (args.root / 'groups' / '01_camera' / 'attack' / 'adversarial.png').write_bytes(b'other-image')
    with pytest.raises(ValueError, match='memory changed'):
        reuse.seed_selected_condition(args, manifest)
    assert not destination.exists()


def test_reject_missing_committed_trajectory_even_with_updated_digest(inheritance):
    args, manifest, source, destination, _, _ = inheritance
    trajectory = source / 'shopping' / '02_adversarial.json'
    write(trajectory, {'altered': True})
    for entry in manifest['groups'][0]['files']:
        if Path(entry['path']) == trajectory:
            entry['sha256'] = reuse.driver.digest(trajectory)
    with pytest.raises(ValueError, match='committed ledger attempt'):
        reuse.seed_selected_condition(args, manifest)
    assert not destination.exists()


def test_explicit_unstarted_source_can_have_preparation_metadata(inheritance):
    args, manifest, source, destination, _, _ = inheritance
    for path in list(source.rglob('*.json')):
        if path.name not in ('evaluation_settings.json', 'targeted_tasks.json'):
            path.unlink()
    row = manifest['groups'][0]
    row['status'] = 'not_started'
    row['files'] = [entry for entry in row['files'] if Path(entry['path']).exists()]
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    audit = reuse.seed_selected_condition(args, manifest)
    assert audit['inherited_total'] == 0
    assert read(destination / 'inherited_condition_results.json')['source_not_started']
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert not (destination / 'deferred_529_state.json').exists()


def test_unstarted_label_cannot_discard_committed_results(inheritance):
    args, manifest, source, destination, _, _ = inheritance
    manifest['groups'][0]['status'] = 'not_started'
    with pytest.raises(ValueError, match='contains prior shopping results'):
        reuse.seed_selected_condition(args, manifest)
    assert not destination.exists()
