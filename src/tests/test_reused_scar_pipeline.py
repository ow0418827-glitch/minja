import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.environment.local_webshop import read, write, fingerprint
from src.experiments import reused_scar_pipeline as pipeline
from src.vision.attack_completion import validate_attack_completion


@pytest.fixture
def imported(tmp_path):
    source_root = tmp_path / 'old_ssa'
    report_path = tmp_path / 'optimization' / 'postprocess_scale' / 'report.json'
    groups, rows = [], []
    for group_id, core_relative in pipeline.CORE_METADATA.items():
        folder = source_root / 'groups' / group_id
        folder.mkdir(parents=True)
        source, target = folder / 'source.png', folder / 'query.png'
        output = report_path.parent / group_id / 'selected.png'
        output.parent.mkdir(parents=True)
        input_image = output.parent / 'input.png'
        for path, level in ((source, 60), (target, 140), (input_image, 66), (output, 63)):
            Image.new('RGB', (8, 8), (level,) * 3).save(path)
        source_asset, target_asset = pipeline.asset(source), pipeline.asset(target)
        write(folder / 'history_binding.json', {'source_image_sha256': source_asset['sha256'],
              'reference_query_sha256': target_asset['sha256']})
        groups.append({'id': group_id, 'query_image': str(target)})
        core = report_path.parent.parent / core_relative
        write(core, {'status': 'success', 'source_image_sha256': source_asset['sha256'],
                     'target_image_sha256': target_asset['sha256'],
                     'output_image_sha256': pipeline.asset(input_image)['sha256'],
                     'attack': {'steps_completed': 30, 'best_step': 29, 'selected_step': 30}})
        rows.append({'group': group_id, 'source': str(source), 'source_sha256': source_asset['sha256'],
                     'query_image_sha256': target_asset['sha256'], 'input': str(input_image),
                     'input_sha256': pipeline.asset(input_image)['sha256'],
                     'selected': {'image': str(output), 'image_sha256': pipeline.asset(output)['sha256'],
                                  'scale': 0.5}, 'criteria': {'key_image_weight': 0.9}})
    write(source_root / 'manifest.json', {'groups': groups})
    write(report_path, {'status': 'completed', 'groups': rows})
    manifest = pipeline.precomputed_manifest(source_root, report_path)
    record = manifest['groups']['01_camera']
    audit = pipeline.pixel_audit(record['source']['path'], record['selected']['path'], 16)
    metadata = {'status': 'success', 'method': 'scar', 'generation': 'precomputed_scar_import',
                'attack': {'steps_requested': 0, 'steps_completed': 0, 'best_step': 0,
                           'selected_step': 0, 'stop_reason': 'precomputed_import_completed'},
                'precomputed_import': {'manifest_sha256': fingerprint(manifest), 'group_record': record,
                                       'output': record['selected'], 'pixel_audit': audit}}
    expected = {'method': 'scar', 'steps': 0, 'epsilon_levels': 16,
                'source_sha256': record['source']['sha256'], 'target_sha256': record['target']['sha256'],
                'precomputed_manifest_sha256': fingerprint(manifest), 'precomputed_group': record}
    return source_root, report_path, metadata, expected


def test_import_accepts_true_zero_steps_but_retains_upstream_provenance(imported):
    _, _, metadata, expected = imported
    validate_attack_completion(metadata, 0, expected)
    assert metadata['attack']['steps_completed'] == 0
    assert metadata['precomputed_import']['group_record']['upstream_optimization_steps'] == 30


def test_import_rejects_wrong_query_or_changed_selected_image(imported):
    source_root, report_path, metadata, expected = imported
    report = read(report_path)
    report['groups'][0]['query_image_sha256'] = 'incorrect'
    write(report_path, report)
    with pytest.raises(ValueError, match='digest'):
        pipeline.precomputed_manifest(source_root, report_path)
    with pytest.raises(ValueError, match='source changed'):
        validate_attack_completion(metadata, 0, expected)


@pytest.mark.parametrize('field', ['steps_requested', 'steps_completed', 'best_step', 'selected_step'])
def test_import_cannot_claim_old_optimizer_steps(imported, field):
    _, _, metadata, expected = imported
    changed = copy.deepcopy(metadata)
    changed['attack'][field] = 500
    with pytest.raises(ValueError, match='cannot claim'):
        validate_attack_completion(changed, 0, expected)


def test_evaluate_uses_existing_imports_without_calling_optimizer(tmp_path, monkeypatch):
    from contextlib import nullcontext
    args = SimpleNamespace(root=tmp_path, precomputed_manifest={'groups': {}})
    write(tmp_path / 'run_settings.json', {'controlled': True})
    write(tmp_path / 'scar_import_manifest.json', args.precomputed_manifest)
    write(tmp_path / 'history_reuse_provenance.json', {'source': 'old'})
    write(tmp_path / 'scar_preparation_result.json', {'status': 'ready_for_shopping'})
    calls = []
    monkeypatch.setattr(pipeline.driver, 'load_plan', lambda a: (tmp_path, {}, []))
    monkeypatch.setattr(pipeline.driver, 'execution_settings', lambda a, p: {'controlled': True})
    monkeypatch.setattr(pipeline, 'verify_reused_artifacts', lambda *a, **k: calls.append('audit'))
    monkeypatch.setattr(pipeline.driver, 'running_server', lambda a: nullcontext())
    monkeypatch.setattr(pipeline.driver, 'evaluate', lambda a: calls.append('evaluate'))
    monkeypatch.setattr(pipeline.driver, 'status', lambda a: {'status': 'incomplete'})
    monkeypatch.setattr(pipeline.driver, 'attack', lambda a: pytest.fail('Existing images must not be optimized'))
    pipeline.evaluate(args)
    assert calls == ['audit', 'evaluate', 'audit']
    assert read(tmp_path / 'pipeline_result.json')['status'] == 'incomplete'
