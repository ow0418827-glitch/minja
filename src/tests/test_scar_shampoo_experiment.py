from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.environment.local_webshop import read, write
from src.experiments import scar_shampoo_experiment as pipeline


def test_pipeline_finishes_images_and_indices_before_shopping():
    calls = []
    core = object()
    target = object()
    result = pipeline.run(core, 'final',
        optimize_fn=lambda a, b: calls.append('optimize'),
        select_fn=lambda a: calls.append('select') or {'selected': True},
        shopping_args_fn=lambda *a: calls.append('bind') or target,
        prepare_fn=lambda a: calls.append('prime') or {'verified': True},
        evaluate_fn=lambda a, p: calls.append('shopping') or {'status': 'running'})
    assert calls == ['optimize', 'select', 'bind', 'prime', 'shopping']
    assert result['status'] == 'running'


def test_optimizer_failure_cannot_start_shopping():
    def fail(*args):
        raise RuntimeError('unfinished image optimization')
    with pytest.raises(RuntimeError, match='unfinished'):
        pipeline.run(object(), 'final', optimize_fn=fail,
                     evaluate_fn=lambda *a: pytest.fail('Shopping must not start before a complete image'))


@pytest.fixture
def amplitude_case(tmp_path, monkeypatch):
    root, source_root = tmp_path / 'core', tmp_path / 'source'
    folder = source_root / 'groups' / pipeline.GROUP
    source_image, target, raw = tmp_path / 'source.png', tmp_path / 'target.png', tmp_path / 'raw.png'
    normal, old_attack = tmp_path / 'normal.png', tmp_path / 'old_attack.png'
    for path, level in ((source_image, 50), (target, 200), (raw, 58), (normal, 25), (old_attack, 54)):
        Image.new('RGB', (8, 8), (level,) * 3).save(path)
    bank = [{'Id': f'memory-{i}', 'Instruction': f'normal {i}', 'Actions': [],
             'PrimaryImage': {'LocalPath': str(normal)}} for i in range(14)]
    bank.append({'Id': 'attacker', 'Instruction': 'wrong product', 'Actions': [],
                 'PrimaryImage': {'LocalPath': str(old_attack)}})
    write(folder / 'memory.json', bank)
    write(folder / 'memory_ready.json', {'source_memory_id': 'attacker'})
    variant = folder / 'evaluations' / 'source_variant'
    write(variant / 'targeted_tasks.json', {'trials': [{'instruction': f'query {i}'} for i in range(20)]})
    write(root / 'groups' / pipeline.GROUP / 'attack_ready.json', {'image': str(raw)})
    write(root / 'groups' / pipeline.GROUP / 'attack/metadata.json', {'status': 'success'})
    group = {'id': pipeline.GROUP, 'source_image': str(source_image), 'query_image': str(target)}
    args = SimpleNamespace(root=root, source_root=source_root, optimization_config=None, device='cpu',
                           source_evaluation_overrides={pipeline.GROUP: str(variant)})
    monkeypatch.setattr(pipeline.driver, 'load_plan', lambda a: (root, {}, [group]))
    monkeypatch.setattr(pipeline.driver, 'resolve_history_binding', lambda *a, **k: group)
    modes = {'strong_normals': False}

    class Encoder:
        def encode(self, path):
            path = str(path)
            level = float(np.asarray(Image.open(path)).mean())
            if modes['strong_normals'] and path == str(normal):
                level = 200
            return np.array([level / 255, 1], dtype=np.float32)

        def encode_text(self, text):
            return np.array([1, 0], dtype=np.float32)

        def close(self):
            pass

    monkeypatch.setattr(pipeline, 'make_joint_encoder', lambda **k: Encoder())
    return args, raw, modes


def test_amplitude_selection_preserves_raw_and_finds_minimum_feasible_change(amplitude_case):
    args, raw, _ = amplitude_case
    original_sha = pipeline.driver.digest(raw)
    report = pipeline.select_amplitude(args)
    assert report['selected']['scale'] == 0.5
    assert report['selected']['source_top1'] == 20
    assert report['selected']['pixel_audit']['max_levels'] == 4
    assert pipeline.driver.digest(raw) == original_sha
    assert report['key_image_weight'] == report['query_image_weight'] == 0.5
    assert report['index_text'] == 'full'


def test_no_feasible_scale_keeps_raw_without_claiming_rank_success(amplitude_case):
    args, raw, modes = amplitude_case
    modes['strong_normals'] = True
    report = pipeline.select_amplitude(args)
    assert report['fallback_used']
    assert report['selected']['scale'] == 1
    assert not report['selected']['eligible']
    assert report['selected']['source_top1'] == 0
    assert np.array_equal(np.asarray(Image.open(raw)), np.asarray(Image.open(report['selected']['image'])))


def test_saved_selected_bytes_cannot_be_silently_reused_if_changed(amplitude_case):
    args, _, _ = amplitude_case
    report = pipeline.select_amplitude(args)
    Image.new('RGB', (8, 8), (90,) * 3).save(report['selected']['image'])
    with pytest.raises(ValueError, match='Selected SCAR image changed'):
        pipeline.select_amplitude(args)
