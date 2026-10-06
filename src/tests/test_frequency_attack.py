"""Frequency optimizer behavior and unchanged legacy delegation."""

from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def frequency_module():
    return importlib.import_module("src.vision.frequency_attack")


@pytest.fixture
def settings_config(tmp_path):
    path = tmp_path / "optimization.json"
    path.write_text(json.dumps({"frequency": {"initial_axis_ratio": 0.25}}), encoding="utf-8")
    return path


@pytest.mark.parametrize("method", ["legacy_pixel", "fair_pixel", "frequency"])
def test_existing_methods_delegate_with_original_arguments(frequency_module, monkeypatch, tmp_path, method):
    from src.vision import dual_rag_bridge

    calls = []
    expected = (tmp_path / "unchanged.png", {"original": True})

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(dual_rag_bridge, "generate_attack", original)
    result = frequency_module.generate_attack(
        "source.png", "target.png", tmp_path / "output", root=tmp_path / "dual",
        optimization_config="original.json", method=method, steps=7, epsilon_levels=12, device="cpu",
    )
    assert result is expected
    assert calls == [(("source.png", "target.png", tmp_path / "output"), {
        "root": tmp_path / "dual", "optimization_config": "original.json", "method": method,
        "steps": 7, "epsilon_levels": 12, "device": "cpu",
    })]


def test_import_settings_and_old_dispatch_need_no_heavy_dependencies(tmp_path):
    config_path = tmp_path / "optimization.json"
    config_path.write_text('{"frequency":{"initial_axis_ratio":0.25}}', encoding="utf-8")
    program = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'dual_rag'}:
        raise AssertionError('Unexpected heavy dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from src.vision import dual_rag_bridge
from src.vision import frequency_attack
sentinel = object()
dual_rag_bridge.generate_attack = lambda *args, **kwargs: sentinel
for method in ('legacy_pixel', 'fair_pixel', 'frequency'):
    assert frequency_attack.generate_attack('a', 'b', 'out', method=method) is sentinel
settings = frequency_attack.frequency_settings(optimization_config=sys.argv[1])
assert settings['frequency_axis_ratio'] == 0.25
assert len(settings['frequency_implementation_sha256']) == 64
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program, str(config_path)], cwd=PROJECT,
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("method", ["legacy_pixel", "fair_pixel", "frequency"])
@pytest.mark.parametrize("option", ["frequency_axis_ratio", "frequency_momentum"])
def test_existing_methods_reject_ignored_frequency_overrides(frequency_module, monkeypatch, method, option):
    from src.vision import dual_rag_bridge

    monkeypatch.setattr(dual_rag_bridge, "generate_attack", lambda *args, **kwargs: pytest.fail("Old engine ran"))
    with pytest.raises(ValueError):
        frequency_module.generate_attack("a", "b", "out", method=method, **{option: 0.5})


def test_settings_override_config_and_identify_implementation(frequency_module, settings_config):
    defaults = frequency_module.frequency_settings(optimization_config=settings_config)
    assert set(defaults) == {
        "frequency_axis_ratio", "frequency_momentum", "frequency_implementation_sha256",
    }
    assert defaults["frequency_axis_ratio"] == 0.25
    assert 0 <= defaults["frequency_momentum"] <= 1
    assert len(defaults["frequency_implementation_sha256"]) == 64
    changed = frequency_module.frequency_settings(
        optimization_config=settings_config, frequency_axis_ratio=0.5, frequency_momentum=0.0,
    )
    assert changed["frequency_axis_ratio"] == 0.5
    assert changed["frequency_momentum"] == 0.0
    assert changed != defaults


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, float("nan"), float("inf"), True, "bad"])
def test_invalid_axis_ratios_fail_before_models(frequency_module, settings_config, ratio):
    with pytest.raises((ValueError, TypeError)):
        frequency_module.frequency_settings(optimization_config=settings_config, frequency_axis_ratio=ratio)


@pytest.mark.parametrize("momentum", [-0.1, 1.1, float("nan"), float("inf"), True, "bad"])
def test_invalid_momentum_fails_before_models(frequency_module, settings_config, momentum):
    with pytest.raises((ValueError, TypeError)):
        frequency_module.frequency_settings(optimization_config=settings_config, frequency_momentum=momentum)


@pytest.fixture
def torch_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from src.vision.dual_rag_bridge import activate

    activate()
    return torch


@pytest.fixture
def small_config(torch_runtime):
    from src.vision.dual_rag_bridge import DEFAULT_DUAL_ROOT
    from dual_rag.optimization.common.config import load_config

    config = load_config(DEFAULT_DUAL_ROOT / "configs" / "optimization.json")
    return replace(
        config, device="cpu", attack=replace(
            config.attack, reference_steps=20, spatial_step_size=2 / 255, clusters=2,
            cluster_iterations=2, sinkhorn_iterations=8, sinkhorn_tolerance=0,
            text_weight=0.0, log_every=5,
        ), runtime=replace(
            config.runtime, allow_downloads=False, hash_model_weights=False,
            allow_device_fallback=False,
        ),
    )


@pytest.fixture
def model_factory(torch_runtime):
    torch = torch_runtime
    functional = torch.nn.functional
    from dual_rag.optimization.attack_core.surrogates import ImageFeatures

    class TinySurrogate:
        def __init__(self, index, constant=False):
            self.index = index
            self.constant = constant
            self.device = torch.device("cpu")
            self.cache_identity = f"test-frequency-surrogate-{index}-{constant}"

        def encode_image(self, image):
            means = image.mean(dim=(-2, -1))
            squares = image.square().mean(dim=(-2, -1))
            embedding = torch.cat((means, squares), dim=-1)
            patches = functional.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
            tokens = torch.cat((patches, patches.square()), dim=-1)
            if self.constant:
                embedding = embedding * 0 + 1
                tokens = tokens * 0 + 1
            return ImageFeatures(
                functional.normalize(torch.roll(embedding, self.index, dims=-1), dim=-1),
                functional.normalize(torch.roll(tokens, self.index, dims=-1), dim=-1),
            )

        def snapshot(self):
            return {"kind": "differentiable_test_model", "index": self.index}

    return lambda constant=False: [TinySurrogate(index, constant) for index in range(3)]


def test_quantized_forward_matches_saved_pixels_and_identity_gradient(frequency_module, torch_runtime):
    torch = torch_runtime
    from dual_rag.optimization.common.io import pil_to_tensor, tensor_to_pil

    image = torch.linspace(0, 1, 3 * 5 * 7).reshape(1, 3, 5, 7).requires_grad_(True)
    quantized = frequency_module.quantize_image(image)
    decoded = pil_to_tensor(tensor_to_pil(image))
    torch.testing.assert_close(quantized, decoded, rtol=0, atol=1e-7)
    gradient = torch.autograd.grad(quantized.sum(), image)[0]
    torch.testing.assert_close(gradient, torch.ones_like(image), rtol=0, atol=0)


def _images():
    from PIL import Image

    return Image.new("RGB", (11, 9), (70, 130, 190)), Image.new("RGB", (7, 13), (190, 80, 60))


def test_real_frequency_loop_improves_saved_score_and_preserves_budget(
    frequency_module, torch_runtime, small_config, model_factory, tmp_path,
):
    import numpy as np
    from PIL import Image

    source, target = _images()
    output, metadata = frequency_module.run_frequency_attack(
        source, target, small_config, steps=20, output_directory=tmp_path / "attack",
        frequency_axis_ratio=0.25, frequency_momentum=0.9, surrogates=model_factory(),
    )
    assert output.is_file()
    saved = Image.open(output).convert("RGB")
    assert saved.size == source.size
    differences = np.asarray(saved, dtype=np.int16) - np.asarray(source, dtype=np.int16)
    assert abs(differences).max() <= 16
    assert metadata["attack"]["steps_requested"] == 20
    assert metadata["attack"]["steps_completed"] == 20
    assert 0 <= metadata["attack"]["best_step"] <= 20
    before = metadata["before_attack"]["selection_score"]
    after = metadata["after_attack_saved"]["selection_score"]
    assert after > before + 1e-3
    from dual_rag.optimization.common.io import pil_to_tensor

    saved_tensor, target_tensor = pil_to_tensor(saved), pil_to_tensor(target)
    independently_scored_global = []
    with torch_runtime.no_grad():
        for model in model_factory():
            saved_feature = model.encode_image(saved_tensor).global_embedding
            target_feature = model.encode_image(target_tensor).global_embedding
            independently_scored_global.append(float(torch_runtime.nn.functional.cosine_similarity(
                saved_feature, target_feature,
            ).mean()))
    assert metadata["after_attack_saved"]["global_similarity"] == pytest.approx(independently_scored_global)
    record = metadata["after_attack_saved"]
    independently_fixed_score = sum(record["global_similarity"]) / 3 + small_config.attack.local_weight * (
        sum(record["local_similarity"]) / 3
    )
    assert after == pytest.approx(independently_fixed_score)
    assert metadata["decoded_pixel_audit"]["within_budget"]
    assert metadata["frequency"]["settings"]["frequency_axis_ratio"] == 0.25
    assert metadata["frequency"]["settings"]["frequency_momentum"] == 0.9
    for state in ("raw_synthesized", "after_range_clip", "after_png_reload"):
        assert 0 <= metadata["frequency"][state]["out_of_band_energy_ratio"] <= 1
    recorded = json.loads((output.parent / "metadata.json").read_text(encoding="utf-8"))
    assert recorded["after_attack_saved"] == metadata["after_attack_saved"]


def test_zero_gradient_returns_source_without_nonfinite_scores(
    frequency_module, torch_runtime, small_config, model_factory, tmp_path,
):
    import numpy as np
    from PIL import Image

    source, target = _images()
    output, metadata = frequency_module.run_frequency_attack(
        source, target, small_config, steps=4, output_directory=tmp_path / "zero",
        frequency_axis_ratio=0.25, frequency_momentum=1.0, surrogates=model_factory(constant=True),
    )
    assert metadata["attack"]["steps_completed"] == 4
    assert metadata["attack"]["best_step"] == 0
    np.testing.assert_array_equal(np.asarray(Image.open(output).convert("RGB")), np.asarray(source))
    assert metadata["after_attack_saved"]["selection_score"] >= metadata["before_attack"]["selection_score"]
    json.dumps(metadata, allow_nan=False)


def test_best_saved_candidate_survives_later_overshooting_updates(
    frequency_module, small_config, model_factory, tmp_path,
):
    from PIL import Image

    source = Image.new("RGB", (11, 9), (70, 130, 190))
    target = Image.new("RGB", (11, 9), (78, 122, 182))
    config = replace(small_config, attack=replace(small_config.attack, spatial_step_size=10 / 255))
    _, metadata = frequency_module.run_frequency_attack(
        source, target, config, steps=4, output_directory=tmp_path / "overshoot",
        frequency_axis_ratio=0.01, frequency_momentum=0.0, surrogates=model_factory(),
    )
    assert metadata["attack"]["steps_completed"] == 4
    assert 0 < metadata["attack"]["best_step"] < 4
    assert metadata["after_attack_saved"]["selection_score"] > (
        metadata["last_step_saved_score"]["selection_score"] + 1e-5
    )


@pytest.mark.parametrize("model_count", [0, 1, 2, 4])
def test_frequency_loop_requires_three_surrogates(
    frequency_module, small_config, model_factory, tmp_path, model_count,
):
    source, target = _images()
    models = model_factory()
    supplied = [models[index % 3] for index in range(model_count)]
    with pytest.raises(ValueError):
        frequency_module.run_frequency_attack(
            source, target, small_config, steps=1, output_directory=tmp_path / "invalid",
            frequency_axis_ratio=0.25, frequency_momentum=0.9, surrogates=supplied,
        )
