"""频谱模拟的官方变换语义、全像素更新和落盘预算。"""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ssa():
    return importlib.import_module("src.vision.ssa_attack")


def test_defaults_do_not_need_an_optimization_file_or_heavy_dependencies(tmp_path):
    program = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'dual_rag'}:
        raise AssertionError('Unexpected heavy dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from src.vision.ssa_attack import ssa_settings
import sys
options = ssa_settings(root=sys.argv[1])
assert options['ssa_samples'] == 20
assert options['ssa_rho'] == .5
assert options['ssa_sigma'] == 16
assert options['ssa_momentum'] == 1
assert len(options['ssa_implementation_sha256']) == 64
"""
    completed = subprocess.run([sys.executable, "-B", "-c", program, str(tmp_path)],
        cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_settings_read_only_ssa_section_and_allow_overrides(ssa, tmp_path):
    path = tmp_path / "optimization.json"
    path.write_text(json.dumps({"frequency": {"initial_axis_ratio": 0.125},
                               "ssa": {"samples": 4, "rho": 0.2, "sigma": 8, "momentum": 0.7}}),
                    encoding="utf-8")
    settings = ssa.ssa_settings(path)
    assert settings["ssa_samples"] == 4
    assert settings["ssa_rho"] == 0.2
    assert settings["ssa_sigma"] == 8
    assert settings["ssa_momentum"] == 0.7
    overridden = ssa.ssa_settings(path, ssa_samples=2, ssa_rho=0, ssa_sigma=0, ssa_momentum=0)
    assert overridden == {**settings, "ssa_samples": 2, "ssa_rho": 0,
                           "ssa_sigma": 0, "ssa_momentum": 0}


@pytest.mark.parametrize("option,value", [
    ("ssa_samples", 0), ("ssa_samples", True), ("ssa_samples", 2.5),
    ("ssa_rho", -0.1), ("ssa_rho", 1.1), ("ssa_rho", float("nan")),
    ("ssa_sigma", -1), ("ssa_sigma", float("inf")), ("ssa_sigma", True),
    ("ssa_momentum", -0.1), ("ssa_momentum", 1.1), ("ssa_momentum", True),
])
def test_invalid_settings_fail_before_models(ssa, option, value):
    with pytest.raises(ValueError):
        ssa.ssa_settings(**{option: value})


@pytest.mark.parametrize("method", ["legacy_pixel", "fair_pixel", "frequency", "frequency_momentum"])
def test_old_methods_delegate_without_ssa_settings(ssa, monkeypatch, tmp_path, method):
    from src.vision import frequency_attack
    calls = []
    expected = (tmp_path / "image.png", {})

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(frequency_attack, "generate_attack", original)
    assert ssa.generate_attack("a", "b", "out", method=method, steps=7, device="cpu") is expected
    assert calls[0][1]["method"] == method
    assert calls[0][1]["steps"] == 7
    with pytest.raises(ValueError):
        ssa.generate_attack("a", "b", "out", method=method, ssa_rho=0.5)


def test_common_attack_entry_forwards_ssa_arguments(ssa, monkeypatch, tmp_path):
    from src.vision import frequency_attack
    calls = []
    expected = (tmp_path / "image.png", {"method": "ssa"})
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return expected
    monkeypatch.setattr(ssa, "generate_attack", fake)
    result = frequency_attack.generate_attack("a", "b", "out",
        root=tmp_path / "dual", optimization_config="optimization.json",
        method="ssa", steps=7, epsilon_levels=12, device="cpu",
        ssa_samples=3, ssa_rho=0.4, ssa_sigma=8, ssa_momentum=0)
    assert result is expected
    assert calls == [(("a", "b", "out"), {
        "root": tmp_path / "dual", "optimization_config": "optimization.json",
        "method": "ssa", "steps": 7, "epsilon_levels": 12, "device": "cpu",
        "ssa_samples": 3, "ssa_rho": 0.4, "ssa_sigma": 8, "ssa_momentum": 0,
    })]


@pytest.mark.parametrize("method,option", [
    ("ssa", "frequency_axis_ratio"), ("ssa", "frequency_momentum"),
    ("frequency_momentum", "ssa_samples"), ("frequency_momentum", "ssa_rho"),
    ("frequency_momentum", "ssa_sigma"), ("frequency_momentum", "ssa_momentum"),
    ("legacy_pixel", "ssa_samples"), ("legacy_pixel", "ssa_rho"),
    ("legacy_pixel", "ssa_sigma"), ("legacy_pixel", "ssa_momentum"),
])
def test_common_entry_rejects_mutually_exclusive_parameters(ssa, monkeypatch, method, option):
    from src.vision import frequency_attack, dual_rag_bridge
    monkeypatch.setattr(ssa, "generate_attack", lambda *args, **kwargs: pytest.fail("SSA ran"))
    monkeypatch.setattr(dual_rag_bridge, "generate_attack", lambda *args, **kwargs: pytest.fail("Old engine ran"))
    with pytest.raises(ValueError):
        frequency_attack.generate_attack("a", "b", "out", method=method, **{option: 1})


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
    config = load_config(DEFAULT_DUAL_ROOT / "configs/optimization.json")
    return replace(config, device="cpu", attack=replace(config.attack,
        spatial_step_size=2 / 255, clusters=2, cluster_iterations=2,
        sinkhorn_iterations=8, sinkhorn_tolerance=0, text_weight=0.0, log_every=5),
        runtime=replace(config.runtime, allow_downloads=False, hash_model_weights=False))


@pytest.fixture
def model_factory(torch_runtime):
    torch = torch_runtime
    functional = torch.nn.functional
    from dual_rag.optimization.attack_core.surrogates import ImageFeatures

    class TinySurrogate:
        def __init__(self, index, constant):
            self.index, self.constant = index, constant
            self.device = torch.device("cpu")
            self.cache_identity = f"ssa-test-{index}-{constant}"

        def encode_image(self, image):
            means = image.mean(dim=(-2, -1))
            squares = image.square().mean(dim=(-2, -1))
            embedding = torch.cat((means, squares), dim=-1)
            patches = functional.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
            tokens = torch.cat((patches, patches.square()), dim=-1)
            if self.constant:
                embedding, tokens = embedding * 0 + 1, tokens * 0 + 1
            return ImageFeatures(
                functional.normalize(torch.roll(embedding, self.index, dims=-1), dim=-1),
                functional.normalize(torch.roll(tokens, self.index, dims=-1), dim=-1))

        def snapshot(self):
            return {"index": self.index, "kind": "test_surrogate"}

    return lambda constant=False: [TinySurrogate(index, constant) for index in range(3)]


@pytest.mark.parametrize("height,width", [(1, 1), (4, 6), (5, 7), (9, 11)])
def test_dct_matches_direct_definition_and_round_trips(ssa, torch_runtime, height, width):
    torch = torch_runtime
    image = torch.linspace(-0.7, 1.1, 3 * height * width, dtype=torch.float64).reshape(1, 3, height, width)
    def basis(length):
        spatial = torch.arange(length, dtype=torch.float64) + 0.5
        frequency = torch.arange(length, dtype=torch.float64)
        return 2 * torch.cos(torch.pi / length * frequency[:, None] * spatial[None, :])
    expected = torch.einsum("kh,bchw,lw->bckl", basis(height), image, basis(width))
    torch.testing.assert_close(ssa.dct_2d(image), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(ssa.idct_2d(ssa.dct_2d(image)), image, rtol=1e-10, atol=1e-10)


def test_simulation_uses_official_mask_noise_and_detached_unclipped_output(ssa, torch_runtime, monkeypatch):
    torch = torch_runtime
    image = torch.full((1, 3, 5, 7), 0.99, requires_grad=True)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.full_like(tensor, 0.75))
    simulated = ssa.spectrum_simulation(image, rho=0.5, sigma=16)
    expected = (image.detach() + 16 / 255) * 1.25
    torch.testing.assert_close(simulated, expected, rtol=1e-5, atol=1e-6)
    assert not simulated.requires_grad
    assert simulated.is_leaf
    assert simulated.min() > 1


def _images():
    from PIL import Image
    return Image.new("RGB", (11, 9), (70, 130, 190)), Image.new("RGB", (7, 13), (190, 80, 60))


def test_full_pixel_loop_improves_saved_score_and_preserves_decoded_budget(
    ssa, torch_runtime, small_config, model_factory, tmp_path,
):
    import numpy as np
    from PIL import Image
    source, target = _images()
    image_path, metadata = ssa.run_ssa_attack(source, target, small_config,
        steps=10, output_directory=tmp_path / "attack", ssa_samples=2,
        ssa_rho=0.3, ssa_sigma=4, ssa_momentum=1, surrogates=model_factory())
    assert image_path.is_file()
    saved = Image.open(image_path).convert("RGB")
    assert saved.size == source.size
    differences = np.asarray(saved, dtype=np.int16) - np.asarray(source, dtype=np.int16)
    assert abs(differences).max() <= 16
    assert metadata["attack"]["steps_completed"] == 10
    assert 0 <= metadata["attack"]["best_step"] <= 10
    assert metadata["after_attack_saved"]["selection_score"] > metadata["before_attack"]["selection_score"] + 1e-3
    assert metadata["decoded_pixel_audit"]["within_budget"]
    assert metadata["ssa"]["parameter_count"] == 3 * 11 * 9
    assert not metadata["ssa"]["band_dropout"]
    assert not metadata["ssa"]["backpropagate_through_transform"]
    assert metadata["ssa"]["momentum_is_project_extension"]
    assert metadata["ssa"]["upstream_commit"] == "c955cf07c8372bfc4e9f17e647042e027f9f3b1d"
    from dual_rag.optimization.common.io import pil_to_tensor
    saved_tensor, target_tensor = pil_to_tensor(saved), pil_to_tensor(target)
    independent_scores = []
    with torch_runtime.no_grad():
        for model in model_factory():
            independent_scores.append(float(torch_runtime.nn.functional.cosine_similarity(
                model.encode_image(saved_tensor).global_embedding,
                model.encode_image(target_tensor).global_embedding).mean()))
    assert metadata["after_attack_saved"]["global_similarity"] == pytest.approx(independent_scores)
    recorded = json.loads((image_path.parent / "metadata.json").read_text(encoding="utf-8"))
    assert recorded["after_attack_saved"] == metadata["after_attack_saved"]


def test_each_step_averages_detached_simulated_input_gradients(
    ssa, torch_runtime, small_config, model_factory, monkeypatch, tmp_path,
):
    torch = torch_runtime
    from dual_rag.optimization.attack_core import objective
    actual_gradient = objective.sequential_image_gradient
    calls = []
    def observed(image_leaf, *args):
        assert image_leaf.is_leaf and image_leaf.requires_grad
        assert image_leaf.grad_fn is None
        gradient, values = actual_gradient(image_leaf, *args)
        calls.append(image_leaf.detach().clone())
        # Direct simulated-image gradients should update full pixels; their
        # magnitude must not be remultiplied by the random frequency mask.
        gradient = torch.ones_like(gradient) * (1 if len(calls) % 3 else -1.5)
        return gradient, values
    monkeypatch.setattr(objective, "sequential_image_gradient", observed)
    source, target = _images()
    output, metadata = ssa.run_ssa_attack(source, target, small_config,
        steps=2, output_directory=tmp_path / "semantics", ssa_samples=3,
        ssa_rho=0.5, ssa_sigma=16, ssa_momentum=0, surrogates=model_factory())
    assert len(calls) == 6
    assert not torch.equal(calls[0], calls[1])
    assert metadata["history"][0]["linf_after_update"] == pytest.approx(2 / 255)
    assert output.is_file()


def test_zero_gradient_saves_the_original_with_finite_metadata(
    ssa, torch_runtime, small_config, model_factory, tmp_path,
):
    import numpy as np
    from PIL import Image
    source, target = _images()
    output, metadata = ssa.run_ssa_attack(source, target, small_config,
        steps=3, output_directory=tmp_path / "zero", ssa_samples=2,
        ssa_rho=0.5, ssa_sigma=16, surrogates=model_factory(constant=True))
    np.testing.assert_array_equal(np.asarray(Image.open(output).convert("RGB")), np.asarray(source))
    assert metadata["attack"]["best_step"] == 0
    assert metadata["attack"]["zero_gradient_steps"] == 3
    json.dumps(metadata, allow_nan=False)


def test_best_quantized_candidate_survives_later_overshoot(
    ssa, small_config, model_factory, tmp_path,
):
    from PIL import Image
    source = Image.new("RGB", (11, 9), (70, 130, 190))
    target = Image.new("RGB", (11, 9), (78, 122, 182))
    config = replace(small_config, attack=replace(small_config.attack, spatial_step_size=10 / 255))
    _, metadata = ssa.run_ssa_attack(source, target, config,
        steps=4, output_directory=tmp_path / "overshoot", ssa_samples=1,
        ssa_rho=0, ssa_sigma=0, ssa_momentum=0, surrogates=model_factory())
    assert 0 < metadata["attack"]["best_step"] < 4
    assert metadata["after_attack_saved"]["selection_score"] > metadata["last_step_saved_score"]["selection_score"] + 1e-5


@pytest.mark.parametrize("model_count", [0, 1, 2, 4])
def test_requires_exactly_three_surrogates(ssa, small_config, model_factory, tmp_path, model_count):
    source, target = _images()
    models = model_factory()
    with pytest.raises(ValueError):
        ssa.run_ssa_attack(source, target, small_config, steps=1,
            output_directory=tmp_path / "invalid", ssa_samples=1,
            surrogates=[models[index % 3] for index in range(model_count)])


def test_existing_output_is_not_overwritten(ssa, small_config, model_factory, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "adversarial.png"
    sentinel.write_bytes(b"existing-image")
    source, target = _images()
    with pytest.raises(ValueError):
        ssa.run_ssa_attack(source, target, small_config, steps=1,
            output_directory=output, ssa_samples=1, surrogates=model_factory())
    assert sentinel.read_bytes() == b"existing-image"
