"""扩散引导符号、反演重建、八位重评分与失败保护。"""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def diffusion():
    return importlib.import_module("src.vision.diffusion_attack")


@pytest.fixture
def torch_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from src.vision.dual_rag_bridge import activate
    activate()
    return torch


@pytest.fixture
def config(torch_runtime):
    from src.vision.dual_rag_bridge import DEFAULT_DUAL_ROOT
    from dual_rag.optimization.common.config import load_config
    original = load_config(DEFAULT_DUAL_ROOT / "configs/optimization.json")
    return replace(original, device="cpu", attack=replace(original.attack,
        clusters=2, cluster_iterations=2, sinkhorn_iterations=8,
        sinkhorn_tolerance=0, text_weight=0),
        runtime=replace(original.runtime, allow_downloads=False, hash_model_weights=False))


@pytest.fixture
def models(torch_runtime):
    torch = torch_runtime
    from dual_rag.optimization.attack_core.surrogates import ImageFeatures

    class SmallSurrogate:
        device = torch.device("cpu")

        def __init__(self, index):
            self.cache_identity = f"diffusion-test-{index}"

        def encode_image(self, image):
            pooled = image.mean((-2, -1))
            embedding = torch.cat((pooled, image.square().mean((-2, -1))), dim=-1)
            patches = torch.nn.functional.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
            tokens = torch.cat((patches, patches.square()), dim=-1)
            return ImageFeatures(torch.nn.functional.normalize(embedding, dim=-1),
                                 torch.nn.functional.normalize(tokens, dim=-1))

    return [SmallSurrogate(index) for index in range(3)]


@pytest.fixture
def components(diffusion, torch_runtime):
    torch = torch_runtime
    diffusers = pytest.importorskip("diffusers")

    class ZeroNoiseNetwork:
        dtype = torch.float32
        calls = 0

        def __call__(self, latents, timestep, encoder_hidden_states):
            # 真实噪声预测必须没有反向图；测试只替代昂贵权重。
            assert not torch.is_grad_enabled()
            self.calls += 1
            return SimpleNamespace(sample=torch.zeros_like(latents))

    class IdentityVAE:
        config = SimpleNamespace(scaling_factor=1.0)
        differentiable_decode_calls = 0

        def encode(self, image):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: image))

        def decode(self, latents):
            if latents.requires_grad:
                self.differentiable_decode_calls += 1
            return SimpleNamespace(sample=latents)

    scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000,
        beta_start=.00085, beta_end=.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=True, steps_offset=0, timestep_spacing="leading")
    inverse = diffusers.DDIMInverseScheduler.from_config(scheduler.config)
    return diffusion.DiffusionComponents(ZeroNoiseNetwork(), IdentityVAE(), scheduler,
        inverse, torch.zeros(1, 1, 1), {"test_only": True, "not_a_real_model_result": True})


def images():
    from PIL import Image
    return Image.new("RGB", (13, 17), (70, 130, 190)), Image.new("RGB", (11, 9), (190, 80, 60))


def settings(diffusion, tmp_path, **overrides):
    return diffusion.diffusion_settings(root=tmp_path, diffusion_options={
        "image_size": 64, "strength": .5, "source_weight": 0,
        "guidance_scale": 5, "latent_step_clip": .1, **overrides})


def test_config_is_lightweight_without_model_dependencies(tmp_path):
    code = """
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'diffusers', 'transformers', 'dual_rag'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from src.vision.diffusion_attack import diffusion_settings
import sys
options = diffusion_settings(root=sys.argv[1])
assert options['initialization'] == 'ddim_inversion'
assert options['image_size'] == 256
assert len(options['implementation_sha256']) == 64
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], cwd=PROJECT,
        text=True, encoding="utf-8", capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("option,value", [
    ("image_size", True), ("image_size", 63), ("image_size", 255),
    ("seed", -1), ("seed", True), ("strength", 0), ("strength", 1.1),
    ("guidance_scale", -1), ("guidance_scale", float("nan")),
    ("latent_step_clip", 0), ("source_weight", float("inf")),
    ("normalize_gradient", 1), ("prompt_guidance_scale", .5),
    ("initialization", "strict_manifold"), ("unknown", 1),
    ("early_stop", 1), ("min_guidance_steps", 0), ("min_guidance_steps", True),
    ("check_every", 0), ("patience", 1.5), ("min_delta", 0),
    ("audit_after_stop", True),
])
def test_invalid_config_fails_before_loading(diffusion, tmp_path, option, value):
    with pytest.raises(ValueError):
        diffusion.diffusion_settings(root=tmp_path, diffusion_options={option: value})


def test_real_component_loader_uses_supported_attention_slice_api_on_cpu(
    diffusion, torch_runtime, tmp_path, monkeypatch,
):
    """小型真实组件走生产加载器，防止不存在的注意力切片方法回归。"""
    torch = torch_runtime
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    unet = diffusers.UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4,
        layers_per_block=1, block_out_channels=(8, 8), norm_num_groups=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=8, attention_head_dim=2)
    vae = diffusers.AutoencoderKL(in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 8), layers_per_block=1, latent_channels=4,
        norm_num_groups=4, sample_size=32)
    text_encoder = transformers.CLIPTextModel(transformers.CLIPTextConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
        bos_token_id=0, eos_token_id=2, pad_token_id=1))

    class SmallTokenizer:
        model_max_length = 8

        def __call__(self, prompts, **kwargs):
            return SimpleNamespace(input_ids=torch.zeros(len(prompts), 8, dtype=torch.long))

    load_calls = []

    def loader_for(component, expected_subfolder):
        def load(path, **kwargs):
            assert kwargs["subfolder"] == expected_subfolder
            assert kwargs["local_files_only"] is True
            load_calls.append(expected_subfolder)
            return component
        return load

    monkeypatch.setattr(diffusers.UNet2DConditionModel, "from_pretrained",
                        loader_for(unet, "unet"))
    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained",
                        loader_for(vae, "vae"))
    monkeypatch.setattr(transformers.CLIPTextModel, "from_pretrained",
                        loader_for(text_encoder, "text_encoder"))
    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained",
                        loader_for(SmallTokenizer(), "tokenizer"))
    scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000,
        clip_sample=False, set_alpha_to_one=True, steps_offset=0, timestep_spacing="leading")
    monkeypatch.setattr(diffusers.DDIMScheduler, "from_pretrained",
                        loader_for(scheduler, "scheduler"))
    slice_calls = []
    real_set_attention_slice = unet.set_attention_slice

    def observed_slice(size):
        slice_calls.append(size)
        real_set_attention_slice(size)

    monkeypatch.setattr(unet, "set_attention_slice", observed_slice)
    model_root = tmp_path / "tiny-local-model"
    model_root.mkdir()
    options = settings(diffusion, tmp_path, model_path=model_root, attention_slicing=True)
    loaded = diffusion.load_diffusion_components(options, torch.device("cpu"))

    assert slice_calls == ["auto"]
    assert load_calls == ["unet", "vae", "text_encoder", "tokenizer", "scheduler"]
    assert loaded.unet is unet and loaded.vae is vae
    assert loaded.text_embeddings.shape == (1, 8, 8)
    assert loaded.text_embeddings.device.type == "cpu"
    assert loaded.text_embeddings.requires_grad is False
    assert all(getattr(processor, "slice_size", 0) > 0
               for processor in unet.attn_processors.values())
    for component in (unet, vae, text_encoder):
        assert component.training is False
        assert all(parameter.device.type == "cpu" and not parameter.requires_grad
                   for parameter in component.parameters())


def test_partial_schedule_has_matching_inversion_and_denoising_levels(
    diffusion, components, config, models, tmp_path,
):
    source, target = images()
    path, metadata = diffusion.run_diffusion_attack(source, target, config,
        steps=10, output_directory=tmp_path / "run", settings=settings(diffusion, tmp_path),
        surrogates=models, components=components, comparison_epsilon_levels=1)
    assert path.is_file()
    assert metadata["attack"]["inversion_timesteps"] == [0, 100, 200, 300, 400]
    assert metadata["attack"]["denoising_timesteps"] == [400, 300, 200, 100, 0]
    assert metadata["attack"]["steps_completed"] == 5
    assert components.unet.calls == 15  # 反演、无引导对照和有引导生成各五步。
    assert components.vae.differentiable_decode_calls == 5
    assert metadata["after_attack_saved"]["selection_score"] > metadata["before_attack"]["selection_score"] + .005
    assert metadata["decoded_pixel_audit"]["max_absolute_pixel_difference_levels"] > 1
    assert metadata["comparison_budget"]["enforced"] is False
    assert metadata["method_claims"]["strict_manifold_guarantee"] is False
    assert metadata["method_claims"]["faithful_advdiffvlm_reproduction"] is False
    assert max(row["guidance_displacement_linf"] for row in metadata["history"]) <= .1000001


def test_guidance_sign_increases_linear_target_and_bounds_real_ddim_displacement(
    diffusion, torch_runtime, components,
):
    torch = torch_runtime
    scheduler = components.scheduler
    scheduler.set_timesteps(10)
    timestep = scheduler.timesteps[2]
    previous = int(timestep) - 100
    alpha, alpha_previous = scheduler.alphas_cumprod[timestep], scheduler.alphas_cumprod[previous]
    latents = torch.zeros(1, 3, 2, 2)
    noise = torch.zeros_like(latents)
    gradient = torch.arange(1, 13, dtype=torch.float32).reshape_as(latents)
    guided, record = diffusion.guided_noise_prediction(noise, gradient, alpha, alpha_previous,
        guidance_scale=100, latent_step_clip=.03)
    baseline = scheduler.step(noise, timestep, latents, eta=0).prev_sample
    modified = scheduler.step(guided, timestep, latents, eta=0).prev_sample
    displacement = modified - baseline
    assert float((displacement * gradient).sum()) > 0
    assert displacement.abs().max().item() == pytest.approx(.03, abs=1e-6)
    assert record["guidance_displacement_linf"] == pytest.approx(.03)
    assert record["guidance_clip_factor"] < 1
    zero_result, zero_record = diffusion.guided_noise_prediction(noise, gradient * 0,
        alpha, alpha_previous, guidance_scale=100, latent_step_clip=.03)
    torch.testing.assert_close(zero_result, noise)
    assert zero_record["zero_gradient"]
    with pytest.raises(FloatingPointError):
        diffusion.guided_noise_prediction(noise, gradient * float("nan"), alpha, alpha_previous,
            guidance_scale=1, latent_step_clip=.03)


def test_no_guidance_matches_saved_reconstruction_at_original_dimensions(
    diffusion, config, models, components, tmp_path,
):
    import numpy as np
    from PIL import Image
    source, target = images()
    path, metadata = diffusion.run_diffusion_attack(source, target, config,
        steps=6, output_directory=tmp_path / "zero", settings=settings(diffusion, tmp_path, guidance_scale=0),
        surrogates=models, components=components)
    saved = Image.open(path).convert("RGB")
    reconstruction = Image.open(metadata["output_reconstruction"]).convert("RGB")
    assert saved.size == source.size == reconstruction.size
    np.testing.assert_array_equal(np.asarray(saved), np.asarray(reconstruction))
    np.testing.assert_array_equal(np.asarray(saved), np.asarray(source))
    assert metadata["similarity_gain_over_reconstruction"] == pytest.approx(0)
    assert metadata["decoded_pixel_audit"]["budget_levels"] is None
    assert metadata["decoded_pixel_audit"]["within_budget"] is None
    assert components.vae.differentiable_decode_calls == 0


def test_final_scores_come_from_eight_bit_saved_image(
    diffusion, config, models, components, torch_runtime, tmp_path,
):
    from dual_rag.optimization.common.io import load_image, pil_to_tensor
    source, target = images()
    path, metadata = diffusion.run_diffusion_attack(source, target, config,
        steps=6, output_directory=tmp_path / "score", settings=settings(diffusion, tmp_path),
        surrogates=models, components=components)
    torch = torch_runtime
    saved = pil_to_tensor(load_image(path))
    target_tensor = pil_to_tensor(target)
    independent = []
    with torch.no_grad():
        for surrogate in models:
            independent.append(float(torch.nn.functional.cosine_similarity(
                surrogate.encode_image(saved).global_embedding,
                surrogate.encode_image(target_tensor).global_embedding).mean()))
    assert metadata["after_attack_saved"]["global_similarity"] == pytest.approx(independent)
    recorded = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    assert recorded["after_attack_saved"] == metadata["after_attack_saved"]
    json.dumps(recorded, allow_nan=False)


def test_existing_output_is_preserved_without_failure_file(diffusion, config, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "adversarial.png"
    sentinel.write_bytes(b"original-file")
    with pytest.raises(ValueError, match="必须为空"):
        diffusion.run_diffusion_attack(*images(), config, steps=10, output_directory=output)
    assert sentinel.read_bytes() == b"original-file"
    assert not (output / "failure.json").exists()


def test_model_failure_is_auditable(diffusion, config, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("intentional test-only model loading failure")
    monkeypatch.setattr(diffusion, "load_diffusion_components", fail)
    output = tmp_path / "failure"
    with pytest.raises(RuntimeError):
        diffusion.run_diffusion_attack(*images(), config, steps=10,
            settings=settings(diffusion, tmp_path), output_directory=output)
    record = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "intentional test-only" in record["error"]
    assert not (output / "adversarial.png").exists()


def test_common_entry_forwards_diffusion_settings(diffusion, tmp_path, monkeypatch):
    from src.vision import frequency_attack
    calls = []
    expected = (tmp_path / "image.png", {"method": "diffusion"})
    options = {"model_path": str(tmp_path / "model"), "strength": .4}

    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(diffusion, "generate_attack", observed)
    assert frequency_attack.generate_attack("a", "b", "out", root=tmp_path,
        optimization_config="optimization.json", method="diffusion", steps=30,
        epsilon_levels=16, device="cpu", diffusion_options=options) is expected
    assert calls == [(("a", "b", "out"), {"root": tmp_path,
        "optimization_config": "optimization.json", "method": "diffusion", "steps": 30,
        "device": "cpu", "diffusion_options": options, "epsilon_levels": 16})]


@pytest.mark.parametrize("option", ["frequency_axis_ratio", "frequency_momentum", "ssa_samples",
                                   "ssa_rho", "ssa_sigma", "ssa_momentum", "ssa_options", "initial_image"])
def test_diffusion_rejects_old_method_options_before_any_dispatch(diffusion, monkeypatch, option):
    from src.vision import frequency_attack, dual_rag_bridge
    monkeypatch.setattr(diffusion, "generate_attack", lambda *args, **kwargs: pytest.fail("diffusion ran"))
    monkeypatch.setattr(dual_rag_bridge, "generate_attack", lambda *args, **kwargs: pytest.fail("old engine ran"))
    with pytest.raises(ValueError):
        frequency_attack.generate_attack("a", "b", "out", method="diffusion", **{option: 1})


@pytest.mark.parametrize("method", ["legacy_pixel", "fair_pixel", "frequency", "frequency_momentum", "ssa"])
def test_old_methods_reject_diffusion_options(diffusion, monkeypatch, method):
    from src.vision import frequency_attack, dual_rag_bridge
    monkeypatch.setattr(diffusion, "generate_attack", lambda *args, **kwargs: pytest.fail("diffusion ran"))
    monkeypatch.setattr(dual_rag_bridge, "generate_attack", lambda *args, **kwargs: pytest.fail("old engine ran"))
    with pytest.raises(ValueError):
        frequency_attack.generate_attack("a", "b", "out", method=method, diffusion_options={"strength": .4})


@pytest.mark.parametrize("method", ["legacy_pixel", "fair_pixel", "frequency"])
def test_old_methods_keep_original_arguments(diffusion, tmp_path, monkeypatch, method):
    from src.vision import frequency_attack, dual_rag_bridge
    expected = (tmp_path / "original.png", {"method": method})
    calls = []

    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(dual_rag_bridge, "generate_attack", observed)
    assert frequency_attack.generate_attack("a", "b", "out", method=method, steps=9,
                                             epsilon_levels=12, device="cpu") is expected
    assert calls[0][1]["method"] == method
    assert calls[0][1]["steps"] == 9
    assert calls[0][1]["epsilon_levels"] == 12
    assert "diffusion_options" not in calls[0][1]


def early_options(diffusion, tmp_path, **overrides):
    return settings(diffusion, tmp_path, **{"early_stop": True, "min_guidance_steps": 2,
                    "check_every": 1, "patience": 2, "min_delta": .002, **overrides})


def test_plateau_starts_fresh_at_minimum_and_uses_three_disjoint_windows(diffusion, tmp_path):
    configured = settings(diffusion, tmp_path, early_stop=True, min_guidance_steps=100,
                          check_every=25, patience=3)
    stopper = diffusion.DiffusionPlateauStopper(configured)
    steps = [0, 25, 50, 75, 100, 125, 150, 175]
    assert [stopper.observe(step, .5) for step in steps] == [False] * 7 + [True]
    assert [row["low_gain_windows"] for row in stopper.checks] == [0, 0, 0, 0, 0, 1, 2, 3]
    assert stopper.checks[4]["anchor"]
    assert [(row["window_start_step"], row["window_end_step"]) for row in stopper.checks[-3:]] == [
        (100, 125), (125, 150), (150, 175)]


def test_tiny_new_bests_do_not_accumulate_to_reset_patience(diffusion, tmp_path):
    stopper = diffusion.DiffusionPlateauStopper(settings(diffusion, tmp_path,
        early_stop=True, min_guidance_steps=1, check_every=1, patience=3, min_delta=.002))
    decisions = [stopper.observe(step, .5 + .001 * step) for step in range(5)]
    assert decisions == [False, False, False, False, True]
    assert stopper.checks[-1]["best_score"] > stopper.checks[1]["best_score"] + .002
    assert stopper.checks[-1]["low_gain_windows"] == 3


def test_large_window_gain_resets_low_gain_counter(diffusion, tmp_path):
    stopper = diffusion.DiffusionPlateauStopper(settings(diffusion, tmp_path,
        early_stop=True, min_guidance_steps=2, check_every=2, patience=2, min_delta=.002))
    scores = [(0, .5), (2, .5), (4, .501), (6, .510), (8, .511), (10, .512)]
    decisions = [stopper.observe(step, score) for step, score in scores]
    assert decisions == [False] * 5 + [True]
    assert [row["low_gain_windows"] for row in stopper.checks] == [0, 0, 1, 0, 1, 2]


def test_complete_tail_denoises_clone_to_final_state_without_changing_schedule(
    diffusion, components, torch_runtime,
):
    torch = torch_runtime
    components.scheduler.set_timesteps(10)
    original_schedule = components.scheduler.timesteps.clone()
    latent = torch.full((1, 3, 4, 4), .1)
    original_latent = latent.clone()
    remaining = components.scheduler.timesteps[-3:]

    class ConstantNoise:
        dtype = torch.float32

        def __call__(self, sample, timestep, encoder_hidden_states):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(sample=torch.full_like(sample, .2))

    components.unet = ConstantNoise()
    candidate = diffusion.fully_denoised_candidate(components, latent, remaining,
                                                    (1, 3, 13, 17), 1.0)
    expected_latent = latent.clone()
    for timestep in remaining:
        expected_latent = components.scheduler.step(torch.full_like(latent, .2), timestep,
                                                    expected_latent, eta=0).prev_sample
    expected = torch.nn.functional.interpolate((expected_latent / 2 + .5).clamp(0, 1),
        size=(13, 17), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    torch.testing.assert_close(candidate, expected)
    torch.testing.assert_close(latent, original_latent)
    torch.testing.assert_close(components.scheduler.timesteps, original_schedule)
    assert components.noise_forward_calls == 3
    assert abs(float(candidate.mean()) - .55) > .01
    # 末步尾程为空时，当前潜变量已经是最终状态，可以直接解码。
    last = diffusion.fully_denoised_candidate(components, expected_latent, remaining[:0],
                                               (1, 3, 13, 17), 1.0)
    torch.testing.assert_close(last, candidate)
    assert components.noise_forward_calls == 3


def test_zero_guidance_early_stop_keeps_baseline_and_real_saved_scores(
    diffusion, components, config, models, tmp_path, torch_runtime,
):
    import numpy as np
    from PIL import Image
    from dual_rag.optimization.common.io import load_image, pil_to_tensor
    source, target = images()
    options = early_options(diffusion, tmp_path, guidance_scale=0)
    path, metadata = diffusion.run_diffusion_attack(source, target, config, steps=12,
        output_directory=tmp_path / "early-zero", settings=options,
        surrogates=models, components=components)
    assert metadata["attack"]["steps_requested"] == 12
    assert metadata["attack"]["max_guidance_steps"] == 6
    assert metadata["attack"]["steps_completed"] == 4
    assert metadata["attack"]["executed_guidance_steps"] == 4
    assert metadata["attack"]["selected_step"] == 0
    assert metadata["attack"]["stop_reason"] == "early_stop_plateau"
    assert metadata["attack"]["remaining_no_guidance_steps"] == 2
    assert metadata["attack"]["target_gradient_steps"] == 0
    assert len(metadata["history"]) == 4
    checks = metadata["checkpoint_checks"]
    assert [row["actual_step"] for row in checks] == [0, 1, 2, 3, 4]
    assert checks[0]["baseline"] and checks[0]["reused_reconstruction"]
    assert checks[0]["candidate_tail_steps"] == 6
    assert checks[0]["tail_forward_calls"] == 0
    assert [row["candidate_tail_steps"] for row in checks[1:]] == [5, 4, 3, 2]
    np.testing.assert_array_equal(np.asarray(Image.open(path)), np.asarray(source))
    for row in checks:
        saved = pil_to_tensor(load_image(row["image"]))
        expected_score = float(torch_runtime.nn.functional.cosine_similarity(
            models[1].encode_image(saved).global_embedding,
            models[1].encode_image(pil_to_tensor(target)).global_embedding).mean())
        assert row["actual_model2_target_cosine"] == pytest.approx(expected_score)
    assert metadata["attack"]["selected_actual_model2_target_cosine"] == pytest.approx(
        metadata["after_attack_saved"]["global_similarity"][1])
    assert metadata["noise_network_forward_counts"]["total"] == 6 + 6 + 4 + 5 + 4 + 3 + 2
    progress = json.loads((path.parent / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete" and progress["steps_completed"] == 4
    history = json.loads((path.parent / "checkpoint_history.json").read_text(encoding="utf-8"))
    assert history["early_stop_decision"]["trigger_step"] == 4
    assert len(history["gradient_history"]) == 4


def test_early_stop_returns_best_completed_candidate_after_later_degradation(
    diffusion, components, config, models, tmp_path, monkeypatch,
):
    from dual_rag.optimization.attack_core import objective
    real_gradient = objective.sequential_image_gradient
    calls = []

    def change_direction(*args):
        gradient, values = real_gradient(*args)
        calls.append(1)
        return gradient if len(calls) == 1 else -gradient, values

    monkeypatch.setattr(objective, "sequential_image_gradient", change_direction)
    options = early_options(diffusion, tmp_path, min_delta=1e-5)
    path, metadata = diffusion.run_diffusion_attack(*images(), config, steps=20,
        output_directory=tmp_path / "best", settings=options, surrogates=models, components=components)
    assert metadata["attack"]["selected_step"] == 1
    assert metadata["attack"]["steps_completed"] == 4
    assert len(calls) == 4
    assert path.read_bytes() == Path(metadata["checkpoint_checks"][1]["image"]).read_bytes()
    assert metadata["after_attack_saved"]["global_similarity"][1] > metadata["checkpoint_checks"][-1]["actual_model2_target_cosine"]
    assert Path(metadata["output_stopped_candidate"]).read_bytes() == Path(metadata["checkpoint_checks"][-1]["image"]).read_bytes()


def test_audit_runs_full_trajectory_but_freezes_production_selection_and_counts(
    diffusion, components, config, models, tmp_path, monkeypatch,
):
    from dual_rag.optimization.attack_core import objective
    real_gradient = objective.sequential_image_gradient
    calls = []

    def first_worse_then_better(*args):
        gradient, values = real_gradient(*args)
        calls.append(1)
        return -gradient if len(calls) <= 4 else gradient, values

    monkeypatch.setattr(objective, "sequential_image_gradient", first_worse_then_better)
    options = early_options(diffusion, tmp_path, min_delta=2.0, audit_after_stop=True)
    path, metadata = diffusion.run_diffusion_attack(*images(), config, steps=20,
        output_directory=tmp_path / "audit", settings=options, surrogates=models, components=components)
    assert len(calls) == 10
    assert metadata["attack"]["steps_completed"] == 4
    assert metadata["attack"]["executed_guidance_steps"] == 10
    assert metadata["attack"]["selected_step"] == 0
    assert metadata["early_stop_decision"]["trigger_step"] == 4
    assert metadata["early_stop_decision"]["audit_continues_computation"]
    assert len(metadata["history"]) == 10
    assert [row["phase"] for row in metadata["history"]] == ["guided_optimization"] * 4 + ["post_stop_audit"] * 6
    audit = metadata["post_stop_audit"]
    assert audit["audit_selected_step"] > 4
    assert audit["audit_total_guidance_steps"] == 10
    assert audit["audit_additional_guidance_steps"] == 6
    assert audit["audit_gain_over_early_stop"] > .001
    assert path.read_bytes() == Path(metadata["checkpoint_checks"][0]["image"]).read_bytes()
    assert Path(audit["audit_best_image"]).read_bytes() != path.read_bytes()
    assert Path(audit["audit_final_image"]).read_bytes() == Path(metadata["checkpoint_checks"][-1]["image"]).read_bytes()
    assert metadata["noise_network_forward_counts"]["primary_guided_denoising"] == 4
    assert metadata["noise_network_forward_counts"]["post_stop_audit_guided_denoising"] == 6
    assert metadata["timing"]["post_stop_audit_guidance_seconds"] > 0
    progress = json.loads((path.parent / "progress.json").read_text(encoding="utf-8"))
    assert progress["steps_completed"] == 4 and progress["executed_guidance_steps"] == 10
    assert progress["selected_step"] == 0


def test_disabled_early_stop_does_not_evaluate_candidates_or_write_progress(
    diffusion, components, config, models, tmp_path, monkeypatch,
):
    def unexpected(*args):
        pytest.fail("disabled early stop must not execute candidate selection")
    monkeypatch.setattr(diffusion, "fully_denoised_candidate", unexpected)
    monkeypatch.setattr(diffusion, "model2_target_cosine", unexpected)
    path, metadata = diffusion.run_diffusion_attack(*images(), config, steps=10,
        output_directory=tmp_path / "disabled", settings=settings(diffusion, tmp_path),
        surrogates=models, components=components)
    assert components.unet.calls == 15
    assert metadata["attack"]["steps_completed"] == 5
    assert metadata["attack"]["selected_step"] == 5
    assert metadata["checkpoint_checks"] == []
    assert not (path.parent / "progress.json").exists()
    assert not (path.parent / "checkpoints").exists()


def test_schedule_limit_fails_before_model_loading(diffusion, tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    (model_root / "scheduler").mkdir(parents=True)
    (model_root / "scheduler/scheduler_config.json").write_text(
        json.dumps({"num_train_timesteps": 17}), encoding="utf-8")
    monkeypatch.setattr(diffusion, "load_diffusion_components",
                        lambda *args: pytest.fail("model loaded before schedule validation"))
    with pytest.raises(ValueError, match="上限 17"):
        diffusion.generate_attack("a", "b", tmp_path / "out", steps=18, root=tmp_path,
                                  diffusion_options={"model_path": model_root})
