"""本地扩散先验约束的图像对齐接口，保留旧攻击接口的独立实现。

参考 AdvDiffVLM 和 Hugging Face 的 clip_guided_stable_diffusion 示例。
这里使用 SD 1.5 的潜变量扩散及项目现有三代理图像目标；并非
AdvDiffVLM 官方 AEGE/GCMG 的完整复现，也不保证严格位于自然图像流形。
噪声网络不参与反向传播，解码器保持可微，逐代理累计图像梯度。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

from src.paths import PROJECT
from src.vision.dual_rag_bridge import DEFAULT_DUAL_ROOT


METHOD = "diffusion"
REFERENCE_URL = "https://github.com/gq-max/AdvDiffVLM"
GUIDANCE_REFERENCE = (
    "https://github.com/huggingface/diffusers/blob/main/examples/community/"
    "clip_guided_stable_diffusion.py"
)


def diffusion_settings(optimization_config=None, *, root=DEFAULT_DUAL_ROOT,
                       diffusion_options=None):
    """先验证轻量配置；此入口不导入扩散模型或张量运行时。"""
    path = Path(optimization_config) if optimization_config else Path(root) / "configs/optimization.json"
    if optimization_config is not None and not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(raw, dict) or not isinstance(raw.get("diffusion", {}), dict):
        raise ValueError("优化配置与 diffusion 节点必须为对象")
    settings = {
        "model_path": str(PROJECT / "tmp/models/stable-diffusion-v1-5"),
        "variant": "fp16", "image_size": 256, "strength": 0.35,
        "guidance_scale": 5.0, "latent_step_clip": 0.1, "source_weight": 1.0,
        "prompt": "a product photograph on a white background", "prompt_guidance_scale": 1.0,
        "seed": 42, "initialization": "ddim_inversion", "normalize_gradient": True,
        "attention_slicing": True,
        "early_stop": False, "audit_after_stop": False, "min_guidance_steps": 100,
        "check_every": 25, "patience": 3, "min_delta": 0.002,
    }
    if diffusion_options is not None and not isinstance(diffusion_options, dict):
        raise ValueError("diffusion_options 必须为对象")
    configured = {**raw.get("diffusion", {}), **(diffusion_options or {})}
    unknown = set(configured) - set(settings)
    if unknown:
        raise ValueError(f"未知扩散配置：{sorted(unknown)}")
    settings.update(configured)
    for name in ("image_size", "seed", "min_guidance_steps", "check_every", "patience"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须为整数")
    if settings["image_size"] < 64 or settings["image_size"] % 8:
        raise ValueError("image_size 必须至少为 64，且为 8 的倍数")
    if not 0 <= settings["seed"] < 2**63:
        raise ValueError("seed 必须位于 [0, 2**63)")
    for name in ("min_guidance_steps", "check_every", "patience"):
        if settings[name] < 1:
            raise ValueError(f"{name} 必须为正整数")
    for name in ("strength", "guidance_scale", "latent_step_clip", "source_weight", "prompt_guidance_scale", "min_delta"):
        value = settings[name]
        if isinstance(value, bool):
            raise ValueError(f"{name} 不能为布尔值")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} 必须为有限非负数")
        settings[name] = value
    if not 0 < settings["strength"] <= 1:
        raise ValueError("strength 必须位于 (0, 1]")
    if settings["latent_step_clip"] <= 0 or settings["prompt_guidance_scale"] < 1:
        raise ValueError("latent_step_clip 必须为正数，prompt_guidance_scale 必须至少为 1")
    if settings["min_delta"] <= 0:
        raise ValueError("min_delta 必须为有限正数")
    for name in ("normalize_gradient", "attention_slicing", "early_stop", "audit_after_stop"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"{name} 必须为布尔值")
    if settings["audit_after_stop"] and not settings["early_stop"]:
        raise ValueError("audit_after_stop 仅可在 early_stop 开启时使用")
    if settings["initialization"] not in {"ddim_inversion", "add_noise"}:
        raise ValueError("initialization 必须为 ddim_inversion 或 add_noise")
    if not isinstance(settings["prompt"], str):
        raise ValueError("prompt 必须为字符串")
    if not isinstance(settings["model_path"], (str, Path)) or not str(settings["model_path"]).strip():
        raise ValueError("model_path 必须为非空本地路径")
    if settings["variant"] not in {None, "fp16"}:
        raise ValueError("variant 必须为 fp16 或 None")
    settings["model_path"] = str(Path(settings["model_path"]).resolve())
    settings["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return settings


def validate_diffusion_schedule(steps, settings, *, components=None):
    """在昂贵加载前读取本地日程上限；注入组件时以其实际配置为准。"""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须为正整数")
    maximum = 1000  # 本接口固定使用 SD1.5，标准训练日程为 1000 步。
    if components is not None:
        maximum = int(components.scheduler.config.num_train_timesteps)
    else:
        path = Path(settings["model_path"]) / "scheduler/scheduler_config.json"
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            maximum = int(raw.get("num_train_timesteps", maximum))
    if steps > maximum:
        raise ValueError(f"完整扩散日程 steps={steps} 超过模型训练日程上限 {maximum}")
    return maximum


class DiffusionPlateauStopper:
    """每次检查划定一个新窗口；窗口间不累计微小增量来重置耐心。"""
    def __init__(self, settings):
        self.settings = settings
        self.checks = []
        self.low_gain_windows = 0
        self.anchor_step = None

    def observe(self, completed_steps, best_score, *, is_final=False):
        if not math.isfinite(best_score):
            raise FloatingPointError("早停候选评分必须为有限数值")
        if not self.checks:
            if completed_steps != 0:
                raise ValueError("早停检查必须从零步重建基线开始")
            record = {"completed_steps": 0, "window_start_step": 0, "window_end_step": 0,
                "best_score": float(best_score), "window_best_gain": None,
                "low_gain_windows": 0, "baseline": True, "stop_eligible": False,
                "is_final_checkpoint": is_final, "should_stop": False, "anchor": False}
        else:
            previous = self.checks[-1]
            if completed_steps <= previous["completed_steps"]:
                raise ValueError("早停检查步数必须严格递增")
            if best_score < previous["best_score"]:
                raise ValueError("早停必须使用历史最佳分数")
            gain = float(best_score) - previous["best_score"]
            anchor = False
            if completed_steps < self.settings["min_guidance_steps"]:
                self.low_gain_windows = 0
            elif self.anchor_step is None:
                self.anchor_step = completed_steps
                self.low_gain_windows = 0
                anchor = True
            else:
                self.low_gain_windows = self.low_gain_windows + 1 if gain < self.settings["min_delta"] else 0
            eligible = self.anchor_step is not None and completed_steps > self.anchor_step
            stop = eligible and self.low_gain_windows >= self.settings["patience"]
            record = {"completed_steps": completed_steps,
                "window_start_step": previous["completed_steps"], "window_end_step": completed_steps,
                "best_score": float(best_score), "window_best_gain": gain,
                "low_gain_windows": self.low_gain_windows, "baseline": False,
                "stop_eligible": eligible, "is_final_checkpoint": is_final, "should_stop": stop,
                "anchor": anchor}
        self.checks.append(record)
        return bool(record["should_stop"])


@dataclass
class DiffusionComponents:
    """支持真实组件和仅用于单元验证的微型组件注入。"""
    unet: Any
    vae: Any
    scheduler: Any
    inverse_scheduler: Any
    text_embeddings: Any
    manifest: dict[str, Any]
    noise_forward_calls: int = 0


def _model_manifest(model_root):
    """记录本地权重与配置内容，避免只凭目录名宣称模型身份。"""
    files = {}
    for subfolder in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        for path in sorted((model_root / subfolder).glob("*")):
            if path.is_file() and path.suffix in {".json", ".txt", ".safetensors", ".bin"}:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                files[str(path.relative_to(model_root)).replace("\\", "/")] = {
                    "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
                }
    return files


def load_diffusion_components(settings, device):
    """仅加载本地必需组件；图像编码解码使用单精度，噪声网络用半精度。"""
    import torch
    from diffusers import AutoencoderKL, DDIMInverseScheduler, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    model_root = Path(settings["model_path"])
    if not model_root.is_dir():
        raise FileNotFoundError(f"本地扩散模型目录不存在：{model_root}")
    network_dtype = torch.float16 if device.type == "cuda" else torch.float32
    common = {"local_files_only": True, "use_safetensors": True}
    if settings["variant"]:
        common["variant"] = settings["variant"]
    unet = UNet2DConditionModel.from_pretrained(
        model_root, subfolder="unet", torch_dtype=network_dtype, **common)
    vae = AutoencoderKL.from_pretrained(
        model_root, subfolder="vae", torch_dtype=torch.float32, **common)
    text_encoder = CLIPTextModel.from_pretrained(
        model_root, subfolder="text_encoder", torch_dtype=network_dtype, **common)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_root, subfolder="tokenizer", local_files_only=True)
    scheduler = DDIMScheduler.from_pretrained(
        model_root, subfolder="scheduler", local_files_only=True,
        clip_sample=False, set_alpha_to_one=True, steps_offset=0, timestep_spacing="leading")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("当前扩散引导仅支持 epsilon 噪声预测模型")
    inverse = DDIMInverseScheduler.from_config(scheduler.config)
    for model in (unet, vae, text_encoder):
        model.eval().requires_grad_(False).to(device)
    if settings["attention_slicing"]:
        unet.set_attention_slice("auto")
    prompts = [settings["prompt"]]
    if settings["prompt_guidance_scale"] > 1:
        prompts.insert(0, "")
    tokenized = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                          truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = text_encoder(tokenized.input_ids.to(device))[0].detach()
    text_encoder.to("cpu")
    del text_encoder, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return DiffusionComponents(unet, vae, scheduler, inverse, embeddings,
        {"local_path": str(model_root), "local_files_only": True,
         "variant": settings["variant"], "files": _model_manifest(model_root)})


def diffusion_working_shape(source_shape, image_size):
    """长边对齐分辨率，短边近似保持比例，满足潜变量编码的八倍尺寸约束。"""
    height, width = source_shape[-2:]
    scale = image_size / max(height, width)
    return tuple(max(64, int(round(size * scale / 8)) * 8) for size in (height, width))


def guidance_sample(latents, noise_prediction, alpha):
    """社区示例的稳定混合；噪声预测在本实现中已停止梯度。"""
    beta_sqrt = (1 - alpha).sqrt()
    predicted_clean = (latents - beta_sqrt * noise_prediction) / alpha.sqrt()
    return beta_sqrt * predicted_clean + (1 - beta_sqrt) * latents


def guided_noise_prediction(noise_prediction, gradient, alpha, alpha_previous, *,
                            guidance_scale, latent_step_clip, normalize_gradient=True):
    """最大化目标的得分引导，并限制其带来的实际单步潜变量位移。

    DDIM 的上一时刻样本中，噪声预测的系数为
    sqrt(1-a_prev)-sqrt(a_prev)*sqrt((1-a)/a)，在去噪方向小于零。
    所以 epsilon_hat = epsilon - scale*sqrt(1-a)*grad(J) 会沿 +grad(J)
    方向移动。这里 J 是相似度减重建惩罚，不是待最小化的损失。
    """
    import torch
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("扩散引导梯度包含非有限数值")
    rms = gradient.square().mean().sqrt()
    if float(rms) <= 1e-12 or guidance_scale == 0:
        return noise_prediction, {"gradient_rms": float(rms), "guidance_displacement_linf": 0.0,
                                  "guidance_clip_factor": 1.0, "zero_gradient": float(rms) <= 1e-12}
    direction = gradient / rms.clamp_min(1e-12) if normalize_gradient else gradient
    correction = -float(guidance_scale) * (1 - alpha).sqrt() * direction
    coefficient = (1 - alpha_previous).sqrt() - alpha_previous.sqrt() * ((1 - alpha) / alpha).sqrt()
    displacement_max = float((coefficient * correction).abs().max())
    clip_factor = min(1.0, float(latent_step_clip) / max(displacement_max, 1e-12))
    correction = correction * clip_factor
    result = noise_prediction + correction
    if not torch.isfinite(result).all():
        raise FloatingPointError("引导后的噪声预测包含非有限数值")
    return result.detach(), {"gradient_rms": float(rms),
        "guidance_displacement_linf": displacement_max * clip_factor,
        "guidance_clip_factor": clip_factor, "zero_gradient": False}


def _predict_noise(components, latents, timestep, prompt_scale):
    import torch
    with torch.no_grad():
        model_input = latents
        if prompt_scale > 1:
            model_input = torch.cat([latents, latents])
        model_input = components.scheduler.scale_model_input(model_input, timestep)
        components.noise_forward_calls += 1
        noise = components.unet(model_input.to(components.unet.dtype), timestep,
                                encoder_hidden_states=components.text_embeddings).sample.float()
        if prompt_scale > 1:
            unconditional, conditional = noise.chunk(2)
            noise = unconditional + prompt_scale * (conditional - unconditional)
        return noise.detach()


def _decode(components, latents):
    return (components.vae.decode(latents / components.vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)


def fully_denoised_candidate(components, latents, remaining_timesteps, source_shape, prompt_scale):
    """从当前潜变量完整走完无引导尾程；不改动主日程或攻击潜变量。"""
    import torch
    import torch.nn.functional as functional
    with torch.no_grad():
        candidate_latents = latents.detach().clone()
        for timestep in remaining_timesteps:
            noise = _predict_noise(components, candidate_latents, timestep, prompt_scale)
            candidate_latents = components.scheduler.step(noise, timestep, candidate_latents,
                                                           eta=0.0).prev_sample.float()
        image = _decode(components, candidate_latents)
        image = functional.interpolate(image, size=source_shape[-2:], mode="bicubic",
                                       align_corners=False, antialias=True).clamp(0, 1)
    return image


def model2_target_cosine(image, surrogates, targets):
    """固定第二代理的目标图余弦，仅用于早停与候选选择。"""
    import torch
    with torch.no_grad():
        feature = surrogates[1].encode_image(image).global_embedding
        value = torch.nn.functional.cosine_similarity(feature, targets.image_features[1].global_embedding).mean()
    return float(value)


def _scored_record(objective, local_weight):
    record = objective.detached_record()
    record["selection_score"] = (
        sum(record["global_similarity"]) / len(record["global_similarity"])
        + local_weight * sum(record["local_similarity"]) / len(record["local_similarity"]))
    return record


def generate_attack(source, target, output_dir, *, root=DEFAULT_DUAL_ROOT,
                    optimization_config=None, method=METHOD, steps=30, device=None,
                    diffusion_options=None, epsilon_levels=None):
    """独立新接口：返回落盘图路径和可审计元数据，不投影到像素预算。"""
    from src.vision import dual_rag_bridge
    if method != METHOD:
        raise ValueError("扩散接口仅支持 diffusion 方法；旧方法继续使用原入口")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须为正整数")
    if epsilon_levels is not None and (
            isinstance(epsilon_levels, bool) or not isinstance(epsilon_levels, int) or not 0 < epsilon_levels <= 255):
        raise ValueError("epsilon_levels 仅作为比较预算，必须为 [1, 255] 内整数")
    settings = diffusion_settings(optimization_config, root=root, diffusion_options=diffusion_options)
    validate_diffusion_schedule(steps, settings)
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("扩散输出目录必须为空，避免覆盖已有结果")
    root = dual_rag_bridge.activate(root)
    from dual_rag.original_attack import resolve_optimization
    path = Path(optimization_config).resolve() if optimization_config else root / "configs/optimization.json"
    config = resolve_optimization(path)
    config = replace(config, device=device or config.device, seed=settings["seed"],
        attack=replace(config.attack, text_weight=0.0),
        runtime=replace(config.runtime, allow_downloads=False))
    try:
        return run_diffusion_attack(source, target, config, steps=steps, output_directory=output,
                                    settings=settings, comparison_epsilon_levels=epsilon_levels)
    finally:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_diffusion_attack(source_image, target_image, config, *, steps, output_directory,
                         settings=None, comparison_epsilon_levels=None, surrogates=None,
                         components=None):
    """可复用模型组件；失败时保存诊断，且不覆盖已有结果目录。"""
    output = Path(output_directory).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("扩散输出目录必须为空，避免覆盖已有结果")
    try:
        return _run_diffusion_attack(source_image, target_image, config, steps=steps,
            output_directory=output, settings=settings, comparison_epsilon_levels=comparison_epsilon_levels,
            surrogates=surrogates, components=components)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        diagnostic = {"status": "failed", "method": METHOD, "error_type": type(error).__name__,
                      "error": str(error), "traceback": traceback.format_exc(), "steps": steps,
                      "settings": settings}
        (output / "failure.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2,
            default=str) + "\n", encoding="utf-8")
        raise


def _run_diffusion_attack(source_image, target_image, config, *, steps, output_directory,
                          settings=None, comparison_epsilon_levels=None, surrogates=None,
                          components=None):
    """运行反演、同初值重建与得分引导；可注入微型组件验证计算语义。"""
    import torch
    import torch.nn.functional as functional
    from dual_rag.optimization.attack_core.objective import (
        TargetCache, prepare_targets, sequential_image_gradient, evaluate_objective)
    from dual_rag.optimization.attack_core.surrogates import load_surrogates, resolve_device
    from dual_rag.optimization.common.io import (
        load_image, pil_to_tensor, save_tensor_png, save_json, sha256_file, decoded_image_audit)

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须为正整数")
    config.validate()
    settings = settings or diffusion_settings()
    validate_diffusion_schedule(steps, settings, components=components)
    output = Path(output_directory).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("扩散输出目录必须为空，避免覆盖已有结果")
    device = resolve_device(config.device, allow_fallback=config.runtime.allow_device_fallback)
    torch.manual_seed(settings["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings["seed"])
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if components is None:
        components = load_diffusion_components(settings, device)
    validate_diffusion_schedule(steps, settings, components=components)
    loading_seconds = time.perf_counter() - started
    surrogate_started = time.perf_counter()
    if surrogates is None:
        surrogates = load_surrogates(config.surrogates, device=str(device), precision="fp32",
            allow_downloads=False, allow_device_fallback=config.runtime.allow_device_fallback,
            require_true_local_tokens=config.runtime.require_true_local_tokens,
            hash_weights=config.runtime.hash_model_weights)
    if len(surrogates) != 3:
        raise ValueError("兼容项目目标需要恰好三个代理模型")
    surrogate_loading_seconds = time.perf_counter() - surrogate_started
    attack = replace(config.attack, text_weight=0.0)
    source_pil, target_pil = load_image(source_image), load_image(target_image)
    source, target = pil_to_tensor(source_pil, device), pil_to_tensor(target_pil, device)
    working_shape = diffusion_working_shape(source.shape, settings["image_size"])
    working_source = functional.interpolate(source, size=working_shape, mode="bicubic",
        align_corners=False, antialias=True).clamp(0, 1)
    targets = prepare_targets(surrogates, target, None, attack, cache=TargetCache())
    weights = source.new_full((len(surrogates),), 1 / len(surrogates))
    uniform_weights = weights.clone()
    before = _scored_record(evaluate_objective(source, surrogates, targets, uniform_weights, attack),
                            attack.local_weight)
    forward_count_at_start = components.noise_forward_calls
    components.scheduler.set_timesteps(steps, device=device)
    components.inverse_scheduler.set_timesteps(steps, device=device)
    active_steps = max(1, min(steps, int(round(steps * settings["strength"]))))
    timesteps = components.scheduler.timesteps[-active_steps:]
    inverse_steps = components.inverse_scheduler.timesteps[:active_steps]
    # 编码器后验均值不会额外引入随机采样；反演本身仍是数值近似。
    with torch.no_grad():
        original_latent = components.vae.encode(working_source * 2 - 1).latent_dist.mode()
        original_latent = original_latent.float() * components.vae.config.scaling_factor
        latents = original_latent.clone()
        inversion_started = time.perf_counter()
        if settings["initialization"] == "ddim_inversion":
            for timestep in inverse_steps:
                noise = _predict_noise(components, latents, timestep, settings["prompt_guidance_scale"])
                latents = components.inverse_scheduler.step(noise, timestep, latents).prev_sample.float()
        else:
            generator = torch.Generator(device=device).manual_seed(settings["seed"])
            noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            latents = components.scheduler.add_noise(latents, noise, timesteps[:1])
        initialized_latent = latents.detach().clone()
    inversion_seconds = time.perf_counter() - inversion_started
    # 无对抗引导重建与攻击严格共用编码、初值、文本条件和时间步。
    reconstruction_started = time.perf_counter()
    with torch.no_grad():
        reconstruction_latents = initialized_latent.clone()
        for timestep in timesteps:
            noise = _predict_noise(components, reconstruction_latents, timestep, settings["prompt_guidance_scale"])
            reconstruction_latents = components.scheduler.step(noise, timestep, reconstruction_latents,
                                                                eta=0.0).prev_sample.float()
        reconstruction = _decode(components, reconstruction_latents)
        reconstruction = functional.interpolate(reconstruction, size=source.shape[-2:], mode="bicubic",
                                                align_corners=False, antialias=True).clamp(0, 1)
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    history = []
    early_enabled = settings.get("early_stop", False)
    audit_enabled = settings.get("audit_after_stop", False)
    stopper = DiffusionPlateauStopper(settings) if early_enabled else None
    best_score, selected_step, best_candidate_path = -math.inf, 0, None
    early_stop_decision = None
    checkpoint_tail_seconds = 0.0
    checkpoint_evaluation_seconds = 0.0
    checkpoint_saving_seconds = 0.0
    checkpoint_progress_seconds = 0.0
    primary_guidance_seconds = 0.0
    post_stop_guidance_seconds = 0.0
    primary_checkpoint_seconds = 0.0
    post_stop_checkpoint_seconds = 0.0
    physical_completed_steps = 0
    stop_reason = "requested_steps_completed"
    stopped_candidate_path = None
    reconstruction_path = None

    def synchronized_time():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def capture_candidate(completed_steps, *, use_reconstruction=False):
        nonlocal best_score, selected_step, best_candidate_path
        nonlocal checkpoint_tail_seconds, checkpoint_evaluation_seconds
        nonlocal checkpoint_saving_seconds, checkpoint_progress_seconds
        nonlocal primary_checkpoint_seconds, post_stop_checkpoint_seconds
        check_started = synchronized_time()
        phase = "post_stop_audit" if early_stop_decision is not None else "guided_optimization"
        remaining = timesteps[completed_steps:]
        candidate = reconstruction if use_reconstruction else fully_denoised_candidate(
            components, latents, remaining, source.shape, settings["prompt_guidance_scale"])
        tail_seconds = synchronized_time() - check_started
        saving_started = time.perf_counter()
        path = save_tensor_png(output / "checkpoints" / f"step_{completed_steps:06d}.png", candidate)
        # 保存后重新解码；候选选择和早停只比较真实八位图片的固定代理2余弦。
        decoded_candidate = pil_to_tensor(load_image(path), device)
        image_hash = sha256_file(path)
        saving_seconds = synchronized_time() - saving_started
        evaluation_started = time.perf_counter()
        score = model2_target_cosine(decoded_candidate, surrogates, targets)
        evaluation_seconds = synchronized_time() - evaluation_started
        if score > best_score:
            best_score, selected_step, best_candidate_path = score, completed_steps, path
        should_stop = stopper.observe(completed_steps, best_score,
                                      is_final=completed_steps == active_steps)
        check = stopper.checks[-1]
        check.update({"actual_step": completed_steps, "phase": phase,
            "image": str(path), "image_sha256": image_hash,
            "actual_model2_target_cosine": score, "selected_step": selected_step,
            "candidate_tail_steps": len(remaining),
            "remaining_no_guidance_steps": len(remaining),
            "tail_forward_calls": 0 if use_reconstruction else len(remaining),
            "reused_reconstruction": use_reconstruction,
            "tail_seconds": tail_seconds, "evaluation_seconds": evaluation_seconds,
            "saving_seconds": saving_seconds,
            "check_elapsed_seconds": synchronized_time() - check_started,
            "cumulative_elapsed_seconds": time.perf_counter() - started})
        checkpoint_tail_seconds += tail_seconds
        checkpoint_evaluation_seconds += evaluation_seconds
        checkpoint_saving_seconds += saving_seconds
        if phase == "post_stop_audit":
            post_stop_checkpoint_seconds += check["check_elapsed_seconds"]
        else:
            primary_checkpoint_seconds += check["check_elapsed_seconds"]
        progress_started = time.perf_counter()
        save_json(output / "checkpoint_history.json", {
            "selection_metric": "actual_model2_target_cosine", "checkpoint_checks": stopper.checks,
            "gradient_history": history, "early_stop_decision": early_stop_decision})
        save_json(output / "progress.json", {
            "status": phase, "steps_requested": steps, "max_guidance_steps": active_steps,
            "steps_completed": early_stop_decision["trigger_step"] if early_stop_decision else completed_steps,
            "executed_guidance_steps": completed_steps,
            "remaining_no_guidance_steps": len(remaining),
            "selected_step": early_stop_decision["selected_step"] if early_stop_decision else selected_step,
            "selected_score": early_stop_decision["selected_score"] if early_stop_decision else best_score,
            "selection_metric": "actual_model2_target_cosine",
            "latest_checkpoint": check, "checkpoint_checks": stopper.checks,
            "early_stop_decision": early_stop_decision,
            "audit_after_stop": audit_enabled, "elapsed_seconds": time.perf_counter() - started})
        checkpoint_progress_seconds += time.perf_counter() - progress_started
        print(f"diffusion checkpoint {completed_steps}/{active_steps}, phase {phase}, "
              f"model2 cosine {score:.6f}, best {best_score:.6f}, "
              f"low-gain windows {check['low_gain_windows']}", flush=True)
        return should_stop, check

    if early_enabled:
        output.mkdir(parents=True, exist_ok=True)
        reconstruction_path = save_tensor_png(output / "reconstruction.png", reconstruction)
        capture_candidate(0, use_reconstruction=True)
    attack_started = time.perf_counter()
    alphas = components.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    timestep_delta = components.scheduler.config.num_train_timesteps // steps
    for index, timestep in enumerate(timesteps):
        guided_step_started = synchronized_time() if early_enabled else time.perf_counter()
        phase = "post_stop_audit" if early_stop_decision is not None else "guided_optimization"
        noise = _predict_noise(components, latents, timestep, settings["prompt_guidance_scale"])
        timestep_int = int(timestep)
        previous = timestep_int - timestep_delta
        alpha = alphas[timestep_int]
        alpha_previous = alphas[previous] if previous >= 0 else components.scheduler.final_alpha_cumprod.to(device)
        if settings["guidance_scale"] > 0:
            with torch.enable_grad():
                latent_leaf = latents.detach().requires_grad_(True)
                image_graph = _decode(components, guidance_sample(latent_leaf, noise, alpha))
                image_leaf = image_graph.detach().requires_grad_(True)
                image_gradient, objective = sequential_image_gradient(image_leaf, surrogates, targets, weights, attack)
                source_mse = (image_leaf.detach() - working_source).square().mean()
                image_gradient = image_gradient - (2 * settings["source_weight"] / image_leaf.numel()) * (
                    image_leaf.detach() - working_source)
                latent_gradient = torch.autograd.grad(image_graph, latent_leaf, grad_outputs=image_gradient)[0]
            guided_noise, record = guided_noise_prediction(noise, latent_gradient, alpha, alpha_previous,
                guidance_scale=settings["guidance_scale"], latent_step_clip=settings["latent_step_clip"],
                normalize_gradient=settings["normalize_gradient"])
            record.update(objective.detached_record())
            record["source_mse"] = float(source_mse)
            record["regularized_objective"] = float(objective.loss) - settings["source_weight"] * float(source_mse)
            weights = torch.softmax(-objective.global_similarities.detach(), dim=0)
            del image_graph, image_leaf, image_gradient, latent_leaf, latent_gradient
        else:
            guided_noise = noise
            record = {"guidance_displacement_linf": 0.0, "guidance_clip_factor": 1.0,
                      "zero_gradient": True, "gradient_rms": 0.0}
        with torch.no_grad():
            latents = components.scheduler.step(guided_noise, timestep, latents, eta=0.0).prev_sample.float()
        if not torch.isfinite(latents).all():
            raise FloatingPointError("去噪后的潜变量包含非有限数值")
        physical_completed_steps = index + 1
        history_record = {"step": index + 1, "timestep": timestep_int, **record}
        if early_enabled:
            history_record["phase"] = phase
        history.append(history_record)
        guided_step_seconds = synchronized_time() - guided_step_started if early_enabled else time.perf_counter() - guided_step_started
        if phase == "post_stop_audit":
            post_stop_guidance_seconds += guided_step_seconds
        else:
            primary_guidance_seconds += guided_step_seconds
        # 可在运行过程中审计进度，且不保留任何代理反向计算图。
        if index == 0 or (index + 1) % 5 == 0 or index + 1 == active_steps:
            print(f"diffusion step {index + 1}/{active_steps}, timestep {timestep_int}", flush=True)
        if early_enabled and (index + 1) % 5 == 0 and (index + 1) % settings["check_every"] != 0 and index + 1 != active_steps:
            progress_started = time.perf_counter()
            save_json(output / "progress.json", {
                "status": phase, "steps_requested": steps, "max_guidance_steps": active_steps,
                "steps_completed": early_stop_decision["trigger_step"] if early_stop_decision else index + 1,
                "executed_guidance_steps": index + 1,
                "selected_step": early_stop_decision["selected_step"] if early_stop_decision else selected_step,
                "selected_score": early_stop_decision["selected_score"] if early_stop_decision else best_score,
                "remaining_no_guidance_steps": active_steps - index - 1,
                "selection_metric": "actual_model2_target_cosine", "latest_checkpoint": stopper.checks[-1],
                "checkpoint_history_path": str(output / "checkpoint_history.json"),
                "early_stop_decision": early_stop_decision, "audit_after_stop": audit_enabled,
                "elapsed_seconds": time.perf_counter() - started})
            checkpoint_progress_seconds += time.perf_counter() - progress_started
        if early_enabled and ((index + 1) % settings["check_every"] == 0 or index + 1 == active_steps):
            should_stop, check = capture_candidate(index + 1)
            if should_stop and index + 1 < active_steps and early_stop_decision is None:
                early_stop_decision = {"trigger_step": index + 1, "selected_step": selected_step,
                    "selected_score": best_score, "selected_image": str(best_candidate_path),
                    "trigger_candidate_image": check["image"],
                    "trigger_candidate_score": check["actual_model2_target_cosine"],
                    "window_best_gain": check["window_best_gain"],
                    "low_gain_windows": check["low_gain_windows"],
                    "criterion": "actual_model2_target_cosine", "audit_continues_computation": audit_enabled}
                stop_reason = "early_stop_plateau"
                # 即使审计继续，生产图片、选中步、触发步与当时分数均被冻结。
                stopped_candidate_path = Path(check["image"])
                save_json(output / "early_stop_decision.json", early_stop_decision)
                save_json(output / "checkpoint_history.json", {
                    "selection_metric": "actual_model2_target_cosine", "checkpoint_checks": stopper.checks,
                    "gradient_history": history, "early_stop_decision": early_stop_decision})
                save_json(output / "progress.json", {
                    "status": "post_stop_audit" if audit_enabled else "early_stop_plateau",
                    "steps_requested": steps, "max_guidance_steps": active_steps,
                    "steps_completed": index + 1, "executed_guidance_steps": index + 1,
                    "selected_step": selected_step, "selected_score": best_score,
                    "remaining_no_guidance_steps": active_steps - index - 1,
                    "early_stop_decision": early_stop_decision, "checkpoint_checks": stopper.checks,
                    "audit_after_stop": audit_enabled, "elapsed_seconds": time.perf_counter() - started})
                if not audit_enabled:
                    break
    attack_seconds = time.perf_counter() - attack_started
    output_started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    production_completed_steps = early_stop_decision["trigger_step"] if early_stop_decision else physical_completed_steps
    production_selected_step = early_stop_decision["selected_step"] if early_stop_decision else (
        selected_step if early_enabled else physical_completed_steps)
    production_selected_score = early_stop_decision["selected_score"] if early_stop_decision else (
        best_score if early_enabled else None)
    adversarial_path = output / "adversarial.png"
    audit_best_path, audit_final_path, stopped_output_path = None, None, None
    if early_enabled:
        production_path = Path(early_stop_decision["selected_image"]) if early_stop_decision else best_candidate_path
        shutil.copyfile(production_path, adversarial_path)
        stopped_output_path = output / "stopped_candidate.png"
        shutil.copyfile(stopped_candidate_path or Path(stopper.checks[-1]["image"]), stopped_output_path)
        if audit_enabled:
            audit_best_path, audit_final_path = output / "audit_best.png", output / "audit_final.png"
            shutil.copyfile(best_candidate_path, audit_best_path)
            shutil.copyfile(Path(stopper.checks[-1]["image"]), audit_final_path)
    else:
        # 默认关闭路径仍只保存最终完整去噪图，不检查尾程、不进行候选选优。
        with torch.no_grad():
            final = _decode(components, latents)
            final = functional.interpolate(final, size=source.shape[-2:], mode="bicubic",
                                           align_corners=False, antialias=True).clamp(0, 1)
        save_tensor_png(adversarial_path, final)
    if reconstruction_path is None:
        reconstruction_path = save_tensor_png(output / "reconstruction.png", reconstruction)
    # 只以实际八位落盘后重新解码的图片给出最终可比较相似度。
    saved = pil_to_tensor(load_image(adversarial_path), device)
    saved_reconstruction = pil_to_tensor(load_image(reconstruction_path), device)
    after = _scored_record(evaluate_objective(saved, surrogates, targets, uniform_weights, attack), attack.local_weight)
    reconstruction_scores = _scored_record(evaluate_objective(saved_reconstruction, surrogates, targets,
                                                             uniform_weights, attack), attack.local_weight)
    if production_selected_score is None:
        production_selected_score = after["global_similarity"][1]
    audit = decoded_image_audit(source_pil, adversarial_path, maximum_levels=comparison_epsilon_levels or 255)
    reconstruction_audit = decoded_image_audit(source_pil, reconstruction_path,
                                              maximum_levels=comparison_epsilon_levels or 255)
    if comparison_epsilon_levels is None:
        for item in (audit, reconstruction_audit):
            item.update(budget_levels=None, within_budget=None)
    manifest = {"source": sha256_file(source_image) if isinstance(source_image, (str, Path)) else None,
                "target": sha256_file(target_image) if isinstance(target_image, (str, Path)) else None}
    metadata = {
        "schema_version": 1, "status": "success", "method": METHOD, "seed": settings["seed"],
        "settings": settings, "model": components.manifest, "input_sha256": manifest,
        "surrogates": [model.snapshot() if callable(getattr(model, "snapshot", None)) else
            {"index": index + 1, "kind": type(model).__name__,
             "cache_identity": str(getattr(model, "cache_identity", "unknown"))}
            for index, model in enumerate(surrogates)],
        "source_shape_bchw": list(source.shape), "working_shape_hw": list(working_shape),
        "output_image": str(adversarial_path), "output_image_sha256": sha256_file(adversarial_path),
        "output_reconstruction": str(reconstruction_path), "reconstruction_sha256": sha256_file(reconstruction_path),
        "output_metadata": str(output / "metadata.json"),
        "attack": {"steps_requested": steps, "max_guidance_steps": active_steps,
            "steps_completed": production_completed_steps,
            "executed_guidance_steps": physical_completed_steps,
            "selected_step": production_selected_step,
            "selected_actual_model2_target_cosine": production_selected_score,
            "stop_reason": stop_reason,
            "remaining_no_guidance_steps": active_steps - production_completed_steps,
            "selection_metric": "actual_model2_target_cosine" if early_enabled else "final_denoised_image",
            "early_stop_enabled": early_enabled,
            "steps_meaning": "full_schedule_steps; actual_denoising_steps=round(steps*strength)",
            "initialization": settings["initialization"], "inversion_timesteps": [int(t) for t in inverse_steps]
                if settings["initialization"] == "ddim_inversion" else [],
            "denoising_timesteps": [int(t) for t in timesteps], "local_weight": attack.local_weight,
            "text_weight": 0.0, "gradient_execution": "sequential_surrogate_image_gradient_then_vae_vjp",
            "objective_direction": "maximize_global_plus_local_minus_source_mse",
            "zero_gradient_steps": sum(int(record["zero_gradient"]) for record in history[:production_completed_steps]),
            "target_gradient_steps": production_completed_steps if settings["guidance_scale"] > 0 else 0},
        "selection_metric": "actual_model2_target_cosine" if early_enabled else "final_denoised_image",
        "checkpoint_checks": stopper.checks if early_enabled else [],
        "early_stop_decision": early_stop_decision,
        "output_stopped_candidate": str(stopped_output_path) if stopped_output_path else None,
        "stopped_candidate_sha256": sha256_file(stopped_output_path) if stopped_output_path else None,
        "post_stop_audit": {"enabled": audit_enabled, "triggered": early_stop_decision is not None,
            "production_result_frozen": early_stop_decision is not None,
            "audit_selected_step": selected_step if audit_enabled else None,
            "audit_total_guidance_steps": physical_completed_steps if audit_enabled else None,
            "audit_additional_guidance_steps": physical_completed_steps - production_completed_steps if audit_enabled else 0,
            "audit_best_actual_model2_target_cosine": best_score if audit_enabled else None,
            "audit_final_actual_model2_target_cosine": stopper.checks[-1]["actual_model2_target_cosine"] if audit_enabled else None,
            "audit_gain_over_early_stop": best_score - production_selected_score if audit_enabled else None,
            "audit_best_image": str(audit_best_path) if audit_best_path else None,
            "audit_best_sha256": sha256_file(audit_best_path) if audit_best_path else None,
            "audit_final_image": str(audit_final_path) if audit_final_path else None,
            "audit_final_sha256": sha256_file(audit_final_path) if audit_final_path else None,
            "meaning": "continues_the_same_guided_trajectory_after_production_stop; extra_compute_is_not_early_stopped"},
        "before_attack": before, "after_attack_saved": after,
        "reconstruction_saved": reconstruction_scores,
        "similarity_gain_over_source": after["selection_score"] - before["selection_score"],
        "similarity_gain_over_reconstruction": after["selection_score"] - reconstruction_scores["selection_score"],
        "decoded_pixel_audit": audit, "reconstruction_audit": reconstruction_audit,
        "comparison_budget": {"epsilon_levels": comparison_epsilon_levels, "enforced": False,
            "meaning": "diagnostic_reference_only; no_pixel_projection"},
        "timing": {"diffusion_loading_seconds": loading_seconds,
            "surrogate_loading_seconds": surrogate_loading_seconds, "inversion_seconds": inversion_seconds,
            "reconstruction_seconds": reconstruction_seconds, "optimization_loop_seconds": attack_seconds,
            "primary_guidance_loop_seconds": primary_guidance_seconds,
            "post_stop_audit_guidance_seconds": post_stop_guidance_seconds,
            "primary_checkpoint_seconds": primary_checkpoint_seconds,
            "post_stop_audit_checkpoint_seconds": post_stop_checkpoint_seconds,
            "checkpoint_tail_seconds": checkpoint_tail_seconds,
            "checkpoint_evaluation_seconds": checkpoint_evaluation_seconds,
            "checkpoint_saving_seconds": checkpoint_saving_seconds,
            "checkpoint_progress_write_seconds": checkpoint_progress_seconds,
            "final_output_and_recompute_seconds": time.perf_counter() - output_started,
            "total_seconds_before_metadata": time.perf_counter() - started},
        "noise_network_forward_counts": {
            "inversion": active_steps if settings["initialization"] == "ddim_inversion" else 0,
            "reconstruction": active_steps, "primary_guided_denoising": production_completed_steps,
            "post_stop_audit_guided_denoising": physical_completed_steps - production_completed_steps,
            "checkpoint_tail_denoising": sum(check["tail_forward_calls"] for check in stopper.checks) if early_enabled else 0,
            "total": components.noise_forward_calls - forward_count_at_start,
            "count_unit": "actual_unet_forward_calls; classifier_free_guidance_uses_one_batched_call"},
        "memory": {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                   if device.type == "cuda" else None},
        "references": [REFERENCE_URL, GUIDANCE_REFERENCE], "history": history,
        "method_claims": {"faithful_advdiffvlm_reproduction": False, "strict_manifold_guarantee": False,
            "lossless_inversion": False, "pixel_budget_enforced": False,
            "unet_gradient_enabled": False, "vae_gradient_enabled": True,
            "extensions": ["SD1.5 text-conditioned latent diffusion instead of official class-conditional LDM",
                "three project CLIP image-target surrogates instead of four official CLIP image-target models",
                "project global+local image objective instead of official objective",
                "sequential three-surrogate gradients and softmax model weighting",
                "stop-gradient noise predictor; gradient RMS normalization; bounded latent displacement",
                "source MSE regularization; approximate DDIM inversion; no GCMG mask"]},
        "primary_evaluator_used_during_attack": True,
        "evaluator": "surrogate_2_shared_with_optimization; not_black_box_transfer",
    }
    if early_enabled:
        save_json(output / "checkpoint_history.json", {
            "selection_metric": "actual_model2_target_cosine", "checkpoint_checks": stopper.checks,
            "gradient_history": history, "early_stop_decision": early_stop_decision,
            "post_stop_audit": metadata["post_stop_audit"]})
        save_json(output / "progress.json", {
            "status": "complete", "steps_requested": steps, "max_guidance_steps": active_steps,
            "steps_completed": production_completed_steps,
            "executed_guidance_steps": physical_completed_steps,
            "selected_step": production_selected_step, "selected_score": production_selected_score,
            "stop_reason": stop_reason,
            "remaining_no_guidance_steps": active_steps - production_completed_steps,
            "selection_metric": "actual_model2_target_cosine", "output_image": str(adversarial_path),
            "checkpoint_checks": stopper.checks, "early_stop_decision": early_stop_decision,
            "post_stop_audit": metadata["post_stop_audit"], "elapsed_seconds": time.perf_counter() - started})
    save_json(output / "metadata.json", metadata)
    return adversarial_path, metadata
