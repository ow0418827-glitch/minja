"""全像素频谱模拟攻击，适配本项目的三代理图像对齐目标。

来源：https://github.com/yuyang-long/SSA，attack.py 与 dct.py。
上游提交：c955cf07c8372bfc4e9f17e647042e027f9f3b1d。
原项目采用 Apache-2.0；许可证副本见 docs/third_party/SSA_LICENSE。
本文件为修改版：保留官方高斯噪声、连续频谱掩码、变换图叶子梯度和
样本均值，加入本项目目标、可选动量、原图尺寸、八位候选选优与预算审计。
官方 attack.py 中动量段默认注释；这里 ssa_momentum=0 可关闭跨步动量。
并未加入频带丢弃，也没有通过频谱变换对原图做链式反向传播。
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

from src.vision.dual_rag_bridge import DEFAULT_DUAL_ROOT


METHOD = "ssa"
UPSTREAM_URL = "https://github.com/yuyang-long/SSA"
UPSTREAM_COMMIT = "c955cf07c8372bfc4e9f17e647042e027f9f3b1d"


def ssa_settings(optimization_config=None, ssa_samples=None, ssa_rho=None,
                 ssa_sigma=None, ssa_momentum=None, root=DEFAULT_DUAL_ROOT):
    """解析独立默认值和可选 ssa 配置，不导入模型或张量运行时。"""
    path = Path(optimization_config) if optimization_config else Path(root) / "configs/optimization.json"
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if optimization_config is not None and not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("ssa", {}), dict):
        raise ValueError("优化配置及 ssa 节点必须为对象")
    configured = raw.get("ssa", {})
    samples = configured.get("samples", 20) if ssa_samples is None else ssa_samples
    rho = configured.get("rho", 0.5) if ssa_rho is None else ssa_rho
    sigma = configured.get("sigma", 16.0) if ssa_sigma is None else ssa_sigma
    momentum = configured.get("momentum", 1.0) if ssa_momentum is None else ssa_momentum
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("ssa_samples 必须为正整数")
    if any(isinstance(value, bool) for value in (rho, sigma, momentum)):
        raise ValueError("频谱模拟数值参数不能为布尔值")
    rho, sigma, momentum = float(rho), float(sigma), float(momentum)
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("ssa_rho 必须为 [0, 1] 内的有限数值")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("ssa_sigma 必须为有限非负数，单位为八位灰度级")
    if not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError("ssa_momentum 必须为 [0, 1] 内的有限数值")
    return {"ssa_samples": samples, "ssa_rho": rho, "ssa_sigma": sigma,
            "ssa_momentum": momentum,
            "ssa_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _dct_1d(image):
    """官方 dct.py 的非归一化二型变换，整理为无需 NumPy 的实现。"""
    import torch
    length = image.shape[-1]
    flattened = image.reshape(-1, length)
    reordered = torch.cat((flattened[:, ::2], flattened[:, 1::2].flip((1,))), dim=1)
    transformed = torch.fft.fft(reordered, dim=-1)
    angle = -torch.arange(length, dtype=image.dtype, device=image.device) * math.pi / (2 * length)
    values = transformed.real * angle.cos() - transformed.imag * angle.sin()
    return (2 * values).reshape(image.shape)


def _idct_1d(coefficients):
    """官方 dct.py 的对应逆变换；保留任意奇偶轴长支持。"""
    import torch
    length = coefficients.shape[-1]
    values = coefficients.reshape(-1, length) / 2
    angle = torch.arange(length, dtype=coefficients.dtype, device=coefficients.device) * math.pi / (2 * length)
    imaginary = torch.cat((torch.zeros_like(values[:, :1]), -values.flip((1,))[:, :-1]), dim=1)
    real = values * angle.cos() - imaginary * angle.sin()
    imaginary = values * angle.sin() + imaginary * angle.cos()
    reordered = torch.fft.ifft(torch.complex(real, imaginary), dim=-1).real
    result = torch.empty_like(reordered)
    result[:, ::2] = reordered[:, :length - length // 2]
    result[:, 1::2] = reordered.flip((1,))[:, :length // 2]
    return result.reshape(coefficients.shape)


def dct_2d(image):
    return _dct_1d(_dct_1d(image).transpose(-1, -2)).transpose(-1, -2)


def idct_2d(coefficients):
    return _idct_1d(_idct_1d(coefficients).transpose(-1, -2)).transpose(-1, -2)


def spectrum_simulation(image, *, rho, sigma):
    """官方模拟：逆变换(变换(图像+高斯噪声)×均匀连续掩码)。"""
    import torch
    # 官方不裁剪模拟图；预算仅约束更新后的完整像素图。
    with torch.no_grad():
        gaussian = torch.randn_like(image) * (sigma / 255.0)
        mask = torch.rand_like(image) * (2 * rho) + (1 - rho)
        return idct_2d(dct_2d(image + gaussian) * mask).detach()


def quantize_image(image):
    rounded = image.mul(255).round().div(255)
    return image + (rounded - image).detach()


def generate_attack(source, target, output_dir, *, root=DEFAULT_DUAL_ROOT,
                    optimization_config=None, method=METHOD, steps=1000,
                    epsilon_levels=16, device=None, ssa_samples=None,
                    ssa_rho=None, ssa_sigma=None, ssa_momentum=None):
    from src.vision import dual_rag_bridge
    if method != METHOD:
        if any(value is not None for value in (ssa_samples, ssa_rho, ssa_sigma, ssa_momentum)):
            raise ValueError("频谱模拟参数仅可用于 ssa 方法")
        from src.vision.frequency_attack import generate_attack as original
        return original(source, target, output_dir, root=root, optimization_config=optimization_config,
                        method=method, steps=steps, epsilon_levels=epsilon_levels, device=device)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须为正整数")
    if isinstance(epsilon_levels, bool) or not isinstance(epsilon_levels, int) or not 0 < epsilon_levels <= 255:
        raise ValueError("epsilon_levels 必须为 [1, 255] 内的整数")
    options = ssa_settings(optimization_config, ssa_samples, ssa_rho, ssa_sigma, ssa_momentum, root)
    root = dual_rag_bridge.activate(root)
    from dual_rag.original_attack import resolve_optimization
    path = Path(optimization_config).resolve() if optimization_config else root / "configs/optimization.json"
    config = resolve_optimization(path)
    config = replace(config, device=device or config.device,
        attack=replace(config.attack, epsilon=epsilon_levels / 255, text_weight=0.0),
        acceptance=replace(config.acceptance, decoded_max_pixel_difference=epsilon_levels),
        runtime=replace(config.runtime, allow_downloads=False))
    try:
        return run_ssa_attack(source, target, config, steps=steps, output_directory=output_dir,
                             **{key: value for key, value in options.items() if key != "ssa_implementation_sha256"})
    finally:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_ssa_attack(source_image, target_image, config, *, steps,
                   output_directory, ssa_samples=None, ssa_rho=None,
                   ssa_sigma=None, ssa_momentum=None, surrogates=None):
    """完整像素变量的官方模拟梯度，与原三代理目标及严格选优兼容。"""
    import torch
    from dual_rag.optimization.attack_core.objective import (
        TargetCache, prepare_targets, sequential_image_gradient, evaluate_objective)
    from dual_rag.optimization.attack_core.surrogates import load_surrogates, resolve_device
    from dual_rag.optimization.common.io import (
        load_image, pil_to_tensor, save_tensor_png, save_json, sha256_file, decoded_image_audit)

    config.validate()
    options = ssa_settings(optimization_config=getattr(config, "config_path", None),
                           ssa_samples=ssa_samples, ssa_rho=ssa_rho,
                           ssa_sigma=ssa_sigma, ssa_momentum=ssa_momentum)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须为正整数")
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("频谱模拟输出目录必须为空，避免覆盖已有对抗图")
    attack = replace(config.attack, reference_steps=steps, text_weight=0.0)
    device = resolve_device(config.device, allow_fallback=config.runtime.allow_device_fallback)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    started = time.perf_counter()
    if surrogates is None:
        surrogates = load_surrogates(config.surrogates, device=str(device),
            precision=config.attack_precision, allow_downloads=False,
            allow_device_fallback=config.runtime.allow_device_fallback,
            require_true_local_tokens=config.runtime.require_true_local_tokens,
            hash_weights=config.runtime.hash_model_weights)
    if len(surrogates) != 3:
        raise ValueError("频谱模拟与原优化相同，要求恰好三个代理模型")
    if any(torch.device(getattr(model, "device", device)) != device for model in surrogates):
        raise ValueError("所有代理模型与攻击张量必须位于同一设备")
    source_pil, target_pil = load_image(source_image), load_image(target_image)
    source, target = pil_to_tensor(source_pil, device), pil_to_tensor(target_pil, device)
    targets = prepare_targets(surrogates, target, None, attack, cache=TargetCache(),
        prepare_cluster_centers=config.shared_engineering.cache_target_cluster_centers)
    uniform_weights = source.new_full((3,), 1 / 3)
    model_weights = uniform_weights.clone()
    pixels = source.detach().clone()
    momentum = torch.zeros_like(pixels)
    minimum, maximum = (source - attack.epsilon).clamp(0, 1), (source + attack.epsilon).clamp(0, 1)
    best_score, best_step = -math.inf, 0
    best_pixels = pixels.clone()
    history, zero_gradient_steps = [], 0
    use_cached_centers = bool(targets.cluster_centers)

    def fixed_record(objective):
        record = objective.detached_record()
        score = objective.global_similarities.mean() + attack.local_weight * objective.local_similarities.mean()
        record["selection_score"] = float(score.detach().cpu())
        scalar_values = [record["loss"], record["selection_score"], record["text_similarity"],
                         *record["global_similarity"], *record["local_similarity"]]
        if not all(math.isfinite(value) for value in scalar_values):
            raise FloatingPointError("频谱模拟目标评分包含非有限值")
        return record

    def consider(record, step):
        nonlocal best_score, best_step, best_pixels
        if record["selection_score"] > best_score:
            best_score, best_step = record["selection_score"], step
            best_pixels = pixels.detach().clone()

    def score_pixels():
        return evaluate_objective(quantize_image(pixels), surrogates, targets,
            uniform_weights, attack, use_cached_target_centers=use_cached_centers)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loop_started = time.perf_counter()
    before = None
    for step in range(1, steps + 1):
        original_objective = score_pixels()
        record = fixed_record(original_objective)
        if before is None:
            before = record
        consider(record, step - 1)
        accumulated = torch.zeros_like(pixels)
        transformed_objective = None
        for _ in range(options["ssa_samples"]):
            simulated = spectrum_simulation(pixels, rho=options["ssa_rho"], sigma=options["ssa_sigma"])
            image_leaf = simulated.requires_grad_(True)
            gradient, objective = sequential_image_gradient(
                image_leaf, surrogates, targets, model_weights, attack)
            # 官方 attack.py 累加模拟图本身梯度，无变换雅可比回传。
            accumulated.add_(gradient)
            values = torch.cat((objective.loss.reshape(1), objective.global_similarities,
                                objective.local_similarities, objective.text_similarity.reshape(1)))
            transformed_objective = values if transformed_objective is None else transformed_objective + values
            del simulated, image_leaf, gradient, objective
        gradient = accumulated.div_(options["ssa_samples"])
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("频谱模拟梯度包含非有限值")
        if not torch.isfinite(transformed_objective).all():
            raise FloatingPointError("频谱模拟的采样目标包含非有限值")
        zero_gradient_steps += int(not bool(gradient.abs().max()))
        normalized = gradient / gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(
            torch.finfo(gradient.dtype).tiny)
        momentum = (options["ssa_momentum"] * momentum + normalized).detach()
        proposed = pixels + attack.spatial_step_size * momentum.sign()
        pixels = torch.maximum(torch.minimum(proposed, maximum), minimum).detach()
        model_weights = torch.softmax(-original_objective.global_similarities.detach(), dim=0)
        if step == 1 or step == steps or step % attack.log_every == 0:
            history.append({"step": step, "scored_state_step": step - 1, **record,
                "simulated_mean_objective_terms": (transformed_objective / options["ssa_samples"]).cpu().tolist(),
                "linf_after_update": float((pixels - source).abs().max().cpu()),
                "weights_for_next_step": model_weights.cpu().tolist()})

    final_record = fixed_record(score_pixels())
    consider(final_record, steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loop_seconds = time.perf_counter() - loop_started
    image_path = save_tensor_png(output / "adversarial.png", quantize_image(best_pixels))
    decoded = pil_to_tensor(load_image(image_path), device)
    audit = decoded_image_audit(source_pil, image_path,
        maximum_levels=config.acceptance.decoded_max_pixel_difference)
    if not audit["within_budget"]:
        raise RuntimeError("频谱模拟落盘图片超过像素扰动预算")
    saved_record = fixed_record(evaluate_objective(decoded, surrogates, targets,
        uniform_weights, attack, use_cached_target_centers=use_cached_centers))
    metadata = {"schema_version": 1, "status": "success", "method": METHOD,
        "seed": config.seed, "target_text": None,
        "source_shape_bchw": list(source.shape), "target_shape_bchw": list(target.shape),
        "source_image_sha256": sha256_file(source_image) if not hasattr(source_image, "convert") else None,
        "target_image_sha256": sha256_file(target_image) if not hasattr(target_image, "convert") else None,
        "implementation_sha256": options["ssa_implementation_sha256"],
        "attack": {**asdict(attack), "steps_requested": steps, "steps_completed": steps,
            "best_step": best_step, "zero_gradient_steps": zero_gradient_steps,
            "stop_reason": "requested_steps_completed",
            "gradient_execution": "sequential_surrogate_gradient_of_detached_simulated_image",
            "initialization": "zero"},
        "before_attack": before, "after_attack_saved": saved_record,
        "last_step_saved_score": final_record,
        "ssa": {"settings": options, "upstream_url": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source_files": ["attack.py", "dct.py"], "upstream_license": "Apache-2.0",
            "parameterization": "full_pixel_space", "parameter_count": source.numel(),
            "transform": "official_unnormalized_dct_ii_and_inverse",
            "noise_distribution": "normal_zero_mean_sigma_divided_by_255",
            "mask_distribution": "uniform_1_minus_rho_to_1_plus_rho",
            "band_dropout": False, "simulated_image_clipped": False,
            "backpropagate_through_transform": False,
            "gradient_averaging": "mean_of_simulated_input_gradients",
            "momentum_is_project_extension": True,
            "upstream_momentum_enabled_in_default_script": False,
            "candidate_selection": "fixed_equal_weight_global_plus_original_local",
            "candidate_scoring": "round_8bit_before_evaluation",
            "projection": "source_plus_minus_epsilon_intersect_0_1",
            "saved_image_claimed_strictly_bandlimited": False},
        "decoded_pixel_audit": audit, "history": history,
        "surrogates": [model.snapshot() if callable(getattr(model, "snapshot", None))
                       else {"class": type(model).__name__} for model in surrogates],
        "primary_evaluator_used_during_attack": False,
        "timing": {"optimization_loop_seconds": loop_seconds,
                   "total_attack_seconds": time.perf_counter() - started},
        "output_image": str(image_path), "output_image_sha256": sha256_file(image_path)}
    save_json(output / "metadata.json", metadata)
    return image_path, metadata
