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
                 ssa_sigma=None, ssa_momentum=None, root=DEFAULT_DUAL_ROOT,
                 ssa_options=None):
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
    controls = {"early_stop": False, "min_steps": 200, "check_every": 50,
        "window": 50, "patience": 2, "min_delta": 0.002, "refine_start": 0.6,
        "step_scale_end": 1.0, "momentum_end": momentum, "rho_end": rho,
        "sigma_end": sigma, "clean_gradient_weight": 0.0, "mse_weight": 0.0,
        "selection_tolerance": 0.0}
    if ssa_options is not None and not isinstance(ssa_options, dict):
        raise ValueError("ssa_options 必须为对象")
    supplied = ssa_options or {}
    unknown = set(supplied) - controls.keys()
    if unknown:
        raise ValueError(f"未知 SSA 控制参数：{sorted(unknown)}")
    controls.update({key: configured[key] for key in controls if key in configured})
    controls.update(supplied)
    if not isinstance(controls["early_stop"], bool):
        raise ValueError("early_stop 必须为布尔值")
    for key in ("min_steps", "check_every", "window", "patience"):
        value = controls[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} 必须为正整数")
    if controls["check_every"] != controls["window"]:
        raise ValueError("check_every 必须等于 window，确保连续窗口互不重叠")
    for key in ("min_delta", "refine_start", "step_scale_end", "momentum_end",
                "rho_end", "sigma_end", "clean_gradient_weight", "mse_weight",
                "selection_tolerance"):
        value = controls[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} 不能为布尔值")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} 必须为有限非负数")
        if key in ("refine_start", "step_scale_end", "momentum_end", "rho_end",
                   "clean_gradient_weight") and value > 1:
            raise ValueError(f"{key} 必须位于 [0, 1]")
        if key == "step_scale_end" and value == 0:
            raise ValueError("step_scale_end 必须大于零")
        if key == "min_delta" and value == 0:
            raise ValueError("min_delta 必须大于零")
        controls[key] = value
    return {"ssa_samples": samples, "ssa_rho": rho, "ssa_sigma": sigma,
            "ssa_momentum": momentum, **{f"ssa_{key}": value for key, value in controls.items()},
            "ssa_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class PlateauStopper:
    """固定候选评分的历史最佳累计收益；不因微小创新高清空耐心。"""

    def __init__(self, settings):
        self.settings = settings
        self.best_scores = []
        self.low_gain_windows = 0
        self.checks = []

    def observe(self, completed_steps, best_score):
        if completed_steps != len(self.best_scores):
            raise ValueError("早停观察必须从零开始逐步记录")
        if not math.isfinite(best_score):
            raise ValueError("早停评分必须有限")
        if self.best_scores and best_score < self.best_scores[-1]:
            raise ValueError("早停必须使用非递减的历史最佳评分")
        self.best_scores.append(best_score)
        options = self.settings
        window = options["ssa_window"]
        if (not options["ssa_early_stop"] or completed_steps < options["ssa_min_steps"]
                or completed_steps < window or completed_steps % options["ssa_check_every"]):
            return False
        baseline = self.best_scores[completed_steps - window]
        improvement = best_score - baseline
        self.low_gain_windows = self.low_gain_windows + 1 if improvement < options["ssa_min_delta"] else 0
        stopped = self.low_gain_windows >= options["ssa_patience"]
        self.checks.append({"completed_steps": completed_steps, "best_score": best_score,
            "baseline_best_score": baseline, "improvement": improvement,
            "low_gain_windows": self.low_gain_windows, "triggered": stopped})
        return stopped


def refinement_schedule(settings, completed_steps, requested_steps):
    """后期按已完成步数线性调度；默认所有端值相同，数值路径不变。"""
    progress = completed_steps / max(1, requested_steps - 1)
    start = settings["ssa_refine_start"]
    fraction = 0.0 if start == 1 else max(0.0, min(1.0, (progress - start) / (1 - start)))
    pairs = {"step_scale": (1.0, settings["ssa_step_scale_end"]),
        "momentum": (settings["ssa_momentum"], settings["ssa_momentum_end"]),
        "rho": (settings["ssa_rho"], settings["ssa_rho_end"]),
        "sigma": (settings["ssa_sigma"], settings["ssa_sigma_end"])}
    return {key: first if first == last else first + (last - first) * fraction
            for key, (first, last) in pairs.items()}


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
                    ssa_rho=None, ssa_sigma=None, ssa_momentum=None, ssa_options=None,
                    initial_image=None):
    from src.vision import dual_rag_bridge
    if method != METHOD:
        if any(value is not None for value in (ssa_samples, ssa_rho, ssa_sigma, ssa_momentum,
                                              ssa_options, initial_image)):
            raise ValueError("频谱模拟参数仅可用于 ssa 方法")
        from src.vision.frequency_attack import generate_attack as original
        return original(source, target, output_dir, root=root, optimization_config=optimization_config,
                        method=method, steps=steps, epsilon_levels=epsilon_levels, device=device)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须为正整数")
    if isinstance(epsilon_levels, bool) or not isinstance(epsilon_levels, int) or not 0 < epsilon_levels <= 255:
        raise ValueError("epsilon_levels 必须为 [1, 255] 内的整数")
    options = ssa_settings(optimization_config, ssa_samples, ssa_rho, ssa_sigma,
                           ssa_momentum, root, ssa_options)
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
            ssa_samples=options["ssa_samples"], ssa_rho=options["ssa_rho"],
            ssa_sigma=options["ssa_sigma"], ssa_momentum=options["ssa_momentum"],
            ssa_options={key[4:]: value for key, value in options.items()
                         if key not in {"ssa_samples", "ssa_rho", "ssa_sigma", "ssa_momentum",
                                        "ssa_implementation_sha256"}}, initial_image=initial_image)
    finally:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_ssa_attack(source_image, target_image, config, *, steps,
                   output_directory, ssa_samples=None, ssa_rho=None,
                   ssa_sigma=None, ssa_momentum=None, surrogates=None,
                   ssa_options=None, initial_image=None):
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
                           ssa_sigma=ssa_sigma, ssa_momentum=ssa_momentum,
                           ssa_options=ssa_options)
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
    minimum, maximum = (source - attack.epsilon).clamp(0, 1), (source + attack.epsilon).clamp(0, 1)
    initialization = None
    if initial_image is not None:
        initial_pil = load_image(initial_image)
        initial_audit = decoded_image_audit(source_pil, initial_pil,
            maximum_levels=config.acceptance.decoded_max_pixel_difference)
        if not initial_audit["within_budget"]:
            raise ValueError("初始候选图相对原甲图超过扰动预算")
        pixels = pil_to_tensor(initial_pil, device)
        if bool((pixels < minimum - 1e-7).any()) or bool((pixels > maximum + 1e-7).any()):
            raise ValueError("初始候选图相对原甲图超过浮点扰动预算")
        initialization = {"image_sha256": sha256_file(initial_image)
            if not hasattr(initial_image, "convert") else None,
            "image_path": str(Path(initial_image).resolve()) if not hasattr(initial_image, "convert") else None,
            "budget_reference": "original_source_image", "decoded_pixel_audit": initial_audit}
    momentum = torch.zeros_like(pixels)
    best_score, best_step = -math.inf, 0
    best_pixels = pixels.clone()
    history, zero_gradient_steps = [], 0
    use_cached_centers = bool(targets.cluster_centers)
    stopper = PlateauStopper(options)
    steps_completed, stop_reason = 0, "requested_steps_completed"
    # 容忍度开启时仅保存未被支配的八位候选到主机，避免显卡驻留全部历史。
    selection_frontier = []

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
        score = record["selection_score"]
        if score > best_score:
            best_score, best_step = score, step
            best_pixels = pixels.detach().clone()
        tolerance = options["ssa_selection_tolerance"]
        if tolerance > 0:
            threshold = best_score - tolerance
            selection_frontier[:] = [item for item in selection_frontier if item["score"] >= threshold]
            if score < threshold:
                return
            rounded = quantize_image(pixels).detach()
            mse = float((rounded - source).square().mean().cpu())
            if any(item["score"] >= score and item["mse"] <= mse for item in selection_frontier):
                return
            selection_frontier[:] = [item for item in selection_frontier
                if not (score >= item["score"] and mse <= item["mse"])]
            selection_frontier.append({"score": score, "mse": mse, "step": step,
                "pixels": rounded.mul(255).round().to(device="cpu", dtype=torch.uint8)})

    def score_pixels():
        return evaluate_objective(quantize_image(pixels), surrogates, targets,
            uniform_weights, attack, use_cached_target_centers=use_cached_centers)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loop_started = time.perf_counter()
    before = None
    if initial_image is not None:
        before = fixed_record(evaluate_objective(source, surrogates, targets,
            uniform_weights, attack, use_cached_target_centers=use_cached_centers))
    initial_record = None
    final_record = None
    for step in range(1, steps + 1):
        original_objective = score_pixels()
        record = fixed_record(original_objective)
        if before is None:
            before = record
        if initial_record is None:
            initial_record = record
        consider(record, step - 1)
        # 此处已完整评分第 step-1 次更新后的图，停止可省去下一整轮频谱采样。
        if stopper.observe(step - 1, best_score):
            stop_reason = "early_stop_plateau"
            final_record = record
            history.append({"step": step - 1, "scored_state_step": step - 1,
                **record, "early_stop": True, "linf_after_update": float((pixels - source).abs().max().cpu()),
                "weights_for_next_step": model_weights.cpu().tolist()})
            break
        scheduled = refinement_schedule(options, step - 1, steps)
        accumulated = torch.zeros_like(pixels)
        transformed_objective = None
        for _ in range(options["ssa_samples"]):
            simulated = spectrum_simulation(pixels, rho=scheduled["rho"], sigma=scheduled["sigma"])
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
        clean_weight = options["ssa_clean_gradient_weight"]
        if clean_weight > 0:
            clean_gradient, _ = sequential_image_gradient(
                pixels.detach().requires_grad_(True), surrogates, targets, model_weights, attack)
            gradient = (1 - clean_weight) * gradient + clean_weight * clean_gradient
            del clean_gradient
        if options["ssa_mse_weight"] > 0:
            # 平均平方差按预算归一化，系数跨分辨率/预算具有明确尺度。
            gradient = gradient - (2 * options["ssa_mse_weight"] / (attack.epsilon ** 2 * pixels.numel())) * (pixels - source)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("频谱模拟梯度包含非有限值")
        if not torch.isfinite(transformed_objective).all():
            raise FloatingPointError("频谱模拟的采样目标包含非有限值")
        zero_gradient_steps += int(not bool(gradient.abs().max()))
        normalized = gradient / gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(
            torch.finfo(gradient.dtype).tiny)
        momentum = (scheduled["momentum"] * momentum + normalized).detach()
        proposed = pixels + (attack.spatial_step_size * scheduled["step_scale"]) * momentum.sign()
        pixels = torch.maximum(torch.minimum(proposed, maximum), minimum).detach()
        steps_completed = step
        model_weights = torch.softmax(-original_objective.global_similarities.detach(), dim=0)
        if step == 1 or step == steps or step % attack.log_every == 0:
            history.append({"step": step, "scored_state_step": step - 1, **record,
                "simulated_mean_objective_terms": (transformed_objective / options["ssa_samples"]).cpu().tolist(),
                "linf_after_update": float((pixels - source).abs().max().cpu()),
                "scheduled_parameters": scheduled,
                "weights_for_next_step": model_weights.cpu().tolist()})

    if final_record is None:
        final_record = fixed_record(score_pixels())
        consider(final_record, steps_completed)
        # 最后一步也记录观察值，但已到上限时停止原因仍为完成请求步数。
        stopper.observe(steps_completed, best_score)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loop_seconds = time.perf_counter() - loop_started
    selected_step = best_step
    if selection_frontier:
        selected = min(selection_frontier, key=lambda item: (item["mse"], -item["score"], item["step"]))
        best_pixels = selected["pixels"].to(device=device, dtype=source.dtype).div(255)
        selected_step = selected["step"]
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
        "attack": {**asdict(attack), "steps_requested": steps, "steps_completed": steps_completed,
            "best_step": best_step, "selected_step": selected_step,
            "best_selection_score": best_score, "zero_gradient_steps": zero_gradient_steps,
            "stop_reason": stop_reason,
            "gradient_execution": "sequential_surrogate_gradient_of_detached_simulated_image",
            "initialization": "provided_image" if initialization else "zero"},
        "before_attack": before, "after_attack_saved": saved_record,
        "initial_candidate": initial_record, "initialization_provenance": initialization,
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
            "candidate_selection": "minimum_decoded_mse_within_fixed_score_tolerance"
                if options["ssa_selection_tolerance"] > 0 else "fixed_equal_weight_global_plus_original_local",
            "early_stop_checks": stopper.checks,
            "early_stop_score": "historical_maximum_of_fixed_equal_weight_8bit_candidates",
            "mse_penalty": "mean_square_difference_normalized_by_original_source_budget",
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
