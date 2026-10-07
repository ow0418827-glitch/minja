"""可选的频域动量优化；旧方法透明委托给原桥接模块。"""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

from src.vision.dual_rag_bridge import DEFAULT_DUAL_ROOT


METHOD = "frequency_momentum"


def frequency_settings(optimization_config=None, frequency_axis_ratio=None,
                       frequency_momentum=None, root=DEFAULT_DUAL_ROOT):
    """只读取配置与实现摘要，不加载模型；用于冻结和恢复校验。"""
    path = Path(optimization_config) if optimization_config else Path(root) / "configs/optimization.json"
    if frequency_axis_ratio is None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        frequency_axis_ratio = raw.get("frequency", {}).get("initial_axis_ratio", 0.25)
    if isinstance(frequency_axis_ratio, bool) or isinstance(frequency_momentum, bool):
        raise ValueError("频率比例和动量必须为数值，不能为布尔值")
    ratio = float(frequency_axis_ratio)
    momentum = 1.0 if frequency_momentum is None else float(frequency_momentum)
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("frequency_axis_ratio 必须是 (0, 1] 内的有限数值")
    if not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError("frequency_momentum 必须是 [0, 1] 内的有限数值")
    return {"frequency_axis_ratio": ratio, "frequency_momentum": momentum,
            "frequency_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def quantize_image(image):
    """八位取整前向、恒等反向；直通估计是近似梯度。"""
    rounded = image.mul(255).round().div(255)
    return image + (rounded - image).detach()


def generate_attack(source, target, output_dir, *, root=DEFAULT_DUAL_ROOT,
                    optimization_config=None, method="legacy_pixel", steps=1000,
                    epsilon_levels=16, device=None, frequency_axis_ratio=None,
                    frequency_momentum=None, ssa_samples=None, ssa_rho=None,
                    ssa_sigma=None, ssa_momentum=None):
    from src.vision import dual_rag_bridge
    ssa_options = dict(ssa_samples=ssa_samples, ssa_rho=ssa_rho,
                       ssa_sigma=ssa_sigma, ssa_momentum=ssa_momentum)
    if method == "ssa":
        if frequency_axis_ratio is not None or frequency_momentum is not None:
            raise ValueError("低频参数仅可用于 frequency_momentum 方法")
        from src.vision.ssa_attack import generate_attack as generate_ssa_attack
        return generate_ssa_attack(
            source, target, output_dir, root=root, optimization_config=optimization_config,
            method=method, steps=steps, epsilon_levels=epsilon_levels, device=device,
            **ssa_options)
    if any(value is not None for value in ssa_options.values()):
        raise ValueError("频谱模拟参数仅可用于 ssa 方法")
    if method != METHOD:
        if frequency_axis_ratio is not None or frequency_momentum is not None:
            raise ValueError("频域增强参数仅可用于 frequency_momentum 方法")
        return dual_rag_bridge.generate_attack(
            source, target, output_dir, root=root, optimization_config=optimization_config,
            method=method, steps=steps, epsilon_levels=epsilon_levels, device=device)

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须为正整数")
    if isinstance(epsilon_levels, bool) or not isinstance(epsilon_levels, int) or not 0 < epsilon_levels <= 255:
        raise ValueError("epsilon_levels 必须为 [1, 255] 内的整数")
    options = frequency_settings(optimization_config, frequency_axis_ratio, frequency_momentum, root)
    root = dual_rag_bridge.activate(root)
    from dual_rag.original_attack import resolve_optimization
    path = Path(optimization_config).resolve() if optimization_config else root / "configs/optimization.json"
    config = resolve_optimization(path)
    config = replace(config, device=device or config.device,
        attack=replace(config.attack, epsilon=epsilon_levels / 255, text_weight=0.0),
        acceptance=replace(config.acceptance, decoded_max_pixel_difference=epsilon_levels),
        runtime=replace(config.runtime, allow_downloads=False))
    # 返回的仅是路径和元数据；模型和计算图在运行器返回后即释放。
    try:
        return run_frequency_attack(source, target, config, steps=steps,
            output_directory=output_dir, frequency_axis_ratio=options["frequency_axis_ratio"],
            frequency_momentum=options["frequency_momentum"])
    finally:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_frequency_attack(source_image, target_image, config, *, steps,
                         output_directory, frequency_axis_ratio, frequency_momentum,
                         surrogates=None):
    """沿用三个代理和原目标，仅改变低频更新、量化前向和候选选择。"""
    import torch
    from dual_rag.optimization.attack_core.objective import (
        TargetCache, prepare_targets, sequential_image_gradient, evaluate_objective)
    from dual_rag.optimization.attack_core.surrogates import load_surrogates, resolve_device
    from dual_rag.optimization.common.io import (
        load_image, pil_to_tensor, save_tensor_png, save_json, sha256_file, decoded_image_audit)
    from dual_rag.optimization.frequency_attack.dct import (
        DCTBasisCache, axis_frequency_count, synthesize, update_coefficients, perturbation_diagnostics)

    config.validate()
    options = frequency_settings(frequency_axis_ratio=frequency_axis_ratio,
                                 frequency_momentum=frequency_momentum)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须为正整数")
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("频域增强输出目录必须为空，避免覆盖已有对抗图")
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
        raise ValueError("频域增强与原优化相同，要求恰好三个代理模型")
    if any(torch.device(getattr(model, "device", device)) != device for model in surrogates):
        raise ValueError("所有代理模型与攻击张量必须位于同一设备")
    source_pil, target_pil = load_image(source_image), load_image(target_image)
    source, target = pil_to_tensor(source_pil, device), pil_to_tensor(target_pil, device)
    cache = TargetCache()
    targets = prepare_targets(surrogates, target, None, attack, cache=cache,
        prepare_cluster_centers=config.shared_engineering.cache_target_cluster_centers)
    uniform_weights = source.new_full((3,), 1 / 3)
    model_weights = uniform_weights.clone()
    height, width = source.shape[-2:]
    ratio = options["frequency_axis_ratio"]
    kh, kw = axis_frequency_count(height, ratio), axis_frequency_count(width, ratio)
    bases = DCTBasisCache()
    bh = bases.get(height, kh, device=device, dtype=source.dtype)
    bw = bases.get(width, kw, device=device, dtype=source.dtype)
    coefficients = source.new_zeros(1, 3, kh, kw)
    momentum = torch.zeros_like(coefficients)
    best_score, best_step = -math.inf, 0
    best_raw, best_float = torch.zeros_like(source), source.detach().clone()
    history, zero_gradient_steps = [], 0
    before = None

    def fixed_record(objective):
        record = objective.detached_record()
        # 动态权重用于梯度；候选比较始终等权，保证跨步评分可比。
        score = objective.global_similarities.mean() + attack.local_weight * objective.local_similarities.mean()
        record["selection_score"] = float(score.detach().cpu())
        if not all(math.isfinite(v) for v in (record["loss"], record["selection_score"])):
            raise FloatingPointError("目标评分包含非有限值")
        return record

    def consider(record, step, raw, floating):
        nonlocal best_score, best_step, best_raw, best_float
        if record["selection_score"] > best_score:
            best_score, best_step = record["selection_score"], step
            best_raw, best_float = raw.detach().clone(), floating.detach().clone()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loop_started = time.perf_counter()
    for step in range(1, steps + 1):
        parameter = coefficients.detach().requires_grad_(True)
        raw = synthesize(parameter, bh, bw)
        floating = (source + raw).clamp(0, 1)
        graph = quantize_image(floating)
        image_leaf = graph.detach().requires_grad_(True)
        image_gradient, objective = sequential_image_gradient(
            image_leaf, surrogates, targets, model_weights, attack)
        record = fixed_record(objective)
        if before is None:
            before = record
        consider(record, step - 1, raw, floating)
        gradient = torch.autograd.grad(graph, parameter, grad_outputs=image_gradient)[0].detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("频率系数梯度包含非有限值")
        if not config.frequency.constant_component:
            gradient[..., 0, 0] = 0
        normalized = gradient / gradient.abs().mean().clamp_min(torch.finfo(gradient.dtype).tiny)
        momentum = (options["frequency_momentum"] * momentum + normalized).detach()
        update = update_coefficients(coefficients, momentum, bh, bw,
            spatial_step_size=attack.spatial_step_size, epsilon=attack.epsilon,
            keep_constant_component=config.frequency.constant_component,
            maximum_reprojection_iterations=config.frequency.maximum_reprojection_iterations,
            reprojection_tolerance=config.frequency.reprojection_tolerance)
        coefficients = update.coefficients
        model_weights = torch.softmax(-objective.global_similarities.detach(), dim=0)
        zero_gradient_steps += int(update.zero_gradient)
        if step == 1 or step == steps or step % attack.log_every == 0:
            history.append({"step": step, "scored_state_step": step - 1, **record,
                "linf_after_update": update.resulting_linf, "radial_scale": update.radial_scale,
                "weights_for_next_step": model_weights.detach().cpu().tolist()})

    raw = synthesize(coefficients, bh, bw)
    floating = (source + raw).clamp(0, 1)
    final_record = fixed_record(evaluate_objective(quantize_image(floating), surrogates,
        targets, uniform_weights, attack, use_cached_target_centers=bool(targets.cluster_centers)))
    consider(final_record, steps, raw, floating)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loop_seconds = time.perf_counter() - loop_started
    image_path = save_tensor_png(output / "adversarial.png", quantize_image(best_float))
    decoded = pil_to_tensor(load_image(image_path), device)
    audit = decoded_image_audit(source_pil, image_path,
        maximum_levels=config.acceptance.decoded_max_pixel_difference)
    if not audit["within_budget"]:
        raise RuntimeError("频域增强落盘图片超过像素扰动预算")
    saved_record = fixed_record(evaluate_objective(decoded, surrogates, targets,
        uniform_weights, attack, use_cached_target_centers=bool(targets.cluster_centers)))
    diagnostics = perturbation_diagnostics(source, best_raw, best_float, decoded, bh, bw)
    diagnostics["settings"] = options
    diagnostics.update({"height_frequencies": kh, "width_frequencies": kw,
        "parameter_count": coefficients.numel(), "quantization_forward": "round_8bit_straight_through",
        "candidate_selection": "fixed_equal_weight_global_plus_original_local",
        "final_saved_image_strictly_bandlimited": False})
    metadata = {"schema_version": 1, "status": "success", "method": METHOD,
        "seed": config.seed, "target_text": None,
        "source_shape_bchw": list(source.shape), "target_shape_bchw": list(target.shape),
        "source_image_sha256": sha256_file(source_image) if not hasattr(source_image, "convert") else None,
        "target_image_sha256": sha256_file(target_image) if not hasattr(target_image, "convert") else None,
        "implementation_sha256": options["frequency_implementation_sha256"],
        "attack": {**asdict(attack), "steps_requested": steps, "steps_completed": steps,
            "best_step": best_step, "zero_gradient_steps": zero_gradient_steps,
            "stop_reason": "requested_steps_completed",
            "gradient_execution": "sequential_surrogate_image_gradient"},
        "before_attack": before, "after_attack_saved": saved_record,
        "last_step_saved_score": final_record, "frequency": diagnostics,
        "decoded_pixel_audit": audit, "history": history,
        "surrogates": [model.snapshot() if callable(getattr(model, "snapshot", None))
                       else {"class": type(model).__name__} for model in surrogates],
        "timing": {"optimization_loop_seconds": loop_seconds,
                   "total_attack_seconds": time.perf_counter() - started},
        "output_image": str(image_path), "output_image_sha256": sha256_file(image_path)}
    save_json(output / "metadata.json", metadata)
    return image_path, metadata
