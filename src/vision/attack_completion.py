"""攻击完成审计：兼容旧满步元数据，并严格区分合法平台早停与中断。"""
import math


def _validate_precomputed_scar(metadata, requested_steps, expected):
    """零步仅允许带来源、图片摘要和像素审计的预生成导入。"""
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    from PIL import Image
    if (requested_steps != 0 or metadata.get('method') != 'scar'
            or metadata.get('generation') != 'precomputed_scar_import'
            or metadata.get('status') != 'success'):
        raise ValueError('Zero-step completion requires an explicit successful precomputed SCAR import')
    attack = metadata.get('attack', {})
    for key in ('steps_requested', 'steps_completed', 'best_step', 'selected_step'):
        if isinstance(attack.get(key), bool) or attack.get(key) != 0:
            raise ValueError('Precomputed import cannot claim optimization steps')
    if attack.get('stop_reason') != 'precomputed_import_completed':
        raise ValueError('Precomputed import has an invalid completion reason')
    imported = metadata.get('precomputed_import', {})
    if not isinstance(imported, dict) or not isinstance(imported.get('group_record'), dict):
        raise ValueError('Precomputed import is missing its provenance')
    record = imported['group_record']
    if expected is not None and (expected.get('method') != 'scar' or expected.get('steps') != 0
            or imported.get('manifest_sha256') != expected.get('precomputed_manifest_sha256')
            or record != expected.get('precomputed_group')
            or record['source']['sha256'] != expected.get('source_sha256')
            or record['target']['sha256'] != expected.get('target_sha256')):
        raise ValueError('Precomputed import differs from the frozen image/manifest settings')
    for key in ('source', 'target', 'selected', 'input', 'report', 'core_metadata'):
        asset = record.get(key, {})
        path = Path(asset.get('path', ''))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset.get('sha256'):
            raise ValueError('Precomputed import source changed: ' + key)
    output = imported.get('output', {})
    path = Path(output.get('path', ''))
    if (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['selected']['sha256']
            or output.get('sha256') != record['selected']['sha256']):
        raise ValueError('Imported output bytes differ from the selected SCAR image')
    core = json.loads(Path(record['core_metadata']['path']).read_text(encoding='utf-8'))
    if (core.get('status') != 'success' or core.get('source_image_sha256') != record['source']['sha256']
            or core.get('target_image_sha256') != record['target']['sha256']
            or core.get('output_image_sha256') != record['input']['sha256']):
        raise ValueError('SCAR upstream optimizer metadata does not bind the supplied images')
    with Image.open(record['source']['path']) as image:
        source = np.asarray(image.convert('RGB'), dtype=np.int16)
    with Image.open(path) as image:
        selected = np.asarray(image.convert('RGB'), dtype=np.int16)
    audit = imported.get('pixel_audit', {})
    budget = expected.get('epsilon_levels', 16) if expected is not None else audit.get('epsilon_levels')
    if (source.shape != selected.shape or isinstance(budget, bool) or not isinstance(budget, int)
            or not 0 < budget <= 16):
        raise ValueError('Precomputed SCAR dimensions or pixel budget are invalid')
    maximum = int(np.abs(selected - source).max())
    if (maximum > budget or audit.get('max_pixel_difference') != maximum
            or audit.get('epsilon_levels') != budget
            or audit.get('source_sha256') != record['source']['sha256']
            or audit.get('adversarial_sha256') != record['selected']['sha256']):
        raise ValueError('Precomputed SCAR pixel audit or perturbation budget does not match')


def validate_attack_completion(metadata, requested_steps, expected_ssa_settings=None):
    """成功返回 None；不合法完成、参数错配或伪造早停轨迹抛出 ValueError。

    不传 expected 时仅审计结果合法完成，可读取旧完整结果。
    传 expected 时严格匹配实际记录的 SSA 实现与控制参数；旧结果不因此
    获得跨实现续跑资格，缺失新控制参数也不擅自补成当前默认值。
    """
    if requested_steps == 0 and not isinstance(requested_steps, bool) and isinstance(metadata, dict):
        return _validate_precomputed_scar(metadata, requested_steps, expected_ssa_settings)
    if isinstance(requested_steps, bool) or not isinstance(requested_steps, int) or requested_steps < 1:
        raise ValueError("requested_steps 必须为正整数")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("attack"), dict):
        raise ValueError("攻击元数据缺少 attack 对象")
    attack = metadata["attack"]
    completed = attack.get("steps_completed")
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= requested_steps:
        raise ValueError("攻击实际完成步数无效")
    if attack.get("steps_requested", requested_steps) != requested_steps:
        raise ValueError("攻击请求步数与冻结配置不符")
    if metadata.get("status", "success") != "success":
        raise ValueError("攻击没有正常完成")
    for key in ("best_step", "selected_step"):
        value = attack.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= completed:
            raise ValueError(f"攻击 {key} 超过完成步数")
    ssa_metadata = metadata.get("ssa", {})
    if not isinstance(ssa_metadata, dict):
        raise ValueError("攻击 ssa 元数据必须为对象")
    settings = ssa_metadata.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("攻击 SSA 设置必须为对象")
    if expected_ssa_settings is not None:
        if not isinstance(expected_ssa_settings, dict):
            raise ValueError("expected_ssa_settings 必须为对象")
        if metadata.get("method") == "ssa" and not settings and any(
                key.startswith("ssa_") for key in expected_ssa_settings):
            raise ValueError("SSA 结果缺少冻结设置，无法作为当前配置续跑复用")
        if settings:
            implementation_key = "ssa_implementation_sha256"
            if (implementation_key in expected_ssa_settings
                    and settings.get(implementation_key) != expected_ssa_settings[implementation_key]):
                raise ValueError("SSA 实现摘要与冻结配置不符：旧结果可独立读取，但不能跨实现作为同配置续跑复用")
            for key, value in expected_ssa_settings.items():
                if key.startswith("ssa_") and (key not in settings or settings[key] != value):
                    raise ValueError(f"SSA 完成元数据参数与冻结配置不符：{key}；旧结果缺省控制不能自动视为当前设置")
    reason = attack.get("stop_reason")
    if completed == requested_steps:
        if reason not in (None, "requested_steps_completed"):
            raise ValueError("攻击虽达到请求步数，但停止原因为异常或不一致")
        return
    if (metadata.get("method") != "ssa" or reason != "early_stop_plateau"
            or metadata.get("status") != "success" or settings.get("ssa_early_stop") is not True):
        raise ValueError("攻击未完成请求步数且不是合法 SSA 平台早停")
    for key in ("ssa_min_steps", "ssa_check_every", "ssa_window", "ssa_patience"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"早停设置无效：{key}")
    delta = settings.get("ssa_min_delta")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("早停最小改善量无效")
    minimum, interval, window, patience = (settings[key] for key in
        ("ssa_min_steps", "ssa_check_every", "ssa_window", "ssa_patience"))
    if interval != window or completed < max(minimum, window) or completed % interval:
        raise ValueError("早停完成步数不满足窗口与最少步数设置")
    checks = ssa_metadata.get("early_stop_checks")
    if not isinstance(checks, list) or len(checks) < patience:
        raise ValueError("早停元数据缺少连续低收益检查记录")
    first_check = ((max(minimum, window) + interval - 1) // interval) * interval
    low_count, prior_best = 0, -math.inf
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or check.get("completed_steps") != first_check + index * interval:
            raise ValueError("早停检查时间序列不连续")
        score, baseline, improvement = (check.get(key) for key in
            ("best_score", "baseline_best_score", "improvement"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (score, baseline, improvement)):
            raise ValueError("早停检查评分无效")
        if score < baseline or score < prior_best or not math.isclose(score - baseline, improvement,
                rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("早停窗口收益与历史最佳评分不一致")
        if index and not math.isclose(baseline, prior_best, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("连续早停窗口的基准最佳评分不连续")
        low_count = low_count + 1 if improvement < delta else 0
        triggered = low_count >= patience
        if check.get("low_gain_windows") != low_count or check.get("triggered") is not triggered:
            raise ValueError("早停计数与收益轨迹不一致")
        if triggered and index != len(checks) - 1:
            raise ValueError("早停触发之后仍包含继续运行检查")
        prior_best = score
    if checks[-1]["completed_steps"] != completed or low_count < patience:
        raise ValueError("实际完成步数没有对应的早停触发记录")
    if "best_selection_score" in attack and not math.isclose(
            attack["best_selection_score"], checks[-1]["best_score"], rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("最终历史最佳评分与停止轨迹不一致")
