"""旧满步结果与冻结 SSA 合法早停的完成审计。"""
from copy import deepcopy

import pytest

from src.vision.attack_completion import validate_attack_completion
from src.vision.ssa_attack import PlateauStopper, ssa_settings


def plateau_metadata():
    settings = ssa_settings(ssa_options={"early_stop": True, "min_steps": 2,
        "check_every": 2, "window": 2, "patience": 2, "min_delta": .002})
    stopper = PlateauStopper(settings)
    for step in range(5):
        stopper.observe(step, 1 + .0001 * step)
    return {"method": "ssa", "status": "success",
        "attack": {"steps_requested": 10, "steps_completed": 4,
            "best_step": 4, "stop_reason": "early_stop_plateau"},
        "ssa": {"settings": settings, "early_stop_checks": stopper.checks}}


def test_old_full_step_metadata_needs_no_new_fields():
    validate_attack_completion({"attack": {"steps_completed": 500}}, 500,
        {"method": "ssa", **ssa_settings()})


def old_ssa_metadata():
    # 对应真实旧 500 步结果的结构，旧节点仅含四个数值及实现摘要。
    return {"schema_version": 1, "status": "success", "method": "ssa",
        "attack": {"steps_requested": 500, "steps_completed": 500,
            "best_step": 500, "stop_reason": "requested_steps_completed"},
        "ssa": {"settings": {"ssa_samples": 20, "ssa_rho": .5,
            "ssa_sigma": 16., "ssa_momentum": 1.,
            "ssa_implementation_sha256": "04f2461d30356cbc787e1410667e49cbd018b8daaf23813b0f857db34273f578"}}}


def test_actual_old_ssa_structure_can_be_read_without_current_implementation_identity():
    validate_attack_completion(old_ssa_metadata(), 500)


def test_old_ssa_structure_can_be_audited_against_its_original_frozen_settings():
    metadata = old_ssa_metadata()
    validate_attack_completion(metadata, 500, metadata["ssa"]["settings"])


def test_old_ssa_structure_cannot_be_reused_as_a_current_implementation_run():
    with pytest.raises(ValueError, match="不能跨实现"):
        validate_attack_completion(old_ssa_metadata(), 500, ssa_settings())


@pytest.mark.parametrize("key,value", [("ssa_early_stop", False), ("ssa_early_stop", True),
    ("ssa_mse_weight", 0.), ("ssa_mse_weight", .1), ("ssa_selection_tolerance", .002)])
def test_missing_old_controls_are_not_silently_replaced_even_with_matching_implementation(key, value):
    metadata = old_ssa_metadata()
    expected = {**metadata["ssa"]["settings"], key: value}
    with pytest.raises(ValueError, match="缺省控制"):
        validate_attack_completion(metadata, 500, expected)


def test_full_ssa_without_settings_cannot_claim_strict_frozen_reuse():
    metadata = old_ssa_metadata()
    metadata["ssa"] = {}
    with pytest.raises(ValueError, match="缺少冻结设置"):
        validate_attack_completion(metadata, 500, ssa_settings())


def test_legitimate_plateau_is_a_completed_frozen_attack():
    metadata = plateau_metadata()
    validate_attack_completion(metadata, 10, {"source": "fixture", **metadata["ssa"]["settings"]})


@pytest.mark.parametrize("reason", [None, "time_budget_exhausted", "exception", "requested_steps_completed"])
def test_incomplete_or_interrupted_run_cannot_be_reused(reason):
    metadata = plateau_metadata()
    metadata["attack"]["stop_reason"] = reason
    with pytest.raises(ValueError):
        validate_attack_completion(metadata, 10)


@pytest.mark.parametrize("mutation", [
    lambda m: m["ssa"]["settings"].update(ssa_early_stop=False),
    lambda m: m["attack"].update(steps_completed=3),
    lambda m: m["attack"].update(steps_requested=9),
    lambda m: m["attack"].update(best_step=5),
    lambda m: m.update(status="partial"),
    lambda m: m["ssa"]["early_stop_checks"].pop(),
    lambda m: m["ssa"]["early_stop_checks"][-1].update(triggered=False),
    lambda m: m["ssa"]["early_stop_checks"][-1].update(low_gain_windows=1),
    lambda m: m["ssa"]["early_stop_checks"][-1].update(improvement=.02),
    lambda m: m["ssa"]["early_stop_checks"][0].update(completed_steps=1),
])
def test_invalid_plateau_evidence_is_rejected(mutation):
    metadata = deepcopy(plateau_metadata())
    mutation(metadata)
    with pytest.raises(ValueError):
        validate_attack_completion(metadata, 10)


def test_completion_requires_same_effective_frozen_parameters():
    metadata = plateau_metadata()
    changed = {**metadata["ssa"]["settings"], "ssa_min_delta": .005}
    with pytest.raises(ValueError, match="冻结"):
        validate_attack_completion(metadata, 10, changed)


def test_full_steps_with_abnormal_stop_reason_is_rejected():
    metadata = {"attack": {"steps_completed": 10, "stop_reason": "time_budget_exhausted"}}
    with pytest.raises(ValueError):
        validate_attack_completion(metadata, 10)
