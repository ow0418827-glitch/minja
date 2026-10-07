"""离线验证原需求延后、两轮预算与人工续跑；没有模型或网络调用。"""
import copy
import importlib
import json

import pytest


CONDITIONS = ["no_memory", "clean", "adversarial"]


def trials(count):
    return [{"task_id": f"task-{index}", "instruction": f"原需求 {index}"} for index in range(count)]


def ledger(tmp_path, count=4, **kwargs):
    cls = importlib.import_module("src.experiments.deferred_529").Deferred529Evaluation
    return cls(tmp_path / "ledger.json", trials(count), CONDITIONS, {"model": "frozen"},
               primary_count=count, **kwargs)


def report(plan, *, overload=False, other_error=False, marker=None):
    return {"task_id": plan["task_id"], "condition": plan["condition"],
            "error": "service overloaded 529" if overload else "connection failed" if other_error else None,
            "stop_reason": "technical_error" if overload or other_error else "step_limit",
            "target_purchased": False, "marker": marker}


def commit(book, *, overload=False, other_error=False, marker=None,
           deferred_after_529=False, error_message=None):
    plan = book.next_attempt()
    assert plan is not None
    running = book.begin(plan["candidate_index"], plan["condition"])
    saved = report(plan, overload=overload, other_error=other_error, marker=marker)
    if deferred_after_529:
        saved["deferred_after_529"] = True
    if error_message is not None:
        saved["error"], saved["stop_reason"] = error_message, "technical_error"
    trace = {"task_id": plan["task_id"], "condition": plan["condition"], "marker": marker,
             "actions": ["原轨迹"]}
    book.record_result(plan["candidate_index"], plan["condition"], saved,
                       trajectory=trace, is_529=overload)
    return plan, running, saved, trace


def finish_valid(book):
    seen = []
    while book.next_attempt() is not None:
        seen.append(commit(book)[0])
    return seen


def test_formal_twenty_slots_keep_step_limit_and_never_resample(tmp_path):
    book = ledger(tmp_path, 20)
    book.start_batch("manual-1")
    seen = finish_valid(book)
    assert len(seen) == 60
    assert [p["candidate_index"] for p in seen] == [i for i in range(20) for _ in CONDITIONS]
    assert book.audit()["status"] == "completed"
    assert book.audit()["accepted_pairs"] == 20
    first_results = book.final_results()
    restored = ledger(tmp_path, 20)
    restored.start_batch("manual-2")
    assert restored.next_attempt() is None
    assert restored.final_results() == first_results
    assert all(row["stop_reason"] == "step_limit" for values in first_results.values() for row in values)
    assert len(restored.saved_trajectories()) == 60


def test_primary_529_skips_demand_but_finishes_other_demands_before_retry(tmp_path):
    book = ledger(tmp_path, 3)
    book.start_batch("manual-1")
    initial = commit(book, overload=True)[0]
    assert initial["candidate_index"] == 0
    assert book.next_attempt()["candidate_index"] == 1
    for _ in range(6):
        plan = commit(book)[0]
        assert plan["phase"] == "primary"
    plan = book.next_attempt()
    assert plan == {"candidate_index": 0, "slot": 1, "task_id": "task-0",
                    "condition": "no_memory", "phase": "deferred", "round": 1}
    assert book.audit()["completed_initial_pass"]
    finish_valid(book)
    assert book.complete
    assert len(book.error_events()) == 1


def test_three_distinct_consecutive_demands_halt_same_batch_and_resume_new_batch(tmp_path):
    book = ledger(tmp_path, 4)
    book.start_batch("manual-1")
    for index in range(3):
        assert commit(book, overload=True)[0]["candidate_index"] == index
    assert book.audit()["status"] == "halted"
    assert book.audit()["halt_reason"] == "three_consecutive_primary_529"
    assert len(book.error_events()) == 3
    before = copy.deepcopy(book.state["attempts"])
    book.start_batch("manual-1")
    assert book.next_attempt() is None
    restored = ledger(tmp_path, 4)
    restored.start_batch("manual-2")
    assert restored.next_attempt()["candidate_index"] == 3
    for _ in range(3):
        assert commit(restored)[0]["phase"] == "primary"
    assert restored.next_attempt()["phase"] == "deferred"
    assert restored.next_attempt()["round"] == 1
    finish_valid(restored)
    assert restored.complete
    assert restored.state["attempts"][:3] == before
    assert restored.state["events"][-1]["new_retry_allowance"] is True
    assert restored.state["events"][-1]["previous_initial_529_streak"] == 3


def test_valid_demand_breaks_primary_529_streak(tmp_path):
    book = ledger(tmp_path, 4)
    book.start_batch("batch")
    commit(book, overload=True)
    for _ in CONDITIONS:
        commit(book)
    commit(book, overload=True)
    commit(book, overload=True)
    assert book.audit()["initial_529_streak"] == 2
    assert book.audit()["status"] == "deferred"
    assert book.audit()["halt_reason"] is None


def test_deferred_round_runs_every_missing_condition_despite_529(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book, overload=True)
    observed = []
    for _ in range(6):
        plan = commit(book, overload=True)[0]
        observed.append((plan["round"], plan["condition"]))
    assert observed == [(r, c) for r in (1, 2) for c in CONDITIONS]
    assert book.audit()["status"] == "halted"
    assert book.audit()["halt_reason"] == "deferred_rounds_exhausted"
    assert book.audit()["rounds_attempted"] == 2
    assert book.audit()["committed_attempts"] == 7
    assert len(book.saved_trajectories()) == 7
    assert len({r["attempt_id"] for r in book.saved_trajectories()}) == 7
    assert book.next_attempt() is None
    book.start_batch("batch")
    assert book.next_attempt() is None
    book.start_batch("another-manual-run")
    assert book.next_attempt()["round"] == 1
    assert book.audit()["overload_529_errors"] == 7


def test_normal_conditions_retained_and_only_missing_conditions_retried(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    first, _, saved, trace = commit(book, marker="first normal")
    commit(book, overload=True)
    assert book.next_attempt()["condition"] == "clean"
    assert commit(book, overload=True)[0]["round"] == 1
    assert commit(book, marker="third condition normal")[0]["condition"] == "adversarial"
    assert book.next_attempt()["round"] == 2
    assert book.next_attempt()["condition"] == "clean"
    commit(book)
    assert book.complete
    assert book.state["slots"][0]["results"]["no_memory"] == saved
    assert book.state["slots"][0]["trajectories"]["no_memory"] == trace
    assert len(book.state["attempts"]) == 5
    assert len(book.error_events()) == 2
    assert book.state["slots"][0]["latest_529"] == {}
    with pytest.raises(ValueError):
        book.record_result(first["candidate_index"], first["condition"], {**saved, "marker": "resampled"})


def test_uncommitted_primary_request_can_resume_without_duplicate_commits(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    plan = book.next_attempt()
    running = book.begin(plan["candidate_index"], plan["condition"])
    restored = ledger(tmp_path, 1)
    restored.start_batch("resume-after-interruption")
    assert restored.begin(0, "no_memory")["attempt_id"] == running["attempt_id"]
    assert restored.audit()["committed_attempts"] == 0
    assert restored.state["events"][-1]["new_retry_allowance"] is False
    restored.record_result(0, "no_memory", report(plan))
    same = ledger(tmp_path, 1)
    assert same.next_attempt()["condition"] == "clean"
    assert same.audit()["committed_attempts"] == 1


def test_interruption_keeps_deferred_round_budget_and_queue_cursor(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book, overload=True)
    commit(book, overload=True)
    plan = book.next_attempt()
    assert (plan["round"], plan["condition"]) == (1, "clean")
    running = book.begin(0, "clean")
    old_errors = book.error_events()
    restored = ledger(tmp_path, 1)
    restored.start_batch("new-process-after-interruption")
    assert restored.audit()["rounds_attempted"] == 1
    assert restored.next_attempt()["condition"] == "clean"
    assert restored.begin(0, "clean")["attempt_id"] == running["attempt_id"]
    restored.record_result(0, "clean", report(plan))
    assert commit(restored)[0]["condition"] == "adversarial"
    assert commit(restored, overload=True)[0]["round"] == 2
    assert restored.audit()["status"] == "halted"
    assert restored.error_events()[:2] == old_errors
    assert restored.audit()["committed_attempts"] == 5


def test_import_shuffled_sparse_original_reports_and_529_without_replacing_tasks(tmp_path):
    inputs = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        inputs[condition].append({"task_id": "task-1", "condition": condition,
                                  "error": None, "marker": "legacy normal"})
    inputs["adversarial"].append({"task_id": "task-0", "condition": "adversarial",
                                   "error": "overload", "status_code": 529})
    book = ledger(tmp_path, 2, initial_results=inputs, is_529=lambda r: r.get("status_code") == 529)
    book.start_batch("batch")
    assert book.next_attempt()["phase"] == "deferred"
    assert book.next_attempt()["candidate_index"] == 0
    old_normal = copy.deepcopy(book.state["slots"][1]["results"])
    finish_valid(book)
    assert book.complete
    assert book.state["slots"][1]["results"] == old_normal
    assert [r["task_id"] for r in book.final_results()["adversarial"]] == ["task-0", "task-1"]
    restored = ledger(tmp_path, 2, initial_results=inputs, is_529=lambda r: r.get("status_code") == 529)
    assert restored.complete
    assert restored.audit()["committed_attempts"] == book.audit()["committed_attempts"]


def test_non_529_technical_results_are_committed_for_old_reserve_bridge(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book, other_error=True)
    finish_valid(book)
    assert book.audit()["status"] == "completed"
    assert book.audit()["other_technical_errors"] == 1
    assert book.audit()["accepted_pairs"] == 0
    assert book.final_results()["no_memory"][0]["error"] == "connection failed"
    assert book.audit()["rounds_attempted"] == 0


def test_committed_report_and_trajectory_recovery_are_idempotent(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    plan, _, saved, trace = commit(book)
    book.record_result(0, plan["condition"], saved, trajectory=trace)
    assert book.audit()["committed_attempts"] == 1
    restored = ledger(tmp_path, 1)
    assert restored.saved_trajectories()[0]["trajectory"] == trace
    assert restored.next_attempt()["condition"] == "clean"


@pytest.mark.parametrize("changed", ["trials", "conditions", "bindings"])
def test_frozen_identity_mismatch_rejected(tmp_path, changed):
    book = ledger(tmp_path, 1)
    cls = type(book)
    t, c, b = trials(1), CONDITIONS, {"model": "frozen"}
    if changed == "trials": t[0]["instruction"] = "changed demand"
    if changed == "conditions": c = list(reversed(c))
    if changed == "bindings": b = {"model": "changed"}
    with pytest.raises(ValueError):
        cls(book.path, t, c, b, primary_count=1)


def test_task_identity_and_commit_order_validated_before_any_result(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    with pytest.raises(ValueError): book.begin(0, "adversarial")
    with pytest.raises(ValueError): book.record_result(0, "no_memory", {"task_id": "wrong"})
    with pytest.raises(ValueError): book.record_result(0, "no_memory", {"error": None})
    assert book.audit()["committed_attempts"] == 0


def test_invalid_import_never_guesses_position(tmp_path):
    with pytest.raises(ValueError):
        ledger(tmp_path, 1, initial_results={"no_memory": [{"error": None}]})
    assert not (tmp_path / "ledger.json").exists()


def test_all_529_history_kept_when_final_valid_results_exist(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book, overload=True, marker="first failure")
    commit(book, overload=True, marker="second failure")
    finish_valid(book)
    assert book.complete
    errors = book.error_events()
    assert [r["report"]["marker"] for r in errors] == ["first failure", "second failure"]
    assert all(r["trajectory"]["actions"] == ["原轨迹"] for r in errors)
    loaded = json.loads(book.path.read_text(encoding="utf-8"))
    assert len(loaded["attempts"]) == 5
    assert loaded["slots"][0]["latest_529"] == {}


def test_frozen_rotated_condition_orders_survive_deferred_rounds(tmp_path):
    orders = [CONDITIONS[1:] + CONDITIONS[:1], CONDITIONS[2:] + CONDITIONS[:2]]
    book = ledger(tmp_path, 2, condition_orders=orders)
    book.start_batch("batch")
    assert commit(book, overload=True)[0]["condition"] == "clean"
    observed = [commit(book)[0]["condition"] for _ in range(3)]
    assert observed == orders[1]
    assert [commit(book)[0]["condition"] for _ in range(3)] == orders[0]
    assert book.complete
    restored = ledger(tmp_path, 2, condition_orders=orders)
    assert restored.complete
    with pytest.raises(ValueError):
        ledger(tmp_path, 2, condition_orders=[CONDITIONS, CONDITIONS])


def test_prior_error_import_is_archived_after_valid_result_without_overwriting(tmp_path):
    inputs = {"no_memory": [
        {"task_id": "task-0", "condition": "no_memory", "error": None, "marker": "valid first"},
        {"task_id": "task-0", "condition": "no_memory", "error": "529 old", "status_code": 529},
    ]}
    book = ledger(tmp_path, 1, initial_results=inputs, is_529=lambda r: r.get("status_code") == 529)
    assert book.state["slots"][0]["results"]["no_memory"]["marker"] == "valid first"
    assert len(book.error_events()) == 1
    book.start_batch("batch")
    assert book.next_attempt()["condition"] == "clean"
    finish_valid(book)
    assert book.complete


def test_identical_529_commit_is_idempotent_and_attempt_numbers_are_per_condition(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    plan, _, saved, trace = commit(book, overload=True)
    book.record_result(0, plan["condition"], saved, trajectory=trace, is_529=True)
    assert book.audit()["committed_attempts"] == 1
    commit(book, overload=True)
    finish_valid(book)
    traces = book.saved_trajectories()
    assert [r["attempt_number"] for r in traces if r["condition"] == "no_memory"] == [1, 2, 3]
    assert [r["attempt_number"] for r in traces if r["condition"] == "clean"] == [1]


def test_atomic_commit_failure_preserves_running_snapshot_and_can_retry(tmp_path, monkeypatch):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    plan = book.next_attempt()
    book.begin(0, "no_memory")
    original_bytes = book.path.read_bytes()
    module = importlib.import_module("src.experiments.deferred_529")
    real_replace = module.os.replace
    def fail_replace(*args):
        raise OSError("injected atomic replacement failure")
    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        book.record_result(0, "no_memory", report(plan))
    assert book.path.read_bytes() == original_bytes
    assert book.audit()["committed_attempts"] == 0
    assert book.audit()["running"]["condition"] == "no_memory"
    monkeypatch.setattr(module.os, "replace", real_replace)
    book.record_result(0, "no_memory", report(plan))
    assert book.audit()["committed_attempts"] == 1
    assert book.next_attempt()["condition"] == "clean"


def test_module_import_uses_standard_library_only():
    import subprocess
    import sys
    program = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','numpy','PIL','requests','flask'}:
        raise AssertionError('Heavy dependency: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from src.experiments.deferred_529 import Deferred529Evaluation, POLICY_VERSION
assert POLICY_VERSION == 'deferred_529_same_demand_v1'
"""
    completed = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_non529_reserve_bridge_excludes_remaining_conditions_without_placeholders(tmp_path):
    book = ledger(tmp_path, 2)
    book.start_batch("batch")
    commit(book, other_error=True)
    book.exclude_other_error(0)
    assert book.next_attempt()["candidate_index"] == 1
    finish_valid(book)
    assert book.complete
    assert book.audit()["other_excluded_slots"] == [1]
    assert len(book.final_results()["no_memory"]) == 2
    assert len(book.final_results()["clean"]) == 1
    assert len(book.final_results()["adversarial"]) == 1
    restored = ledger(tmp_path, 2)
    assert restored.complete
    assert restored.audit()["other_excluded_slots"] == [1]
    before = len(restored.state["events"])
    restored.exclude_other_error(0)
    assert len(restored.state["events"]) == before


def test_reserve_bridge_never_excludes_a_slot_with_unresolved_529(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book, overload=True)
    commit(book, overload=True)
    commit(book, other_error=True)
    with pytest.raises(ValueError):
        book.exclude_other_error(0)
    assert book.state["slots"][0]["latest_529"]
    assert book.audit()["other_excluded_slots"] == []
    finish_valid(book)
    assert book.complete


def test_reserve_bridge_does_not_exclude_valid_step_limit(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book)
    with pytest.raises(ValueError):
        book.exclude_other_error(0)


def test_initial_529_then_round_one_500_remains_pending_until_round_two_normal(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book, overload=True, marker="original 529")
    first, _, mixed, trace = commit(book, deferred_after_529=True,
                                    error_message="HTTP 500", marker="round one 500")
    assert (first["phase"], first["round"], first["condition"]) == ("deferred", 1, "no_memory")
    assert "no_memory" not in book.state["slots"][0]["results"]
    assert book.state["slots"][0]["latest_529"]["no_memory"] == mixed
    assert book.state["attempts"][-1]["is_529"] is False
    assert book.audit()["overload_529_errors"] == 1
    assert book.audit()["deferred_after_529_errors"] == 1
    restored = ledger(tmp_path, 1)
    restored.start_batch("resume-interrupted-round")
    assert restored.audit()["rounds_attempted"] == 1
    for condition in ("clean", "adversarial"):
        assert commit(restored)[0]["condition"] == condition
    final = commit(restored)[0]
    assert (final["round"], final["condition"]) == (2, "no_memory")
    assert restored.complete
    assert restored.audit()["other_excluded_slots"] == []
    errors = restored.error_events()
    assert [row["is_529"] for row in errors] == [True, False]
    assert errors[1]["report"] == mixed
    assert errors[1]["trajectory"] == trace
    assert restored.audit()["technical_errors_total"] == 2
    assert restored.audit()["overload_529_errors"] == 1
    assert len(restored.state["attempts"]) == 5
    assert {row["task_id"] for row in restored.state["attempts"]} == {"task-0"}


def test_529_recovered_condition_keeps_same_demand_pinned_for_another_timeout(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book, overload=True)
    normal, _, saved, trace = commit(book, marker="first valid recovery")
    assert book.state["slots"][0]["latest_529"] == {}
    assert book.has_529_history(0)
    timeout = commit(book, deferred_after_529=True, error_message="request timed out")[2]
    assert "clean" not in book.state["slots"][0]["results"]
    assert book.state["slots"][0]["latest_529"]["clean"] == timeout
    with pytest.raises(ValueError, match="不得交备用"):
        book.exclude_other_error(0)
    assert commit(book)[0]["condition"] == "adversarial"
    final = commit(book)[0]
    assert (final["round"], final["condition"]) == (2, "clean")
    assert book.complete
    assert book.state["slots"][0]["results"][normal["condition"]] == saved
    assert book.state["slots"][0]["trajectories"][normal["condition"]] == trace
    assert book.audit()["other_excluded_slots"] == []
    assert book.audit()["overload_529_errors"] == 1
    assert book.audit()["deferred_after_529_errors"] == 1


def test_reserve_bridge_rejects_all_real_529_history_even_when_latest_is_empty(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book, overload=True)
    commit(book)
    commit(book, other_error=True)
    assert book.state["slots"][0]["latest_529"] == {}
    assert book.has_529_history(0)
    with pytest.raises(ValueError, match="不得交备用"):
        book.exclude_other_error(0)
    finish_valid(book)
    restored = ledger(tmp_path, 1)
    assert restored.has_529_history(0)
    with pytest.raises(ValueError, match="不得交备用"):
        restored.exclude_other_error(0)
    assert restored.audit()["other_excluded_slots"] == []


@pytest.mark.parametrize("history,error,flag", [(False, "HTTP 500", True),
                                               (True, None, True),
                                               (True, "HTTP 500", 1)])
def test_deferred_after_529_requires_real_history_error_and_boolean(tmp_path, history, error, flag):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    if history:
        commit(book, overload=True)
    plan = book.next_attempt()
    book.begin(plan["candidate_index"], plan["condition"])
    saved = {**report(plan), "error": error, "deferred_after_529": flag}
    before = book.path.read_bytes()
    with pytest.raises(ValueError):
        book.record_result(0, plan["condition"], saved)
    assert book.path.read_bytes() == before
    assert book.audit()["committed_attempts"] == int(history)
    assert book.audit()["running"] is not None


def test_import_pending_mixed_error_requires_real_529_history_and_preserves_type(tmp_path):
    inputs = {"clean": [{"task_id": "task-0", "condition": "clean", "error": "HTTP 500",
                         "deferred_after_529": True}],
              "no_memory": [{"task_id": "task-0", "condition": "no_memory",
                             "error": "original overload", "status_code": 529}]}
    detector = lambda row: row.get("status_code") == 529
    book = ledger(tmp_path, 1, initial_results=inputs, is_529=detector)
    assert book.has_529_history(0)
    assert book.state["slots"][0]["results"] == {}
    assert book.state["slots"][0]["latest_529"]["clean"]["error"] == "HTTP 500"
    assert [row["is_529"] for row in book.state["attempts"]] == [True, False]
    book.start_batch("batch")
    assert book.next_attempt()["phase"] == "deferred"
    restored = ledger(tmp_path, 1, initial_results=inputs, is_529=detector)
    assert restored.audit()["committed_attempts"] == 2
    finish_valid(restored)
    assert restored.complete
    assert restored.audit()["deferred_after_529_errors"] == 1


def test_mixed_errors_exhaust_two_rounds_without_replacement_or_false_529_count(tmp_path):
    book = ledger(tmp_path, 1)
    book.start_batch("batch")
    commit(book, overload=True)
    seen = [commit(book, deferred_after_529=True, error_message="request timed out")[0]
            for _ in range(6)]
    assert [(row["round"], row["condition"]) for row in seen] == [(r, c) for r in (1, 2) for c in CONDITIONS]
    assert book.audit()["status"] == "halted"
    assert book.audit()["halt_reason"] == "deferred_rounds_exhausted"
    assert book.audit()["overload_529_errors"] == 1
    assert book.audit()["deferred_after_529_errors"] == 6
    assert book.audit()["other_excluded_slots"] == []
    assert book.audit()["pending_slots"] == [1]
    restored = ledger(tmp_path, 1)
    assert restored.next_attempt() is None
    assert restored.state["slots"][0]["results"] == {}
