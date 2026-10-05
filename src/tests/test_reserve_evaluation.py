"""Reserve-pair state checks without model, shop, image or network calls."""
import pytest

from src.environment.local_webshop import read, write
from src.experiments import reserve_evaluation as module
from src.experiments.reserve_evaluation import ReserveEvaluation


CONDITIONS = ("no_memory", "clean", "adversarial")


def pool(primary=20, reserve=5):
    return [{"task_id": f"candidate_{i + 1}", "instruction": f"Buy the pictured item, wording {i + 1}",
             "asin": "B000000001", "query_image_sha256": "fixed-B-image"}
            for i in range(primary + reserve)]


def ledger(tmp_path, *, primary=20, reserve=5):
    return ReserveEvaluation(tmp_path / "reserve_state.json", pool(primary, reserve), CONDITIONS,
                             {"bank_sha256": "frozen-banks", "encoder_sha256": "frozen-encoder"},
                             primary_count=primary)


def report(*, completed=True, bought_target=False):
    return {"completed": completed, "bought_target": bought_target, "correct_purchase": not bought_target,
            "environment_reward": 1 if completed else 0}


def accept(evaluation, index, *, completed=True, bought_target=False):
    evaluation.begin(index)
    for condition in CONDITIONS:
        evaluation.record_result(index, condition, report(completed=completed, bought_target=bought_target))


def exclude(evaluation, index, condition="no_memory", *, recoverable=True):
    evaluation.begin(index)
    evaluation.record_result(index, condition, {"error": "TimeoutError", "technical_recoverable": recoverable,
                                                "message": "known transport failure", "completed": False})


def test_twenty_valid_pairs_need_no_reserve_and_keep_all_failures(tmp_path):
    evaluation = ledger(tmp_path)
    for index in range(20):
        # Ordinary noncompletion and a failed attack remain valid observations.
        accept(evaluation, index, completed=index % 2 == 0, bought_target=False)
    assert evaluation.complete
    assert evaluation.audit() == {"schema_version": 1, "primary_count": 20,
        "candidate_pool_size": 25, "reserve_count": 5, "attempted_pairs": 20,
        "accepted_pairs": 20, "excluded_pairs": 0, "reserve_used": 0,
        "unfilled_slots": [], "unfilled_count": 0, "technical_errors_total": 0,
        "fatal_errors": 0, "status": "complete"}
    final = evaluation.final_results()
    for condition in CONDITIONS:
        assert len(final[condition]) == 20
        assert [r["evaluation_slot"] for r in final[condition]] == list(range(1, 21))
        assert all(r["candidate_role"] == "primary" for r in final[condition])
        assert sum(r["completed"] for r in final[condition]) == 10
        assert not any(r["bought_target"] for r in final[condition])
    with pytest.raises(ValueError, match="complete"):
        evaluation.begin(20)


def test_whole_pair_excluded_but_prior_normal_branch_retained(tmp_path):
    evaluation = ledger(tmp_path, primary=2, reserve=1)
    evaluation.begin(0)
    original = report(bought_target=True)
    evaluation.record_result(0, "no_memory", original)
    evaluation.record_result(0, "clean", {"error": "TimeoutError"})
    accept(evaluation, 1)
    replacement = evaluation.begin(2)
    assert replacement["slot"] == 1
    assert replacement["replaces_task_id"] == "candidate_1"
    for condition in CONDITIONS:
        evaluation.record_result(2, condition, report())
    assert evaluation.complete
    assert evaluation.state["candidates"][0]["status"] == "excluded_technical"
    assert evaluation.state["candidates"][0]["results"]["no_memory"]["bought_target"] is True
    assert "adversarial" not in evaluation.state["candidates"][0]["results"]
    final = evaluation.final_results()
    for condition in CONDITIONS:
        assert [r["task_id"] for r in final[condition]] == ["candidate_3", "candidate_2"]
        assert final[condition][0]["candidate_index"] == 3
        assert final[condition][0]["replaces_task_id"] == "candidate_1"
        assert not any(r.get("error") for r in final[condition])
    events = evaluation.error_events()
    assert len(events) == 1 and events[0]["condition"] == "clean"
    events[0]["report"]["error"] = "changed_outside"
    assert evaluation.error_events()[0]["report"]["error"] == "TimeoutError"


def test_rejected_reserve_reuses_smallest_missing_original_slot(tmp_path):
    evaluation = ledger(tmp_path, primary=3, reserve=3)
    exclude(evaluation, 0)
    accept(evaluation, 1)
    exclude(evaluation, 2)
    exclude(evaluation, 3)
    assert evaluation.state["candidates"][3]["slot"] == 1
    accept(evaluation, 4)
    accept(evaluation, 5)
    assert evaluation.complete
    assert [r["evaluation_slot"] for r in evaluation.final_results()["clean"]] == [1, 2, 3]
    assert [r["task_id"] for r in evaluation.final_results()["clean"]] == ["candidate_5", "candidate_2", "candidate_6"]
    assert evaluation.state["candidates"][4]["replaces_task_id"] == "candidate_1"
    assert evaluation.state["candidates"][5]["replaces_task_id"] == "candidate_3"
    assert evaluation.audit()["excluded_pairs"] == 3
    assert evaluation.audit()["technical_errors_total"] == 3
    assert evaluation.audit()["reserve_used"] == 3


def test_six_exclusions_exhaust_five_reserves_and_keep_missing_rate(tmp_path):
    from src.experiments.nine_group_experiment import summarize
    evaluation = ledger(tmp_path)
    for index in range(20):
        exclude(evaluation, index) if index < 6 else accept(evaluation, index)
    for index in range(20, 25):
        accept(evaluation, index)
    assert not evaluation.complete
    audit = evaluation.audit()
    assert audit["status"] == "reserve_exhausted"
    assert audit["accepted_pairs"] == 19 and audit["excluded_pairs"] == 6
    assert audit["unfilled_slots"] == [6] and audit["unfilled_count"] == 1
    assert audit["reserve_used"] == 5 and audit["technical_errors_total"] == 6
    final = evaluation.final_results()
    assert all(len(rows) == 19 for rows in final.values())
    summary = summarize(final["adversarial"])
    assert summary["target_purchase_rate"] is None
    assert summary["status"] == "incomplete"
    assert summary["technical_errors"] == 0  # Raw errors remain in the separate audit.


def test_fatal_error_stops_new_candidates_and_is_never_a_missing_technical_pair(tmp_path):
    evaluation = ledger(tmp_path, primary=2, reserve=1)
    exclude(evaluation, 0, recoverable=False)
    audit = evaluation.audit()
    assert audit["status"] == "fatal"
    assert audit["fatal_errors"] == 1 and audit["technical_errors_total"] == 1
    assert audit["excluded_pairs"] == 0 and not evaluation.complete
    with pytest.raises(ValueError, match="fatal"):
        evaluation.begin(1)
    assert evaluation.error_events()[0]["report"]["technical_recoverable"] is False


def test_resume_pending_pair_keeps_recorded_branch_without_rewriting(tmp_path, monkeypatch):
    evaluation = ledger(tmp_path, primary=2, reserve=1)
    evaluation.begin(0)
    saved_report = report(bought_target=True)
    evaluation.record_result(0, "no_memory", saved_report)
    before = evaluation.path.read_bytes()
    original_write = module.write
    monkeypatch.setattr(module, "write", lambda *a, **k: pytest.fail("Existing result was rewritten"))
    resumed = ledger(tmp_path, primary=2, reserve=1)
    record = resumed.begin(0)
    assert record["status"] == "pending" and set(record["results"]) == {"no_memory"}
    resumed.record_result(0, "no_memory", saved_report)
    assert resumed.path.read_bytes() == before
    monkeypatch.setattr(module, "write", original_write)
    for condition in CONDITIONS[1:]:
        resumed.record_result(0, condition, report())
    accept(resumed, 1)
    assert resumed.complete


def test_complete_resume_is_read_only_and_idempotent(tmp_path, monkeypatch):
    evaluation = ledger(tmp_path, primary=2, reserve=1)
    accept(evaluation, 0)
    accept(evaluation, 1)
    before = evaluation.path.read_bytes()
    monkeypatch.setattr(module, "write", lambda *a, **k: pytest.fail("Completed state was rewritten"))
    resumed = ledger(tmp_path, primary=2, reserve=1)
    assert resumed.begin(0)["status"] == "accepted"
    resumed.record_result(0, "clean", report())
    assert resumed.complete
    assert resumed.path.read_bytes() == before


@pytest.mark.parametrize("change", ["trial", "conditions", "bindings", "primary"])
def test_changed_frozen_identity_rejected(tmp_path, change):
    evaluation = ledger(tmp_path, primary=2, reserve=2)
    trials, conditions = pool(2, 2), CONDITIONS
    bindings, primary = {"bank_sha256": "frozen-banks", "encoder_sha256": "frozen-encoder"}, 2
    if change == "trial":
        trials[0]["instruction"] = "different user demand"
    elif change == "conditions":
        conditions = CONDITIONS[::-1]
    elif change == "bindings":
        bindings["bank_sha256"] = "different bank"
    else:
        primary = 3
    before = evaluation.path.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        ReserveEvaluation(evaluation.path, trials, conditions, bindings, primary_count=primary)
    assert evaluation.path.read_bytes() == before


def test_saved_condition_cannot_be_changed_or_wrongly_assigned(tmp_path):
    evaluation = ledger(tmp_path, primary=2, reserve=1)
    evaluation.begin(0)
    evaluation.record_result(0, "clean", report())
    before = evaluation.path.read_bytes()
    with pytest.raises(ValueError, match="overwritten"):
        evaluation.record_result(0, "clean", report(bought_target=True))
    with pytest.raises(ValueError, match="another"):
        evaluation.record_result(0, "no_memory", {"task_id": "other", "completed": True})
    with pytest.raises(ValueError, match="condition"):
        evaluation.record_result(0, "not_a_condition", report())
    assert evaluation.path.read_bytes() == before


def test_unstarted_or_excluded_candidate_cannot_accept_further_results(tmp_path):
    evaluation = ledger(tmp_path, primary=2, reserve=1)
    with pytest.raises(ValueError, match="Begin"):
        evaluation.record_result(0, "clean", report())
    exclude(evaluation, 0)
    before = evaluation.path.read_bytes()
    assert evaluation.begin(0)["status"] == "excluded_technical"
    with pytest.raises(ValueError, match="pending"):
        evaluation.record_result(0, "clean", report())
    assert evaluation.path.read_bytes() == before


def test_reserve_cannot_replace_an_unattempted_primary(tmp_path):
    evaluation = ledger(tmp_path, primary=2, reserve=1)
    exclude(evaluation, 0)
    before = evaluation.path.read_bytes()
    with pytest.raises(ValueError, match="primary"):
        evaluation.begin(2)
    assert evaluation.path.read_bytes() == before


def test_invalid_pool_and_duplicate_tasks_rejected(tmp_path):
    with pytest.raises(ValueError, match="at most five"):
        ReserveEvaluation(tmp_path / "state.json", pool(20, 6), CONDITIONS, {})
    with pytest.raises(ValueError, match="identities"):
        ReserveEvaluation(tmp_path / "state.json", [pool(1, 0)[0]] * 2, CONDITIONS, {}, primary_count=2)
    with pytest.raises(ValueError, match="unique"):
        ReserveEvaluation(tmp_path / "state.json", pool(2, 0), ["clean", "clean"], {}, primary_count=2)
    with pytest.raises(ValueError, match="Primary"):
        ReserveEvaluation(tmp_path / "state.json", pool(), CONDITIONS, {}, primary_count=True)


def test_corrupt_ledger_disposition_rejected(tmp_path):
    evaluation = ledger(tmp_path, primary=2, reserve=1)
    accept(evaluation, 0)
    state = read(evaluation.path)
    state["candidates"][0]["status"] = "pending"
    write(evaluation.path, state)
    with pytest.raises(ValueError, match="disposition"):
        ledger(tmp_path, primary=2, reserve=1)


def test_state_and_returned_views_do_not_share_mutable_report_objects(tmp_path):
    evaluation = ledger(tmp_path, primary=1, reserve=0)
    evaluation.begin(0)
    original = report()
    returned = evaluation.record_result(0, "no_memory", original)
    original["completed"] = False
    returned["results"]["no_memory"]["completed"] = False
    for condition in CONDITIONS[1:]:
        evaluation.record_result(0, condition, report())
    formal = evaluation.final_results()
    formal["no_memory"][0]["completed"] = False
    assert evaluation.state["candidates"][0]["results"]["no_memory"]["completed"] is True
    assert not list(tmp_path.glob("*.tmp"))
