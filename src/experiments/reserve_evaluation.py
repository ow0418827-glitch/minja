"""Transactional, paired replacement of technical evaluation failures.

Model mistakes and ordinary step limits are valid results. Only a reported
technical error excludes a candidate pair; every recorded branch remains in the
ledger, including normal branches of a pair later excluded by another branch.
"""
from __future__ import annotations

import copy
from pathlib import Path

from src.environment.local_webshop import fingerprint, read, write


SCHEMA_VERSION = 1
_TERMINAL = {"accepted", "excluded_technical", "fatal"}


class ReserveEvaluation:
    """Persist candidate results before deriving twenty paired formal results.

    Candidate indexes are zero-based; saved slots and candidate indexes in formal
    results are one-based. Call ``begin`` before recording a new candidate. A
    completed candidate and an identical saved branch are idempotent no-ops.
    """

    def __init__(self, path, trials, conditions, bindings, primary_count=20):
        if isinstance(primary_count, bool) or not isinstance(primary_count, int) or not 1 <= primary_count <= 20:
            raise ValueError("Primary evaluation count must be between 1 and 20")
        if not isinstance(trials, list) or not primary_count <= len(trials) <= min(25, primary_count + 5):
            raise ValueError("Candidate pool requires primary tasks and at most five reserve tasks")
        conditions = list(conditions)
        if not conditions or any(not isinstance(c, str) or not c for c in conditions) or len(set(conditions)) != len(conditions):
            raise ValueError("Evaluation conditions must be nonempty and unique")
        task_ids = [t.get("task_id") if isinstance(t, dict) else None for t in trials]
        if any(not isinstance(t, str) or not t for t in task_ids) or len(set(task_ids)) != len(task_ids):
            raise ValueError("Candidate task identities must be nonempty and unique")
        self.path = Path(path)
        self.trials = copy.deepcopy(trials)
        self.conditions = conditions
        self.primary_count = primary_count
        identity = {"schema_version": SCHEMA_VERSION, "primary_count": primary_count,
                    "trials_sha256": fingerprint(trials), "conditions": conditions,
                    "bindings": copy.deepcopy(bindings)}
        self.identity = identity
        if self.path.exists():
            state = read(self.path)
            if state.get("identity") != identity:
                raise ValueError("Frozen reserve evaluation tasks, conditions or bindings changed")
            self.state = state
            self._validate_state(task_ids)
        else:
            self.state = {"identity": identity, "candidates": [
                {"task_id": task_id, "candidate_index": index + 1,
                 "slot": index + 1 if index < primary_count else None,
                 "role": "primary" if index < primary_count else "reserve",
                 "replaces_task_id": None, "started": False,
                 "results": {}, "trajectories": {}, "status": "pending"}
                for index, task_id in enumerate(task_ids)]}
            self._save()

    def _save(self):
        write(self.path, self.state)

    def _candidate(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.trials):
            raise ValueError("Candidate index is outside the frozen pool")
        return self.state["candidates"][index]

    def _validate_state(self, task_ids):
        records = self.state.get("candidates")
        if not isinstance(records, list) or len(records) != len(task_ids):
            raise ValueError("Frozen reserve ledger has an invalid candidate pool")
        accepted = set()
        for index, record in enumerate(records):
            expected_role = "primary" if index < self.primary_count else "reserve"
            if (not isinstance(record, dict) or record.get("task_id") != task_ids[index]
                    or record.get("candidate_index") != index + 1 or record.get("role") != expected_role
                    or record.get("status") not in _TERMINAL | {"pending"}
                    or not isinstance(record.get("started"), bool)):
                raise ValueError("Frozen reserve ledger candidate identity changed")
            slot = record.get("slot")
            if expected_role == "primary":
                if slot != index + 1 or record.get("replaces_task_id") is not None:
                    raise ValueError("Frozen primary evaluation slot changed")
            elif record["started"]:
                if (isinstance(slot, bool) or not isinstance(slot, int) or not 1 <= slot <= self.primary_count
                        or record.get("replaces_task_id") != task_ids[slot - 1]):
                    raise ValueError("Frozen reserve replacement slot changed")
            elif slot is not None or record.get("replaces_task_id") is not None:
                raise ValueError("Unused reserve candidate already has a replacement slot")
            results = record.get("results")
            if not isinstance(results, dict) or set(results) - set(self.conditions):
                raise ValueError("Frozen candidate contains unknown evaluation conditions")
            if results and not record["started"]:
                raise ValueError("Unstarted candidate contains evaluation results")
            trajectories = record.get("trajectories", {})
            if (not isinstance(trajectories, dict) or set(trajectories) - set(results)
                    or any(not isinstance(row, dict) for row in trajectories.values())):
                raise ValueError("Frozen trajectory does not belong to a recorded evaluation result")
            for condition, report in results.items():
                if (not isinstance(report, dict) or report.get("task_id") != record["task_id"]
                        or report.get("condition") != condition):
                    raise ValueError("Frozen evaluation report belongs to another candidate")
            errors = [r for r in results.values() if r.get("error")]
            expected_status = ("fatal" if any(r.get("technical_recoverable") is False for r in errors)
                               else "excluded_technical" if errors
                               else "accepted" if len(results) == len(self.conditions) else "pending")
            if record["status"] != expected_status:
                raise ValueError("Frozen candidate disposition disagrees with its results")
            if expected_status == "accepted":
                if slot in accepted:
                    raise ValueError("Multiple accepted candidates fill the same evaluation slot")
                accepted.add(slot)

    def begin(self, index):
        record = self._candidate(index)
        if record["status"] in _TERMINAL or record["started"]:
            return copy.deepcopy(record)
        if self.audit()["fatal_errors"]:
            raise ValueError("A fatal evaluation error prevents beginning further candidates")
        if self.complete:
            raise ValueError("The complete evaluation needs no additional candidate")
        if record["role"] == "reserve":
            if any(r["status"] == "pending" for r in self.state["candidates"][:self.primary_count]):
                raise ValueError("Finish the primary candidate pool before beginning reserves")
            accepted = {r["slot"] for r in self.state["candidates"] if r["status"] == "accepted"}
            active = {r["slot"] for r in self.state["candidates"] if r["started"] and r["status"] == "pending"}
            available = sorted(set(range(1, self.primary_count + 1)) - accepted - active)
            if not available:
                raise ValueError("No unfilled evaluation slot is available for a reserve")
            record["slot"] = available[0]
            record["replaces_task_id"] = self.trials[record["slot"] - 1]["task_id"]
        record["started"] = True
        self._save()
        return copy.deepcopy(record)

    def record_result(self, index, condition, report, *, trajectory=None):
        record = self._candidate(index)
        if condition not in self.conditions or not isinstance(report, dict):
            raise ValueError("A result must belong to a frozen evaluation condition")
        if trajectory is not None and not isinstance(trajectory, dict):
            raise ValueError("An evaluation trajectory must be a dictionary")
        saved = copy.deepcopy(report)
        saved.setdefault("task_id", record["task_id"])
        saved.setdefault("condition", condition)
        if saved["task_id"] != record["task_id"] or saved["condition"] != condition:
            raise ValueError("Result belongs to another evaluation candidate or condition")
        previous = record["results"].get(condition)
        if previous is not None:
            if previous != saved:
                raise ValueError("A previously recorded evaluation result cannot be overwritten")
            if trajectory is not None and record.get("trajectories", {}).get(condition) != trajectory:
                raise ValueError("A previously recorded evaluation trajectory cannot be overwritten")
            return copy.deepcopy(record)
        if not record["started"] or record["status"] != "pending":
            raise ValueError("Begin a pending candidate before recording a new result")
        record["results"][condition] = saved
        if trajectory is not None:
            record.setdefault("trajectories", {})[condition] = copy.deepcopy(trajectory)
        if saved.get("error"):
            record["status"] = "fatal" if saved.get("technical_recoverable") is False else "excluded_technical"
        elif len(record["results"]) == len(self.conditions):
            record["status"] = "accepted"
        self._save()
        return copy.deepcopy(record)

    def saved_trajectories(self):
        """Recover shopping files after a crash without rerunning recorded model calls."""
        return [{"candidate_index": candidate["candidate_index"], "condition": condition,
                 "trajectory": copy.deepcopy(row)}
                for candidate in self.state["candidates"]
                for condition, row in candidate.get("trajectories", {}).items()]

    def final_results(self):
        result = {condition: [] for condition in self.conditions}
        accepted = sorted((r for r in self.state["candidates"] if r["status"] == "accepted"), key=lambda r: r["slot"])
        for candidate in accepted:
            for condition in self.conditions:
                report = copy.deepcopy(candidate["results"][condition])
                report.update(evaluation_slot=candidate["slot"], candidate_index=candidate["candidate_index"],
                              candidate_role=candidate["role"], replaces_task_id=candidate["replaces_task_id"])
                result[condition].append(report)
        return result

    def error_events(self):
        events = []
        for candidate in self.state["candidates"]:
            for condition, report in candidate["results"].items():
                if report.get("error"):
                    events.append({"task_id": candidate["task_id"], "candidate_index": candidate["candidate_index"],
                                   "evaluation_slot": candidate["slot"], "candidate_role": candidate["role"],
                                   "replaces_task_id": candidate["replaces_task_id"], "condition": condition,
                                   "status": candidate["status"], "report": copy.deepcopy(report)})
        return events

    @property
    def complete(self):
        return self.audit()["status"] == "complete"

    def audit(self):
        records = self.state["candidates"]
        accepted = {r["slot"] for r in records if r["status"] == "accepted"}
        missing = sorted(set(range(1, self.primary_count + 1)) - accepted)
        fatal = sum(bool(report.get("error") and report.get("technical_recoverable") is False)
                    for r in records for report in r["results"].values())
        errors = sum(bool(report.get("error")) for r in records for report in r["results"].values())
        status = ("fatal" if fatal else "complete" if not missing
                  else "reserve_exhausted" if all(r["status"] in _TERMINAL for r in records) else "pending")
        return {"schema_version": SCHEMA_VERSION, "primary_count": self.primary_count,
                "candidate_pool_size": len(records), "reserve_count": len(records) - self.primary_count,
                "attempted_pairs": sum(r["started"] for r in records),
                "accepted_pairs": len(accepted),
                "excluded_pairs": sum(r["status"] == "excluded_technical" for r in records),
                "reserve_used": sum(r["started"] and r["role"] == "reserve" for r in records),
                "unfilled_slots": missing, "unfilled_count": len(missing),
                "technical_errors_total": errors, "fatal_errors": fatal, "status": status}
