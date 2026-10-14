"""Opt-in round driver; core policy does not modify the legacy shopping loop."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace
from typing import Callable

from .adapters import ShoppingEnvironment
from .authorization import intersect_hypotheses
from .compiler import Op, ProcedureHint
from .contracts import (
    AdmittedField, Assessment, Candidate, Contract, Observation, Outcome, Request,
    Status, Target,
)
from .gate import ExecutionGate
from .history import HistoryResolver
from .ledger import EvidenceLedger
from .model import DecisionModel
from .providers import HmacVerifier, SourceProvider, UnavailableSource
from .scheduler import Scheduler


@dataclass(frozen=True)
class RunnerConfig:
    max_candidates: int = 5
    max_pages: int = 2
    max_environment_steps: int = 30
    max_detail_draws: int = 5
    procedure_kl_budget: float = 0.005
    match_threshold: float = 0.95
    allow_uncalibrated_purchase: bool = False
    random_seed: int = 0

    def __post_init__(self) -> None:
        if min(self.max_candidates, self.max_pages, self.max_environment_steps,
               self.max_detail_draws) < 1 or self.procedure_kl_budget < 0:
            raise ValueError("Invalid run budget")


class SafeRecallRunner:
    def __init__(self, environment: ShoppingEnvironment, model: DecisionModel,
                 *, source: SourceProvider | None = None,
                 verifier: HmacVerifier | None = None,
                 clock: Callable[[], float] = time.time,
                 config: RunnerConfig = RunnerConfig()):
        self.environment = environment
        self.model = model
        self.config = config
        self.history = HistoryResolver(source or UnavailableSource(),
                                       verifier or HmacVerifier({}), clock=clock)
        self.gate = ExecutionGate(score_threshold=config.match_threshold)
        self.ledger = EvidenceLedger()
        self.scheduler = Scheduler(config.procedure_kl_budget,
                                   config.max_detail_draws,
                                   random.Random(config.random_seed))
        self._audit: list[dict[str, object]] = []
        self._steps = 0

    def _record(self, event: str, **fields: object) -> None:
        self._audit.append({"event": event, **fields})

    def _result(self, status: Status, reason: str, *,
                final_handle: str | None = None,
                final_options: dict[str, str] | None = None) -> Outcome:
        return Outcome(status, reason, final_handle, final_options or {},
                       self._steps, audit=tuple(self._audit))

    def _observed(self, observation: Observation) -> Observation:
        self._steps += 1
        self.ledger.add(observation)
        return observation

    def _target(self, contract: Contract) -> tuple[Target | None, str]:
        values = dict(contract.current_fields)
        admitted: dict[str, AdmittedField] = {}
        for reference in contract.references:
            result = self.history.resolve(contract, reference)
            self._record("history_admission", field=reference.field,
                         source_kind=reference.kind, status=result.reason)
            if result.field is None:
                return None, result.reason
            if reference.field in values and values[reference.field] != result.field.value:
                return None, "current_history_conflict"
            values[reference.field] = result.field.value
            admitted[reference.field] = result.field
        return Target(contract.subject, values, admitted, contract.version), "accepted"

    def _history_still_valid(self, contract: Contract, target: Target) -> bool:
        for reference in contract.references:
            refreshed = self.history.resolve(contract, reference)
            original = target.admitted.get(reference.field)
            if (refreshed.field is None or original is None
                    or (refreshed.field.source_id, refreshed.field.source_version,
                        refreshed.field.value, refreshed.field.scope,
                        refreshed.field.source_kind, refreshed.field.purpose) !=
                    (original.source_id, original.source_version, original.value,
                     original.scope, original.source_kind, original.purpose)):
                self._record("history_changed", field=reference.field,
                             status=refreshed.reason)
                return False
        return True

    def _inspect(self, request: Request, target: Target,
                 observation: Observation, hint: ProcedureHint
                 ) -> tuple[Observation, Assessment]:
        if (self.scheduler.draws < self.scheduler.max_draws
                and self._steps + 3 < self.config.max_environment_steps):
            op, trace = self.scheduler.choose(
                (Op.ATTRIBUTES, Op.DESCRIPTION, Op.FEATURES), hint)
            self._record("procedure_draw", **trace)
            subpage, current = self.environment.detail(op.value)
            self._steps += 2
            # The detail was reached from this exact item, then the same item
            # was reopened. Keep source text bound to its current handle.
            observation = replace(current, text=current.text + "\n" + subpage.text)
            self.ledger.add(observation)
        assessment = self.model.assess(request, target, observation)
        for option_type, label in sorted(assessment.suggested_options.items()):
            if observation.option_types.get(label) != option_type:
                self._record("option_rejected", reason="not_visible")
                return observation, assessment
            if self._steps + 2 >= self.config.max_environment_steps:
                return observation, assessment
            observation = self._observed(self.environment.select(label))
            # WebShopAdapter.select performs a click and a fresh observation.
            # Counting two calls prevents the safety runner from hiding cost.
            self._steps += 1
        if assessment.suggested_options:
            assessment = self.model.assess(request, target, observation)
        return observation, assessment

    def run(self, request: Request, *,
            ordinary_memory: tuple[str, ...] = ()) -> Outcome:
        # A runner instance may be reused, but authorization evidence and
        # statistical budgets belong to one episode only.
        self.ledger = EvidenceLedger()
        self.scheduler = Scheduler(self.config.procedure_kl_budget,
                                   self.config.max_detail_draws,
                                   random.Random(self.config.random_seed))
        self._audit = []
        self._steps = 0
        try:
            return self._run_once(request, ordinary_memory)
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            # A malformed model reply, failed observation, or exhausted model
            # budget must never fall through to a purchase call.
            self._record("episode_failure", error=type(exc).__name__)
            return self._result(Status.BLOCKED, "episode_validation_failed")

    def _run_once(self, request: Request,
                  ordinary_memory: tuple[str, ...]) -> Outcome:
        self._observed(self.environment.reset())
        try:
            hypotheses = self.model.intent(request)
        except (ValueError, KeyError, TypeError) as exc:
            self._record("intent_error", error=type(exc).__name__)
            return self._result(Status.NEEDS_CLARIFICATION, "intent_unresolved")
        contract = intersect_hypotheses(request.subject, hypotheses, request.text)
        self._record("authorization", reference_fields=[
            ref.field for ref in contract.references],
            unresolved=list(contract.unresolved))
        if contract.unresolved:
            return self._result(Status.NEEDS_CLARIFICATION,
                                "authorization_ambiguous")
        target, reason = self._target(contract)
        if target is None:
            return self._result(Status.NEEDS_EVIDENCE, reason)
        if not target.values and not request.image_bytes:
            return self._result(Status.NEEDS_CLARIFICATION,
                                "target_not_identifiable_from_current_request")
        try:
            hint = self.model.compile_procedure(ordinary_memory)
        except (ValueError, KeyError, TypeError):
            hint = ProcedureHint(())
            self._record("procedure_rejected")
        query = contract.search_terms or target.values.get("target_identity", "")
        if not query:
            try:
                query = self.model.query(request, target)
            except (ValueError, KeyError, TypeError) as exc:
                self._record("query_error", error=type(exc).__name__)
                return self._result(Status.NEEDS_EVIDENCE, "search_query_unresolved")
        if self._steps + 1 >= self.config.max_environment_steps:
            return self._result(Status.BUDGET_EXHAUSTED, "environment_step_limit")
        try:
            search = self._observed(self.environment.search(query))
        except ValueError:
            return self._result(Status.BLOCKED, "search_not_allowed")
        if search.page != "search":
            return self._result(Status.BLOCKED, "search_state_invalid")
        candidates: list[Candidate] = []
        inspected = 0
        pages = 0
        while pages < self.config.max_pages and inspected < self.config.max_candidates:
            pages += 1
            visible_ids = tuple(dict.fromkeys(search.visible_ids))
            for handle in visible_ids:
                if (inspected >= self.config.max_candidates
                        or self._steps + 4 >= self.config.max_environment_steps):
                    break
                inspected += 1
                try:
                    observation = self._observed(self.environment.open(handle))
                except ValueError:
                    self._record("candidate_rejected", reason="invalid_current_handle")
                    continue
                if observation.page != "item" or observation.handle != handle:
                    return self._result(Status.BLOCKED, "candidate_binding_invalid")
                try:
                    observation, assessment = self._inspect(request, target,
                                                            observation, hint)
                except (ValueError, KeyError, TypeError) as exc:
                    self._record("candidate_rejected", handle=handle,
                                 reason="assessment_invalid",
                                 error=type(exc).__name__)
                    search = self._observed(self.environment.back())
                    continue
                verdict = self.gate.check(request, target, observation,
                                          assessment, self.ledger)
                self._record("candidate", handle=handle,
                             score=assessment.match_score, decision=verdict.reason)
                if verdict.allowed:
                    candidates.append(Candidate(handle, observation, assessment))
                search = self._observed(self.environment.back())
            if (not search.has_next or pages >= self.config.max_pages
                    or inspected >= self.config.max_candidates
                    or self._steps + 5 >= self.config.max_environment_steps):
                break
            search = self._observed(self.environment.next_page())
        if not candidates:
            status = (Status.BUDGET_EXHAUSTED if self._steps + 4 >=
                      self.config.max_environment_steps else Status.NO_MATCH)
            return self._result(status, "no_eligible_current_candidate")
        candidates.sort(key=lambda item: (-item.assessment.match_score, item.handle))
        for choice in candidates:
            if self._steps + 4 >= self.config.max_environment_steps:
                break
            while search.page_number > choice.observation.page_number:
                search = self._observed(self.environment.previous_page())
            while search.page_number < choice.observation.page_number:
                search = self._observed(self.environment.next_page())
            # Reopen and recheck. Search-result order, an old handle, or a
            # previous option/price snapshot cannot authorize a purchase.
            current = self._observed(self.environment.open(choice.handle))
            try:
                current, assessment = self._inspect(request, target, current,
                                                    ProcedureHint(()))
            except (ValueError, KeyError, TypeError) as exc:
                self._record("submit_rejected", handle=choice.handle,
                             reason="assessment_invalid",
                             error=type(exc).__name__)
                search = self._observed(self.environment.back())
                continue
            verdict = self.gate.check(request, target, current, assessment,
                                      self.ledger)
            if not verdict.allowed:
                self._record("submit_rejected", handle=choice.handle,
                             reason=verdict.reason)
                search = self._observed(self.environment.back())
                continue
            if not self._history_still_valid(contract, target):
                return self._result(Status.NEEDS_EVIDENCE,
                                    "authorized_history_changed")
            if not self.config.allow_uncalibrated_purchase:
                self._record("submit_rejected", handle=choice.handle,
                             reason="complete_policy_risk_not_calibrated")
                return self._result(Status.BLOCKED,
                                    "complete_policy_risk_not_calibrated")
            # The adapter checks version, handle and selected options again at
            # the actual irreversible environment call.
            try:
                done, actual_handle = self.environment.buy(current)
            except ValueError:
                return self._result(Status.BLOCKED, "submit_state_changed")
            self._steps += 1
            if not done or actual_handle != choice.handle:
                return self._result(Status.BLOCKED, "purchase_result_unverified")
            self._record("purchase", handle=actual_handle)
            return self._result(Status.PURCHASED, "current_evidence_accepted",
                                final_handle=actual_handle,
                                final_options=dict(current.selected_options))
        return self._result(Status.BUDGET_EXHAUSTED, "submit_budget_exhausted")
