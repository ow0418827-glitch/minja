"""Independent field, evidence, option and page-state execution gate."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .contracts import Assessment, Observation, Request, Target
from .ledger import EvidenceLedger


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str


class ExecutionGate:
    def __init__(self, *, score_threshold: float = 0.95):
        if not 0 <= score_threshold <= 1:
            raise ValueError("Score threshold outside [0, 1]")
        self.score_threshold = score_threshold

    def check(self, request: Request, target: Target, observation: Observation,
              assessment: Assessment, ledger: EvidenceLedger) -> GateDecision:
        if observation.page != "item" or not observation.handle:
            return GateDecision(False, "not_on_current_item")
        if ledger.current_item(observation.handle) != observation:
            return GateDecision(False, "stale_candidate_evidence")
        if request.image_bytes and not observation.image_bytes:
            return GateDecision(False, "current_product_image_missing")
        if assessment.conflict or assessment.missing_fields:
            return GateDecision(False, "conflict_or_required_evidence_missing")
        if assessment.match_score < self.score_threshold:
            return GateDecision(False, "absolute_match_score_below_threshold")
        for field, required in target.values.items():
            if field == "max_price":
                if observation.price is None:
                    return GateDecision(False, "current_price_missing")
                try:
                    if observation.price > float(required):
                        return GateDecision(False, "budget_exceeded")
                except ValueError:
                    return GateDecision(False, "invalid_budget")
                continue
            if field == "size" and target.admitted.get("size", None):
                scope = target.admitted["size"].scope
                if scope.get("kind") == "body_measurement":
                    option = assessment.suggested_options.get("Size")
                    bounds = assessment.size_chart.get(option or "")
                    span = assessment.evidence_spans.get("size_chart", "")
                    if not option or bounds is None or not ledger.cited_here(observation, span):
                        return GateDecision(False, "current_size_chart_missing")
                    quoted_numbers = [float(x) for x in re.findall(
                        r"\d+(?:\.\d+)?", span)]
                    if (option.casefold() not in span.casefold()
                            or scope["dimension"].casefold() not in span.casefold()
                            or len(quoted_numbers) < 2
                            or quoted_numbers[-2:] != list(bounds)):
                        return GateDecision(False, "size_chart_citation_invalid")
                    if not (bounds[0] <= float(scope["min_cm"])
                            and float(scope["max_cm"]) <= bounds[1]):
                        return GateDecision(False, "body_range_not_covered")
                    continue
                if scope.get("kind") == "brand_label":
                    brand = scope.get("brand", "")
                    observed_brand = assessment.observed_fields.get("brand", "")
                    if (observed_brand.casefold() != brand.casefold()
                            or not ledger.cited_here(
                                observation, assessment.evidence_spans.get("brand", ""))):
                        return GateDecision(False, "size_brand_scope_mismatch")
            observed = assessment.observed_fields.get(field)
            if observed is None or observed.casefold() != required.casefold():
                return GateDecision(False, "target_field_not_matched:" + field)
            span = assessment.evidence_spans.get(field, "")
            if not ledger.cited_here(observation, span):
                return GateDecision(False, "uncited_current_evidence:" + field)
        for option_type, label in assessment.suggested_options.items():
            if observation.option_types.get(label) != option_type:
                return GateDecision(False, "option_not_currently_visible")
            if observation.selected_options.get(option_type) != label:
                return GateDecision(False, "required_option_not_selected")
        if "size" in target.values:
            size_options = {
                label: option_type for label, option_type in observation.option_types.items()
                if option_type.casefold() == "size"
            }
            if size_options:
                selected = observation.selected_options.get(next(iter(size_options.values())))
                scope = target.admitted.get("size")
                if scope and scope.scope.get("kind") == "body_measurement":
                    if selected != assessment.suggested_options.get("Size"):
                        return GateDecision(False, "size_chart_option_not_selected")
                elif selected != target.values["size"]:
                    return GateDecision(False, "target_size_option_not_selected")
        if "variant" in target.values:
            # A page can list several variants in its description. If the
            # requested variant is a selectable option, its presence alone is
            # insufficient: the actual selected option must match it.
            matching = [
                option_type for label, option_type in observation.option_types.items()
                if label.casefold() == target.values["variant"].casefold()
            ]
            if matching and not any(
                observation.selected_options.get(option_type, "").casefold()
                == target.values["variant"].casefold()
                for option_type in matching
            ):
                return GateDecision(False, "target_variant_option_not_selected")
        return GateDecision(True, "eligible")
