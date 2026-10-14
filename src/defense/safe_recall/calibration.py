"""Statistical primitives used only with properly separated calibration data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SplitConformalIntent:
    """Marginally valid only under exchangeability and candidate coverage.

    This helper does not grant authority by itself. A downstream authorizer
    intersects permissions across *all* retained hypotheses.
    """

    threshold: float
    alpha: float
    calibration_count: int

    @classmethod
    def fit(cls, true_intent_scores: tuple[float, ...],
            alpha: float) -> "SplitConformalIntent":
        if not true_intent_scores or not 0 < alpha < 1:
            raise ValueError("Independent calibration scores and alpha required")
        if any(not math.isfinite(x) for x in true_intent_scores):
            raise ValueError("Scores must be finite")
        n = len(true_intent_scores)
        rank = math.ceil((n + 1) * (1 - alpha))
        threshold = (math.inf if rank > n
                     else sorted(true_intent_scores)[rank - 1])
        return cls(threshold, alpha, n)

    def plausible(self, candidate_scores: Mapping[object, float]
                  ) -> tuple[object, ...]:
        return tuple(key for key, score in candidate_scores.items()
                     if score <= self.threshold)
