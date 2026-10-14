"""Actual sampled observation policies with a natural-log KL budget."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from .compiler import Op, ProcedureHint


def divergence(p: Mapping[Op, float], q: Mapping[Op, float]) -> float:
    if set(p) != set(q):
        raise ValueError("Policy support differs")
    return sum(value * math.log(value / q[op]) for op, value in p.items()
               if value > 0)


def project(reference: Mapping[Op, float], advantage: Mapping[Op, float],
            budget: float) -> dict[Op, float]:
    if budget < 0 or not reference or any(v <= 0 for v in reference.values()):
        raise ValueError("Invalid reference distribution or budget")
    total = sum(reference.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Reference distribution must sum to one")
    if budget == 0 or len(reference) == 1:
        return dict(reference)
    values = {op: max(-1.0, min(1.0, advantage.get(op, 0.0)))
              for op in reference}
    if len(set(values.values())) == 1:
        return dict(reference)

    def tilted(inverse_temp: float) -> dict[Op, float]:
        logits = {op: math.log(q) + inverse_temp * values[op]
                  for op, q in reference.items()}
        offset = max(logits.values())
        weights = {op: math.exp(value - offset) for op, value in logits.items()}
        normalizer = sum(weights.values())
        return {op: weight / normalizer for op, weight in weights.items()}

    low, high = 0.0, 1.0
    while high < 1e6 and divergence(tilted(high), reference) < budget:
        high *= 2
    if high >= 1e6 and divergence(tilted(high), reference) <= budget:
        return tilted(high)
    for _ in range(70):
        mid = (low + high) / 2
        if divergence(tilted(mid), reference) <= budget:
            low = mid
        else:
            high = mid
    answer = tilted(low)
    if divergence(answer, reference) > budget + 1e-10:
        raise ArithmeticError("Numerical projection exceeded KL budget")
    return answer


@dataclass
class Scheduler:
    total_budget: float
    max_draws: int
    rng: random.Random
    spent_upper_bound: float = 0.0
    draws: int = 0

    def choose(self, allowed: tuple[Op, ...], hint: ProcedureHint) -> tuple[Op, dict[str, object]]:
        if not allowed or self.draws >= self.max_draws:
            raise ValueError("No observation draw available")
        reference = {op: 1.0 / len(allowed) for op in allowed}
        remaining = max(0.0, self.total_budget - self.spent_upper_bound)
        allocation = remaining / (self.max_draws - self.draws)
        benefit = {op: 1.0 if op in hint.preferred else 0.0 for op in allowed}
        actual = project(reference, benefit, allocation)
        value = self.rng.random()
        cumulative = 0.0
        selected = allowed[-1]
        for op, probability in actual.items():
            cumulative += probability
            if value < cumulative:
                selected = op
                break
        measured = divergence(actual, reference)
        self.draws += 1
        # Allocate per-path upper bounds, not only realized divergence.
        self.spent_upper_bound += allocation
        if self.spent_upper_bound > self.total_budget + 1e-10:
            raise ArithmeticError("Path KL allocation exceeded budget")
        return selected, {
            "operation": selected.value,
            "reference": {op.value: v for op, v in reference.items()},
            "actual": {op.value: v for op, v in actual.items()},
            "kl": measured,
            "allocated": allocation,
            "path_allocated": self.spent_upper_bound,
        }
