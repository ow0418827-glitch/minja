"""Score use and complete-policy finite-sample audit.

The raw match score is not treated as a calibrated per-item probability.
"""

from __future__ import annotations

import math


def exact_upper(wrong: int, purchased: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper limit for P(wrong | purchased)."""
    if not 0 < delta < 1 or not 0 <= wrong <= purchased:
        raise ValueError("Invalid binomial counts or failure probability")
    if purchased == 0 or wrong == purchased:
        return 1.0
    if wrong == 0:
        return 1.0 - delta ** (1.0 / purchased)
    try:
        from scipy.stats import beta
        return float(beta.ppf(1.0 - delta, wrong + 1, purchased - wrong))
    except ImportError:
        # The binomial lower tail decreases in p. This small no-dependency
        # fallback is suitable for ordinary audit counts.
        def tail(p: float) -> float:
            return sum(math.comb(purchased, k) * p ** k *
                       (1.0 - p) ** (purchased - k)
                       for k in range(wrong + 1))
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = (low + high) / 2
            if tail(mid) > delta:
                low = mid
            else:
                high = mid
        return high


def audit_policy(purchased: tuple[bool, ...], wrong: tuple[bool, ...],
                 delta: float) -> dict[str, float | int]:
    if len(purchased) != len(wrong) or not purchased:
        raise ValueError("Complete independent task outcomes are required")
    if any(bad and not bought for bought, bad in zip(purchased, wrong)):
        raise ValueError("Wrong purchase cannot occur without a purchase")
    n = sum(purchased)
    e = sum(wrong)
    return {
        "tasks": len(purchased), "purchases": n, "wrong_purchases": e,
        "coverage": n / len(purchased),
        "task_wrong_rate": e / len(purchased),
        "conditional_wrong_rate": e / n if n else None,
        "conditional_upper": exact_upper(e, n, delta),
    }


def clustered_upper(groups: tuple[tuple[tuple[bool, bool], ...], ...],
                    delta: float) -> float:
    """Conservative purchase-conditional bound for repeated trials per family.

    Families must be sampled independently and each must have the same
    predeclared number of repetitions. Rewordings of one item stay one group.
    """
    if not groups or not 0 < delta < 1:
        raise ValueError("Independent groups and delta required")
    repeats = len(groups[0])
    if repeats < 1 or any(len(group) != repeats for group in groups):
        raise ValueError("Equal predeclared repetitions are required")
    if any(wrong and not purchased for group in groups
           for purchased, wrong in group):
        raise ValueError("Wrong purchase without purchase")
    a = [sum(p for p, _ in group) / repeats for group in groups]
    b = [sum(w for _, w in group) / repeats for group in groups]
    radius = math.sqrt(math.log(2 / delta) / (2 * len(groups)))
    mean_a = sum(a) / len(groups)
    mean_b = sum(b) / len(groups)
    if mean_a <= radius:
        return 1.0
    return min(1.0, (mean_b + radius) / (mean_a - radius))
