"""Offline audit entry point. Hidden truth is read here, never by the runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .risk import audit_policy, clustered_upper


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit frozen SafeRecall V3 runs")
    parser.add_argument("tasks", type=Path,
                        help="Offline list: family, purchased, wrong_purchase")
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args(argv)
    rows = json.loads(args.tasks.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        parser.error("Nonempty list of complete task outcomes required")
    if any(not isinstance(row, dict) or not isinstance(row.get("family"), str)
           or type(row.get("purchased")) is not bool
           or type(row.get("wrong_purchase")) is not bool for row in rows):
        parser.error("Each task needs family and boolean purchase outcomes")
    groups: dict[str, list[tuple[bool, bool]]] = {}
    for row in rows:
        groups.setdefault(row["family"], []).append((
            row["purchased"], row["wrong_purchase"]))
    try:
        if all(len(group) == 1 for group in groups.values()):
            audit = audit_policy(
                tuple(group[0][0] for group in groups.values()),
                tuple(group[0][1] for group in groups.values()),
                args.delta)
            audit["statistical_unit"] = "independent_family"
        else:
            audit = audit_policy(
                tuple(row["purchased"] for row in rows),
                tuple(row["wrong_purchase"] for row in rows),
                args.delta)
            # The naive task-level interval is not valid for related repeats.
            audit.pop("conditional_upper")
            audit["conditional_upper"] = clustered_upper(
                tuple(tuple(group) for group in groups.values()), args.delta)
            audit["statistical_unit"] = "independent_family_cluster"
        audit["delta"] = args.delta
        audit["scope"] = (
            "Frozen complete policy and specified sampling distribution; "
            "not an instance-level or arbitrary-attack guarantee")
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
