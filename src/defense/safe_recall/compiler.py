"""A tiny operation grammar for ordinary, potentially poisoned memories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Op(str, Enum):
    ATTRIBUTES = "Attributes"
    DESCRIPTION = "Description"
    FEATURES = "Features"


@dataclass(frozen=True)
class ProcedureHint:
    preferred: tuple[Op, ...]


def compile_procedure(records: Iterable[object]) -> ProcedureHint:
    """Accept only pre-typed observation hints; never parse commands or values.

    Raw text and image records deliberately produce no capability. A separate
    controlled compiler may propose this enum, but it cannot carry an ASIN,
    search string, target field, threshold, purchase or authorization.
    """
    hints: list[Op] = []
    for record in records:
        if isinstance(record, ProcedureHint):
            hints.extend(record.preferred)
    return ProcedureHint(tuple(dict.fromkeys(hints)))
