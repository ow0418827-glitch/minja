"""Small immutable contracts at every trust boundary.

Only Request fields may cross from the experiment driver into the defense.
Hidden goals, rewards, attack metadata and filesystem names have no slot here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class Mode(str, Enum):
    CURRENT = "current"
    REPURCHASE = "repurchase"
    PREFERENCE = "preference"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class Status(str, Enum):
    PURCHASED = "purchased"
    NEEDS_EVIDENCE = "needs_evidence"
    NEEDS_CLARIFICATION = "needs_clarification"
    NO_MATCH = "no_match"
    BUDGET_EXHAUSTED = "budget_exhausted"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Request:
    subject: str
    text: str
    image_bytes: bytes | None = field(default=None, repr=False)
    image_mime: str | None = None

    def __post_init__(self) -> None:
        if not self.subject or not self.text.strip():
            raise ValueError("A subject and the actual current user text are required")
        if self.image_bytes is not None and not self.image_mime:
            raise ValueError("Image MIME type is required with image bytes")


@dataclass(frozen=True, order=True)
class FieldReference:
    """A narrow query authorized by a *current* request interpretation."""

    kind: str  # order or confirmation
    field: str  # target_identity, size, caffeine, ...
    category: str
    selector: str  # last_completed_order or latest_confirmation
    purpose: str = "target"
    quote: str = ""  # exact supporting words in the current user request

    def __post_init__(self) -> None:
        if self.kind not in {"order", "confirmation"}:
            raise ValueError("Unsupported history source")
        allowed = {
            "order": {"target_identity", "variant"},
            "confirmation": {"size", "caffeine", "preference"},
        }
        if self.field not in allowed[self.kind]:
            raise ValueError("Historical field cannot inherit this authority")
        if self.selector not in {"last_completed_order", "latest_confirmation"}:
            raise ValueError("Unsupported history reference")
        if self.kind == "order" and self.selector != "last_completed_order":
            raise ValueError("Order selector mismatch")
        if self.kind == "confirmation" and self.selector != "latest_confirmation":
            raise ValueError("Confirmation selector mismatch")
        if not all((self.field, self.category, self.purpose)):
            raise ValueError("Field reference must be scoped")


@dataclass(frozen=True)
class IntentHypothesis:
    mode: Mode
    current_fields: tuple[tuple[str, str], ...] = ()
    references: tuple[FieldReference, ...] = ()
    search_terms: str = ""

    def __post_init__(self) -> None:
        if len({key for key, _ in self.current_fields}) != len(self.current_fields):
            raise ValueError("Duplicate current field")
        if any(not key or not value for key, value in self.current_fields):
            raise ValueError("Empty current field")


@dataclass(frozen=True)
class Contract:
    subject: str
    current_fields: Mapping[str, str]
    references: tuple[FieldReference, ...]
    search_terms: str
    unresolved: tuple[str, ...]
    version: int = 1


@dataclass(frozen=True)
class AdmittedField:
    field: str
    value: str
    purpose: str
    source_id: str
    source_version: int
    source_kind: str
    risk_score: float
    scope: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Target:
    subject: str
    values: Mapping[str, str]
    admitted: Mapping[str, AdmittedField]
    authorization_version: int


@dataclass(frozen=True)
class Observation:
    page: str
    text: str
    version: int
    page_number: int = 1
    has_next: bool = False
    visible_ids: tuple[str, ...] = ()
    handle: str | None = None
    price: float | None = None
    option_types: Mapping[str, str] = field(default_factory=dict)
    selected_options: Mapping[str, str] = field(default_factory=dict)
    image_bytes: bytes | None = field(default=None, repr=False)
    image_mime: str | None = None


@dataclass(frozen=True)
class Assessment:
    """A model score is a ranking feature, not a certified error probability."""

    match_score: float
    observed_fields: Mapping[str, str]
    evidence_spans: Mapping[str, str]
    missing_fields: tuple[str, ...] = ()
    conflict: bool = False
    suggested_options: Mapping[str, str] = field(default_factory=dict)
    size_chart: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.match_score <= 1:
            raise ValueError("Match score outside [0, 1]")


@dataclass(frozen=True)
class Candidate:
    handle: str
    observation: Observation
    assessment: Assessment


@dataclass(frozen=True)
class Outcome:
    status: Status
    reason: str
    final_handle: str | None = None
    final_options: Mapping[str, str] = field(default_factory=dict)
    steps: int = 0
    protocol: str = "safe_recall_v3"
    audit: tuple[Mapping[str, object], ...] = ()
