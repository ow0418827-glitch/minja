"""Authenticated source contracts.

An HMAC authenticates what a configured source asserted. It does not prove
that an order was semantically correct, a preference remains applicable, or
that the user authorized this task. Query receipts also authenticate scope
and completeness; a signed old order alone cannot establish "last".
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol

from .contracts import FieldReference


def canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


class SourceState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SignedRecord:
    record_id: str
    issuer: str
    subject: str
    kind: str
    category: str
    claims: Mapping[str, str]
    event_time: float
    known_time: float
    valid_from: float
    valid_until: float | None
    version: int
    status: str = "active"
    fixture: bool = False
    signature: str = field(default="", repr=False)

    def payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id, "issuer": self.issuer,
            "subject": self.subject, "kind": self.kind,
            "category": self.category, "claims": dict(self.claims),
            "event_time": self.event_time, "known_time": self.known_time,
            "valid_from": self.valid_from, "valid_until": self.valid_until,
            "version": self.version, "status": self.status,
            "fixture": self.fixture,
        }


@dataclass(frozen=True)
class QueryReceipt:
    issuer: str
    subject: str
    kind: str
    category: str
    selector: str
    snapshot: str
    record_ids: tuple[str, ...]
    complete: bool
    issued_at: float
    record_digests: Mapping[str, str]
    fixture: bool = False
    signature: str = field(default="", repr=False)

    def payload(self) -> dict[str, object]:
        return {
            "issuer": self.issuer, "subject": self.subject,
            "kind": self.kind, "category": self.category,
            "selector": self.selector, "snapshot": self.snapshot,
            "record_ids": list(self.record_ids), "complete": self.complete,
            "issued_at": self.issued_at,
            "record_digests": dict(self.record_digests),
            "fixture": self.fixture,
        }


@dataclass(frozen=True)
class SourceResponse:
    state: SourceState
    records: tuple[SignedRecord, ...] = ()
    receipt: QueryReceipt | None = None


class SourceProvider(Protocol):
    def fetch(self, subject: str, reference: FieldReference) -> SourceResponse: ...


class UnavailableSource:
    def fetch(self, subject: str, reference: FieldReference) -> SourceResponse:
        return SourceResponse(SourceState.UNAVAILABLE)


class SignedJsonSource:
    """Read an externally issued, signed snapshot. This class cannot sign it.

    The file is re-read for every reference and before submission. Its receipt
    must bind subject, query scope, complete record IDs and issue time.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def fetch(self, subject: str, reference: FieldReference) -> SourceResponse:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return SourceResponse(SourceState.UNAVAILABLE)
        if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
            return SourceResponse(SourceState.UNAVAILABLE)
        for row in data["queries"]:
            try:
                receipt = QueryReceipt(**row["receipt"])
                if (receipt.subject, receipt.kind, receipt.category,
                        receipt.selector) != (subject, reference.kind,
                                             reference.category,
                                             reference.selector):
                    continue
                records = tuple(SignedRecord(**item)
                                for item in row["records"])
                return SourceResponse(SourceState.AVAILABLE, records, receipt)
            except (KeyError, TypeError, ValueError):
                return SourceResponse(SourceState.UNAVAILABLE)
        return SourceResponse(SourceState.UNAVAILABLE)


class HmacVerifier:
    """Pinned source keys supplied by the host, never by a memory record."""

    def __init__(self, keys: Mapping[str, bytes], *, allow_fixture: bool = False):
        self._keys = dict(keys)
        self.allow_fixture = allow_fixture

    def _valid(self, issuer: str, payload: Mapping[str, object],
               signature: str, fixture: bool) -> bool:
        key = self._keys.get(issuer)
        if not key or not signature or (fixture and not self.allow_fixture):
            return False
        expected = hmac.new(key, canonical(payload), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def record(self, record: SignedRecord) -> bool:
        return self._valid(record.issuer, record.payload(), record.signature,
                           record.fixture)

    def receipt(self, receipt: QueryReceipt) -> bool:
        return self._valid(receipt.issuer, receipt.payload(), receipt.signature,
                           receipt.fixture)


def sign_fixture_record(record: SignedRecord, key: bytes) -> SignedRecord:
    """Test helper; the fixture marker is authenticated and rejected by default."""
    if not record.fixture:
        raise ValueError("Fixture signing requires fixture=True")
    signature = hmac.new(key, canonical(record.payload()), hashlib.sha256).hexdigest()
    return replace(record, signature=signature)


def sign_fixture_receipt(receipt: QueryReceipt, key: bytes) -> QueryReceipt:
    if not receipt.fixture:
        raise ValueError("Fixture signing requires fixture=True")
    signature = hmac.new(key, canonical(receipt.payload()), hashlib.sha256).hexdigest()
    return replace(receipt, signature=signature)
