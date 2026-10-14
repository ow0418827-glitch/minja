"""Scope-checked historical reference resolution and field admission."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Callable

from .authorization import permits
from .contracts import AdmittedField, Contract, FieldReference
from .providers import HmacVerifier, SourceProvider, SourceState, canonical


@dataclass(frozen=True)
class Admission:
    field: AdmittedField | None
    reason: str


class HistoryResolver:
    def __init__(self, provider: SourceProvider, verifier: HmacVerifier,
                 *, clock: Callable[[], float], initial_risk: float = 0.02,
                 confirmation_hazard_per_day: float = 0.001,
                 max_risk: float = 0.25,
                 max_receipt_age_seconds: float = 300):
        if not 0 <= initial_risk <= 1 or not 0 <= max_risk <= 1:
            raise ValueError("Risk settings must be in [0, 1]")
        self.provider = provider
        self.verifier = verifier
        self.clock = clock
        self.initial_risk = initial_risk
        self.hazard = confirmation_hazard_per_day
        self.max_risk = max_risk
        self.max_receipt_age_seconds = max_receipt_age_seconds

    def resolve(self, contract: Contract, reference: FieldReference) -> Admission:
        if not permits(contract, reference, subject=contract.subject,
                       purpose=reference.purpose):
            return Admission(None, "field_not_authorized")
        response = self.provider.fetch(contract.subject, reference)
        if response.state != SourceState.AVAILABLE:
            return Admission(None, "source_unavailable")
        receipt = response.receipt
        if receipt is None or not self.verifier.receipt(receipt):
            return Admission(None, "query_receipt_unverified")
        if not receipt.complete:
            return Admission(None, "query_incomplete")
        if (receipt.subject, receipt.kind, receipt.category, receipt.selector) != (
            contract.subject, reference.kind, reference.category, reference.selector
        ):
            return Admission(None, "query_scope_mismatch")
        now = self.clock()
        if (receipt.issued_at > now
                or now - receipt.issued_at > self.max_receipt_age_seconds):
            return Admission(None, "query_receipt_stale")
        records = response.records
        if tuple(sorted(receipt.record_ids)) != tuple(sorted(r.record_id for r in records)):
            return Admission(None, "query_record_set_mismatch")
        digests = {record.record_id: hashlib.sha256(
            canonical(record.payload())).hexdigest() for record in records}
        if dict(receipt.record_digests) != digests:
            return Admission(None, "query_record_digest_mismatch")
        if len({r.record_id for r in records}) != len(records):
            return Admission(None, "duplicate_record")
        scoped = []
        for record in records:
            if not self.verifier.record(record):
                return Admission(None, "record_unverified")
            if (record.issuer != receipt.issuer
                    or record.subject != contract.subject
                    or record.kind != reference.kind
                    or record.category != reference.category):
                return Admission(None, "record_scope_mismatch")
            if record.event_time > now or record.known_time > now:
                return Admission(None, "record_from_future")
            if record.version < 1:
                return Admission(None, "invalid_version")
            if record.status not in {"active", "completed", "revoked", "canceled"}:
                return Admission(None, "unknown_record_status")
            if reference.kind == "order":
                if record.status == "completed":
                    scoped.append(record)
            elif reference.field in record.claims or record.status in {
                "revoked", "canceled"
            }:
                scoped.append(record)
        if not scoped:
            return Admission(None, "no_active_matching_record")
        latest_time = max(r.event_time for r in scoped)
        latest = [r for r in scoped if r.event_time == latest_time]
        if any(r.status in {"revoked", "canceled"} for r in latest):
            return Admission(None, "reference_revoked")
        if reference.kind == "confirmation" and any(
            r.valid_from > now or
            (r.valid_until is not None and now >= r.valid_until)
            for r in latest
        ):
            return Admission(None, "confirmation_expired")
        if any(reference.field not in r.claims for r in latest):
            return Admission(None, "latest_record_field_missing")
        values = {r.claims[reference.field] for r in latest}
        if len(values) != 1:
            return Admission(None, "reference_conflict")
        # A signed event fact does not decay. Current applicability of an old
        # confirmation does; this model is a configurable estimate, not a
        # statistical certificate.
        age_days = max(0.0, (now - latest_time) / 86400)
        stability = (math.exp(-self.hazard * age_days)
                     if reference.kind == "confirmation" else 1.0)
        risk = min(1.0, self.initial_risk + 1.0 - stability)
        if risk > self.max_risk:
            return Admission(None, "historical_applicability_uncertain")
        selected = min(latest, key=lambda r: r.record_id)
        scope = {}
        if reference.field == "size":
            size_scope = selected.claims.get("size_scope")
            if size_scope == "brand_label":
                brand = selected.claims.get("size_brand", "").strip()
                if not brand:
                    return Admission(None, "size_brand_scope_missing")
                scope = {"kind": "brand_label", "brand": brand}
            elif size_scope == "body_measurement":
                try:
                    low = float(selected.claims["size_min_cm"])
                    high = float(selected.claims["size_max_cm"])
                except (KeyError, TypeError, ValueError):
                    return Admission(None, "body_measurement_invalid")
                dimension = selected.claims.get("size_dimension", "")
                if dimension not in {"chest", "waist", "hip"} or not 30 <= low <= high <= 200:
                    return Admission(None, "body_measurement_invalid")
                scope = {
                    "kind": "body_measurement", "dimension": dimension,
                    "min_cm": str(low), "max_cm": str(high),
                }
            else:
                return Admission(None, "size_scope_unverified")
        return Admission(
            AdmittedField(reference.field, selected.claims[reference.field],
                          reference.purpose, selected.record_id, selected.version,
                          selected.kind, risk, scope),
            "accepted",
        )
