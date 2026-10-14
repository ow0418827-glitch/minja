"""Authorize a use of a historical *field*, never an entire memory item."""

from __future__ import annotations

from .contracts import Contract, FieldReference, IntentHypothesis, Mode


def intersect_hypotheses(subject: str, hypotheses: tuple[IntentHypothesis, ...],
                         current_text: str) -> Contract:
    if not hypotheses:
        return Contract(subject, {}, (), "", ("intent_unresolved",))
    fields = [dict(h.current_fields) for h in hypotheses]
    common_fields = {
        key: value for key, value in fields[0].items()
        if all(other.get(key) == value for other in fields[1:])
    }
    common_refs = set(hypotheses[0].references)
    for hypothesis in hypotheses[1:]:
        common_refs.intersection_update(hypothesis.references)
    # Explicit current values override only their own fields.
    allowed_refs = tuple(sorted(
        ref for ref in common_refs
        if ref.field not in common_fields and ref.quote
        and ref.quote.casefold() in current_text.casefold()
    ))
    all_fields = set().union(*(set(dict(h.current_fields)) for h in hypotheses))
    all_ref_fields = {ref.field for h in hypotheses for ref in h.references}
    resolved = set(common_fields) | {ref.field for ref in allowed_refs}
    unresolved = tuple(sorted((all_fields | all_ref_fields) - resolved))
    if any(h.mode == Mode.UNCERTAIN for h in hypotheses):
        unresolved = tuple(sorted(set(unresolved) | {"intent_unresolved"}))
    terms = hypotheses[0].search_terms if all(
        h.search_terms == hypotheses[0].search_terms for h in hypotheses
    ) else ""
    if len(hypotheses) > 1 and not resolved and not terms:
        unresolved = tuple(sorted(set(unresolved) | {"intent_unresolved"}))
    return Contract(subject, common_fields, allowed_refs, terms, unresolved)


def permits(contract: Contract, reference: FieldReference, *, subject: str,
            purpose: str) -> bool:
    return (
        contract.subject == subject
        and reference in contract.references
        and reference.purpose == purpose
    )
