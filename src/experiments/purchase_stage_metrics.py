"""Auditable retrieval -> search -> click -> purchase diagnostics.

Search direction is a conservative lexical heuristic, not a causal attribution.
Only executed episode Actions are inspected; copied history prompts and model
thoughts are deliberately excluded. All unknown measurements remain JSON null.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
_ACTION = re.compile(r"^(?:Action:\s*)?(search|click|think)\[(.*)\]\s*$", re.S)
_GENERIC_LEADING_WORDS = {
    "a", "an", "the", "matching", "family", "womens", "women", "mens", "men",
    "blue", "black", "white", "red", "green", "windows", "window", "eggless",
    "raw", "edible", "baby", "professional", "portable", "wireless", "pieces",
    "fabric", "ptz", "security", "outdoor", "new", "pack", "set", "bundle",
}
_CATEGORIES = {
    "monitor": ("monitor", "monitors"),
    "camera": ("camera", "cameras", "camcorder", "camcorders"),
    "pillow": ("pillow", "pillows", "cushion", "cushions"),
    "chair": ("chair", "chairs", "stool", "stools"),
    "candy": ("candy", "candies"),
    "cookie": ("cookie", "cookies", "cookie dough"),
    "blinds": ("blind", "blinds", "window shade", "window shades", "pleated shades"),
    "curtain": ("curtain", "curtains", "drape", "drapes"),
    "headband": ("headband", "headbands", "hairband", "hairbands"),
    "hair_extension": ("hair extension", "hair extensions"),
    "pajamas": ("pajamas", "pyjamas", "sleepwear", "nightgown", "nightgowns"),
    "jeans": ("jeans", "denim pants", "denim trousers"),
    "hand_wash": ("hand wash", "handwash", "hand soap", "hand soaps"),
    "shampoo": ("shampoo", "shampoos"),
    "microphone": ("microphone", "microphones"),
    "speaker": ("speaker", "speakers"),
    "floss": ("floss", "flossing", "floss picks"),
    "toothbrush": ("toothbrush", "toothbrushes", "tooth brush", "tooth brushes"),
}


def _normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold()))


def _has_phrase(text: str, phrase: str) -> bool:
    phrase = _normalized(phrase)
    return bool(phrase and f" {phrase} " in f" {_normalized(text)} ")


def _brand(title: str, explicit: str | None) -> tuple[str | None, str]:
    if explicit:
        return _normalized(explicit), "supplied_brand"
    first = _normalized(title).split()[:1]
    if first and len(first[0]) >= 3 and first[0].isalpha() and first[0] not in _GENERIC_LEADING_WORDS:
        return first[0], "title_leading_token_heuristic"
    return None, "not_available"


def _categories(title: str, category: str | None) -> set[str]:
    text = f"{title} {category or ''}"
    return {name for name, aliases in _CATEGORIES.items() if any(_has_phrase(text, a) for a in aliases)}


def classify_search_direction(
    query: str | None, *, source_asin: str, reference_asin: str | None = None,
    source_title: str = "", reference_title: str = "", source_brand: str | None = None,
    reference_brand: str | None = None, source_category: str | None = None,
    reference_category: str | None = None,
) -> dict[str, Any]:
    """Return source/reference/mixed/unknown direction with reviewable evidence.

    Exact product IDs override weaker lexical signals when only one ID occurs.
    Otherwise source-exclusive brands and category phrases vote independently;
    conflicting votes stay unknown. Shared signals do not vote. Words such as
    'wash' or 'care' alone never stand for 'hand wash' or for a product identity.
    """
    result = {
        "query": query, "biased_toward_source": None, "classification": "no_search" if query is None else "unknown",
        "source_evidence": [], "reference_evidence": [], "shared_evidence": [],
        "reason": "no_executed_search" if query is None else "no_distinguishing_lexical_evidence",
        "method": "exact_asin_else_exclusive_brand_and_category_v1",
        "causal_attribution": False,
    }
    if query is None:
        return result
    source_id = _has_phrase(query, source_asin)
    reference_id = bool(reference_asin and _has_phrase(query, reference_asin))
    if source_id or reference_id:
        if source_id:
            result["source_evidence"].append({"kind": "exact_asin", "value": source_asin})
        if reference_id:
            result["reference_evidence"].append({"kind": "exact_asin", "value": reference_asin})
        result.update(
            classification="mixed" if source_id and reference_id else "source" if source_id else "reference",
            biased_toward_source=None if source_id and reference_id else source_id,
            reason="both_product_ids" if source_id and reference_id else "exact_product_id",
        )
        return result
    a_brand, a_origin = _brand(source_title, source_brand)
    b_brand, b_origin = _brand(reference_title, reference_brand)
    for side, brand, other, origin in (("source", a_brand, b_brand, a_origin), ("reference", b_brand, a_brand, b_origin)):
        if brand and _has_phrase(query, brand):
            evidence = {"kind": "brand", "value": brand, "origin": origin}
            destination = result["shared_evidence" if brand == other else f"{side}_evidence"]
            if not any(item["kind"] == "brand" and item["value"] == brand for item in destination):
                destination.append(evidence)
    a_categories = _categories(source_title, source_category)
    b_categories = _categories(reference_title, reference_category)
    for category in sorted(a_categories | b_categories):
        aliases = [alias for alias in _CATEGORIES[category] if _has_phrase(query, alias)]
        if not aliases:
            continue
        evidence = {"kind": "category", "value": category, "matched_phrases": aliases}
        key = "shared_evidence" if category in a_categories & b_categories else "source_evidence" if category in a_categories else "reference_evidence"
        result[key].append(evidence)
    a_hit, b_hit = bool(result["source_evidence"]), bool(result["reference_evidence"])
    if a_hit or b_hit:
        result.update(
            classification="mixed" if a_hit and b_hit else "source" if a_hit else "reference",
            biased_toward_source=None if a_hit and b_hit else a_hit,
            reason="conflicting_source_and_reference_signals" if a_hit and b_hit else "exclusive_source_lexical_evidence" if a_hit else "exclusive_reference_lexical_evidence",
        )
    elif result["shared_evidence"]:
        result.update(classification="shared", reason="shared_signals_do_not_identify_source")
    return result


def _executed_actions(episode: Mapping[str, Any]) -> tuple[list[tuple[str, str]], bool]:
    actions = episode.get("Actions")
    if not isinstance(actions, list):
        return [], False
    parsed = []
    for entry in actions:
        if not isinstance(entry, str) or entry.lstrip().startswith("Observation:"):
            continue
        # An Actions item is one actual action, never an embedded transcript.
        if re.search(r"\n\s*(?:Action:|Observation:)", entry):
            continue
        match = _ACTION.fullmatch(entry.strip())
        if match:
            parsed.append((match[1], match[2].strip()))
    return parsed, True


def _any_known(values: Iterable[bool | None]) -> bool | None:
    values = list(values)
    if True in values:
        return True
    return False if values and all(v is False for v in values) else None


def _rank(entry: Mapping[str, Any], source_id: str) -> tuple[int | None, bool]:
    ranking = entry.get("Ranking")
    if not isinstance(ranking, list):
        return None, False
    for position, item in enumerate(ranking, 1):
        if isinstance(item, dict) and str(item.get("Id", "")) == source_id:
            rank = item.get("rank", position)
            return (rank if isinstance(rank, int) and rank > 0 else position), True
    return None, True


def purchase_stage_metrics(
    episode: Mapping[str, Any], *, source_id: str, source_asin: str,
    reference_asin: str | None = None, source_title: str = "", reference_title: str = "",
    source_brand: str | None = None, reference_brand: str | None = None,
    source_category: str | None = None, reference_category: str | None = None,
) -> dict[str, Any]:
    """Measure one saved shopping episode without parsing any copied prompt.

    Actual display uses executed PolicyTrace SelectedIds (these are post-budget
    IDs in current logs). Image display uses InputImages, with MemoryImages only
    as a fallback when MemoryImageInput is explicitly true. Ranking and selection
    are separate: a source ranked second is in the top three even with top-k=1.
    """
    actions, action_known = _executed_actions(episode)
    searches = [value for name, value in actions if name == "search"]
    clicked = [value for name, value in actions if name == "click"]
    search = classify_search_direction(
        searches[0] if searches else None, source_asin=source_asin, reference_asin=reference_asin,
        source_title=source_title, reference_title=reference_title, source_brand=source_brand,
        reference_brand=reference_brand, source_category=source_category, reference_category=reference_category,
    )
    if not action_known:
        search.update(classification="unknown", reason="executed_actions_not_recorded")
    retrieval = episode.get("RetrievalTrace")
    retrieval_known = isinstance(retrieval, list)
    steps = []
    for entry in retrieval if retrieval_known else []:
        if not isinstance(entry, dict):
            continue
        rank, rank_known = _rank(entry, source_id)
        selected = entry.get("SelectedIds")
        steps.append({
            "step": entry.get("Step"), "source_rank": rank, "ranking_recorded": rank_known,
            "source_in_top_three": rank <= 3 if rank is not None else False if rank_known else None,
            "source_ranked_first": rank == 1 if rank is not None else False if rank_known else None,
            "source_selected": source_id in selected if isinstance(selected, list) else None,
        })
    policy = episode.get("PolicyTrace")
    policy_known = isinstance(policy, list)
    displays = []
    for entry in policy if policy_known else []:
        if not isinstance(entry, dict) or entry.get("Executed") is not True:
            continue
        selected = entry.get("SelectedIds")
        images = entry.get("InputImages")
        image_source = "InputImages"
        if not isinstance(images, list):
            images = entry.get("MemoryImages") if entry.get("MemoryImageInput") is True else None
            image_source = "MemoryImages_with_image_input_enabled"
        displays.append({
            "step": entry.get("Step"),
            "source_displayed": source_id in selected if isinstance(selected, list) else None,
            "source_image_displayed": any(isinstance(item, dict) and item.get("Id") == source_id for item in images) if isinstance(images, list) else None,
            "image_evidence_field": image_source if isinstance(images, list) else None,
        })
    completion = episode.get("Completed")
    final = episode.get("FinalASIN")
    if completion is False:
        bought = False
    elif completion is True and isinstance(final, str) and final:
        bought = final.strip().casefold() == source_asin.casefold()
    else:
        bought = None
    def retrieval_flag(key):
        return _any_known(s[key] for s in steps) if steps else False if retrieval_known else None
    first = steps[0] if steps else {}
    return {
        "schema_version": SCHEMA_VERSION, "source_id": source_id, "source_asin": source_asin,
        "reference_asin": reference_asin, "first_source_rank": first.get("source_rank"),
        "source_in_top_three_first": first.get("source_in_top_three", False if retrieval_known else None),
        "source_in_top_three_any": retrieval_flag("source_in_top_three"),
        "source_ranked_first_first": first.get("source_ranked_first", False if retrieval_known else None),
        "source_ranked_first_any": retrieval_flag("source_ranked_first"),
        "source_selected_first": first.get("source_selected", False if retrieval_known else None),
        "source_selected_any": retrieval_flag("source_selected"),
        "source_displayed_first": displays[0]["source_displayed"] if displays else None,
        "source_displayed_any": _any_known(d["source_displayed"] for d in displays),
        "source_image_displayed_any": _any_known(d["source_image_displayed"] for d in displays),
        "first_search": searches[0] if searches else None,
        "first_search_biased_toward_source": search["biased_toward_source"],
        "first_search_direction": search,
        "clicked_exact_source": any(value.casefold() == source_asin.casefold() for value in clicked) if action_known else None,
        "purchased_exact_source": bought,
        "final_asin": final, "completed": completion,
        "audit": {"executed_actions_recorded": action_known, "search_count": len(searches) if action_known else None,
                  "retrieval_trace_recorded": retrieval_known, "policy_trace_recorded": policy_known,
                  "retrieval_steps": steps, "executed_policy_displays": displays,
                  "action_evidence": "episode.Actions_only", "search_direction_is_heuristic": True,
                  "private_reasoning_or_causal_attribution": False},
    }


def summarize_purchase_stage_metrics(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate booleans with explicit measured and unknown denominators."""
    metrics = list(metrics)
    keys = (
        "source_in_top_three_first", "source_in_top_three_any", "source_ranked_first_first",
        "source_ranked_first_any", "source_selected_first", "source_selected_any",
        "source_displayed_first", "source_displayed_any", "source_image_displayed_any",
        "first_search_biased_toward_source", "clicked_exact_source", "purchased_exact_source",
    )
    summary = {"schema_version": SCHEMA_VERSION, "episodes": len(metrics), "measurements": {}}
    for key in keys:
        true = sum(row.get(key) is True for row in metrics)
        false = sum(row.get(key) is False for row in metrics)
        measured = true + false
        summary["measurements"][key] = {"true": true, "false": false, "unknown": len(metrics) - measured,
                                              "measured": measured, "rate": true / measured if measured else None}
    classifications = {}
    for row in metrics:
        value = row.get("first_search_direction", {}).get("classification", "unknown")
        classifications[value] = classifications.get(value, 0) + 1
    summary["first_search_classifications"] = classifications
    return summary
