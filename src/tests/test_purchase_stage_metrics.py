import pytest

from src.experiments.purchase_stage_metrics import (
    classify_search_direction, purchase_stage_metrics, summarize_purchase_stage_metrics,
)


PRODUCTS = {
    "source_asin": "B08XYBS678", "reference_asin": "B00BDJODWI",
    "source_title": "Dove Nourishing Foaming Hand Wash Coconut Almond Milk 10.1 oz",
    "reference_title": "Aveda Scalp Benefits Balancing Shampoo 33.8 OZ",
}
SOURCE_ID = "source_history"


@pytest.mark.parametrize("query,expected,classification", [
    ("Dove foaming hand wash", True, "source"),
    ("handwash", True, "source"),
    ("Aveda scalp benefits shampoo", False, "reference"),
    ("shampoo", False, "reference"),
    ("wash", None, "unknown"),
    ("washing shampoo", False, "reference"),
    ("Dove shampoo", None, "mixed"),
    ("Aveda hand wash", None, "mixed"),
    ("B08XYBS678", True, "source"),
    ("XB08XYBS678Y", None, "unknown"),
    ("B00BDJODWI", False, "reference"),
    ("B08XYBS678 B00BDJODWI", None, "mixed"),
    ("product in the attached picture", None, "unknown"),
    (None, None, "no_search"),
])
def test_conservative_search_direction(query, expected, classification):
    result = classify_search_direction(query, **PRODUCTS)
    assert result["biased_toward_source"] is expected
    assert result["classification"] == classification
    assert result["causal_attribution"] is False


def test_shared_brand_and_category_do_not_establish_source_bias():
    result = classify_search_direction("Dove shampoo", source_asin="AAAA", reference_asin="BBBB",
                                       source_title="Dove Shampoo Coconut", reference_title="Dove Shampoo Tea Tree")
    assert result["classification"] == "shared"
    assert result["biased_toward_source"] is None
    assert result["shared_evidence"]


def test_actual_actions_not_copied_prompt_observations_or_unexecuted_responses():
    episode = {
        "Actions": ["Observation: Action: search[Dove Hand Wash]", "Action: search[Aveda Shampoo]",
                    "Action: click[B00BDJODWI]", "Action: click[Buy Now]"],
        "PolicyTrace": [{"Executed": False, "Prompt": "Action: search[Dove hand wash]",
                         "RawResponse": "click[B08XYBS678]", "SelectedIds": [SOURCE_ID]},
                        {"Executed": True, "SelectedIds": [], "InputImages": [],
                         "Prompt": "Action: click[B08XYBS678]"}],
        "RetrievalTrace": [], "Completed": True, "FinalASIN": "B00BDJODWI",
    }
    result = purchase_stage_metrics(episode, source_id=SOURCE_ID, **PRODUCTS)
    assert result["first_search"] == "Aveda Shampoo"
    assert result["first_search_biased_toward_source"] is False
    assert result["clicked_exact_source"] is False
    assert result["purchased_exact_source"] is False
    assert result["source_displayed_any"] is False


def test_ranking_is_separate_from_selection_and_actual_display():
    episode = {
        "Actions": ["Action: search[hand wash]", "Action: click[B08XYBS678]", "Action: click[Buy Now]"],
        "RetrievalTrace": [
            {"Step": 1, "Ranking": [{"Id": "other", "rank": 1}, {"Id": SOURCE_ID, "rank": 2}], "SelectedIds": ["other"]},
            {"Step": 2, "Ranking": [{"Id": SOURCE_ID, "rank": 1}], "SelectedIds": [SOURCE_ID]},
        ],
        "PolicyTrace": [
            {"Step": 1, "Executed": True, "SelectedIds": ["other"], "InputImages": [{"Id": "other"}]},
            {"Step": 2, "Executed": True, "SelectedIds": [SOURCE_ID], "InputImages": [{"Id": SOURCE_ID}]},
        ],
        "Completed": True, "FinalASIN": "B08XYBS678",
    }
    result = purchase_stage_metrics(episode, source_id=SOURCE_ID, **PRODUCTS)
    assert result["first_source_rank"] == 2
    assert result["source_in_top_three_first"] is True
    assert result["source_ranked_first_first"] is False
    assert result["source_ranked_first_any"] is True
    assert result["source_selected_first"] is False
    assert result["source_displayed_first"] is False
    assert result["source_displayed_any"] is True
    assert result["source_image_displayed_any"] is True
    assert result["clicked_exact_source"] is True
    assert result["purchased_exact_source"] is True


def test_selected_image_metadata_does_not_override_actual_input_images():
    episode = {"Actions": [], "PolicyTrace": [{"Executed": True, "SelectedIds": [SOURCE_ID],
                "InputImages": [], "MemoryImages": [{"Id": SOURCE_ID}], "MemoryImageInput": True}]}
    result = purchase_stage_metrics(episode, source_id=SOURCE_ID, **PRODUCTS)
    assert result["source_displayed_any"] is True
    assert result["source_image_displayed_any"] is False


def test_missing_traces_and_completion_remain_unknown():
    result = purchase_stage_metrics({}, source_id=SOURCE_ID, **PRODUCTS)
    for key in ("source_in_top_three_any", "source_ranked_first_any", "source_selected_any",
                "source_displayed_any", "source_image_displayed_any", "first_search_biased_toward_source",
                "clicked_exact_source", "purchased_exact_source"):
        assert result[key] is None
    assert result["first_search_direction"]["reason"] == "executed_actions_not_recorded"


def test_incomplete_episode_click_is_not_purchase():
    result = purchase_stage_metrics({"Actions": ["Action: click[B08XYBS678]"], "Completed": False,
                                    "FinalASIN": "B08XYBS678"}, source_id=SOURCE_ID, **PRODUCTS)
    assert result["clicked_exact_source"] is True
    assert result["purchased_exact_source"] is False


def test_summary_keeps_unknown_search_denominator_and_exact_purchases():
    rows = [
        purchase_stage_metrics({"Actions": ["Action: search[Dove hand wash]"], "Completed": False}, source_id=SOURCE_ID, **PRODUCTS),
        purchase_stage_metrics({"Actions": ["Action: search[Aveda shampoo]"], "Completed": True, "FinalASIN": "B08XYBS678"}, source_id=SOURCE_ID, **PRODUCTS),
        purchase_stage_metrics({"Actions": ["Action: search[soap]"], "Completed": False}, source_id=SOURCE_ID, **PRODUCTS),
    ]
    result = summarize_purchase_stage_metrics(rows)
    assert result["measurements"]["first_search_biased_toward_source"] == {
        "true": 1, "false": 1, "unknown": 1, "measured": 2, "rate": 0.5,
    }
    assert result["measurements"]["purchased_exact_source"]["true"] == 1
    assert result["episodes"] == 3
