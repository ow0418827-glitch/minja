"""Weighted fusion contracts and index identity, without loading real models."""
import copy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.vision.joint_encoder import ChunkedClipEncoder, GMEProcessEncoder, normalized
from src.vision.visual_retrieval import VisualMemoryRetriever, memory_index_text, sha256
from src.vision import visual_retrieval


def tiny_clip():
    calls = []
    tokenizer = SimpleNamespace(context_length=77, sot_token_id=1, eot_token_id=2,
                                encode=lambda text: list(text))
    base = SimpleNamespace(snapshot={"model": "tiny"}, device="cpu",
                           surrogate=SimpleNamespace(tokenizer=tokenizer),
                           encode=lambda path: calls.append("image") or np.array([1., 0.], dtype=np.float32))
    encoder = ChunkedClipEncoder(base)
    encoder.encode_text = lambda text: calls.append("text") or np.array([0., 1.], dtype=np.float32)
    return encoder, calls


def test_fusion_equal_weight_keeps_exact_legacy_arithmetic():
    encoder, calls = tiny_clip()
    legacy = normalized(np.array([1., 0.], dtype=np.float32) + np.array([0., 1.], dtype=np.float32))
    assert np.array_equal(encoder.encode_joint("text", "unused"), legacy)
    assert np.array_equal(encoder.encode_joint("text", "unused", .5), legacy)
    assert calls == ["image", "text", "image", "text"]


def test_fusion_weight_is_applied_before_final_normalization():
    encoder, calls = tiny_clip()
    vector = encoder.encode_joint("text", "unused", .8)
    assert np.array_equal(vector, normalized(np.array([.8, .2], dtype=np.float32)))
    assert calls == ["image", "text"]


@pytest.mark.parametrize("weight,expected,modality", [(0., [0., 1.], "text"), (1., [1., 0.], "image")])
def test_endpoint_skips_other_modality(weight, expected, modality):
    encoder, calls = tiny_clip()
    assert np.array_equal(encoder.encode_joint("text", "unused", weight), expected)
    assert calls == [modality]


def test_empty_text_has_explicit_image_fallback_except_text_only():
    encoder, calls = tiny_clip()
    assert np.array_equal(encoder.encode_joint(" ", "unused", .2), [1., 0.])
    assert calls == ["image"]
    with pytest.raises(ValueError, match="nonempty text"):
        encoder.encode_joint("", "unused", 0)


@pytest.mark.parametrize("weight", [-.01, 1.01, float("nan"), float("inf"), None, "wrong"])
def test_invalid_weights_fail_before_model_encoding(weight):
    encoder, calls = tiny_clip()
    with pytest.raises(ValueError, match="image_weight"):
        encoder.encode_joint("text", "unused", weight)
    assert not calls


def observed_entry(path):
    return {"Id": "history", "Instruction": "买黑色可调扶手椅", "Actions": ["click[Buy Now]", "other monitor"],
            "FinalASIN": "CHAIR123", "FinalOptions": {"Color": "Black"}, "FinalPrice": 99,
            "PrimaryImage": {"Title": "Adjustable chair", "ASIN": "CHAIR123", "LocalPath": path.name,
                             "SHA256": sha256(path)},
            "Reward": 1., "Attack": {"target": "leak"}}


def test_semantic_index_uses_observed_purchase_and_preserves_history(tmp_path):
    path = tmp_path / "picture.png"
    Image.new("RGB", (2, 2)).save(path)
    entry = observed_entry(path)
    before = copy.deepcopy(entry)
    text = memory_index_text(entry, "semantic")
    assert all(value in text for value in [entry["Instruction"], "Adjustable chair", "Black"])
    assert all(value not in text for value in ["click", "monitor", "CHAIR123", "FinalPrice", "99", "leak"])
    assert entry == before
    entry["PrimaryImage"]["ASIN"] = "OTHER"
    with pytest.raises(ValueError, match="does not match FinalASIN"):
        memory_index_text(entry, "semantic")


@pytest.mark.parametrize("missing", ["Title", "ASIN", "FinalASIN"])
def test_semantic_index_refuses_unverified_purchase_title(missing):
    entry = {"Instruction": "x", "PrimaryImage": {"Title": "title", "ASIN": "A"}, "FinalASIN": "A"}
    (entry if missing == "FinalASIN" else entry["PrimaryImage"]).pop(missing)
    with pytest.raises(ValueError, match="requires a purchased Title"):
        memory_index_text(entry, "semantic")


class WeightedEncoder:
    snapshot = {"model": "weighted-test"}
    supports_image_weight = True

    def __init__(self):
        self.calls = []

    def encode(self, path):
        self.calls.append(("image", path))
        return np.array([1., 0.], dtype=np.float32)

    def encode_joint(self, text, path, image_weight=.5):
        self.calls.append((text, image_weight))
        return normalized([image_weight, 1 - image_weight])


def test_independent_weights_and_policy_isolate_disk_caches(tmp_path):
    path = tmp_path / "picture.png"
    Image.new("RGB", (2, 2)).save(path)
    entries = [observed_entry(path)]
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(entries), encoding="utf-8")
    first = WeightedEncoder()
    retriever = VisualMemoryRetriever(first, bank, "joint", "joint", index_text="semantic",
                                      key_image_weight=.8, query_image_weight=.6)
    ranking = retriever.rank(entries, "query", path)
    assert [call[1] for call in first.calls] == [.6, .8]
    assert ranking[0]["key_image_weight"] == .8
    assert ranking[0]["query_image_weight"] == .6
    assert retriever.snapshot["index_text_policy"]["fields"] == ["Instruction", "PrimaryImage.Title", "FinalOptions"]
    second = WeightedEncoder()
    VisualMemoryRetriever(second, bank, "joint", "joint", "semantic",
                           key_image_weight=.8, query_image_weight=.6).rank(entries, "query", path)
    assert len(second.calls) == 1  # Only the query; the key is read from the disk cache.
    third = WeightedEncoder()
    changed = VisualMemoryRetriever(third, bank, "joint", "joint", "semantic",
                                    key_image_weight=.9, query_image_weight=.6)
    changed.rank(entries, "query", path)
    assert len(third.calls) == 2
    assert changed.index_identity(entries) != retriever.index_identity(entries)
    assert len(list((tmp_path / "retrieval_index_cache").glob("*.json"))) == 2
    query_changed = VisualMemoryRetriever(WeightedEncoder(), bank, "joint", "joint", "semantic",
                                          key_image_weight=.8, query_image_weight=.9)
    assert query_changed.index_identity(entries) != retriever.index_identity(entries)


def test_native_encoder_rejects_manual_fusion_before_request(tmp_path):
    encoder = GMEProcessEncoder.__new__(GMEProcessEncoder)
    encoder.snapshot = {"model": "native-test"}
    encoder._request = lambda request: pytest.fail("unsupported manual fusion must not call worker")
    with pytest.raises(ValueError, match="native joint fusion"):
        encoder.encode_joint("text", "picture.png", .8)
    with pytest.raises(ValueError, match="native joint fusion"):
        VisualMemoryRetriever(encoder, tmp_path / "bank.json", "joint", "joint", key_image_weight=.8)


def test_decoder_build_change_isolates_cache_with_same_raw_image_and_model(tmp_path, monkeypatch):
    class LocalClip(WeightedEncoder):
        snapshot = {"model": "same-model", "provider": "clip"}

    decoder = {"schema_version": 1, "scope": "local_encoder_process", "library": "Pillow",
               "verified": True, "pillow_version": "12.0.0", "jpeg_version": "9.0",
               "libjpeg_turbo_enabled": False, "libjpeg_turbo_version": None}
    inspections = []

    def inspect_decoder():
        inspections.append(True)
        return dict(decoder)

    monkeypatch.setattr(visual_retrieval, "pillow_decoder_signature", inspect_decoder)
    path = tmp_path / "picture.png"
    Image.new("RGB", (2, 2)).save(path)
    entries = [observed_entry(path)]
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(entries), encoding="utf-8")
    first = VisualMemoryRetriever(LocalClip(), bank, "joint", "joint")
    first.rank(entries, "query", path)
    first.rank(entries, "query", path)
    assert len(inspections) == 1  # Captured once, never decoding or inspecting per image.
    first_query_key = first._encode("query", path, "joint")[1]
    first_identity = first.index_identity(entries)
    second_encoder = LocalClip()
    second = VisualMemoryRetriever(second_encoder, bank, "joint", "joint")
    second.rank(entries, "query", path)
    assert len(second_encoder.calls) == 1  # Same decoder reuses disk keys.
    decoder.update(pillow_version="12.3.0", jpeg_version="8.0", libjpeg_turbo_enabled=True,
                   libjpeg_turbo_version="3.1.4")
    third_encoder = LocalClip()
    third = VisualMemoryRetriever(third_encoder, bank, "joint", "joint")
    third.rank(entries, "query", path)
    assert len(third_encoder.calls) == 2  # Changed decoder recomputes the identical raw image.
    assert third.encoder.snapshot == first.encoder.snapshot
    assert third.index_identity(entries) != first_identity
    assert third._encode("query", path, "joint")[1] != first_query_key
    assert third.snapshot["image_decoder"]["libjpeg_turbo_version"] == "3.1.4"
    assert len(list((tmp_path / "retrieval_index_cache").glob("*.json"))) == 2


def test_worker_decoder_is_not_inferred_from_parent_pillow(tmp_path, monkeypatch):
    monkeypatch.setattr(visual_retrieval, "pillow_decoder_signature",
                        lambda: pytest.fail("GME decoder cannot be inferred from parent interpreter"))
    encoder = GMEProcessEncoder.__new__(GMEProcessEncoder)
    encoder.snapshot = {"provider": "gme", "model": "worker"}
    retriever = VisualMemoryRetriever(encoder, tmp_path / "bank.json", "joint", "joint")
    assert retriever.snapshot["image_decoder"] == {
        "schema_version": 1, "scope": "encoder_worker_process", "verified": False,
        "reason": "worker_image_decoder_runtime_not_reported"}


def test_real_local_decoder_signature_reports_build_without_model_loading():
    signature = visual_retrieval.pillow_decoder_signature()
    assert signature["verified"] is True
    assert signature["scope"] == "local_encoder_process"
    assert isinstance(signature["pillow_version"], str)
    assert "jpeg_version" in signature
    assert isinstance(signature["libjpeg_turbo_enabled"], bool)
    assert "libjpeg_turbo_version" in signature
