"""Joint retrieval checks: both modalities, complete history, cache and identity."""
import hashlib
import json
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest

from src.vision.joint_encoder import (ChunkedClipEncoder, GMEProcessEncoder,
                                     make_joint_encoder, split_token_bounded_text)
from src.vision.visual_retrieval import (VisualMemoryRetriever, memory_index_text,
                                        sha256)
from src.vision.joint_encoder_worker import CheckedProcessor, FullTextGME, TextContextOverflow


def fake_clip():
    import torch

    class Tokenizer:
        context_length = 6
        sot_token_id = 90
        eot_token_id = 99

        def encode(self, text):
            return [ord(character) % 9 + 1 for character in text]

    class Model:
        def __init__(self):
            self.seen = []

        def encode_text(self, tokens):
            self.seen.extend(tokens.cpu().tolist())
            actual = tokens.float() * (tokens < 90)
            return torch.stack([actual.sum(dim=-1) + 1,
                                actual.square().sum(dim=-1) + 1], dim=-1)

    model = Model()
    base = SimpleNamespace(device="cpu", snapshot={"model": "fake-clip"},
                           surrogate=SimpleNamespace(tokenizer=Tokenizer(), model=model),
                           encode=lambda path: np.array([1., 0.], dtype=np.float32))
    return ChunkedClipEncoder(base), model


def test_clip_long_text_covers_every_token_and_tail_changes_vector():
    encoder, model = fake_clip()
    first = encoder.encode_text("aaaaaaaaaaab")
    assert encoder.last_encoding_audit["encoded_tokens"] == 12
    assert encoder.last_encoding_audit["chunks"] == 3
    assert encoder.last_encoding_audit["truncated"] is False
    payload = [token for row in model.seen for token in row if 0 < token < 90]
    assert payload == encoder.tokenizer.encode("aaaaaaaaaaab")
    second = encoder.encode_text("aaaaaaaaaaac")
    assert not np.allclose(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.)


def test_clip_joint_includes_text_and_image_equal_weights():
    encoder, _ = fake_clip()
    text = encoder.encode_text("abcd")
    image = encoder.encode("unused.png")
    expected = text + image
    expected /= np.linalg.norm(expected)
    assert np.allclose(encoder.encode_joint("abcd", "unused.png"), expected)
    assert np.allclose(encoder.encode_joint("", "unused.png"), image)
    encoder.close()
    encoder.close()
    with pytest.raises(RuntimeError, match="closed"):
        encoder.encode_joint("x", "unused.png")


def test_token_checked_character_chunks_preserve_unicode_and_tail():
    text = "牙刷 😀 颜色绿色。" * 30 + "最后必须保留"
    count = lambda value: list(value.encode("utf-8"))
    parts = split_token_bounded_text(text, count, 40)
    assert "".join(parts) == text
    assert all(len(count(part)) <= 40 for part in parts)
    assert parts[-1].endswith("最后必须保留")


def test_full_index_includes_all_actions_and_excludes_experiment_labels():
    entry = {"Id": "x", "Instruction": "Buy B", "Actions": ["think", "tail observation"],
             "FinalASIN": "B", "FinalOptions": {"color": "green"}, "FinalPrice": 20,
             "GroundTruth": "LEAK_GOAL", "Role": "LEAK_ROLE", "Attack": "LEAK_ATTACK",
             "PolicyTrace": [{"Prompt": "LEAK_TRACE"}], "Reward": 1.}
    text = memory_index_text(entry)
    assert all(part in text for part in ("Buy B", "think", "tail observation", "green"))
    assert "LEAK" not in text and "Reward" not in text
    assert memory_index_text(entry, "instruction") == "Buy B"
    with pytest.raises(ValueError, match="index_text"):
        memory_index_text(entry, "bad")


class CountingEncoder:
    def __init__(self):
        self.snapshot = {"model": "counting", "weights": 1}
        self.calls = []
        self.last_encoding_audit = None

    def encode_joint(self, text, path):
        self.calls.append((text, str(path)))
        with Image.open(path) as image:
            channel = image.getpixel((0, 0))[0]
        # Both text and image alter the direction, including historical tail.
        vector = np.array([channel + 1, sum(map(ord, text)) % 101 + 1], dtype=np.float32)
        self.last_encoding_audit = {"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                    "truncated": False}
        return vector

    def encode(self, path):
        return self.encode_joint("", path)


@pytest.fixture
def joint_bank(tmp_path):
    entries = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), (index * 30 + 10, 0, 0)).save(path)
        entries.append({"Id": f"m{index}", "Instruction": "Buy brush",
                        "Actions": [f"Observation final {index}"], "FinalASIN": f"B{index}",
                        "PrimaryImage": {"LocalPath": path.name, "SHA256": sha256(path)}})
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(entries), encoding="utf-8")
    return entries, memory_path, tmp_path / "0.png"


def test_joint_rank_uses_full_history_and_both_query_modalities(joint_bank):
    entries, memory_path, query = joint_bank
    encoder = CountingEncoder()
    retriever = VisualMemoryRetriever(encoder, memory_path, "joint", "joint")
    ranking = retriever.rank(entries, "Buy query", query)
    assert encoder.calls[0] == ("Buy query", str(query))
    assert "Observation final" in encoder.calls[1][0]
    assert len(ranking) == 5
    assert all(row["text_sha256"] and row["image_sha256"] for row in ranking)
    assert all(row["encoding_audit"]["truncated"] is False for row in ranking)
    assert all(row["query_encoding_audit"]["truncated"] is False for row in ranking)
    assert retriever.snapshot["query_encoding_audit"]["text_sha256"] == hashlib.sha256(b"Buy query").hexdigest()
    assert retriever.snapshot["index_text_policy"]["coverage"].startswith("complete")


def test_persistent_cache_reuses_keys_but_query_is_separate(joint_bank):
    entries, memory_path, query = joint_bank
    first = CountingEncoder()
    VisualMemoryRetriever(first, memory_path, "joint", "joint").rank(entries, "Buy query", query)
    assert len(first.calls) == 6
    second = CountingEncoder()
    VisualMemoryRetriever(second, memory_path, "joint", "joint").rank(entries, "Buy query", query)
    assert len(second.calls) == 1
    cached = list((memory_path.parent / "retrieval_index_cache").glob("*.json"))
    document = json.loads(cached[0].read_text(encoding="utf-8"))
    assert len(document["vectors"]) == 5
    assert "Observation final" not in cached[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("changed", ["bank", "text", "image", "model", "text_strategy"])
def test_cache_invalidation_checks_each_actual_input(joint_bank, changed):
    entries, memory_path, query = joint_bank
    encoder = CountingEncoder()
    retriever = VisualMemoryRetriever(encoder, memory_path, "joint", "joint")
    retriever.rank(entries, "query", query)
    count = len(encoder.calls)
    if changed == "bank":
        memory_path.write_text(json.dumps(entries, indent=4), encoding="utf-8")
    elif changed == "text":
        entries[0]["Actions"].append("New action at historical tail")
    elif changed == "image":
        path = memory_path.parent / entries[4]["PrimaryImage"]["LocalPath"]
        Image.new("RGB", (8, 8), (199, 0, 0)).save(path)
        entries[4]["PrimaryImage"]["SHA256"] = sha256(path)
    elif changed == "model":
        encoder.snapshot["weights"] = 2
    else:
        retriever = VisualMemoryRetriever(encoder, memory_path, "joint", "joint", index_text="instruction")
    retriever.rank(entries, "query", query)
    assert len(encoder.calls) > count


def test_top_three_keep_corresponding_whole_memories_and_embeddings(joint_bank):
    entries, memory_path, query = joint_bank
    retriever = VisualMemoryRetriever(CountingEncoder(), memory_path, "joint", "joint")
    embeddings = {"Instructions": np.arange(10).reshape(5, 2),
                  "Actions": [[i] for i in range(5)]}
    selected, encoded, ranking = retriever.shortlist(entries, embeddings, "query", query, 3)
    indexes = [row["index"] for row in ranking[:3]]
    assert selected == [entries[i] for i in indexes]
    assert len(selected) == 3
    assert encoded["Actions"] == [[i] for i in indexes]
    assert np.array_equal(encoded["Instructions"], embeddings["Instructions"][indexes])
    assert np.allclose(encoded["VisualScores"], [row["score"] for row in ranking[:3]])


def test_image_digest_failure_is_not_hidden_by_cached_embedding(joint_bank):
    entries, memory_path, query = joint_bank
    retriever = VisualMemoryRetriever(CountingEncoder(), memory_path, "joint", "joint")
    retriever.rank(entries, "query", query)
    changed = memory_path.parent / entries[4]["PrimaryImage"]["LocalPath"]
    Image.new("RGB", (8, 8), (200, 0, 0)).save(changed)
    with pytest.raises(ValueError, match="digest mismatch"):
        retriever.rank(entries, "query", query)


def test_corrupt_cached_vector_is_recomputed(joint_bank):
    entries, memory_path, query = joint_bank
    first = CountingEncoder()
    VisualMemoryRetriever(first, memory_path, "joint", "joint").rank(entries, "query", query)
    cached = next((memory_path.parent / "retrieval_index_cache").glob("*.json"))
    document = json.loads(cached.read_text(encoding="utf-8"))
    record = next(iter(document["vectors"].values()))
    record["vector"] = [8, 1]
    cached.write_text(json.dumps(document), encoding="utf-8")
    second = CountingEncoder()
    VisualMemoryRetriever(second, memory_path, "joint", "joint").rank(entries, "query", query)
    assert len(second.calls) == 2  # one query and precisely one invalid key


def test_gme_process_error_is_clear_and_worker_is_closed(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text("import sys,json\nfor line in sys.stdin:\n r=json.loads(line)\n"
                      " if r['op']=='close': break\n"
                      " print(json.dumps({'snapshot':{'provider':'fake-gme'}} if r['op']=='init' "
                      "else {'error':'ValueError: local GME test failure'}),flush=True)\n", encoding="utf-8")
    encoder = GMEProcessEncoder(python=sys.executable, worker=worker, timeout=10)
    with pytest.raises(RuntimeError, match="local GME test failure"):
        encoder.encode_joint("query", tmp_path / "image.png")
    encoder.close()
    encoder.close()
    assert encoder.process.poll() is not None
    with pytest.raises(RuntimeError, match="closed"):
        encoder.encode("x")


def test_gme_missing_environment_and_invalid_selection_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="gme_python"):
        GMEProcessEncoder(python=tmp_path / "missing_python.exe")
    with pytest.raises(ValueError, match="clip or gme"):
        make_joint_encoder(kind="other")


def test_native_gme_processor_never_silently_truncates():
    seen = []

    def process(**kwargs):
        seen.append(kwargs)
        return {"input_ids": np.zeros((1, 11), dtype=np.int64)}

    checked = CheckedProcessor(process, 10)
    with pytest.raises(TextContextOverflow):
        checked(text=["long"], truncation=True, max_length=10)
    assert seen[0]["truncation"] is False
    assert "max_length" not in seen[0]


def test_native_gme_chunks_keep_complete_text_and_native_pair_fusion():
    class Processor:
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: list(text))

        def __call__(self, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": np.zeros((1, 523 + len(kwargs["text"][0])), dtype=np.int64)}

    class Base:
        snapshot = {"model": "native-fake-gme"}

        def __init__(self):
            self.model = SimpleNamespace(processor=Processor(), max_length=530)
            self.admitted_texts = []

        def encode_joint(self, text, path):
            self.model.processor(text=[text], truncation=True, max_length=530)
            self.admitted_texts.append((text, path))
            return np.array([len(text) + 1, sum(map(ord, text)) % 99 + 1])

    base = Base()
    encoder = FullTextGME(base)
    text = "完整历史 😀 必须保留所有字符。" * 4 + "最终尾部"
    vector, audit = encoder.encode_joint(text, "same-image.png")
    assert "".join(part for part, _ in base.admitted_texts) == text
    assert all(path == "same-image.png" for _, path in base.admitted_texts)
    assert all(count <= 530 for count in audit["chunk_multimodal_tokens"])
    assert audit["encoded_characters"] == len(text)
    assert audit["chunks"] > 1 and audit["truncated"] is False
    assert np.linalg.norm(vector) == pytest.approx(1.)


def test_gme_timeout_terminates_worker_instead_of_falling_back(tmp_path):
    worker = tmp_path / "hanging_worker.py"
    worker.write_text("import sys,json,time\nfor line in sys.stdin:\n r=json.loads(line)\n"
                      " if r['op']=='close': break\n"
                      " if r['op']=='init': print(json.dumps({'snapshot':{'provider':'fake-gme'}}),flush=True)\n"
                      " else: time.sleep(60)\n", encoding="utf-8")
    encoder = GMEProcessEncoder(python=sys.executable, worker=worker, timeout=10)
    encoder.timeout = .1
    with pytest.raises(TimeoutError, match="exceeded"):
        encoder.encode_joint("query", "unused.png")
    assert encoder.closed and encoder.process.poll() is not None
