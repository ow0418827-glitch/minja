"""Image-only or joint text/image memory retrieval with auditable indexing."""
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
        raise ValueError("Invalid retrieval embedding")
    return vector / np.linalg.norm(vector)


INDEX_TEXT_POLICY = {
    "version": 1,
    "full_fields": ["Instruction", "Actions", "FinalASIN", "FinalOptions", "FinalPrice"],
    "coverage": "complete_instruction_and_all_actions_observations_no_truncation",
    "excluded": "audit_labels_source_role_attack_metadata_reward_policy_trace_task_goal",
}


def canonical_digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def memory_index_text(entry, mode="full"):
    """Use observed history only, excluding experiment labels and scoring goals."""
    if mode == "instruction":
        return str(entry.get("Instruction", ""))
    if mode != "full":
        raise ValueError("index_text must be full or instruction")
    sections = ["Instruction:\n" + str(entry.get("Instruction", ""))]
    actions = entry.get("Actions", [])
    if isinstance(actions, (list, tuple)):
        actions = "\n".join(item if isinstance(item, str) else json.dumps(
            item, ensure_ascii=False, sort_keys=True) for item in actions)
    elif not isinstance(actions, str):
        actions = json.dumps(actions, ensure_ascii=False, sort_keys=True)
    sections.append("Actions and observations:\n" + actions)
    purchase = {key: entry[key] for key in ("FinalASIN", "FinalOptions", "FinalPrice")
                if key in entry and entry[key] is not None}
    if purchase:
        sections.append("Observed purchase:\n" + json.dumps(purchase, ensure_ascii=False,
                                                              sort_keys=True))
    return "\n\n".join(sections)


def resolve_image(entry, memory_path):
    # An explicitly present adversarial key must never silently fall back to A.
    field = "RetrievalImage" if entry.get("RetrievalImage") is not None else "PrimaryImage"
    record = entry.get(field)
    if not record or not record.get("LocalPath"):
        if field == "RetrievalImage":
            raise ValueError("RetrievalImage must reference a local image")
        return None
    path = Path(record["LocalPath"])
    if not path.is_absolute():
        path = Path(memory_path).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if record.get("SHA256") and sha256(path) != record["SHA256"]:
        raise ValueError(f"Image digest mismatch: {path}")
    return path


class VisualMemoryRetriever:
    def __init__(self, encoder, memory_path, key_mode="image", query_mode="image",
                 index_text="full", cache_dir="auto"):
        if key_mode not in {"image", "joint"} or query_mode not in {"image", "joint"}:
            raise ValueError("key/query mode must be image or joint")
        self.encoder = encoder
        self.memory_path = Path(memory_path).resolve()
        self.key_mode, self.query_mode = key_mode, query_mode
        if index_text not in {"full", "instruction"}:
            raise ValueError("index_text must be full or instruction")
        self.index_text = index_text
        self.cache_dir = (self.memory_path.parent / "retrieval_index_cache"
                          if cache_dir == "auto" else Path(cache_dir).resolve()
                          if cache_dir is not None else None)
        self.cache = {}
        self.audits = {}
        self.snapshot = {"encoder": encoder.snapshot, "key_mode": key_mode,
                         "query_mode": query_mode, "index_text": index_text,
                         "index_text_policy": INDEX_TEXT_POLICY if index_text == "full"
                         else {"version": 1, "fields": ["Instruction"]},
                         "cache_schema": 1}

    def _encode(self, text, path, mode, *, scope="query"):
        digest = sha256(path)
        key = canonical_digest({"mode": mode, "image_sha256": digest,
                                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
                                if mode == "joint" else None,
                                "model_sha256": canonical_digest(self.encoder.snapshot),
                                "scope": scope})
        if key not in self.cache:
            value = self.encoder.encode_joint(text, path) if mode == "joint" else self.encoder.encode(path)
            self.cache[key] = unit(value)
            audit = getattr(self.encoder, "last_encoding_audit", None) if mode == "joint" else None
            self.audits[key] = json.loads(json.dumps(audit)) if audit is not None else None
        return self.cache[key], key

    def _load_index_cache(self, path, identity):
        if not path or not path.is_file():
            return
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("identity") != identity:
                return
            for key, record in document.get("vectors", {}).items():
                vector = record["vector"]
                if canonical_digest(vector) != record["vector_sha256"]:
                    continue
                self.cache[key] = unit(vector)
                self.audits[key] = record.get("encoding_audit")
        except (OSError, ValueError, KeyError, TypeError):
            # A partial/corrupt cache is disposable; the bank and image checks are
            # independent and must never be bypassed by cache recovery.
            return

    def _save_index_cache(self, path, identity, keys):
        if path is None:
            return
        vectors = {}
        for key in keys:
            vector = self.cache[key].tolist()
            vectors[key] = {"vector": vector, "vector_sha256": canonical_digest(vector),
                            "encoding_audit": self.audits.get(key)}
        payload = {"schema_version": 1, "identity": identity, "vectors": vectors}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        finally:
            if temporary.is_file():
                temporary.unlink()

    def rank(self, memory, instruction, query_image):
        # Bank bytes, caller entries and encoder configuration are refreshed on
        # every ranking, even if this retriever instance is reused after edits.
        bank_sha = sha256(self.memory_path) if self.memory_path.is_file() else None
        model_sha = canonical_digest(self.encoder.snapshot)
        identity = {"bank_sha256": bank_sha, "entries_sha256": canonical_digest(memory),
                    "model_sha256": model_sha, "key_mode": self.key_mode,
                    "index_text": self.index_text,
                    "index_text_policy_sha256": canonical_digest(self.snapshot["index_text_policy"])}
        scope = canonical_digest(identity)
        cache_path = self.cache_dir / (scope + ".json") if self.cache_dir else None
        self._load_index_cache(cache_path, identity)
        self.snapshot.update(encoder=self.encoder.snapshot, bank_sha256=bank_sha,
                             model_sha256=model_sha, entries_sha256=identity["entries_sha256"])
        query_text = str(instruction or "")
        query_path = Path(query_image).resolve()
        query, query_key = self._encode(query_text, query_path, self.query_mode)
        query_audit = self.audits.get(query_key)
        self.snapshot["query_encoding_audit"] = query_audit
        ranked = []
        keys = []
        for index, entry in enumerate(memory):
            path = resolve_image(entry, self.memory_path)
            if path is None:
                continue
            text = memory_index_text(entry, self.index_text) if self.key_mode == "joint" else ""
            vector, key = self._encode(text, path, self.key_mode, scope=scope)
            keys.append(key)
            if vector.shape != query.shape:
                raise ValueError("Query and memory vector dimensions differ")
            ranked.append({"index": index, "Id": entry["Id"], "score": float(vector @ query),
                           "image_sha256": sha256(path),
                           "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                           "text_characters": len(text), "index_text": self.index_text,
                           "model_sha256": model_sha, "bank_sha256": bank_sha,
                           "query_text_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                           "query_image_sha256": sha256(query_path),
                           "query_encoding_audit": query_audit,
                           "encoding_audit": self.audits.get(key)})
        ranked.sort(key=lambda item: (-item["score"], item["index"]))
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        self._save_index_cache(cache_path, identity, keys)
        return ranked

    def shortlist(self, memory, embeddings, instruction, query_image, candidates):
        if not isinstance(candidates, int) or candidates < 1:
            raise ValueError("Retrieval candidates must be a positive integer")
        ranked = self.rank(memory, instruction, query_image)
        selected = ranked[:candidates]
        indexes = [item["index"] for item in selected]
        subset = {}
        for key, values in embeddings.items():
            if isinstance(values, list):
                subset[key] = [values[i] for i in indexes]
            else:
                subset[key] = values[indexes]
        subset["VisualScores"] = np.asarray([item["score"] for item in selected], dtype=np.float32)
        return [memory[i] for i in indexes], subset, ranked


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
