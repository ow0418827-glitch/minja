"""Line-delimited worker protocol for local GME; stdout is protocol only."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vision.joint_encoder import normalized, split_token_bounded_text


class TextContextOverflow(ValueError):
    pass


class CheckedProcessor:
    """Stop the native loader from silently truncating text or image tokens."""

    def __init__(self, processor, limit):
        self.processor, self.limit = processor, int(limit)
        self.last_tokens = None

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, **kwargs):
        kwargs["truncation"] = False
        kwargs.pop("max_length", None)
        value = self.processor(**kwargs)
        self.last_tokens = int(value["input_ids"].shape[-1])
        if self.last_tokens > self.limit:
            raise TextContextOverflow("Native multimodal sequence exceeds context")
        return value


class FullTextGME:
    def __init__(self, base):
        self.base = base
        self.tokenizer = base.model.processor.tokenizer
        self.limit = int(base.model.max_length)
        self.processor = CheckedProcessor(base.model.processor, self.limit)
        base.model.processor = self.processor
        # Reserve native prompt + image tokens. Exact processor checks every call;
        # any unusual image expansion triggers further splitting before inference.
        self.payload = max(1, self.limit - 512)
        self.snapshot = {**base.snapshot, "text_policy": {
            "algorithm": "tokenizer_checked_character_chunks_v1",
            "native_context_length": self.limit,
            "initial_payload_tokens": self.payload,
            "native_processor_truncation": False,
            "coverage": "all_original_characters_no_truncation",
            "aggregation": "unit_token_count_weighted_mean_native_fused_vectors_then_unit",
            "per_chunk_image": "same_input_image",
            "tokenizer": type(self.tokenizer).__name__,
        }}

    def _tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode(self, path):
        return normalized(self.base.encode(path))

    def encode_joint(self, text, path):
        text = str(text or " ")
        pending = split_token_bounded_text(text, self._tokens, self.payload)
        parts = []
        vectors = []
        sequence_tokens = []
        while pending:
            chunk = pending.pop(0)
            try:
                vector = normalized(self.base.encode_joint(chunk, path))
            except TextContextOverflow:
                if len(chunk) < 2:
                    raise ValueError("GME image/prompt leaves no usable text context")
                middle = len(chunk) // 2
                pending[:0] = [chunk[:middle], chunk[middle:]]
                continue
            parts.append(chunk)
            vectors.append(vector)
            sequence_tokens.append(self.processor.last_tokens)
        if "".join(parts) != text:
            raise ValueError("GME text chunk coverage differs from original input")
        weights = [max(1, len(self._tokens(part))) for part in parts]
        result = normalized(np.average(np.stack(vectors), axis=0, weights=weights))
        audit = {"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                 "input_characters": len(text), "encoded_characters": sum(map(len, parts)),
                 "input_tokens": len(self._tokens(text)), "chunks": len(parts),
                 "chunk_text_tokens": weights, "chunk_multimodal_tokens": sequence_tokens,
                 "truncated": False}
        return result, audit


def load_encoder(request):
    root = Path(request["root"]).resolve()
    if not (root / "src/dual_rag/encoder.py").is_file():
        raise FileNotFoundError("Configured dual-rag encoder source is absent")
    sys.path.insert(0, str(root / "src"))
    from dual_rag.encoder import GMEEncoder
    return FullTextGME(GMEEncoder({"provider": "gme", "model": request["model"],
                                   "device": request["device"], "precision": "fp16"}, root))


def main():
    output = sys.stdout
    encoder = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "close":
                break
            # Libraries can print download/loading progress even in offline mode.
            with contextlib.redirect_stdout(sys.stderr):
                if op == "init":
                    if encoder is not None:
                        raise ValueError("GME worker already initialized")
                    encoder = load_encoder(request)
                    response = {"snapshot": encoder.snapshot}
                elif encoder is None:
                    raise ValueError("GME worker is not initialized")
                elif op == "image":
                    response = {"vector": encoder.encode(request["path"]).tolist()}
                elif op == "joint":
                    vector, audit = encoder.encode_joint(request["text"], request["path"])
                    response = {"vector": vector.tolist(), "audit": audit}
                else:
                    raise ValueError("Unknown GME worker operation")
        except Exception as exc:
            # No full request, memory history, API settings or chain-of-thought.
            response = {"error": f"{type(exc).__name__}: {exc}"[:1500]}
        output.write(json.dumps(response, ensure_ascii=False) + "\n")
        output.flush()
    encoder = None


if __name__ == "__main__":
    main()
