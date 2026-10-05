"""Local-only joint text/image encoders with explicit full-text coverage.

CLIP uses the exact retrieval/attack surrogate weights, but does not call its
truncating text helper. GME runs in an isolated compatible Python interpreter.
Both return one normalized vector per complete text/image pair.
"""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import numpy as np

from src.paths import PROJECT
from src.vision.dual_rag_bridge import DEFAULT_DUAL_ROOT, make_encoder


def normalized(value):
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
        raise ValueError("Joint encoder returned an invalid embedding")
    return vector / np.linalg.norm(vector)


def _text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_token_bounded_text(text, encode_tokens, payload_tokens):
    """Split at character boundaries using real token counts; preserve all text.

    Character boundaries avoid decoding partial byte-level tokens into replacement
    characters. Every emitted chunk is rechecked by the actual tokenizer.
    """
    text = str(text or " ")
    if payload_tokens < 1:
        raise ValueError("Text chunk token budget must be positive")
    chunks = []
    offset = 0
    while offset < len(text):
        remaining = text[offset:]
        token_count = len(encode_tokens(remaining))
        if token_count <= payload_tokens:
            chunks.append(remaining)
            break
        count = min(len(remaining), max(1, len(remaining) * payload_tokens // token_count))
        while len(encode_tokens(remaining[:count])) > payload_tokens:
            if count == 1:
                raise ValueError("One character exceeds the text token budget")
            count = max(1, count // 2)
        chunks.append(remaining[:count])
        offset += count
    return chunks or [" "]


class ChunkedClipEncoder:
    """Equal text/image fusion; length-weighted mean of all text token chunks."""

    def __init__(self, base):
        self.base = base
        self.surrogate = base.surrogate
        self.tokenizer = self.surrogate.tokenizer
        self.device = base.device
        self.context_length = int(getattr(self.tokenizer, "context_length", 77))
        self.payload_tokens = self.context_length - 2
        if self.payload_tokens <= 0 or not callable(getattr(self.tokenizer, "encode", None)):
            raise ValueError("CLIP joint encoder requires a non-truncating tokenizer.encode")
        self.sot_token = int(self.tokenizer.sot_token_id)
        self.eot_token = int(self.tokenizer.eot_token_id)
        self.closed = False
        self.last_encoding_audit = None
        self.snapshot = {
            **base.snapshot,
            "fusion": "equal_unit_image_plus_unit_full_text",
            "text_policy": {
                "algorithm": "contiguous_nonoverlapping_raw_token_chunks_v1",
                "context_length": self.context_length,
                "payload_tokens_per_chunk": self.payload_tokens,
                "special_tokens_per_chunk": 2,
                "coverage": "all_token_ids_no_truncation",
                "aggregation": "unit_token_count_weighted_mean_then_unit",
                "tokenizer": type(self.tokenizer).__name__,
            },
        }

    def _assert_open(self):
        if self.closed:
            raise RuntimeError("Joint encoder is closed")

    def encode(self, path):
        self._assert_open()
        return normalized(self.base.encode(path))

    def encode_text(self, text):
        self._assert_open()
        import torch
        text = str(text or " ")
        ids = list(self.tokenizer.encode(text))
        chunks = [ids[start:start + self.payload_tokens]
                  for start in range(0, len(ids), self.payload_tokens)] or [[]]
        weighted = None
        weights = []
        # Small fixed batches avoid growing GPU memory with historical length.
        with torch.no_grad():
            for start in range(0, len(chunks), 16):
                batch = chunks[start:start + 16]
                tokens = torch.zeros((len(batch), self.context_length), dtype=torch.long,
                                     device=self.device)
                for row, chunk in enumerate(batch):
                    tokens[row, :len(chunk) + 2] = torch.tensor(
                        [self.sot_token, *chunk, self.eot_token], dtype=torch.long,
                        device=self.device)
                features = self.surrogate.model.encode_text(tokens).float()
                features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
                for chunk, vector in zip(batch, features):
                    weight = max(1, len(chunk))
                    weighted = vector * weight if weighted is None else weighted + vector * weight
                    weights.append(weight)
        self.last_encoding_audit = {
            "text_sha256": _text_digest(text), "input_tokens": len(ids),
            "encoded_tokens": sum(len(chunk) for chunk in chunks),
            "chunks": len(chunks), "payload_tokens_per_chunk": self.payload_tokens,
            "chunk_token_counts": [len(chunk) for chunk in chunks],
            "truncated": False,
        }
        return normalized(weighted / sum(weights))

    def encode_joint(self, text, path):
        image = self.encode(path)
        if not str(text or "").strip():
            self.last_encoding_audit = {"text_sha256": _text_digest(str(text or "")),
                                        "input_tokens": 0, "encoded_tokens": 0,
                                        "chunks": 0, "truncated": False}
            return image
        return normalized(image + self.encode_text(text))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.base = self.surrogate = self.tokenizer = None
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GMEProcessEncoder:
    """One persistent worker; no installs, downloads, or silent model fallback."""

    def __init__(self, python=None, model=None, device="cuda", timeout=300,
                 root=DEFAULT_DUAL_ROOT, worker=None):
        self.timeout = float(timeout)
        self.closed = False
        self._responses = queue.Queue()
        self._stderr = deque(maxlen=30)
        self._lock = threading.Lock()
        self.last_encoding_audit = None
        python = Path(python) if python else Path(sys.executable).parent.parent / "dual-visual/python.exe"
        if not python.is_file():
            raise FileNotFoundError(f"GME compatible Python does not exist: {python}; configure gme_python")
        worker = Path(worker) if worker else Path(__file__).with_name("joint_encoder_worker.py")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_MODULES_CACHE=str(PROJECT / "src/output/joint_encoder_runtime/hf_modules"))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [str(python), "-X", "utf8", "-u", str(worker)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", env=env, creationflags=flags, cwd=PROJECT)
        self._reader_threads = [threading.Thread(target=self._read_stdout, daemon=True),
                                threading.Thread(target=self._read_stderr, daemon=True)]
        for thread in self._reader_threads:
            thread.start()
        try:
            response = self._request({"op": "init", "root": str(Path(root).resolve()),
                                      "model": str(model or "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"),
                                      "device": device})
            self.snapshot = response["snapshot"]
        except BaseException:
            self.close()
            raise

    def _read_stdout(self):
        for line in self.process.stdout:
            try:
                self._responses.put(json.loads(line))
            except ValueError:
                self._responses.put({"error": "GME worker emitted invalid protocol output"})
        self._responses.put({"error": "GME worker exited before completing request"})

    def _read_stderr(self):
        for line in self.process.stderr:
            self._stderr.append(line.strip()[-500:])

    def _request(self, request):
        with self._lock:
            if self.closed or self.process.poll() is not None:
                raise RuntimeError("GME worker is closed or has exited")
            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                response = self._responses.get(timeout=self.timeout)
            except queue.Empty as exc:
                self.close()
                raise TimeoutError(f"GME encoder request exceeded {self.timeout:g} seconds") from exc
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("GME worker connection failed") from exc
            if response.get("error"):
                # Worker errors contain a concise exception, never complete input.
                raise RuntimeError("GME encoder failed: " + response["error"])
            return response

    def encode(self, path):
        response = self._request({"op": "image", "path": str(Path(path).resolve())})
        return normalized(response["vector"])

    def encode_joint(self, text, path):
        response = self._request({"op": "joint", "text": str(text or ""),
                                  "path": str(Path(path).resolve())})
        self.last_encoding_audit = response.get("audit")
        return normalized(response["vector"])

    def close(self):
        if self.closed:
            return
        self.closed = True
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.write('{"op":"close"}\n')
                process.stdin.flush()
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        for thread in self._reader_threads:
            thread.join(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream:
                stream.close()


def make_joint_encoder(kind="clip", optimization_config=None, device=None,
                       gme_python=None, gme_model=None):
    if kind == "clip":
        return ChunkedClipEncoder(make_encoder(optimization_config=optimization_config,
                                              surrogate=2, device=device))
    if kind == "gme":
        return GMEProcessEncoder(python=gme_python, model=gme_model, device=device or "cuda")
    raise ValueError("Joint retrieval encoder must be clip or gme")
