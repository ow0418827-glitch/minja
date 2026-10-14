"""One-episode opt-in driver for a running WebShop endpoint.

Example:
  python -m src.defense.safe_recall.cli --session demo --subject local-user \
    --request "Buy the same product as this photo" --image current.png \
    --image-host example-cdn.invalid

Source adapters for real orders/confirmations must be injected through the
Python API. This CLI defaults to UnavailableSource and never manufactures one.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .adapters import CurrentImageLoader, WebShopAdapter
from .contracts import Request
from .model import JsonDecisionModel
from .runner import RunnerConfig, SafeRecallRunner
from .providers import HmacVerifier, SignedJsonSource


def _memory_snippets(path: str | None) -> tuple[str, ...]:
    if not path:
        return ()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Memory file must contain a list")
    output = []
    for row in raw[:5]:
        if not isinstance(row, dict):
            continue
        # Only prior action/observation prose is compiled. Reward, outcome,
        # attack tags, filenames and hidden scoring fields are discarded.
        actions = row.get("Actions", [])
        if isinstance(actions, list):
            output.append("\n".join(str(value) for value in actions[:8])[:4000])
    return tuple(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one SafeRecall V3 episode")
    parser.add_argument("--session", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-host", action="append", default=[])
    parser.add_argument("--image-cache-records", type=Path,
                        help="Read verified current-page images from a pinned local cache")
    parser.add_argument("--image-cache-only", action="store_true",
                        help="Do not fetch a missing current product image remotely")
    parser.add_argument("--memory-json")
    parser.add_argument("--source-json", type=Path,
                        help="Externally signed order/confirmation snapshot")
    parser.add_argument("--source-issuer",
                        help="Pinned issuer name for the signed snapshot")
    parser.add_argument("--source-key-env",
                        help="Environment variable holding issuer HMAC key in hex")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-model-calls", type=int, default=24)
    parser.add_argument("--match-threshold", type=float, default=0.95)
    parser.add_argument("--research-execute", action="store_true",
                        help="Allow an uncalibrated simulator purchase for research")
    args = parser.parse_args(argv)
    if args.image_cache_only and args.image_cache_records is None:
        parser.error("--image-cache-only requires --image-cache-records")
    if args.image:
        image_bytes = args.image.read_bytes()
        image_mime = mimetypes.guess_type(args.image.name)[0] or "image/png"
    else:
        image_bytes = None
        image_mime = None
    request = Request(args.subject, args.request, image_bytes, image_mime)
    from src.configuration.llm_config import load_model_config
    from src.agent.minja import webshopEnv
    model_config = load_model_config(args.model_config) if args.model_config else load_model_config()
    model = JsonDecisionModel(model_config.client(), model_config.model,
                              max_calls=args.max_model_calls,
                              provider=model_config.provider)
    env = webshopEnv(instruction_overrides={args.session: args.request})
    adapter = WebShopAdapter(
        env, args.session,
        image_loader=CurrentImageLoader(frozenset(
            host.lower() for host in args.image_host),
            cache_records=args.image_cache_records,
            cache_only=args.image_cache_only),
    )
    source = None
    verifier = None
    if any((args.source_json, args.source_issuer, args.source_key_env)):
        if not all((args.source_json, args.source_issuer, args.source_key_env)):
            parser.error("All source authentication arguments are required together")
        secret_hex = os.environ.get(args.source_key_env, "")
        if not secret_hex:
            parser.error("Pinned source key environment variable is missing")
        try:
            key = bytes.fromhex(secret_hex)
        except ValueError:
            parser.error("Pinned source key must be hex")
        if len(key) < 32:
            parser.error("Pinned source key must be at least 32 bytes")
        source = SignedJsonSource(args.source_json)
        verifier = HmacVerifier({args.source_issuer: key})
    runner = SafeRecallRunner(
        adapter, model, source=source, verifier=verifier,
        config=RunnerConfig(
            max_candidates=args.max_candidates,
            match_threshold=args.match_threshold,
            allow_uncalibrated_purchase=args.research_execute,
        ),
    )
    outcome = runner.run(request, ordinary_memory=_memory_snippets(args.memory_json))
    output = Path("src/output/safe_recall_v3")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = output / f"{stamp}_{uuid4().hex[:8]}.json"
    result_path.write_text(json.dumps({
        "protocol": outcome.protocol, "status": outcome.status.value,
        "reason": outcome.reason, "final_handle": outcome.final_handle,
        "final_options": dict(outcome.final_options), "steps": outcome.steps,
        "model_calls": model.calls, "model": model_config.public(),
        "audit": list(outcome.audit),
        "source_capability": "signed_snapshot" if source else "unavailable",
        "risk_certificate": "none",
        "research_execute": args.research_execute,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": outcome.status.value, "reason": outcome.reason,
        "result_path": str(result_path.resolve()),
    }, ensure_ascii=False))
    return 0 if outcome.status.value == "purchased" else 2


if __name__ == "__main__":
    raise SystemExit(main())
