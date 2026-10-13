"""以落盘真实查询复核图文权重与扰动收缩，不调用购物模型。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from src.vision.joint_encoder import make_joint_encoder
from src.vision.visual_retrieval import memory_index_text, resolve_image


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def blend(image, text, weight):
    if weight == 1:
        return image
    if weight == 0:
        return text
    return unit(image + text) if weight == .5 else unit(weight * image + (1 - weight) * text)


def semantic_text(entry):
    return memory_index_text(entry, "semantic")


def load_group(run, group):
    root = run / "groups" / group
    memory = json.loads((root / "memory.json").read_text(encoding="utf-8"))
    attacker = next(i for i, entry in enumerate(memory) if "_history_" in str(entry["Id"]))
    candidates = [p / "shopping" for p in sorted((root / "evaluations").iterdir())
                  if len(list((p / "shopping").glob("*_adversarial.json"))) == 20]
    if not candidates:
        raise ValueError(f"No complete recorded shopping queries: {group}")
    shop = candidates[0]
    trials = []
    for path in sorted(shop.glob("*_adversarial.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        ranking = doc["RetrievalTrace"][0]["Ranking"]
        mine = next(row for row in ranking if row["Id"] == memory[attacker]["Id"])
        trials.append({"path": str(path), "text": doc["Instruction"],
                       "query_sha": mine["query_image_sha256"], "rank": mine["rank"], "score": mine["score"]})
    return root, memory, attacker, trials


def evaluate(images, texts, qimages, qtexts, attacker, key_weight, query_weight):
    keys = [blend(i, t, key_weight) for i, t in zip(images, texts)]
    ranks, margins, scores = [], [], []
    winners = []
    for qi, qt in zip(qimages, qtexts):
        query = blend(qi, qt, query_weight)
        row = [float(key @ query) for key in keys]
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))
        ranks.append(order.index(attacker) + 1)
        margins.append(row[attacker] - max(value for i, value in enumerate(row) if i != attacker))
        scores.append(row[attacker])
        winners.append(order[0])
    return {"top1": ranks.count(1), "rank_min": min(ranks), "rank_max": max(ranks),
            "mean_margin": float(np.mean(margins)), "min_margin": min(margins),
            "ranks": ranks, "scores": scores, "margins": margins, "winners": winners}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    images_by_sha = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in (args.run / "assets").iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    enc = make_joint_encoder(kind="clip", device=args.device)
    started = time.perf_counter()
    report = {"run": str(args.run.resolve()), "groups": [], "query_source": "recorded_shopping_instruction_and_digest"}
    try:
        for group in ["01_camera", "02_chair", "06_jeans"]:
            print(f"Encoding {group}", flush=True)
            root, memory, attacker, trials = load_group(args.run, group)
            images, full, instruction, semantic = [], [], [], []
            for entry in memory:
                images.append(unit(enc.encode(resolve_image(entry, root / "memory.json"))))
                full.append(unit(enc.encode_text(memory_index_text(entry, "full"))))
                instruction.append(unit(enc.encode_text(memory_index_text(entry, "instruction"))))
                semantic.append(unit(enc.encode_text(semantic_text(entry))))
            image_cache, text_cache = {}, {}
            for trial in trials:
                if trial["query_sha"] not in image_cache:
                    image_cache[trial["query_sha"]] = enc.encode(images_by_sha[trial["query_sha"]])
                if trial["text"] not in text_cache:
                    text_cache[trial["text"]] = enc.encode_text(trial["text"])
            qimages = [image_cache[t["query_sha"]] for t in trials]
            qtexts = [text_cache[t["text"]] for t in trials]
            original = dict(memory[attacker])
            original.pop("RetrievalImage", None)
            source_path = resolve_image(original, root / "memory.json")
            clean_images = list(images)
            clean_images[attacker] = unit(enc.encode(source_path))
            baseline = evaluate(images, full, qimages, qtexts, attacker, .5, .5)
            error = max(abs(a - t["score"]) for a, t in zip(baseline["scores"], trials))
            aligned = baseline["ranks"] == [t["rank"] for t in trials]
            if not aligned or error > 2e-6:
                raise ValueError(f"Recorded baseline mismatch {group}: ranks={aligned}, error={error}")
            rows = []
            for mode, texts in [("full", full), ("instruction", instruction), ("semantic", semantic)]:
                for alpha in [.5, .6, .7, .8, .9, 1.]:
                    for beta in sorted({.5, alpha}):
                        result = evaluate(images, texts, qimages, qtexts, attacker, alpha, beta)
                        clean = evaluate(clean_images, texts, qimages, qtexts, attacker, alpha, beta)
                        rows.append({"index_text": mode, "key_weight": alpha, "query_weight": beta,
                                     **result, "clean_source_top1": clean["top1"], "clean_source_mean_margin": clean["mean_margin"]})
            source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32)
            attacked_path = resolve_image(memory[attacker], root / "memory.json")
            attacked = np.asarray(Image.open(attacked_path).convert("RGB"), dtype=np.float32)
            if source.shape != attacked.shape:
                raise ValueError("Source and attack dimensions differ")
            shrinks = []
            for scale in [1., .75, .5, .25, 0.]:
                decoded = np.clip(np.rint(source + scale * (attacked - source)), 0, 255).astype(np.uint8)
                path = args.output / f"{group}_scale_{scale:g}.png"
                Image.fromarray(decoded).save(path)
                modified = list(images)
                modified[attacker] = unit(enc.encode(path))
                difference = decoded.astype(np.float32) - source
                results = [{"key_weight": alpha, "query_weight": beta,
                            **evaluate(modified, full, qimages, qtexts, attacker, alpha, beta)}
                           for alpha, beta in [(.5, .5), (.7, .5), (.8, .5), (.9, .5), (1., .5), (.8, .8), (1., 1.)]]
                shrinks.append({"scale": scale, "path": str(path.resolve()),
                                "mae_levels": float(np.abs(difference).mean()),
                                "mse_levels": float(np.square(difference).mean()),
                                "max_levels": float(np.abs(difference).max()), "results": results})
            np.savez_compressed(args.output / f"{group}_components.npz", images=images, clean_images=clean_images,
                                full=full, instruction=instruction, semantic=semantic, qimages=qimages, qtexts=qtexts,
                                attacker=attacker)
            record = {"group": group, "attacker_index": attacker, "attacker_id": memory[attacker]["Id"],
                      "trials": trials, "baseline_aligned": aligned, "baseline_max_score_error": error,
                      "source_path": str(source_path), "attack_path": str(attacked_path),
                      "weight_sweep": rows, "shrink": shrinks}
            report["groups"].append(record)
            report["elapsed_seconds"] = time.perf_counter() - started
            (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Verified {group}: baseline top1={baseline['top1']}, max_error={error:.9g}", flush=True)
    finally:
        enc.close()


if __name__ == "__main__":
    main()
