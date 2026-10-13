"""在已有固定记忆库与真实查询上，选出保持排名/相似度的小幅扰动后处理图。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any

import numpy as np
from PIL import Image

from src.experiments.probe_vision_optimization import evaluate
from src.vision.joint_encoder import normalized
from src.vision.visual_retrieval import pillow_decoder_signature


DEFAULT_SCALES = (1., .875, .75, .625, .5, .375, .25)
BUDGET_LEVELS = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: float, label: str, low: float, high: float | None = None) -> float:
    value = float(value)
    if not math.isfinite(value) or value < low or (high is not None and value > high):
        raise ValueError(f"{label} must be finite and in [{low}, {high if high is not None else 'inf'}]")
    return value


def pixel_audit(source: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if source.shape != candidate.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source and candidate must have the same RGB dimensions")
    difference = candidate.astype(np.float64) - source.astype(np.float64)
    absolute = np.abs(difference)
    mse = float(np.mean(difference * difference))
    maximum = int(absolute.max())
    return {
        "shape": list(source.shape), "mae_levels": float(absolute.mean()),
        "mse_levels": mse, "max_levels": maximum, "budget_levels": BUDGET_LEVELS,
        "within_budget": maximum <= BUDGET_LEVELS,
        "budget_saturated_channel_fraction": float(np.mean(absolute == BUDGET_LEVELS)),
        "psnr_db": 10 * math.log10(255**2 / mse) if mse else None,
    }


def scaled_candidate(source: np.ndarray, modified: np.ndarray, scale: float) -> np.ndarray:
    finite_number(scale, "scale", 0., 1.)
    if source.shape != modified.shape:
        raise ValueError("Scaling cannot resize the image")
    return np.clip(np.rint(source.astype(np.float64) + scale *
                          (modified.astype(np.float64) - source.astype(np.float64))), 0, 255).astype(np.uint8)


def check_baseline_audit(actual: dict[str, Any], stored: dict[str, Any]) -> None:
    pairs = {
        "mae_levels": "mean_absolute_pixel_difference_levels",
        "mse_levels": "mean_squared_pixel_difference_levels",
        "max_levels": "max_absolute_pixel_difference_levels",
    }
    for actual_key, stored_key in pairs.items():
        if not math.isclose(actual[actual_key], stored[stored_key], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(
                f"Original 500-step pixel audit differs: {actual_key} "
                f"{actual[actual_key]} != {stored[stored_key]}. Use the original test "
                "image decoder environment, e.g. minja-vision or dual-visual."
            )
    if not actual["within_budget"]:
        raise ValueError("Original 500-step candidate is outside the original-source budget")


def load_inputs(probe: Path, mapping_path: Path) -> list[dict[str, Any]]:
    """Decode and validate inputs/cache without constructing an encoder."""
    from src.vision.dual_rag_bridge import activate
    activate()
    from dual_rag.optimization.common.io import load_image

    mapping = read_json(mapping_path)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("--images must contain a nonempty JSON object mapping group IDs to PNG paths")
    report = read_json(probe / "report.json")
    groups = {group["group"]: group for group in report["groups"]}
    contexts = []
    for group_id, value in mapping.items():
        if group_id not in groups or not re.fullmatch(r"[A-Za-z0-9_-]+", group_id):
            raise ValueError(f"Unknown or unsafe group ID: {group_id}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Image path for {group_id} must be a nonempty string")
        image = Path(value)
        if not image.is_absolute():
            image = mapping_path.parent / image
        image = image.resolve()
        if not image.is_file() or image.suffix.lower() != ".png":
            raise ValueError(f"Input must be an existing PNG: {image}")
        with Image.open(image) as opened:
            if opened.format != "PNG":
                raise ValueError(f"Input file is not a PNG: {image}")
        group = groups[group_id]
        if not group["baseline_aligned"] or group["baseline_max_score_error"] > 2e-6:
            raise ValueError(f"Probe baseline was not verified: {group_id}")
        source = Path(group["source_path"])
        old_image = Path(group["attack_path"])
        metadata_path = Path(report["run"]) / "groups" / group_id / "attack" / "metadata.json"
        metadata = read_json(metadata_path)
        if sha256(source) != metadata["source_image_sha256"]:
            raise ValueError(f"Source image digest differs from the original metadata: {group_id}")
        source_array = np.asarray(load_image(source), dtype=np.uint8)
        old_array = np.asarray(load_image(old_image), dtype=np.uint8)
        original_audit = pixel_audit(source_array, old_array)
        check_baseline_audit(original_audit, metadata["decoded_pixel_audit"])
        input_digest = sha256(image)
        input_array = np.asarray(load_image(image), dtype=np.uint8)
        if sha256(image) != input_digest:
            raise ValueError(f"Input image changed while reading: {image}")
        input_audit = pixel_audit(source_array, input_array)
        if not input_audit["within_budget"]:
            raise ValueError(f"Input candidate exceeds the original-source 16-level budget: {image}")
        component_path = probe / f"{group_id}_components.npz"
        with np.load(component_path, allow_pickle=False) as cache:
            arrays = {key: np.array(cache[key], copy=True) for key in ("images", "full", "qimages", "qtexts")}
            attacker = int(cache["attacker"])
        dimension = arrays["images"].shape[-1]
        if attacker != group["attacker_index"] or not 0 <= attacker < 15:
            raise ValueError(f"Cached attacker index does not match the probe: {group_id}")
        if len(group["trials"]) != 20:
            raise ValueError("This fixed-library tool requires 20 recorded real queries per group")
        for key, expected in (("images", (15, dimension)), ("full", (15, dimension)),
                              ("qimages", (20, dimension)), ("qtexts", (20, dimension))):
            array = arrays[key]
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"Invalid cache shape or values: {group_id}/{key}")
            if not np.allclose(np.linalg.norm(array, axis=1), 1., atol=2e-5):
                raise ValueError(f"Cached vectors must be normalized: {group_id}/{key}")
        if len({trial["query_sha"] for trial in group["trials"]}) != 1:
            raise ValueError("Expected one fixed target image per group")
        if not np.allclose(arrays["qimages"], arrays["qimages"][0], atol=1e-6):
            raise ValueError("Cached target image vectors are inconsistent")
        baseline = evaluate(arrays["images"], arrays["full"], arrays["qimages"], arrays["qtexts"],
                            attacker, .5, .5)
        error = max(abs(score - trial["score"]) for score, trial in zip(baseline["scores"], group["trials"]))
        if baseline["ranks"] != [trial["rank"] for trial in group["trials"]] or error > 2e-6:
            raise ValueError(f"Cached baseline no longer reproduces the recorded queries: {group_id}")
        contexts.append({
            "group": group_id, "source": source, "source_array": source_array,
            "old_image": old_image, "input": image, "input_array": input_array,
            "input_sha256": input_digest, "input_audit": input_audit,
            "old_audit": original_audit, "old_metadata": metadata,
            "components": arrays, "components_path": component_path, "attacker": attacker,
            "old_image_cosine": float(arrays["images"][attacker] @ arrays["qimages"][0]),
            "baseline_error": error, "query_source": report.get("query_source"),
            "query_image_sha256": group["trials"][0]["query_sha"],
        })
    return contexts


def matches_criteria(row: dict[str, Any], criteria: dict[str, Any]) -> bool:
    selected = next(rank for rank in row["rankings"]
                    if rank["key_weight"] == criteria["key_image_weight"]
                    and rank["query_weight"] == criteria["query_image_weight"])
    return (selected["top1"] >= criteria["required_top1"]
            and selected["min_margin"] >= criteria["min_margin"]
            and row["image_cosine"] >= criteria["min_image_cosine"])


def select_minimum(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [row for row in rows if row["eligible"]]
    return min(feasible, key=lambda row: (row["pixel_audit"]["mse_levels"],
               row["pixel_audit"]["mae_levels"], -row["image_cosine"], row["scale"])) if feasible else None


def check_encoder_snapshot(snapshot: dict[str, Any], contexts: list[dict[str, Any]]) -> None:
    current = snapshot.get("surrogate", {})
    fields = ("name", "weight_sha256", "image_size", "preprocess", "precision")
    for context in contexts:
        original = context["old_metadata"]["surrogates"][1]
        if any(not current.get(key) or current.get(key) != original.get(key) for key in fields):
            raise ValueError("Retrieval encoder weights/preprocessing/precision differ from the cache's original proxy")


def run(contexts: list[dict[str, Any]], output: Path, *, scales: list[float], device: str,
        min_margin: float, min_image_cosine: float | None, required_top1: int,
        key_weight: float, query_weight: float, optimization_config: Path | None,
        probe: Path, mapping_path: Path, encoder_factory=None) -> dict[str, Any]:
    if encoder_factory is None:
        from src.vision.joint_encoder import make_joint_encoder
        encoder_factory = make_joint_encoder
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1, "method": "postprocess_original_source_perturbation_amplitude",
        "status": "running", "generated_at_beijing": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "scope": "known_fixed_15_memory_library_and_20_recorded_queries_per_group_only",
        "does_not_change_optimizer": True, "full_shopping_experiment_executed": False,
        "index_text": "full", "probe": str(probe.resolve()), "images_mapping": str(mapping_path.resolve()),
        "images_mapping_sha256": sha256(mapping_path), "scales": scales,
        "decoder": pillow_decoder_signature(), "groups": [],
    }

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                       allow_nan=False), encoding="utf-8")

    encoder = None
    try:
        encoder = encoder_factory(kind="clip", device=device, optimization_config=optimization_config)
        check_encoder_snapshot(encoder.snapshot, contexts)
        report["encoder_snapshot"] = encoder.snapshot
        for context in contexts:
            directory = output / context["group"]
            directory.mkdir()
            criteria = {
                "key_image_weight": key_weight, "query_image_weight": query_weight,
                "required_top1": required_top1, "min_margin": min_margin,
                "min_image_cosine": context["old_image_cosine"] if min_image_cosine is None else min_image_cosine,
                "cosine_reference": "cached_original_500_step_image_dot_fixed_target"
                if min_image_cosine is None else "explicit_override",
                "minimize": "eight_bit_original_source_pixel_mse_among_feasible_candidates",
            }
            record = {
                "group": context["group"], "source": str(context["source"]),
                "source_sha256": sha256(context["source"]),
                "input": str(context["input"]), "input_sha256": context["input_sha256"],
                "input_pixel_audit": context["input_audit"],
                "original_500_pixel_audit": context["old_audit"],
                "old_image_cosine": context["old_image_cosine"],
                "components": str(context["components_path"]),
                "components_sha256": sha256(context["components_path"]),
                "cached_baseline_max_score_error": context["baseline_error"],
                "query_source": context["query_source"], "query_image_sha256": context["query_image_sha256"],
                "criteria": criteria, "candidates": [], "selected": None,
            }
            report["groups"].append(record)
            arrays, attacker = context["components"], context["attacker"]
            weights = list(dict.fromkeys(((.5, .5), (.9, .9), (key_weight, query_weight))))
            for scale in scales:
                image_array = scaled_candidate(context["source_array"], context["input_array"], scale)
                audit = pixel_audit(context["source_array"], image_array)
                if not audit["within_budget"]:
                    raise ValueError("Scaled candidate exceeds the original-source budget")
                path = directory / f"scale_{scale:g}.png"
                Image.fromarray(image_array).save(path, format="PNG")
                encoded = normalized(encoder.encode(path))
                if encoded.shape != arrays["images"][attacker].shape:
                    raise ValueError("New image embedding dimension differs from the cached vectors")
                images = np.array(arrays["images"], copy=True)
                images[attacker] = encoded
                row = {
                    "scale": scale, "image": str(path.resolve()), "image_sha256": sha256(path),
                    "pixel_audit": audit, "image_cosine": float(encoded @ arrays["qimages"][0]),
                    "rankings": [{"key_weight": alpha, "query_weight": beta,
                                  **evaluate(images, arrays["full"], arrays["qimages"], arrays["qtexts"],
                                             attacker, alpha, beta)} for alpha, beta in weights],
                }
                row["eligible"] = matches_criteria(row, criteria)
                record["candidates"].append(row)
                save()
            selected = select_minimum(record["candidates"])
            if selected is None:
                record["selection_status"] = "no_candidate_meets_all_criteria"
            else:
                path = directory / "selected.png"
                shutil.copyfile(selected["image"], path)
                record["selected"] = {**selected, "image": str(path.resolve()), "image_sha256": sha256(path)}
                record["selected_scale"] = selected["scale"]
                record["selection_status"] = "feasible_minimum_mse_among_tested_scales"
            save()
        report["status"] = "completed"
        save()
        return report
    except BaseException:
        report["status"] = "interrupted_or_failed"
        save()
        raise
    finally:
        if encoder is not None:
            encoder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True, help="已有报告与组件缓存的目录")
    parser.add_argument("--images", type=Path, required=True, help="组别到已优化PNG路径的映射文件")
    parser.add_argument("--output", type=Path, required=True, help="必须尚不存在的新结果目录")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--min-margin", type=float, default=.005)
    parser.add_argument("--min-image-cosine", type=float, default=None)
    parser.add_argument("--required-top1", type=int, default=20)
    parser.add_argument("--key-image-weight", type=float, default=.9)
    parser.add_argument("--query-image-weight", type=float, default=.9)
    parser.add_argument("--optimization-config", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true", help="仅检查输入与缓存，不加载模型或创建输出")
    args = parser.parse_args()
    try:
        scales = [finite_number(scale, "scale", 0., 1.) for scale in args.scales]
        if len(set(scales)) != len(scales) or len({f"{scale:g}" for scale in scales}) != len(scales):
            raise ValueError("Scales must be unique and have distinct output filenames")
        min_margin = finite_number(args.min_margin, "min_margin", 0.)
        if args.min_image_cosine is not None:
            finite_number(args.min_image_cosine, "min_image_cosine", -1., 1.)
        if not 1 <= args.required_top1 <= 20:
            raise ValueError("required_top1 must be in [1, 20]")
        key = finite_number(args.key_image_weight, "key_image_weight", 0., 1.)
        query = finite_number(args.query_image_weight, "query_image_weight", 0., 1.)
        contexts = load_inputs(args.probe.resolve(), args.images.resolve())
        if args.validate_only:
            print(json.dumps({"status": "inputs_and_baselines_validated_without_encoder",
                              "groups": [context["group"] for context in contexts]}, ensure_ascii=False))
            return
        if args.output.exists():
            raise ValueError("--output must be a new directory; existing experiment files will not be overwritten")
    except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
        parser.error(str(error))
    report = run(contexts, args.output.resolve(), scales=scales, device=args.device,
                 min_margin=min_margin, min_image_cosine=args.min_image_cosine,
                 required_top1=args.required_top1, key_weight=key, query_weight=query,
                 optimization_config=args.optimization_config, probe=args.probe.resolve(),
                 mapping_path=args.images.resolve())
    print(json.dumps({"report": str((args.output / "report.json").resolve()),
                      "selections": {group["group"]: group.get("selected_scale") for group in report["groups"]}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
