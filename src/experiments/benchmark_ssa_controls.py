"""三代理图片级对照：采样、预算内精修与真实窗口早停；不运行购物。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from src.vision.dual_rag_bridge import activate
from src.vision.ssa_attack import run_ssa_attack, PlateauStopper, ssa_settings
from src.experiments.probe_vision_optimization import evaluate, unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", default="06_jeans")
    parser.add_argument("--cases", nargs="+", choices=["samples20", "samples5", "refine_control", "refine", "early_stop", "optimized"],
                        default=["samples20", "samples5", "refine_control", "refine", "early_stop"])
    parser.add_argument("--short-steps", type=int, default=30)
    parser.add_argument("--optimized-steps", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.probe / "report.json").read_text(encoding="utf-8"))
    group = next(g for g in report["groups"] if g["group"] == args.group)
    components = np.load(args.probe / f"{args.group}_components.npz")
    root = activate()
    from dual_rag.original_attack import resolve_optimization
    from dual_rag.optimization.attack_core.surrogates import load_surrogates
    from dual_rag.optimization.attack_core import objective as obj
    from dual_rag.optimization.common.io import load_image, pil_to_tensor, save_tensor_png
    import torch
    config = resolve_optimization(root / "configs/optimization.json")
    config = replace(config, attack=replace(config.attack, epsilon=16 / 255, text_weight=0.0),
                     acceptance=replace(config.acceptance, decoded_max_pixel_difference=16),
                     runtime=replace(config.runtime, allow_downloads=False))
    print("Loading three local surrogates", flush=True)
    models = load_surrogates(config.surrogates, device="cuda", precision=config.attack_precision,
        allow_downloads=False, allow_device_fallback=False, require_true_local_tokens=True, hash_weights=False)
    target_sha = group["trials"][0]["query_sha"]
    target_path = next(path for path in (Path(report["run"]) / "assets").iterdir()
        if path.is_file() and __import__("hashlib").sha256(path.read_bytes()).hexdigest() == target_sha)
    source_path = Path(group["source_path"])
    original_gradient = obj.sequential_image_gradient
    original_observe = PlateauStopper.observe
    state = {}

    def gradient(*a, **kw):
        result = original_gradient(*a, **kw)
        state["calls"] += 1
        if state["calls"] % state["calls_per_step"] == 0:
            step = state["calls"] // state["calls_per_step"]
            if step == 1 or step % 10 == 0:
                print(f"{state['case']}: completed update {step}, elapsed {time.perf_counter()-state['started']:.1f}s", flush=True)
        return result

    def observe(self, step, score):
        state["best_curve"].append({"step": step, "best_score": score})
        if state.get("alternate") is not None and not state.get("alternate_stopped"):
            if original_observe(state["alternate"], step, score):
                state["alternate_stopped"] = step
        if step % 50 == 0:
            (args.output / f"{state['case']}_progress.json").write_text(json.dumps({
                "case": state["case"], "completed_steps": step, "best_score": score,
                "elapsed_seconds": time.perf_counter() - state["started"],
                "alternate_delta_005_stop": state.get("alternate_stopped"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        return original_observe(self, step, score)

    obj.sequential_image_gradient = gradient
    PlateauStopper.observe = observe
    records = []
    try:
        for case in args.cases:
            samples = 20 if case == "samples20" else 5
            steps = 500 if case == "early_stop" else args.optimized_steps if case == "optimized" else args.short_steps
            options = {}
            initial = None
            if case in ("refine_control", "refine"):
                initial = args.probe / f"{args.group}_scale_0.75.png"
            if case in ("refine", "optimized"):
                options = {"refine_start": 0.0, "step_scale_end": .2, "momentum_end": .8,
                    "rho_end": .1, "sigma_end": 4., "clean_gradient_weight": .5,
                    "mse_weight": .01, "selection_tolerance": .002}
            if case == "optimized":
                options.update(early_stop=True, min_steps=200, check_every=50,
                               window=50, patience=2, min_delta=.002)
            if case == "early_stop":
                options = {"early_stop": True, "min_steps": 200, "check_every": 50,
                           "window": 50, "patience": 2, "min_delta": .002}
            state.clear()
            state.update(case=case, calls=0, calls_per_step=samples + int(options.get("clean_gradient_weight", 0) > 0),
                         started=time.perf_counter(), best_curve=[], alternate=None)
            if case == "early_stop":
                state["alternate"] = PlateauStopper(ssa_settings(ssa_options={**options, "min_delta": .005}))
            print(f"Starting {case}: {steps} max updates, {samples} samples", flush=True)
            path, metadata = run_ssa_attack(source_path, target_path, config,
                steps=steps, output_directory=args.output / case, ssa_samples=samples,
                ssa_options=options, initial_image=initial, surrogates=models)
            with torch.no_grad():
                image = pil_to_tensor(load_image(path), "cuda")
                encoded = unit(models[1].encode_image(image).global_embedding.detach().cpu().numpy().reshape(-1))
            images = list(components["images"])
            images[group["attacker_index"]] = encoded
            rankings = [{"key_weight": alpha, "query_weight": beta,
                **evaluate(images, components["full"], components["qimages"], components["qtexts"],
                           group["attacker_index"], alpha, beta)}
                for alpha, beta in [(.5, .5), (.9, .5), (.9, .9), (1., 1.)]]
            record = {"case": case, "samples": samples, "steps_requested": steps,
                "steps_completed": metadata["attack"]["steps_completed"],
                "stop_reason": metadata["attack"]["stop_reason"], "timing": metadata["timing"],
                "audit": metadata["decoded_pixel_audit"], "after": metadata["after_attack_saved"],
                "rankings": rankings, "best_curve": state["best_curve"],
                "alternate_delta_005_stop": state.get("alternate_stopped"), "image": str(path)}
            records.append(record)
            (args.output / "report.json").write_text(json.dumps({"group": args.group, "cases": records},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: record[key] for key in ("case", "steps_completed", "stop_reason", "audit", "timing")}), flush=True)
    finally:
        obj.sequential_image_gradient = original_gradient
        PlateauStopper.observe = original_observe


if __name__ == "__main__":
    main()
