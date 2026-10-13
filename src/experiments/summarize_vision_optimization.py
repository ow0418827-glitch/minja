"""只用已有报告和图片生成检索、采样、早停与精修的中文汇总，不加载模型。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, features


GROUP_NAMES = {"01_camera": "相机", "02_chair": "椅子", "06_jeans": "牛仔裤"}
CASE_NAMES = {
    "samples20": "20 次采样，从原甲图优化 30 步",
    "samples5": "5 次采样，从原甲图优化 30 步",
    "refine_control": "旧更新规则，从旧 500 步图的 75% 扰动优化 30 步",
    "refine": "组合精修，从旧 500 步图的 75% 扰动优化 30 步",
    "early_stop": "5 次采样，从原甲图优化，最大 500 步并早停",
    "optimized": "5 次采样加原图梯度，从原甲图优化，最大 300 步并精修/早停",
}
EXPECTED_CASES = [
    ("benchmark", "06_jeans", "samples20"),
    ("benchmark_remaining", "06_jeans", "samples5"),
    ("benchmark_remaining", "06_jeans", "refine_control"),
    ("benchmark_remaining", "06_jeans", "refine"),
    ("benchmark_remaining", "06_jeans", "early_stop"),
    ("benchmark_optimized", "06_jeans", "optimized"),
    ("benchmark_refine_01_camera", "01_camera", "refine"),
    ("benchmark_refine_02_chair", "02_chair", "refine"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def provenance(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def picture_metrics(source: Path, candidate: Path, budget: float = 16) -> dict[str, Any]:
    # 与攻击核心共用解码入口；只导入图片工具，不加载模型或调用显卡。
    from src.vision.dual_rag_bridge import activate
    activate()
    from dual_rag.optimization.common.io import load_image
    original = np.asarray(load_image(source), dtype=np.float64)
    modified = np.asarray(load_image(candidate), dtype=np.float64)
    if original.shape != modified.shape:
        raise ValueError(f"Source/candidate shape differs: {source}: {original.shape}, {candidate}: {modified.shape}")
    difference = modified - original
    absolute = np.abs(difference)
    mse = float(np.mean(difference * difference))
    maximum = float(np.max(absolute))
    return {
        "shape": list(original.shape),
        "mae_levels": float(np.mean(absolute)),
        "mse_levels": mse,
        "max_levels": maximum,
        "budget_levels": budget,
        "within_budget": maximum <= budget,
        "budget_saturated_channel_fraction": float(np.mean(absolute == budget)),
        "psnr_db": 10 * math.log10(255**2 / mse) if mse else None,
        "source": str(source.resolve()),
        "image": str(candidate.resolve()),
        "source_decoded_rgb_sha256": hashlib.sha256(original.astype(np.uint8).tobytes()).hexdigest(),
    }


def check_stored_audit(metrics: dict[str, Any], audit: dict[str, Any], candidate: Path) -> None:
    fields = {
        "mae_levels": "mean_absolute_pixel_difference_levels",
        "mse_levels": "mean_squared_pixel_difference_levels",
        "max_levels": "max_absolute_pixel_difference_levels",
    }
    for actual_key, recorded_key in fields.items():
        if not math.isclose(metrics[actual_key], audit[recorded_key], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(
                f"Decoded pixels differ from the recorded audit for {candidate}: "
                f"{actual_key}={metrics[actual_key]} != {audit[recorded_key]}. "
                "Use the original experiment's Pillow/JPEG decoder environment "
                "(for these reports, use D:/Anaconda/envs/minja-vision/python.exe "
                "or D:/Anaconda/envs/dual-visual/python.exe)."
            )


def ranking(rows: list[dict[str, Any]], key: float, query: float, index: str | None = None) -> dict[str, Any]:
    found = [
        row for row in rows
        if math.isclose(row["key_weight"], key)
        and math.isclose(row["query_weight"], query)
        and (index is None or row.get("index_text") == index)
    ]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one ranking for {index}, {key}, {query}; found {len(found)}")
    return found[0]


def compact_rank(row: dict[str, Any]) -> dict[str, Any]:
    fields = ["top1", "rank_min", "rank_max", "mean_margin", "min_margin", "ranks"]
    fields += ["clean_source_top1", "clean_source_mean_margin"]
    return {field: row[field] for field in fields if field in row}


def link(path: str | Path, label: str) -> str:
    return f"[{label}](<{Path(path).resolve().as_posix()}>)"


def fmt(value: float | None, precision: int = 6) -> str:
    return "—" if value is None else f"{value:.{precision}f}"


def rank_range(row: dict[str, Any]) -> str:
    return str(row["rank_min"]) if row["rank_min"] == row["rank_max"] else f"{row['rank_min']}～{row['rank_max']}"


def build_summary(output: Path, probe_path: Path, old_run: Path | None) -> dict[str, Any]:
    probe = read_json(probe_path)
    old_run = (old_run or Path(probe["run"])).resolve()
    groups = probe["groups"]
    summary: dict[str, Any] = {
        "generated_at_beijing": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "output_root": str(output.resolve()),
        "runtime": {
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "pillow_version": Image.__version__,
            "jpeg_codec_version": features.version_codec("jpg"),
            "libjpeg_turbo_version": features.version_feature("libjpeg_turbo"),
        },
        "query_source": probe.get("query_source"),
        "trial_count": sum(len(group["trials"]) for group in groups),
        "all_baseline_ranks_aligned": all(group["baseline_aligned"] for group in groups),
        "baseline_max_score_error": max(group["baseline_max_score_error"] for group in groups),
        "input_reports": [provenance(probe_path)],
        "weight_sweep": [],
        "shrink": [],
        "original_500": [],
        "benchmarks": [],
        "comparisons": [],
        "pending": [],
        "postprocessing": None,
    }
    selections = [
        ("full", .5, .5), ("full", .9, .5), ("full", .8, .8), ("full", .9, .9),
        ("semantic", .5, .5), ("semantic", .9, .9), ("instruction", .5, .5),
    ]
    by_group = {group["group"]: group for group in groups}
    for group in groups:
        name = group["group"]
        source = Path(group["source_path"])
        for index, key, query in selections:
            row = ranking(group["weight_sweep"], key, query, index)
            summary["weight_sweep"].append({
                "group": name, "index_text": index, "key_weight": key, "query_weight": query,
                **compact_rank(row),
            })
        for shrink in group["shrink"]:
            if shrink["scale"] not in (.75, .5):
                continue
            summary["shrink"].append({
                "group": name, "scale": shrink["scale"],
                "metrics": picture_metrics(source, Path(shrink["path"])),
                "old_equal_ranking": compact_rank(ranking(shrink["results"], .5, .5)),
                "weighted_09_ranking": compact_rank(ranking(shrink["results"], .9, .9))
                if any(math.isclose(row["key_weight"], .9) and math.isclose(row["query_weight"], .9)
                       for row in shrink["results"]) else None,
            })
        metadata_path = old_run / "groups" / name / "attack" / "metadata.json"
        metadata = read_json(metadata_path)
        summary["input_reports"].append(provenance(metadata_path))
        old_metrics = picture_metrics(source, Path(group["attack_path"]))
        check_stored_audit(old_metrics, metadata["decoded_pixel_audit"], Path(group["attack_path"]))
        summary["original_500"].append({
            "group": name, "steps_completed": metadata["attack"]["steps_completed"],
            "samples": metadata["ssa"]["settings"]["ssa_samples"],
            "best_step": metadata["attack"]["best_step"],
            "timing": metadata["timing"],
            "global_similarity": metadata["after_attack_saved"]["global_similarity"],
            "local_similarity": metadata["after_attack_saved"]["local_similarity"],
            "cos2": metadata["after_attack_saved"]["global_similarity"][1],
            "stored_pixel_audit": metadata["decoded_pixel_audit"],
            "metrics": old_metrics,
            "old_equal_ranking": compact_rank(ranking(group["weight_sweep"], .5, .5, "full")),
            "weighted_09_ranking": compact_rank(ranking(group["weight_sweep"], .9, .9, "full")),
            "metadata": str(metadata_path.resolve()),
        })
    present = set()
    for report_path in sorted(output.glob("benchmark*/report.json")):
        report = read_json(report_path)
        summary["input_reports"].append(provenance(report_path))
        name = report["group"]
        if name not in by_group:
            raise ValueError(f"Benchmark group missing from probe: {name}")
        for case in report["cases"]:
            metadata_path = report_path.parent / case["case"] / "metadata.json"
            metadata = read_json(metadata_path)
            present.add((report_path.parent.name, name, case["case"]))
            summary["input_reports"].append(provenance(metadata_path))
            metrics = picture_metrics(Path(by_group[name]["source_path"]), Path(case["image"]))
            check_stored_audit(metrics, case["audit"], Path(case["image"]))
            if not metrics["within_budget"]:
                raise ValueError(f"Decoded candidate exceeds original-source budget: {case['image']}")
            summary["benchmarks"].append({
                "directory": report_path.parent.name, "group": name, "case": case["case"],
                "description": CASE_NAMES.get(case["case"], case["case"]),
                "samples": case["samples"], "steps_requested": case["steps_requested"],
                "steps_completed": case["steps_completed"], "stop_reason": case["stop_reason"],
                "best_step": metadata["attack"]["best_step"],
                "selected_step": metadata["attack"].get("selected_step", metadata["attack"]["best_step"]),
                "timing": case["timing"],
                "global_similarity": case["after"]["global_similarity"],
                "local_similarity": case["after"]["local_similarity"],
                "cos2": case["after"]["global_similarity"][1],
                "stored_pixel_audit": case["audit"], "metrics": metrics,
                "old_equal_ranking": compact_rank(ranking(case["rankings"], .5, .5)),
                "weighted_09_ranking": compact_rank(ranking(case["rankings"], .9, .9)),
                "early_stop_checks": metadata["ssa"].get("early_stop_checks", []),
                "alternate_delta_005_stop": case.get("alternate_delta_005_stop"),
                "settings": metadata["ssa"]["settings"],
                "initialization_provenance": metadata.get("initialization_provenance"),
                "metadata": str(metadata_path.resolve()), "image": case["image"],
            })
    for directory, group, case in EXPECTED_CASES:
        if (directory, group, case) in present:
            continue
        progress_path = output / directory / f"{case}_progress.json"
        pending = {
            "directory": directory, "group": group, "case": case,
            "status": "进行中" if progress_path.exists() else "待完成，可能排队或尚未写入报告",
        }
        if progress_path.exists():
            pending["progress"] = read_json(progress_path)
        summary["pending"].append(pending)
    case_map = {(row["group"], row["case"]): row for row in summary["benchmarks"]}
    old_map = {row["group"]: row for row in summary["original_500"]}
    for group in by_group:
        if (group, "samples20") in case_map and (group, "samples5") in case_map:
            control, proposed = case_map[(group, "samples20")], case_map[(group, "samples5")]
            summary["comparisons"].append(compare_records(
                group, "equal_30_updates_20_vs_5_samples", control, proposed))
        if (group, "refine_control") in case_map and (group, "refine") in case_map:
            control, proposed = case_map[(group, "refine_control")], case_map[(group, "refine")]
            summary["comparisons"].append(compare_records(
                group, "same_75_percent_initial_candidate_refinement", control, proposed))
        for case in ("early_stop", "optimized", "refine"):
            if (group, case) in case_map:
                summary["comparisons"].append(compare_records(
                    group, f"original_500_vs_{case}", old_map[group], case_map[(group, case)]))
    postprocess_path = output / "postprocess_scale" / "report.json"
    if postprocess_path.is_file():
        postprocess = read_json(postprocess_path)
        summary["input_reports"].append(provenance(postprocess_path))
        for record in postprocess["groups"]:
            selected = record.get("selected")
            if selected is None:
                continue
            recomputed = picture_metrics(Path(record["source"]), Path(selected["image"]))
            for key in ("mae_levels", "mse_levels", "max_levels"):
                if not math.isclose(recomputed[key], selected["pixel_audit"][key], rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError(f"Postprocessed selected image audit differs: {record['group']}/{key}")
            selected["recomputed_metrics"] = recomputed
            old_metrics = old_map[record["group"]]["metrics"]
            selected["mae_reduction_fraction_vs_original_500"] = 1 - recomputed["mae_levels"] / old_metrics["mae_levels"]
            selected["mse_reduction_fraction_vs_original_500"] = 1 - recomputed["mse_levels"] / old_metrics["mse_levels"]
        summary["postprocessing"] = {**postprocess, "report_path": str(postprocess_path.resolve())}
    return summary


def compare_records(group: str, comparison: str, control: dict[str, Any],
                    proposed: dict[str, Any]) -> dict[str, Any]:
    before_time, after_time = [row["timing"]["optimization_loop_seconds"] for row in (control, proposed)]
    before_mae, after_mae = [row["metrics"]["mae_levels"] for row in (control, proposed)]
    before_mse, after_mse = [row["metrics"]["mse_levels"] for row in (control, proposed)]
    total_time_comparable = comparison != "original_500_vs_refine"
    return {
        "group": group, "comparison": comparison,
        "control_loop_seconds": before_time, "proposed_loop_seconds": after_time,
        "total_time_comparable": total_time_comparable,
        "loop_time_ratio": after_time / before_time if total_time_comparable else None,
        "loop_speed_factor": before_time / after_time if total_time_comparable else None,
        "loop_time_reduction_fraction": 1 - after_time / before_time if total_time_comparable else None,
        "cos2_difference": proposed["cos2"] - control["cos2"],
        "mae_reduction_fraction": 1 - after_mae / before_mae,
        "mse_reduction_fraction": 1 - after_mse / before_mse,
    }


def render(summary: dict[str, Any], test_log: Path) -> str:
    lines = [
        "# 检索权重与图片优化实测汇总", "",
        f"生成时间：{summary['generated_at_beijing']}。此文件由只读汇总器重新生成，精确数值保存在旁边的结构化汇总中。", "",
        "评测范围为相机、椅子、牛仔裤三组，每组固定 15 条记忆、同一张乙图和 20 句实际执行的模糊购买请求，共 60 次既有查询。首位次数表示甲记忆的检索排名，不能解释为购物成功率或一般正常检索准确率。没有启动完整购物回合。", "",
        "全文索引包含历史需求、操作与观察等记录；仅需求索引保留历史需求；语义短索引保留历史需求、真实购买标题和最终选项。所有记忆使用同一索引规则。权重扫描复用已编码向量，没有重复运行模型。", "",
        "## 旧结果复现与可信度", "",
        f"实际查询读取自落盘购物记录，未使用存档的具体需求句。{summary['trial_count']} 次基线名次全部一致：{'是' if summary['all_baseline_ranks_aligned'] else '否'}；分数最大绝对偏差为 {summary['baseline_max_score_error']:.12g}。", "",
        "分差为甲记忆得分减去最强其他记忆得分，正值表示领先。图片与文字加权后再次归一化，权重并非实际得分中的线性占比；文字带来的分差也不是固定扣分。", "",
        "## 提高图片权重", "",
        "记忆端和查询端分别将“图片权重×单位图像向量＋文字权重×单位文本向量”的和再次归一化，文字权重为 1 减图片权重。0.5/0.5 保留原等权算术路径。", "",
        "| 索引内容 | 记忆图片权重/查询图片权重 | 组别 | 首位次数 | 名次范围 | 平均分差 | 最差分差 | 原甲图首位次数 | 原甲图平均分差 |",
        "|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    index_names = {"full": "全文", "semantic": "语义短索引", "instruction": "仅历史需求"}
    for row in summary["weight_sweep"]:
        lines.append(f"| {index_names[row['index_text']]} | {row['key_weight']:.1f}/{row['query_weight']:.1f} | {GROUP_NAMES[row['group']]} | {row['top1']}/20 | {rank_range(row)} | {row['mean_margin']:+.6f} | {row['min_margin']:+.6f} | {row.get('clean_source_top1', '—')}/20 | {row.get('clean_source_mean_margin', 0):+.6f} |")
    lines += ["", "原甲图首位次数只用于对照未扰动的这条甲记忆，不代表全部正常记忆的检索准确率。", "",
        "## 只收缩既有扰动幅度", "",
        "候选图＝原甲图＋比例×（旧扰动图－原甲图），再按八位图取整。尺寸和修改区域掩码没有缩放；部分通道取整后可能恢复原值，但本评测不据此宣称修改位置更少。所有指标相对原甲图计算。", "",
        "| 组别 | 保留幅度 | 平均绝对改动/灰度级 | 均方改动/灰度级平方 | 最大改动 | 原等权首位次数 | 名次范围 | 最差分差 |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary["shrink"]:
        m, r = row["metrics"], row["old_equal_ranking"]
        lines.append(f"| {GROUP_NAMES[row['group']]} | {row['scale']:.0%} | {m['mae_levels']:.6f} | {m['mse_levels']:.6f} | {m['max_levels']:.0f} | {r['top1']}/20 | {rank_range(r)} | {r['min_margin']:+.6f} |")
    lines += ["", "## 三组旧 500 步图与真实图片级对照", "",
        "下表图像相似度取最终保存图的第二个代理全局余弦相似度，直接读取落盘的保存图评分，未将综合目标分数当作图像余弦相似度。时间为优化循环秒数；三代理全局与局部得分、总调用时间、完整配置和源报告摘要另存结构化汇总。", "",
        "| 组别 | 案例与起点 | 采样数 | 实际步数/上限 | 循环时间/秒 | 第二代理余弦 | 平均绝对改动 | 均方改动 | 最大改动 | 原等权首位 | 两端0.9首位 | 两端0.9最差分差 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["original_500"]:
        m, r, w = row["metrics"], row["old_equal_ranking"], row["weighted_09_ranking"]
        lines.append(f"| {GROUP_NAMES[row['group']]} | 旧 500 步，从原甲图 | {row['samples']} | {row['steps_completed']}/500 | {row['timing']['optimization_loop_seconds']:.3f} | {row['cos2']:.9f} | {m['mae_levels']:.6f} | {m['mse_levels']:.6f} | {m['max_levels']:.0f} | {r['top1']}/20 | {w['top1']}/20 | {w['min_margin']:+.6f} |")
    for row in summary["benchmarks"]:
        m, r, w = row["metrics"], row["old_equal_ranking"], row["weighted_09_ranking"]
        lines.append(f"| {GROUP_NAMES[row['group']]} | {row['description']} | {row['samples']} | {row['steps_completed']}/{row['steps_requested']} | {row['timing']['optimization_loop_seconds']:.3f} | {row['cos2']:.9f} | {m['mae_levels']:.6f} | {m['mse_levels']:.6f} | {m['max_levels']:.0f} | {r['top1']}/20 | {w['top1']}/20 | {w['min_margin']:+.6f} |")
    lines += ["", "所有已完成的新图均按实际保存图重新审计 16 灰度级预算。平均绝对改动和均方改动越低，表示幅度更小；这两项不等于人眼感知评分。", "",
        "## 早停的真实触发轨迹", "",
        "最低 200 步，每 50 步检查最近 50 步历史最佳固定八位候选评分增加量，连续两个窗口低于 0.002 才停止。任意微小创新高不会自动清零计数。保存历史最佳或符合小修改选优规则的候选，真实完成步数独立记录。", "",
        "| 组别/案例 | 完成步数 | 窗口改善 | 连续低收益计数 | 是否触发 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["benchmarks"]:
        for check in row["early_stop_checks"]:
            case_name = "纯早停" if row["case"] == "early_stop" else "组合优化"
            lines.append(f"| {GROUP_NAMES[row['group']]}/{case_name} | {check['completed_steps']} | {check['improvement']:.12g} | {check['low_gain_windows']} | {'是' if check['triggered'] else '否'} |")
        if row["case"] == "early_stop":
            lines += ["", f"牛仔裤 5 次采样纯早停实测完成 {row['steps_completed']} 步，优化循环 {row['timing']['optimization_loop_seconds']:.6f} 秒。0.005 门槛在同一真实轨迹中的备用判断于 {row['alternate_delta_005_stop']} 步触发；这是同时观察同一条轨迹，不是另外完成了一次独立优化。", ""]
    lines += [
        "平台早停只证明近期代理评分改善弱，不能证明真实检索已成功。与旧 20 次采样 500 步相比的总耗时差同时包含采样减少和更新次数减少，不能全部归因于早停；若估算同一 5 次采样轨迹节省的 150 步计算，应标为外推而非已完成实测。", "",
        "## 预算边界与峰值信噪比", "",
        "PSNR(峰值信噪比)由八位保存图相对原甲图的均方误差计算，峰值使用 255；它衡量逐通道误差，不替代感知相似度。预算饱和率统计绝对改动恰好等于 16 的颜色通道比例，不统计修改位置数量。", "",
        "| 组别 | 候选 | 预算饱和通道比例 | 峰值信噪比/分贝 |",
        "|---|---|---:|---:|",
    ]
    for row in summary["original_500"]:
        m = row["metrics"]
        lines.append(f"| {GROUP_NAMES[row['group']]} | 旧500步 | {m['budget_saturated_channel_fraction']:.6%} | {fmt(m['psnr_db'])} |")
    for row in summary["benchmarks"]:
        m = row["metrics"]
        lines.append(f"| {GROUP_NAMES[row['group']]} | {row['description']} | {m['budget_saturated_channel_fraction']:.6%} | {fmt(m['psnr_db'])} |")
    lines += ["", "## 计算量与结论的适用范围", "",
        "20 次与 5 次采样案例从相同原甲图出发、都执行 30 步，用于比较等更新次数的采样效果；计算量不同，短程结果不能保证后续 500 步质量。", "",
        "原规则继续优化与组合精修都从旧 500 步图的 75% 扰动开始，都是 30 步；组合精修每步另含一次原图梯度，并包含步长、动量、模拟强度、均方修改量惩罚和候选小修改选优的共同变化。其起点已消耗原先 500 步计算，只能报告增量精修耗时，不能将该耗时当成从零生成的总时间，也不能从此对照确定某个组件单独的贡献。", "",
        "组合参数：步长末端倍率 0.2（0.5 灰度级降至 0.1），动量末端 0.8，频谱幅度末端 0.1，噪声标准差末端 4，未模拟梯度权重 0.5，预算归一化均方惩罚权重 0.01，候选分数容忍度 0.002。始终以原甲图为预算中心。", "",
        "图像权重收益、精修收益、采样收益和早停收益应分别解释。测试覆盖现有三个图对和 60 个既有查询，没有证明一般模型迁移、其他查询或其他图片上的提升。", "",
        "| 组别 | 对比范围 | 循环时间减少比例 | 平均绝对改动减少比例 | 均方改动减少比例 | 第二代理余弦变化 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    comparison_names = {
        "equal_30_updates_20_vs_5_samples": "原甲图同为30步，20次采样→5次采样",
        "same_75_percent_initial_candidate_refinement": "75%起点同为30步，旧规则→组合精修",
        "original_500_vs_early_stop": "旧20次500步→新5次350步纯早停，包含多项差异",
        "original_500_vs_optimized": "旧20次500步→新原甲图组合优化，包含多项差异",
        "original_500_vs_refine": "旧500步图→缩放后精修，仅比较增量耗时",
    }
    for row in summary["comparisons"]:
        time_value = "不作总耗时比较" if row["comparison"] == "original_500_vs_refine" else f"{row['loop_time_reduction_fraction']:.3%}"
        lines.append(f"| {GROUP_NAMES[row['group']]} | {comparison_names[row['comparison']]} | {time_value} | {row['mae_reduction_fraction']:.3%} | {row['mse_reduction_fraction']:.3%} | {row['cos2_difference']:+.9f} |")
    lines += ["", "负的减少比例表示增加。上述百分比由完整浮点值计算，未用表格中取舍小数后的数值二次计算。组合精修依赖旧500步起点，因此与旧500步从零优化不计算时间比或加速率；只记录增量精修耗时。", "",
        "## 新优化图的幅度后处理", "",
    ]
    postprocess = summary["postprocessing"]
    if postprocess is None:
        lines += ["尚无后处理报告，不报告其效果。", ""]
    else:
        lines += [
            "此步骤将新优化图相对原甲图的差值按幅度缩小，再按八位图取整。复用固定库中其他14条记忆、全文文本和20条实际查询向量，只对缩放候选重新计算检索图像编码；未更改优化过程，未加载三代理，未重跑购物回合。", "",
            "默认候选比例为100%、87.5%、75%、62.5%、50%、37.5%、25%。选择条件为两端图片权重0.9下20次全部首位、最差领先至少0.005，且重新编码的图像余弦不低于缓存旧500步图片与乙图的余弦；在满足条件的已测比例中选择均方改动最小者。", "",
            f"后处理状态：{'已完成' if postprocess['status'] == 'completed' else '进行中或未完整完成'}；本次工具累计耗时 {postprocess['elapsed_seconds']:.6f} 秒，包含一个检索代理的加载、候选编码与文件处理。以下相似度来自缩放图的独立重新编码，未套用原优化图评分。", "",
            "| 组别 | 保留新扰动幅度 | 缩放图余弦 | 旧500步图余弦门槛 | 平均绝对改动 | 均方改动 | 最大改动 | 原等权首位 | 两端0.9首位 | 两端0.9最差领先 | 相对旧500步平均改动减少 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for record in postprocess["groups"]:
            selected = record.get("selected")
            if selected is None:
                continue
            m = selected["pixel_audit"]
            r, w = ranking(selected["rankings"], .5, .5), ranking(selected["rankings"], .9, .9)
            lines.append(f"| {GROUP_NAMES[record['group']]} | {selected['scale']:.1%} | {selected['image_cosine']:.9f} | {record['old_image_cosine']:.9f} | {m['mae_levels']:.6f} | {m['mse_levels']:.6f} | {m['max_levels']} | {r['top1']}/20 | {w['top1']}/20 | {w['min_margin']:+.6f} | {selected['mae_reduction_fraction_vs_original_500']:.3%} |")
        lines += ["", "这只是已知固定库与既有查询上的后处理推荐，不能说明任意比例或未知查询同样可行；只在七个已测比例内选择，并不宣称全局最小扰动。最大改动、平均改动和检索图像相似度是独立指标。", ""]
        for record in postprocess["groups"]:
            selected = record.get("selected")
            if selected is None:
                lines.append(f"- {GROUP_NAMES[record['group']]}：未找到同时符合全部门槛的候选，或该组仍未结束。")
            else:
                lines.append(f"- {link(selected['image'], GROUP_NAMES[record['group']] + '推荐后处理图')}。原优化图保留原样；推荐图没有伪造或覆盖攻击生成元数据。")
        lines += ["", link(postprocess["report_path"], "完整后处理候选、输入摘要、配置与编码器快照"), ""]
    lines += [
        "## 尚未完成的报告", "",
    ]
    if not summary["pending"]:
        lines += ["预定案例均已写入报告；此处依据报告存在性判断，未检查其他外部任务。", ""]
    for row in summary["pending"]:
        progress = row.get("progress")
        detail = f"；最近落盘进度 {progress['completed_steps']} 步，耗时 {progress['elapsed_seconds']:.1f} 秒" if progress else ""
        lines.append(f"- {GROUP_NAMES[row['group']]}：{CASE_NAMES[row['case']]}，{row['status']}{detail}。")
    lines += ["", "缺少完整报告的案例不进入已完成结果表，不提前宣称其排名、修改量或时间效果。可以在模型任务完成后重新运行本汇总器，无需重算既有成功评测。", "",
        "## 项目验证与来源", "",
        "此前全项目测试运行 814 项通过，耗时 391.82 秒；该次收集发生在新增入口测试加入之前，新增入口另一次运行 2 项通过。两次运行按各自范围记录，不把历史多次相关子集测试相加当成去重总数。本汇总器不运行或重现这些测试。", "",
        f"本次图片审计使用 Pillow(图片处理库) {summary['runtime']['pillow_version']}，JPEG(联合图像专家组图片格式)解码器版本 {summary['runtime']['jpeg_codec_version']}。汇总器要求重解码的改动指标与落盘审计一致；环境更换可能导致有损图片解码差异，应使用原测试环境复现。", "",
        link(test_log, "任务进度与测试运行记录"), "",
        link(Path(summary["output_root"]) / "summary.json", "结构化汇总（保存完整浮点值与配置）"), "",
    ]
    for item in summary["input_reports"]:
        lines.append(f"- {link(item['path'], '原始报告或元数据')}；文件摘要：`{item['sha256']}`。")
    lines += ["", "汇总器只覆盖本次新输出根中的自身汇总文件，旧实验的图片、元数据与报告始终只读。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path("src/output/vision_optimization_20261008_0200"))
    parser.add_argument("--probe", type=Path, default=None)
    parser.add_argument("--old-run", type=Path, default=None)
    parser.add_argument("--test-log", type=Path,
                        default=Path("C:/Users/22106/.codex/task-notes/minja-20261008-0200/progress.md"))
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_dir():
        parser.error("Output root must already exist; this tool only summarizes existing experiments")
    probe_path = (args.probe or output / "probe" / "report.json").resolve()
    summary = build_summary(output, probe_path, args.old_run)
    summary["validation_runs"] = [
        {"scope": "project_suite_collected_before_new_entry_tests", "passed": 814, "seconds": 391.82},
        {"scope": "new_entry_profile_tests_separate_run", "passed": 2},
    ]
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "RESULTS.md").write_text(render(summary, args.test_log), encoding="utf-8")
    print(json.dumps({
        "summary": str(output / "summary.json"), "report": str(output / "RESULTS.md"),
        "baseline_trials": summary["trial_count"], "benchmark_cases": len(summary["benchmarks"]),
        "pending_cases": len(summary["pending"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
