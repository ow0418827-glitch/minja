"""Plot decoded checkpoint cosines and separate production stopping from audit.

This module reads saved metadata and images only. It never loads a model.
The noisy per-guidance-step training history is deliberately not plotted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from src.paths import PROJECT


OLD_REPORT = PROJECT / "src/output/diffusion_group7_20261011_stronger/comparison/report.json"
OLD_THIRTY_SCORE = 0.7717060162768721
SOURCE_SCORE = 0.6722690085526885
SCORE_KEY = "actual_model2_target_cosine"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_path(attack):
    attack = Path(attack).resolve()
    if attack.is_file():
        return attack
    for path in (attack / "metadata.json", attack / "attack/metadata.json"):
        if path.is_file():
            return path
    raise FileNotFoundError("输入中没有攻击元数据文件")


def resolve_image(value, folder):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (folder / path).resolve()


def check_candidates(metadata, folder):
    checks = metadata.get("checkpoint_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("需要含完整尾程候选评分的 checkpoint_checks；不能用每步训练历史代替")
    result, previous, best = [], -1, -math.inf
    for check in checks:
        step = check.get("actual_step", check.get("completed_steps"))
        score = check.get(SCORE_KEY)
        if isinstance(step, bool) or not isinstance(step, int) or step <= previous:
            raise ValueError("候选的实际引导步数必须为严格递增的非负整数")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("每个检查点必须包含固定第二代理的有限目标余弦")
        if check.get("completed_steps", step) != step:
            raise ValueError("候选实际步数与完成步数字段不一致")
        if not check.get("image") or not check.get("image_sha256"):
            raise ValueError("候选缺少保存图片路径或内容摘要")
        path = resolve_image(check["image"], folder)
        if digest(path) != check["image_sha256"]:
            raise ValueError(f"候选保存图内容摘要不匹配：第{step}步")
        best = max(best, float(score))
        if "best_score" in check and abs(float(check["best_score"]) - best) > 5e-6:
            raise ValueError(f"候选历史最佳评分与真实目标余弦不一致：第{step}步")
        result.append({**check, "actual_step": step, SCORE_KEY: float(score),
                       "image": str(path), "running_best_cosine": best})
        previous = step
    if result[0]["actual_step"] != 0:
        raise ValueError("候选序列必须以零步无引导重建开始")
    return result


def _by_step(checks, step, label):
    match = [item for item in checks if item["actual_step"] == step]
    if len(match) != 1:
        raise ValueError(f"{label}所指步数没有唯一的完整尾程候选")
    return match[0]


def references(metadata):
    result = {"source_cosine": SOURCE_SCORE, "old_thirty_guidance_cosine": OLD_THIRTY_SCORE,
              "source": "previous_group7_saved_image_comparison", "report": str(OLD_REPORT),
              "scores_reencoded_by_this_summary": False}
    if not OLD_REPORT.is_file():
        result["report_available"] = False
        return result
    old = read_json(OLD_REPORT)
    surrogate = metadata.get("surrogates", [{}, {}])
    old_weight = old.get("encoder_snapshot", {}).get("surrogate", {}).get("weight_sha256")
    new_weight = surrogate[1].get("weight_sha256") if len(surrogate) > 1 else None
    if old_weight and new_weight and old_weight != new_weight:
        raise ValueError("旧三十步比较与当前检查点使用不同第二代理权重")
    expected = metadata.get("input_sha256", {})
    for key in ("source", "target"):
        if expected.get(key) and old.get(key, {}).get("sha256") != expected[key]:
            raise ValueError("旧三十步比较与当前实验的原图或目标图不同")
    rows = {row["name"]: row for row in old.get("rows", []) if row.get("defense") == "none"}
    if "source" in rows and "diffusion" in rows:
        result.update(source_cosine=rows["source"]["target_cosine"],
                      old_thirty_guidance_cosine=rows["diffusion"]["target_cosine"],
                      report_sha256=digest(OLD_REPORT), report_available=True,
                      old_image=rows["diffusion"]["input_image"],
                      old_image_sha256=rows["diffusion"]["input_sha256"])
    return result


def build_summary(metadata, checks, source_path):
    attack = metadata.get("attack", {})
    audit = metadata.get("post_stop_audit", {}) or {}
    decision = metadata.get("early_stop_decision") or {}
    if metadata.get("selection_metric", attack.get("selection_metric")) != SCORE_KEY:
        raise ValueError("早停选择指标不是完整候选固定第二代理目标余弦")
    production_steps = attack.get("steps_completed")
    selected_step = attack.get("selected_step")
    if not isinstance(production_steps, int) or isinstance(production_steps, bool):
        raise ValueError("元数据缺少生产主程实际引导步数")
    production = [row for row in checks if row["actual_step"] <= production_steps]
    if not production:
        raise ValueError("生产主程候选为空")
    selected = _by_step(checks, selected_step, "生产选中图")
    if selected_step > production_steps:
        raise ValueError("生产输出不能选择早停后的审计候选")
    production_best = max(production, key=lambda row: row[SCORE_KEY])
    if selected[SCORE_KEY] < production_best[SCORE_KEY] - 5e-6:
        raise ValueError("生产选中图不是停止前的最佳候选")
    triggered_checks = [row for row in production if row.get("should_stop")]
    triggered = bool(decision.get("triggered", bool(triggered_checks)))
    trigger_step = decision.get("trigger_step", decision.get("actual_step", decision.get("completed_steps"))) if triggered else None
    if triggered and trigger_step is None:
        trigger_step = triggered_checks[0]["actual_step"] if triggered_checks else production_steps
    if triggered and trigger_step != production_steps:
        raise ValueError("停止决策步数与生产主程实际完成步数不一致")
    audit_enabled = bool(audit.get("enabled", metadata.get("settings", {}).get("audit_after_stop", False)))
    audit_rows = [row for row in checks if row["actual_step"] > production_steps]
    all_best = max(checks, key=lambda row: row[SCORE_KEY])
    audit_best_step = audit.get("audit_selected_step")
    if audit_best_step is None:
        audit_best_step = all_best["actual_step"]
    audit_best = _by_step(checks, audit_best_step, "审计最佳图")
    if audit_best[SCORE_KEY] < all_best[SCORE_KEY] - 5e-6:
        raise ValueError("审计最佳图与全序列实际最佳分数不一致")
    audit_final = checks[-1]
    final_steps = audit.get("audit_total_guidance_steps")
    if final_steps is None:
        final_steps = audit_final["actual_step"]
    if final_steps != audit_final["actual_step"]:
        raise ValueError("审计总引导步数与最后候选步数不一致")
    if audit_rows and not audit_enabled:
        raise ValueError("存在早停后候选但未声明审计分支")
    if attack.get("executed_guidance_steps", final_steps) != final_steps:
        raise ValueError("实际累计引导步数与最后检查点不一致")
    saved_score = metadata.get("after_attack_saved", {}).get("global_similarity", [])
    if len(saved_score) > 1 and abs(saved_score[1] - selected[SCORE_KEY]) > 2e-5:
        raise ValueError("生产最终保存图重评分与选中检查点评分不一致")
    for path_key, hash_key, expected in (
            ("output_image", "output_image_sha256", selected),
            ("audit_best_image", "audit_best_sha256", audit_best),
            ("audit_final_image", "audit_final_sha256", audit_final)):
        record = metadata if path_key == "output_image" else audit
        if record.get(path_key):
            actual_hash = digest(resolve_image(record[path_key], source_path.parent))
            if actual_hash != record.get(hash_key) or actual_hash != expected["image_sha256"]:
                raise ValueError("生产或审计输出图与对应候选内容摘要不一致")
    delta = audit_best[SCORE_KEY] - selected[SCORE_KEY]
    settings = metadata.get("settings", {})
    threshold = settings.get("min_delta", 0.002)
    production_checks = [row for row in checks if row["actual_step"] <= production_steps]
    refs = references(metadata)

    def point(row):
        return {"actual_step": row["actual_step"], "target_cosine": row[SCORE_KEY],
                "image": row["image"], "image_sha256": row["image_sha256"],
                "cumulative_elapsed_seconds": row.get("cumulative_elapsed_seconds")}

    def sum_time(rows, key):
        values = [row.get(key) for row in rows]
        return sum(float(value) for value in values if isinstance(value, (int, float))) if any(value is not None for value in values) else None

    def tails(rows):
        return sum(int(row.get("candidate_tail_steps", 0)) for row in rows if not row.get("reused_reconstruction", False))

    return {"schema_version": 1, "status": "completed", "input_metadata": str(source_path),
            "input_metadata_sha256": digest(source_path), "score_field": SCORE_KEY,
            "score_semantics": "checkpoint_complete_unguided_tail_original_size_8bit_fixed_surrogate2_target_cosine",
            "training_history_used_as_curve": False, "model_loaded": False,
            "parameters": settings, "attack": attack,
            "early_stop_triggered": triggered, "early_stop_trigger_step": trigger_step,
            "production_selected": point(selected), "audit_best": point(audit_best), "audit_final": point(audit_final),
            "post_stop_audit_enabled": audit_enabled, "audit_has_additional_guidance": bool(audit_rows),
            "post_stop_audit": audit, "actual_total_guidance_steps": final_steps,
            "production_guidance_steps": production_steps,
            "post_stop_audit_additional_guidance_steps": final_steps - production_steps,
            "audit_best_gain_over_production": delta,
            "audit_gain_exceeds_one_window_threshold": delta >= threshold,
            "audit_final_minus_production_cosine": audit_final[SCORE_KEY] - selected[SCORE_KEY],
            "production_gain_over_old_thirty": selected[SCORE_KEY] - refs["old_thirty_guidance_cosine"],
            "audit_best_gain_over_old_thirty": audit_best[SCORE_KEY] - refs["old_thirty_guidance_cosine"],
            "candidate_count": len(checks), "verified_candidate_image_hashes": len(checks),
            "candidate_tail_steps_excluding_reused_reconstruction": tails(checks),
            "production_candidate_tail_steps": tails(production_checks), "audit_candidate_tail_steps": tails(audit_rows),
            "production_checkpoint_tail_seconds": sum_time(production_checks, "tail_seconds"),
            "audit_checkpoint_tail_seconds": sum_time(audit_rows, "tail_seconds"),
            "production_checkpoint_evaluation_seconds": sum_time(production_checks, "evaluation_seconds"),
            "audit_checkpoint_evaluation_seconds": sum_time(audit_rows, "evaluation_seconds"),
            "production_stop_checkpoint_elapsed_seconds": production_checks[-1].get("cumulative_elapsed_seconds"),
            "last_checkpoint_elapsed_seconds": audit_final.get("cumulative_elapsed_seconds"),
            "timing": metadata.get("timing", {}), "references": refs,
            "noise_network_forward_counts": metadata.get("noise_network_forward_counts", {}),
            "scope": "one_schedule_and_pair; maximum_observed_at_saved_checkpoints; no_global_maximum_claim; audit_computation_not_stopped"}


def draw_curve(output, checks, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError("未找到绘制中文科学曲线所需字体")
    font = FontProperties(fname=str(font_path))
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "savefig.dpi": 200})
    figure, (axis, gain_axis) = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1]})
    steps = [row["actual_step"] for row in checks]
    scores = [row[SCORE_KEY] for row in checks]
    best = [row["running_best_cosine"] for row in checks]
    axis.plot(steps, scores, marker="o", markersize=4, color="#236ca0", linewidth=1.6, label="完整尾程候选的目标余弦")
    axis.plot(steps, best, color="#df8e18", linewidth=2, label="检查点历史最佳余弦")
    refs = summary["references"]
    axis.axhline(refs["old_thirty_guidance_cosine"], color="#72843f", linestyle="--", label="此前实际30步图")
    axis.axhline(refs["source_cosine"], color="#858585", linestyle=":", label="原图")
    markers = [("production_selected", "生产主程选中", "s", "#278351"),
               ("audit_best", "审计最佳", "*", "#963a95"),
               ("audit_final", "审计最终", "D", "#b34b3f")]
    for key, label, marker, color in markers:
        point = summary[key]
        axis.scatter([point["actual_step"]], [point["target_cosine"]], s=125 if marker == "*" else 60,
                     marker=marker, color=color, edgecolors="white", zorder=6,
                     label=f"{label}：{point['actual_step']}步，{point['target_cosine']:.6f}")
    trigger = summary["early_stop_trigger_step"]
    if trigger is not None:
        for ax in (axis, gain_axis):
            ax.axvline(trigger, color="#444444", linestyle="--", linewidth=1)
            if trigger < steps[-1]:
                ax.axvspan(trigger, steps[-1], color="#e7dbee", alpha=.3)
        axis.text(trigger, axis.get_ylim()[0], " 生产停止／后续为审计", fontproperties=font,
                  fontsize=9, va="bottom", rotation=90)
    gains = [checks[index].get("window_best_gain") for index in range(1, len(checks))]
    gain_axis.plot(steps[1:], [math.nan if gain is None else gain for gain in gains], marker="o",
                   markersize=3, color="#df8e18", label="每检测窗口历史最佳增量")
    gain_axis.axhline(summary["parameters"].get("min_delta", .002), color="#b34b3f", linestyle="--", label="单窗口早停增量阈值")
    axis.set_title("第七组扩散优化：真实保存候选曲线与早停校准", fontproperties=font, fontsize=16)
    axis.set_ylabel("固定第二代理对目标图的余弦", fontproperties=font)
    gain_axis.set_ylabel("窗口最佳增量", fontproperties=font)
    gain_axis.set_xlabel("实际有梯度引导的去噪步数（零步为无引导重建）", fontproperties=font)
    for ax in (axis, gain_axis):
        ax.grid(alpha=.22)
        ax.legend(prop=font, fontsize=9, loc="best")
    figure.text(.07, .018, "候选已完成剩余无引导去噪并以原尺寸八位图评分；审计继续计算，整次校准并未提前结束。",
                fontproperties=font, fontsize=10)
    figure.tight_layout(rect=(0, .04, 1, 1))
    path = output / "score_curve.png"
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    return path


def write_report(output, checks, summary):
    settings = summary["parameters"]
    production, best, final = [summary[key] for key in ("production_selected", "audit_best", "audit_final")]
    refs = summary["references"]
    trigger = summary["early_stop_trigger_step"]
    stop = f"第{trigger}个实际引导步触发早停" if trigger is not None else "没有触发早停，主程到达实际引导上限"
    lines = [f"公子，本轮{stop}。生产输出选中第{production['actual_step']}步的真实完整候选，目标余弦{production['target_cosine']:.6f}。", "",
             f"本轮实际共计算{summary['actual_total_guidance_steps']}个引导步，其中生产主程{summary['production_guidance_steps']}步、早停后校准审计{summary['post_stop_audit_additional_guidance_steps']}步。审计分支继续计算，用于检验停止规则；不能称整次实验提前停止或把主程时间当作本次总耗时。", "",
             "| 图片／候选 | 实际引导步 | 固定第二代理目标余弦 |", "|---|---:|---:|",
             f"| 原图 | 不适用 | {refs['source_cosine']:.6f} |",
             f"| 旧图 | 30 | {refs['old_thirty_guidance_cosine']:.6f} |",
             f"| 本轮无引导重建 | 0 | {checks[0][SCORE_KEY]:.6f} |",
             f"| 生产主程选中图 | {production['actual_step']} | {production['target_cosine']:.6f} |",
             f"| 全程观测最佳图 | {best['actual_step']} | {best['target_cosine']:.6f} |",
             f"| 全程最终图 | {final['actual_step']} | {final['target_cosine']:.6f} |", "",
             f"审计最佳比生产选中图增加{summary['audit_best_gain_over_production']:.6f}；审计最终比生产选中图增加{summary['audit_final_minus_production_cosine']:.6f}。生产选中图比旧三十步图增加{summary['production_gain_over_old_thirty']:.6f}，审计最佳比旧图增加{summary['audit_best_gain_over_old_thirty']:.6f}。", ""]
    if summary["audit_has_additional_guidance"]:
        qualifier = "达到或超过" if summary["audit_gain_exceeds_one_window_threshold"] else "小于"
        lines.extend([f"停止后观察到的最佳增益{qualifier}一个窗口的{settings.get('min_delta', .002):.6f}阈值。这个差距是本条轨迹上的误停代价；阈值只定义每窗口增量，不保证未来收益为零，也不构成通用显著性检验。", ""])
    lines.extend([f"参数为完整日程{summary['attack'].get('steps_requested')}步、强度{settings.get('strength')}、最低引导步数{settings.get('min_guidance_steps')}、每{settings.get('check_every')}个实际步检查、连续{settings.get('patience')}个低增窗口、单窗口阈值{settings.get('min_delta')}。参数完整原值保存在汇总文件中。", "",
                  f"共读取并校验{len(checks)}张候选保存图。检测复制当前潜变量并完成剩余无引导尾程，再恢复原尺寸、保存八位图并评分；没有使用含噪潜变量的直接解码或逐步训练综合损失作曲线。", "",
                  f"候选评估额外完成{summary['candidate_tail_steps_excluding_reused_reconstruction']}个无引导尾程步，零步重建的既有计算不重复计入。生产检查尾程耗时{_seconds(summary['production_checkpoint_tail_seconds'])}，审计检查尾程耗时{_seconds(summary['audit_checkpoint_tail_seconds'])}。", "",
                  f"生产停止检查点累计耗时字段为{_seconds(summary['production_stop_checkpoint_elapsed_seconds'])}，最终检查点为{_seconds(summary['last_checkpoint_elapsed_seconds'])}；计时起点按原攻击元数据定义。这些检查点计时可能不包含最后输出评分和元数据写入。", "",
                  "攻击元数据记录的耗时：", ""])
    lines.extend(f"- {label}：{_seconds(value)}。" for key, value in summary["timing"].items()
                 if isinstance(value, (int, float)) and (label := _timing_label(key)))
    counts = summary["noise_network_forward_counts"]
    if counts:
        lines.extend(["", "实际噪声网络前向计算次数：", ""])
        count_names = {"inversion": "反演", "reconstruction": "无引导重建",
                       "primary_guided_denoising": "生产主程引导去噪", "post_stop_audit_guided_denoising": "审计分支引导去噪",
                       "checkpoint_tail_denoising": "检查点完整尾程", "total": "全部实际前向总计"}
        lines.extend(f"- {label}：{counts[key]}次。" for key, label in count_names.items() if key in counts)
        lines.extend(["", "前向次数按实际噪声网络调用计算；批处理的条件与无条件预测仍计为一次调用。候选尾程消耗属于额外评估计算，并未重复计算梯度。", ""])
    lines.extend(["", "曲线中的最佳只指这一组图、这一套参数、这一条日程轨迹中已保存检查点的最高分；没有穷举所有步数、种子、提示词、引导强度或模型，因此不能称最大可优化量的全局上限。旧三十步日程与千步日程的离散噪声起点不同，差异也不能全部归因于迭代次数。", "",
                  "评分代理参与优化；本报告不验证黑盒迁移、购物成功率、16灰度级预算、无损反演或严格自然图像流形。", "",
                  f"![真实候选评分曲线](<{Path(summary['curve']).as_posix()}>)", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _seconds(value):
    return "未记录" if value is None else f"{float(value):.3f}秒"


def _timing_label(key):
    names = {"diffusion_loading_seconds": "扩散模型加载", "surrogate_loading_seconds": "代理模型加载",
             "inversion_seconds": "反演", "reconstruction_seconds": "无引导重建",
             "optimization_loop_seconds": "优化循环及检查总计", "total_seconds_before_metadata": "元数据写入前总计",
             "production_optimization_seconds": "生产主程优化及检查", "audit_optimization_seconds": "停止后审计优化及检查",
             "checkpoint_tail_seconds": "候选完整尾程总计", "checkpoint_evaluation_seconds": "候选固定代理评分总计",
             "primary_guidance_loop_seconds": "生产主程引导更新（不含候选检查）",
             "post_stop_audit_guidance_seconds": "审计分支引导更新（不含候选检查）",
             "primary_checkpoint_seconds": "生产主程候选检查", "post_stop_audit_checkpoint_seconds": "审计分支候选检查",
             "checkpoint_saving_seconds": "候选保存及重新读取", "checkpoint_progress_write_seconds": "候选进度写入",
             "final_output_and_recompute_seconds": "最终输出与重评分"}
    return names.get(key, f"{key}(原始计时字段)")


def summarize(attack, output):
    source_path = metadata_path(attack)
    output = Path(output).resolve()
    if output.is_relative_to(source_path.parent):
        raise ValueError("汇总输出必须位于原攻击目录外的独立目录")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("汇总输出必须不存在或完全为空")
    metadata = read_json(source_path)
    if metadata.get("status") not in {"success", "completed"}:
        raise ValueError("原攻击尚未成功完成，不生成最终早停结论")
    checks = check_candidates(metadata, source_path.parent)
    summary = build_summary(metadata, checks, source_path)
    output.mkdir(parents=True, exist_ok=True)
    summary["curve"] = str(draw_curve(output, checks, summary))
    summary["curve_sha256"] = digest(summary["curve"])
    fields = ["actual_step", "phase", SCORE_KEY, "running_best_cosine", "selected_step", "window_best_gain",
              "low_gain_windows", "should_stop", "stop_eligible", "candidate_tail_steps", "reused_reconstruction",
              "tail_seconds", "evaluation_seconds", "saving_seconds", "check_elapsed_seconds", "cumulative_elapsed_seconds",
              "image", "image_sha256"]
    with (output / "checkpoints.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(checks)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(output, checks, summary)
    if digest(source_path) != summary["input_metadata_sha256"]:
        raise ValueError("汇总期间攻击元数据发生变化")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", required=True, type=Path, help="Attack directory, run root, or metadata file")
    parser.add_argument("--output", required=True, type=Path, help="Independent nonexistent or empty output directory")
    args = parser.parse_args()
    summary = summarize(args.attack, args.output)
    print(json.dumps({"status": summary["status"], "curve": summary["curve"],
                      "production_selected": summary["production_selected"],
                      "audit_best": summary["audit_best"], "audit_best_gain_over_production": summary["audit_best_gain_over_production"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
