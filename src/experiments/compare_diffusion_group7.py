"""Compare new group-seven diffusion images with frozen image attack artifacts.

All embeddings are computed in one encoder session, from lossless saved images.
Only candidate images receive defenses; the original target stays unchanged.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, JpegImagePlugin

from src.paths import PROJECT


BASE = PROJECT / "src/output"
ORIGINAL_ROOT = BASE / "nine_groups_attacker_history_joint"
SOURCE = ORIGINAL_ROOT / "assets/9b5f4a94d9ab7f2e001498260243b3d91eedcd774939a4558c4f5786192e3a6a.jpg"
TARGET = ORIGINAL_ROOT / "assets/dd0abf8d2d52aa289346aa2e7219187cec901c901d4f3cce1ef3379691467256.jpg"
DEFENSES = ("none", "gaussian_r4", "jpeg_q25")
DEFENSE_LABELS = {"none": "未处理", "gaussian_r4": "高斯模糊半径4", "jpeg_q25": "有损压缩质量25"}


@dataclass(frozen=True)
class Candidate:
    name: str
    title: str
    path: Path
    note: str


def default_candidates():
    return [
        Candidate("pixel", "原像素方法", ORIGINAL_ROOT / "groups/07_shampoo/attack/adversarial.png",
                  "完成1000次；像素预算16灰度级；使用最终图"),
        Candidate("frequency_momentum", "频域动量", BASE / "nine_groups_frequency_group7_500steps_20261005/groups/07_shampoo/attack/adversarial.png",
                  "完成500次，选中第250次；像素预算16灰度级"),
        Candidate("ssa", "频谱模拟原始图", BASE / "scar_shampoo_core_20261008/groups/07_shampoo/attack/adversarial.png",
                  "完成300次；未缩幅；像素预算16灰度级"),
        Candidate("scar", "谱域协同自适应精修", BASE / "scar_shampoo_old_b_equal05_full_20261008/groups/07_shampoo/attack/adversarial.png",
                  "300次频谱模拟后选择扰动幅度0.625；预算16灰度级"),
    ]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def unit_vector(value):
    vector = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("编码器返回了非法向量")
    return vector / norm


def cosine(left, right):
    if left.shape != right.shape:
        raise ValueError("图片向量维度不一致")
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _gaussian_valid(image, kernel):
    horizontal = np.tensordot(np.lib.stride_tricks.sliding_window_view(image, len(kernel), axis=1), kernel, axes=([-1], [0]))
    return np.tensordot(np.lib.stride_tricks.sliding_window_view(horizontal, len(kernel), axis=0), kernel, axes=([-1], [0]))


def grayscale_ssim(left, right):
    """Gaussian-window luminance SSIM, population covariance, valid interior."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    x, y = left @ weights, right @ weights
    side = min(11, x.shape[0], x.shape[1])
    side -= 1 - side % 2
    if side < 3:
        return None
    axis = np.arange(side, dtype=np.float64) - side // 2
    kernel = np.exp(-(axis ** 2) / (2 * 1.5 ** 2))
    kernel /= kernel.sum()
    mx, my = _gaussian_valid(x, kernel), _gaussian_valid(y, kernel)
    vx = np.maximum(0, _gaussian_valid(x * x, kernel) - mx * mx)
    vy = np.maximum(0, _gaussian_valid(y * y, kernel) - my * my)
    cov = _gaussian_valid(x * y, kernel) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(np.mean(((2 * mx * my + c1) * (2 * cov + c2)) /
                         ((mx * mx + my * my + c1) * (vx + vy + c2))))


def distortion(source, candidate):
    native_size = candidate.size
    aligned = candidate if native_size == source.size else candidate.resize(source.size, Image.Resampling.LANCZOS)
    left, right = np.asarray(source, dtype=np.float64), np.asarray(aligned, dtype=np.float64)
    difference = right - left
    mse = float(np.mean(difference ** 2))
    return {
        "source_size_wh": list(source.size), "candidate_native_size_wh": list(native_size),
        "comparison_size_wh": list(source.size),
        "candidate_resized_for_pixel_metrics": native_size != source.size,
        "pixel_metric_resize": "lanczos_to_original_source_size" if native_size != source.size else "none",
        "mae_levels": float(np.mean(np.abs(difference))),
        "max_absolute_difference_levels": int(np.max(np.abs(difference))),
        "mse_levels": mse, "psnr_db": None if mse == 0 else 10 * math.log10(255 ** 2 / mse),
        "psnr_identical_images": mse == 0, "ssim": grayscale_ssim(left, right),
    }, aligned


def save_defended(image, output, defense):
    record = {"defense": defense, "image_native_size_wh": list(image.size), "resize_before_defense": False}
    if defense == "none":
        result = image
    elif defense == "gaussian_r4":
        result = image.filter(ImageFilter.GaussianBlur(radius=4.0))
        record["gaussian_radius"] = 4.0
    elif defense == "jpeg_q25":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=25)
        jpeg_path = output.with_suffix(".jpg")
        jpeg_path.write_bytes(buffer.getvalue())
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            record["jpeg_chroma_sampling"] = JpegImagePlugin.get_sampling(compressed)
            result = compressed.convert("RGB").copy()
        record.update(jpeg_quality=25, jpeg_encode_count=1, jpeg_file=str(jpeg_path), jpeg_sha256=digest(jpeg_path))
    else:
        raise ValueError(f"不支持的处理方式：{defense}")
    result.save(output, format="PNG")
    record.update(saved_image=str(output), saved_image_sha256=digest(output))
    return record


def parse_extra(value, index):
    if "=" not in value:
        raise argparse.ArgumentTypeError("附加图片格式应为 名称=图片路径")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("附加图片的名称和路径均不能为空")
    return Candidate(f"extra_{index}", name.strip(), Path(path).resolve(), "附加候选；参数和预算以该候选生成记录为准")


def _font(size):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    raise FileNotFoundError("未找到中文字体，无法生成中文比较拼图")


def wrap_text(draw, text, font, width):
    lines, current = [], ""
    for character in text:
        if character == "\n":
            lines.append(current)
            current = ""
        elif current and draw.textlength(current + character, font=font) > width:
            lines.append(current)
            current = character
        else:
            current += character
    if current:
        lines.append(current)
    return lines


def make_panel(output, cards):
    columns, card_width, card_height = 4, 480, 610
    rows = math.ceil(len(cards) / columns)
    panel = Image.new("RGB", (columns * card_width + 40, rows * card_height + 170), "#eef1f5")
    draw = ImageDraw.Draw(panel)
    large, normal, small = _font(33), _font(23), _font(18)
    draw.text((25, 16), "第七组：扩散图片优化与现有方法比较", fill="#17233b", font=large)
    draw.text((25, 60), "同一检索编码器 · 保存图重新读取 · 目标余弦越高，表示目标特征越相似", fill="#445166", font=normal)
    for index, card in enumerate(cards):
        x, y = 20 + index % columns * card_width, 105 + index // columns * card_height
        draw.rounded_rectangle((x, y, x + card_width - 14, y + card_height - 13), radius=14, fill="white")
        title = wrap_text(draw, card["title"], normal, card_width - 44)
        for offset, line in enumerate(title[:2]):
            draw.text((x + 15, y + 11 + offset * 27), line, fill="#17233b", font=normal)
        image = read_rgb(card["image"])
        image.thumbnail((card_width - 40, 420), Image.Resampling.LANCZOS)
        panel.paste(image, (x + (card_width - 14 - image.width) // 2, y + 65 + (420 - image.height) // 2))
        draw.text((x + 15, y + 498), f"目标余弦：{card['target_cosine']:.6f}", fill="#155c95", font=normal)
        draw.text((x + 15, y + 529), f"原图余弦：{card['source_cosine']:.6f}", fill="#445166", font=small)
        note = wrap_text(draw, card["note"], small, card_width - 44)
        for offset, line in enumerate(note[:2]):
            draw.text((x + 15, y + 554 + offset * 22), line, fill="#667284", font=small)
    draw.text((25, rows * card_height + 115), "旧法与扩散法的步数、像素预算不同；此图展示图片特征比较，不代表购物成功率。", fill="#445166", font=small)
    destination = output / "comparison.png"
    panel.save(destination)
    return destination


def write_report(output, report):
    plain = [row for row in report["rows"] if row["defense"] == "none"]
    lines = ["公子，本报告对第七组原图、现有方法和新扩散图片进行了同一次模型加载下的统一比较。", "",
             "检索编码器为第二个对比语言图像预训练代理，采用32像素块的基础视觉变换器。各余弦来自保存图片重新读取后的向量。", "",
             "| 图片 | 对目标余弦 | 对原图余弦 | 平均绝对像素差 | 最大像素差 | 峰值信噪比 | 结构相似性 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in plain:
        if row["name"] == "target":
            continue
        psnr = "无穷（相同图片）" if row["psnr_identical_images"] else f"{row['psnr_db']:.3f}"
        ssim = "未计算（尺寸过小）" if row["ssim"] is None else f"{row['ssim']:.6f}"
        title = row["title"].replace("|", "\\|")
        lines.append(f"| {title} | {row['target_cosine']:.6f} | {row['source_cosine']:.6f} | {row['mae_levels']:.6f} | {row['max_absolute_difference_levels']} | {psnr} | {ssim} |")
    lines.extend(["", "| 图片 | 未处理目标余弦 | 高斯模糊半径4后 | 有损压缩质量25后 |", "|---|---:|---:|---:|"])
    for row in plain:
        if row["name"] == "target":
            continue
        lookup = {item["defense"]: item for item in report["rows"] if item["name"] == row["name"]}
        title = row["title"].replace("|", "\\|")
        lines.append(f"| {title} | {row['target_cosine']:.6f} | {lookup['gaussian_r4']['target_cosine']:.6f} | {lookup['jpeg_q25']['target_cosine']:.6f} |")
    lines.extend(["", "处理仅作用于被比较的候选图，包括原图基线；目标参考图保持原图内容。高斯模糊使用原生尺寸，半径4；有损压缩质量25，色度采样代码记录在机器可读报告中。", "",
                  "像素差以八位灰度级测量，比较尺寸为原图尺寸。候选尺寸不同时，仅为像素指标将候选高质量缩放到原图尺寸，并保存该对齐图；向量和模糊、压缩仍基于候选原生尺寸。结构相似性使用亮度通道、高斯窗口、标准差1.5、总体协方差和有效内部区域；峰值信噪比对完全相同图记为无穷，机器可读值记为空并标记图片相同。", "",
                  "旧法步数及生成记录：", ""])
    lines.extend(f"- {item['title']}：{item['note']}。" for item in report["candidates"] if item["name"] not in {"source", "target"})
    lines.extend(["", "旧法均在16灰度级预算内优化；扩散方法和重建基线没有与旧法相同的像素预算或计算量保证。该表比较这些实际输出的特征相似性、失真和简单处理后的变化，不能据此推出方法在相同预算下优劣、跨模型迁移能力、自然图像流形保证或购物攻击成功率。", "",
                  "若提供了无对抗引导的扩散重建图，应将其与新扩散图一并比较，以分离生成重建本身的影响。", "",
                  f"比较拼图：![第七组比较图](<{Path(report['panel']).as_posix()}>)", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def compare(args):
    started = time.perf_counter()
    source_path, target_path, output = args.source.resolve(), args.target.resolve(), args.output.resolve()
    candidates = [Candidate("source", "原图", source_path, "原始甲商品图片"),
                  Candidate("target", "目标参考图", target_path, "原始乙商品图片"), *default_candidates()]
    if args.reconstruction:
        candidates.append(Candidate("reconstruction", "扩散重建基线", args.reconstruction.resolve(), "无对抗引导的重建对照；参数以生成记录为准"))
    candidates.append(Candidate("diffusion", "扩散潜空间优化", args.diffusion.resolve(), "新生成图片；参数和像素变化以本轮实际记录为准"))
    candidates.extend(parse_extra(value, index) for index, value in enumerate(args.candidate, 1))
    protected = {item.path.resolve().parents[3] for item in default_candidates()} | {ORIGINAL_ROOT.resolve()}
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError("比较输出须位于旧实验目录以外的独立目录")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("比较输出须为不存在或完全空的独立目录")
    images = {item.name: read_rgb(item.path) for item in candidates}
    original_hashes = {str(item.path.resolve()): digest(item.path) for item in candidates}
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from src.vision.dual_rag_bridge import make_encoder
    encoder = make_encoder(surrogate=2, device=args.device)
    cache = {}

    def encode(path):
        sha = digest(path)
        if sha not in cache:
            cache[sha] = unit_vector(encoder.encode(path))
        return cache[sha]

    report = {"schema_version": 1, "status": "running", "group": "07_shampoo", "api_calls": 0,
              "new_optimization_steps": 0, "encoder_snapshot": encoder.snapshot,
              "source": {"path": str(source_path), "sha256": digest(source_path)},
              "target": {"path": str(target_path), "sha256": digest(target_path)},
              "protocol": {"surrogate": 2, "all_images_reencoded_in_this_run": True,
                  "read_saved_images_for_encoding": True, "candidate_only_defense": True,
                  "target_image_defended": False, "gaussian_radius": 4.0, "jpeg_quality": 25,
                  "defense_at_candidate_native_resolution": True, "pixel_comparison_resolution": "original_source",
                  "different_sizes_resize_candidate_for_pixel_metrics_only": "lanczos",
                  "ssim": "BT601_luminance_gaussian_window11_sigma1.5_population_covariance_valid_interior",
                  "ssim_small_image_window": "largest_odd_window_up_to_11; null_below_3",
                  "psnr_identical": "null_psnr_db_and_psnr_identical_images_true",
                  "scope": "image_feature_similarity_and_distortion; no_shopping_or_retrieval_rank_claims",
                  "budget_comparability": "legacy_16_levels; diffusion_and_reconstruction_not_budget_matched; unequal_steps"},
              "candidates": [], "rows": []}
    base_paths = {}
    for index, item in enumerate(candidates):
        image_dir = output / "images/none"
        image_dir.mkdir(parents=True, exist_ok=True)
        base_path = image_dir / f"{index:02d}_{item.name}.png"
        images[item.name].save(base_path, format="PNG")
        base_paths[item.name] = base_path
    source_vector, target_vector = encode(base_paths["source"]), encode(base_paths["target"])
    report["embedding_dimension"] = int(source_vector.size)
    cards = []
    for index, item in enumerate(candidates):
        report["candidates"].append({"name": item.name, "title": item.title, "input": str(item.path.resolve()),
                                     "input_sha256": digest(item.path), "note": item.note})
        for defense in (("none",) if item.name == "target" else DEFENSES):
            destination = output / "images" / defense / f"{index:02d}_{item.name}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            saved = save_defended(images[item.name], destination, defense)
            reread = read_rgb(destination)
            pixel_metrics, aligned = distortion(images["source"], reread)
            if pixel_metrics["candidate_resized_for_pixel_metrics"]:
                metric_dir = output / "images/pixel_metric_aligned" / defense
                metric_dir.mkdir(parents=True, exist_ok=True)
                metric_path = metric_dir / destination.name
                aligned.save(metric_path, format="PNG")
                pixel_metrics.update(pixel_metric_aligned_image=str(metric_path), pixel_metric_aligned_sha256=digest(metric_path))
            vector = encode(destination)
            row = {"name": item.name, "title": item.title, "input_image": str(item.path.resolve()),
                   "input_sha256": digest(item.path), "note": item.note, **saved, **pixel_metrics,
                   "target_cosine": cosine(target_vector, vector), "source_cosine": cosine(source_vector, vector)}
            report["rows"].append(row)
            if defense == "none":
                cards.append({"title": item.title, "image": destination, "note": item.note,
                              "target_cosine": row["target_cosine"], "source_cosine": row["source_cosine"]})
        print(f"完成图片比较：{item.title}", flush=True)
    if any(digest(path) != sha for path, sha in original_hashes.items()):
        raise ValueError("比较期间输入图片内容改变，拒绝将本次比较标为完成")
    report["input_integrity_verified"] = True
    report["panel"] = str(make_panel(output, cards))
    report["panel_sha256"] = digest(report["panel"])
    report.update(status="completed", distinct_image_embeddings=len(cache), elapsed_seconds=time.perf_counter() - started)
    write_json(output / "report.json", report)
    fields = ["name", "title", "defense", "target_cosine", "source_cosine", "mae_levels", "max_absolute_difference_levels",
              "mse_levels", "psnr_db", "psnr_identical_images", "ssim", "candidate_resized_for_pixel_metrics",
              "input_image", "input_sha256", "saved_image", "saved_image_sha256", "note"]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])
    write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion", type=Path, required=True, help="New diffusion image")
    parser.add_argument("--reconstruction", type=Path, help="Optional unguided reconstruction baseline")
    parser.add_argument("--output", type=Path, required=True, help="Independent nonexistent or empty output directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH", help="Extra named image; repeatable")
    args = parser.parse_args()
    report = compare(args)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()), "panel": report["panel"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
