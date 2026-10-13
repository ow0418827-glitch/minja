"""CPU-only metric and saved-image comparison checks; no model weights loaded."""
from argparse import Namespace
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.experiments import compare_diffusion_group7 as comparison


def test_pixel_metrics_match_constant_error_and_ssim_formula():
    source = Image.new("RGB", (20, 20), (100, 100, 100))
    candidate = Image.new("RGB", (20, 20), (110, 110, 110))
    result, _ = comparison.distortion(source, candidate)
    assert result["mae_levels"] == 10
    assert result["max_absolute_difference_levels"] == 10
    assert result["mse_levels"] == 100
    assert result["psnr_db"] == pytest.approx(20 * math.log10(255 / 10))
    c1 = (0.01 * 255) ** 2
    assert result["ssim"] == pytest.approx((2 * 100 * 110 + c1) / (100 ** 2 + 110 ** 2 + c1))
    identical, _ = comparison.distortion(source, source)
    assert identical["ssim"] == pytest.approx(1)
    assert identical["psnr_db"] is None
    assert identical["psnr_identical_images"] is True


def test_changed_size_is_recorded_and_metrics_use_original_size():
    result, aligned = comparison.distortion(Image.new("RGB", (20, 18), "red"), Image.new("RGB", (32, 32), "red"))
    assert aligned.size == (20, 18)
    assert result["candidate_native_size_wh"] == [32, 32]
    assert result["candidate_resized_for_pixel_metrics"] is True
    assert result["mae_levels"] == 0


def test_full_comparison_rereads_saved_defenses_and_keeps_original_target(monkeypatch, tmp_path):
    from src.vision import dual_rag_bridge

    source, target, diffusion = [tmp_path / name for name in ("source.png", "target.png", "diffusion.png")]
    Image.new("RGB", (20, 20), (80, 20, 10)).save(source)
    Image.new("RGB", (20, 20), (10, 160, 30)).save(target)
    Image.new("RGB", (24, 24), (60, 60, 20)).save(diffusion)
    original = {path: path.read_bytes() for path in (source, target, diffusion)}
    calls = []

    class FakeEncoder:
        snapshot = {"provider": "test", "surrogate_number": 2}

        def encode(self, path):
            calls.append(Path(path))
            pixels = np.asarray(comparison.read_rgb(path), dtype=np.float64)
            return np.r_[1.0, pixels.mean(axis=(0, 1))]

    def factory(*, surrogate, device):
        assert surrogate == 2
        assert device == "cpu"
        return FakeEncoder()

    monkeypatch.setattr(dual_rag_bridge, "make_encoder", factory)
    monkeypatch.setattr(comparison, "default_candidates", lambda: [])

    def panel(output, cards):
        path = output / "comparison.png"
        Image.new("RGB", (1, 1)).save(path)
        return path

    monkeypatch.setattr(comparison, "make_panel", panel)
    args = Namespace(source=source, target=target, diffusion=diffusion, reconstruction=None,
                     candidate=[], output=tmp_path / "comparison", device="cpu")
    report = comparison.compare(args)
    assert report["status"] == "completed"
    assert len(report["rows"]) == 7
    assert report["protocol"]["target_image_defended"] is False
    assert all(path.is_relative_to(args.output) and path.is_file() for path in calls)
    target_rows = [row for row in report["rows"] if row["name"] == "target"]
    assert len(target_rows) == 1
    assert target_rows[0]["target_cosine"] == pytest.approx(1)
    diffusion_rows = [row for row in report["rows"] if row["name"] == "diffusion"]
    assert all(row["candidate_native_size_wh"] == [24, 24] for row in diffusion_rows)
    assert all(row["comparison_size_wh"] == [20, 20] for row in diffusion_rows)
    assert all(row["candidate_resized_for_pixel_metrics"] for row in diffusion_rows)
    for row in report["rows"]:
        assert comparison.digest(row["saved_image"]) == row["saved_image_sha256"]
    assert original == {path: path.read_bytes() for path in original}
    assert json.loads((args.output / "report.json").read_text(encoding="utf-8"))["input_integrity_verified"]
    assert (args.output / "comparison.csv").is_file()
    assert (args.output / "report.md").is_file()


def test_nonempty_output_is_rejected_before_loading_encoder(monkeypatch, tmp_path):
    from src.vision import dual_rag_bridge
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20)).save(source)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(comparison, "default_candidates", lambda: [])
    monkeypatch.setattr(dual_rag_bridge, "make_encoder", lambda **_: pytest.fail("Model was loaded"))
    args = Namespace(source=source, target=source, diffusion=source, reconstruction=None,
                     candidate=[], output=output, device="cpu")
    with pytest.raises(ValueError, match="完全空"):
        comparison.compare(args)
    assert sentinel.read_text(encoding="utf-8") == "keep"
