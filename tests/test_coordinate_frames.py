"""Coordinate-frame contract tests (webcam → pad-square → norm_crop)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TORCH = ROOT / "vendor" / "torch_train"
if TORCH.is_dir() and str(TORCH) not in sys.path:
    sys.path.insert(0, str(TORCH))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.coordinate_frames import (  # noqa: E402
    COORD_NORM_CROP,
    CropRect,
    assert_norm_crop,
    pad_square_crop_rect,
    to_norm_crop,
    webcam_pixels_to_norm_crop,
)
from utils.keypoints import transform_keypoints_crop  # noqa: E402


def _pt(x: float, y: float) -> np.ndarray:
    k = np.zeros((37, 4), dtype=np.float32)
    k[15] = [x, y, 1.0, 1.0]
    k[31] = [x, y + 100.0, 1.0, 1.0]
    k[32] = [x - 80.0, y + 120.0, 1.0, 1.0]
    k[34] = [x + 80.0, y + 120.0, 1.0, 1.0]
    return k


def test_pad_square_16x9_center_maps_to_origin() -> None:
    k = _pt(640.0, 360.0)
    n, crop = webcam_pixels_to_norm_crop(k, 1280, 720, image_size=768)
    assert crop.mode == "pad_square"
    assert abs(float(n[15, 0])) < 1e-5
    assert abs(float(n[15, 1])) < 1e-5
    assert float(n[15, 3]) >= 0.5


def test_pad_square_4x3_and_portrait() -> None:
    for w, h, cx, cy in ((640, 480, 320.0, 240.0), (720, 1280, 360.0, 640.0)):
        k = _pt(cx, cy)
        n, crop = webcam_pixels_to_norm_crop(k, w, h, image_size=768)
        assert crop.w == crop.h == float(max(w, h))
        assert abs(float(n[15, 0])) < 1e-4
        assert abs(float(n[15, 1])) < 1e-4


def test_isotropic_deltas_after_pad_square() -> None:
    """Equal pixel dx/dy become equal norm deltas on 16:9 after pad-square."""
    base = _pt(640.0, 360.0)
    moved = base.copy()
    moved[32, 0] += 50.0
    moved[32, 1] += 50.0
    n0, crop = webcam_pixels_to_norm_crop(base, 1280, 720)
    n1 = to_norm_crop(moved, crop)
    dx = float(n1[32, 0] - n0[32, 0])
    dy = float(n1[32, 1] - n0[32, 1])
    assert abs(abs(dx) - abs(dy)) < 1e-5


def test_outside_crop_marked_invisible() -> None:
    # Point far left of a tight square crop.
    k = _pt(-500.0, 100.0)
    crop = CropRect(x0=0.0, y0=0.0, w=200.0, h=200.0, out_w=768, out_h=768, mode="custom")
    n = to_norm_crop(k, crop)
    assert float(n[15, 3]) < 0.5


def test_assert_norm_crop() -> None:
    assert_norm_crop(COORD_NORM_CROP, label="ok")
    try:
        assert_norm_crop("norm_full", label="bad")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_matches_transform_keypoints_crop() -> None:
    k = _pt(100.0, 200.0)
    crop = pad_square_crop_rect(640, 480, image_size=768)
    a = to_norm_crop(k, crop)
    b = transform_keypoints_crop(
        k,
        crop_x0=crop.x0,
        crop_y0=crop.y0,
        crop_w=crop.w,
        crop_h=crop.h,
        out_w=crop.out_w,
        out_h=crop.out_h,
        normalize=True,
    )
    np.testing.assert_allclose(a, b)
