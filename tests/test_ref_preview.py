"""Reference display crop helpers (no detector / GPU required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ref_pose_fit import _align_body_nose_to_face, reference_display_rgb

GIGI = (
    Path(__file__).resolve().parents[1]
    / "test-pose"
    / "input"
    / "vtuber_ref_gigi_mood_green.png"
)


def test_align_body_nose_to_face() -> None:
    k = np.zeros((37, 4), dtype=np.float32)
    k[15] = [0.1, 0.2, 1, 1]
    k[30] = [0.5, 0.6, 0.9, 1]
    out = _align_body_nose_to_face(k)
    assert abs(float(out[30, 0]) - 0.1) < 1e-6
    assert abs(float(out[30, 1]) - 0.2) < 1e-6


def test_reference_display_rgb_greenscreen_square() -> None:
    if not GIGI.is_file():
        return  # optional artifact
    arr = np.asarray(Image.open(GIGI).convert("RGB"))
    # Greenscreen refs use chroma crop (skip_crop=False).
    canvas = reference_display_rgb(arr, skip_crop=False, image_size=768)
    assert canvas.shape == (768, 768, 3)
    assert canvas.dtype == np.uint8
    # Pad-only path must also produce 768².
    padded = reference_display_rgb(arr, skip_crop=True, image_size=768)
    assert padded.shape == (768, 768, 3)
