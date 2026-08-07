"""Bridge JSON compatibility: norm_crop packets and legacy pixel conversion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TORCH = ROOT.parent / "send2pod" / "torch_train"
LIVE = ROOT.parent / "tools" / "live-poser"
for p in (TORCH, ROOT, LIVE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bridge import build_bridge_frame  # noqa: E402
from engine import keypoints_from_json  # noqa: E402
from utils.coordinate_frames import (  # noqa: E402
    COORD_NORM_CROP,
    coord_meta_dict,
    webcam_pixels_to_norm_crop,
)


def _body7() -> np.ndarray:
    b = np.zeros((7, 4), dtype=np.float32)
    for i, (x, y) in enumerate(
        ((640, 360), (640, 420), (520, 450), (480, 520), (760, 450), (800, 520), (640, 500))
    ):
        b[i] = [x, y, 0.9, 1.0]
    return b


def _attach(bridge):
    src_w, src_h = bridge.image_wh
    kps_norm, crop = webcam_pixels_to_norm_crop(bridge.keypoints, src_w, src_h)
    bridge.keypoints_norm = kps_norm
    bridge.coord_space = COORD_NORM_CROP
    bridge.crop = crop
    bridge.image_size = 768
    meta = dict(bridge.meta or {})
    meta.update(
        coord_meta_dict(
            coord_space=COORD_NORM_CROP,
            source_wh=(src_w, src_h),
            crop=crop,
            image_size=768,
            mirrored=bool(bridge.mirrored),
            skeleton_method=str(bridge.skeleton_method or "none"),
            body_lost=bool(meta.get("body_lost", False)),
        )
    )
    bridge.meta = meta
    return bridge


def test_bridge_to_dict_includes_crop_meta() -> None:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 0] = 640.0
    pts[:, 1] = np.linspace(200, 400, 28)
    pts[:, 2] = 1.0
    bridge = build_bridge_frame(
        pts,
        (720, 1280, 3),
        body7=_body7(),
        skeleton_method="mediapipe_pose_lite",
        meta={"body_lost": False, "fps": 30.0},
    )
    _attach(bridge)
    d = bridge.to_dict()
    assert d["coord_space"] == COORD_NORM_CROP
    assert "crop" in d
    assert "keypoints_norm" in d
    assert d["source_wh"] == [1280, 720]
    assert d["meta"]["coord_space"] == COORD_NORM_CROP


def test_keypoints_from_json_norm_crop_packet() -> None:
    payload = {
        "schema": "KEYPOINT_SCHEMA",
        "coord_space": "norm_crop",
        "keypoints": [
            {"i": 15, "x": 0.0, "y": 0.1, "score": 1.0, "visible": True},
            {"i": 31, "x": 0.0, "y": 0.4, "score": 1.0, "visible": True},
        ],
    }
    arr = keypoints_from_json(payload)
    assert arr.shape == (37, 4)
    assert abs(float(arr[15, 1]) - 0.1) < 1e-6


def test_keypoints_from_json_legacy_pixels() -> None:
    payload = {
        "schema": "KEYPOINT_SCHEMA",
        "image_wh": [1280, 720],
        "keypoints": [
            {"i": 15, "x": 640.0, "y": 360.0, "score": 1.0, "visible": True},
        ],
    }
    arr = keypoints_from_json(payload)
    assert abs(float(arr[15, 0])) < 1e-4
    assert abs(float(arr[15, 1])) < 1e-4


def test_keypoints_from_json_rejects_ambiguous_norm() -> None:
    payload = {
        "keypoints": [
            {"i": 15, "x": 0.1, "y": -0.2, "score": 1.0, "visible": True},
        ]
    }
    try:
        keypoints_from_json(payload)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "coord_space" in str(exc).lower() or "norm" in str(exc).lower()


def test_attach_norm_crop_roundtrip_json() -> None:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[15] = [640.0, 360.0, 1.0]
    for i in range(28):
        if pts[i, 2] == 0:
            pts[i] = [600.0 + i, 300.0, 0.8]
    bridge = build_bridge_frame(pts, (720, 1280, 3), body7=_body7())
    _attach(bridge)
    text = json.dumps(bridge.to_dict())
    loaded = keypoints_from_json(json.loads(text))
    assert abs(float(loaded[15, 0]) - float(bridge.keypoints_norm[15, 0])) < 1e-4
