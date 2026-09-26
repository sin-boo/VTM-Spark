"""Hair pose-map + live polygon helpers (no GPU / no YOLO weights required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hair_tracker import clone_hair_segments, flip_hair_pixels, _is_hair3_weights
from backend.live_poser_client import TrackerSnapshot
from utils.hair import (
    flip_hair_horizontal,
    rasterize_hair_maps,
    transform_hair_crop,
    transform_hair_crop_rect,
)
from utils.keypoints import NUM_POSE_CHANNELS, empty_keypoints
from utils.pose_map import rasterize_pose_maps
from utils.coordinate_frames import pad_square_crop_rect


def test_pose_channels_include_hair() -> None:
    assert NUM_POSE_CHANNELS == 11


def test_hair3_weights_name() -> None:
    assert _is_hair3_weights(Path("models/trackers/animeseg_hair3.pt"))
    assert not _is_hair3_weights(Path("models/trackers/other.pt"))


def test_hair_crop_flip_and_pose_channels() -> None:
    segs = [
        {
            "class": "hair_left",
            "polygon": [[100.0, 100.0], [200.0, 100.0], [150.0, 200.0]],
        }
    ]
    cropped = transform_hair_crop(
        segs,
        crop_x0=0,
        crop_y0=0,
        crop_w=400,
        crop_h=400,
        out_w=100,
        out_h=100,
        normalize=True,
    )
    assert cropped[0]["class"] == "hair_left"
    flipped = flip_hair_horizontal(cropped)
    assert flipped[0]["class"] == "hair_right"
    maps = rasterize_hair_maps(cropped, 32)
    assert maps.shape == (3, 32, 32)
    assert float(maps[1].sum()) > 0.0
    k = empty_keypoints()
    k[0] = [0.0, 0.0, 1.0, 1.0]
    batch = torch.from_numpy(k[None].astype(np.float32))
    hair = torch.from_numpy(maps[None].astype(np.float32))
    pose = rasterize_pose_maps(batch, 32, 32, hair_maps=hair)
    assert tuple(pose.shape) == (1, NUM_POSE_CHANNELS, 32, 32)
    assert float(pose[0, 9].sum()) > 0.0


def test_live_hair_pixel_flip_swaps_left_right() -> None:
    segs = [
        {
            "class": "hair_left",
            "polygon": [[10.0, 10.0], [30.0, 10.0], [20.0, 40.0]],
        }
    ]
    flipped = flip_hair_pixels(segs, 100)
    assert flipped[0]["class"] == "hair_right"
    xs = [p[0] for p in flipped[0]["polygon"]]
    assert min(xs) > 60.0
    cloned = clone_hair_segments(flipped)
    assert cloned[0]["class"] == "hair_right"


def test_hair_uses_same_pad_square_crop_as_keypoints() -> None:
    crop = pad_square_crop_rect(1280, 720, image_size=768)
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[640.0, 100.0], [700.0, 100.0], [670.0, 180.0]],
        }
    ]
    normed = transform_hair_crop_rect(segs, crop)
    assert normed[0]["class"] == "hair_middle"
    x = float(normed[0]["polygon"][0][0])
    # Image center x stays near 0 after pad-square.
    assert abs(x) < 0.05


def test_tracker_snapshot_hair_fields_default() -> None:
    snap = TrackerSnapshot(
        t=0.0,
        keypoints_px=None,
        keypoints_norm=None,
        image_wh=(32, 24),
        bridge=None,
        rel=None,
        fps=0.0,
        faces=0,
    )
    assert snap.hair_method == "none"
    assert snap.hair_lost is False
    assert snap.hair_segments_px is None
    assert snap.hair_segments_norm is None


def test_live_poser_skips_webcam_hair_by_default() -> None:
    from backend.live_poser_client import LivePoserTracker

    assert LivePoserTracker().use_hair is False


def _rest_hair_segments() -> list[dict]:
    return [
        {
            "class": "hair_middle",
            "polygon": [[-0.12, -0.55], [0.12, -0.55], [0.0, -0.32]],
        },
        {
            "class": "hair_left",
            "polygon": [[-0.42, -0.40], [-0.18, -0.40], [-0.30, 0.05]],
        },
        {
            "class": "hair_right",
            "polygon": [[0.18, -0.40], [0.42, -0.40], [0.30, 0.05]],
        },
    ]


def test_hair_follow_identity_at_rest() -> None:
    from backend.engine import neutral_keypoints
    from backend.hair_follow import build_hair_rig, follow_hair

    ref = neutral_keypoints()
    segs = _rest_hair_segments()
    rig = build_hair_rig(segs, ref)
    assert rig is not None
    assert len(rig.parts) == 3
    out = follow_hair(rig, ref)
    assert [s["class"] for s in out] == [s["class"] for s in segs]
    for a, b in zip(out, segs):
        np.testing.assert_allclose(a["polygon"], b["polygon"], atol=1e-5)


def test_hair_follow_translates_with_the_head() -> None:
    from backend.engine import neutral_keypoints
    from backend.hair_follow import build_hair_rig, follow_hair

    ref = neutral_keypoints()
    live = ref.copy()
    live[:28, 0] += 0.15
    live[28:30, 0] += 0.15
    rig = build_hair_rig(_rest_hair_segments(), ref)
    moved = follow_hair(rig, live)
    rest = follow_hair(rig, ref)
    dx = float(np.mean(np.asarray(moved[0]["polygon"])[:, 0] - np.asarray(rest[0]["polygon"])[:, 0]))
    assert dx > 0.10


def test_hair_follow_rolls_with_eye_line() -> None:
    from backend.engine import neutral_keypoints
    from backend.hair_follow import build_hair_rig, follow_hair
    from backend.pose_controller import L_EYE, R_EYE

    ref = neutral_keypoints()
    live = ref.copy()
    live[list(L_EYE), 1] += 0.05
    live[list(R_EYE), 1] -= 0.05
    rig = build_hair_rig(_rest_hair_segments(), ref)
    tilted = follow_hair(rig, live)
    rest = follow_hair(rig, ref)
    # Left lock should drop relative to right when the eye line rolls this way.
    rest_dy = float(np.mean(np.asarray(rest[1]["polygon"])[:, 1]) - np.mean(np.asarray(rest[2]["polygon"])[:, 1]))
    tilt_dy = float(np.mean(np.asarray(tilted[1]["polygon"])[:, 1]) - np.mean(np.asarray(tilted[2]["polygon"])[:, 1]))
    assert tilt_dy > rest_dy + 0.02
