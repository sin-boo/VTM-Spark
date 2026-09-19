"""Per-model tracking folders merge into one KEYPOINT_SCHEMA tensor."""

from __future__ import annotations

import numpy as np

from backend.tracking import (
    BodyTrack,
    FaceTrack,
    HairTrack,
    IrisTrack,
    merge_tracks,
)
from backend.tracking.face import FaceTracker
from bridge import IrisPoint


def _pts28() -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(200, 440, 28)
    pts[:, 1] = np.linspace(160, 360, 28)
    pts[:, 2] = 1.0
    return pts


def _body7() -> np.ndarray:
    body = np.zeros((7, 4), dtype=np.float32)
    for i, (x, y) in enumerate(
        ((320, 280), (320, 340), (240, 360), (220, 420), (400, 360), (420, 420), (320, 400))
    ):
        body[i] = [x, y, 0.9, 1.0]
    return body


def test_merge_tracks_fills_face_iris_body() -> None:
    face = FaceTrack(pts28=_pts28(), alive=True)
    iris = IrisTrack(
        right=IrisPoint(x=250.0, y=200.0, score=0.9, visible=True, method="iris_pose"),
        left=IrisPoint(x=390.0, y=200.0, score=0.8, visible=True, method="iris_pose"),
        method="iris_pose",
    )
    body = BodyTrack(joints=_body7(), method="mediapipe_pose_lite", lost=False)
    merged = merge_tracks(
        image_shape=(480, 640, 3),
        face=face,
        iris=iris,
        body=body,
        hair=HairTrack(),
    )
    assert merged.keypoints_px.shape == (37, 4)
    assert merged.keypoints_norm.shape == (37, 4)
    assert merged.bridge.iris_method == "iris_pose"
    assert merged.bridge.skeleton_method == "mediapipe_pose_lite"
    np.testing.assert_allclose(merged.keypoints_px[15, :2], face.pts28[15, :2])
    np.testing.assert_allclose(merged.keypoints_px[28, :2], [250.0, 200.0])
    np.testing.assert_allclose(merged.keypoints_px[29, :2], [390.0, 200.0])
    np.testing.assert_allclose(merged.keypoints_px[30, :2], body.joints[0, :2])


def test_merge_tracks_keeps_hair_polygons() -> None:
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[100.0, 80.0], [200.0, 80.0], [150.0, 160.0]],
            "score": 0.9,
        }
    ]
    merged = merge_tracks(
        image_shape=(480, 640, 3),
        face=FaceTrack(pts28=_pts28(), alive=True),
        hair=HairTrack(segments=segs, method="animeseg_hair3"),
    )
    assert merged.hair_segments_px is not None
    assert merged.hair_segments_px[0]["class"] == "hair_middle"
    assert merged.bridge.hair_method == "animeseg_hair3"
    assert merged.hair_segments_norm is not None


def test_face_tracker_mouth_map_overrides_slot() -> None:
    tracker = FaceTracker()
    mapped = tracker.set_mouth_osf_map({"25": 55})
    assert mapped[25] == 55
    lms = np.zeros((66, 3), dtype=np.float32)
    for i in range(66):
        lms[i] = [float(i), float(100 + i), 1.0]

    class _Face:
        def __init__(self) -> None:
            self.lms = lms

    track = tracker.decode(_Face(), landmark_threshold=0.0)
    assert track.alive
    assert float(track.pts28[25, 0]) == 155.0
    assert float(track.pts28[25, 1]) == 55.0
