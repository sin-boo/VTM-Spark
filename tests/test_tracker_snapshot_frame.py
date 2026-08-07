"""Snapshot frame copy + mirror alignment helpers for camera diagnostic."""

from __future__ import annotations

import numpy as np

from live_poser_client import LivePoserTracker, TrackerSnapshot
from cameras import CameraCapture


def test_tracker_snapshot_frame_bgr_is_independent_copy() -> None:
    """UI must not share a writable buffer with the capture loop."""
    src = np.zeros((24, 32, 3), dtype=np.uint8)
    src[:, :] = (10, 20, 30)
    snap = TrackerSnapshot(
        t=0.0,
        keypoints_px=None,
        keypoints_norm=None,
        image_wh=(32, 24),
        bridge=None,
        rel=None,
        fps=0.0,
        faces=0,
        frame_bgr=np.ascontiguousarray(src.copy()),
        mirrored=False,
    )
    assert snap.frame_bgr is not None
    assert snap.frame_bgr.shape == (24, 32, 3)
    # Mutating the original must not affect the snapshot copy.
    src[:, :] = (99, 99, 99)
    assert int(snap.frame_bgr[0, 0, 0]) == 10
    # Mutating the snapshot must not affect a later UI-held reference if we copy again.
    held = snap.frame_bgr.copy()
    snap.frame_bgr[:, :] = 0
    assert int(held[0, 0, 0]) == 10


def test_tracker_snapshot_mirrored_flag_and_wh() -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    # Paint left edge green so a mirror flip would move it to the right.
    frame[:, 0:4] = (0, 255, 0)
    mirrored = np.ascontiguousarray(frame[:, ::-1].copy())
    snap = TrackerSnapshot(
        t=1.0,
        keypoints_px=None,
        keypoints_norm=None,
        image_wh=(80, 40),
        bridge=None,
        rel=None,
        fps=12.0,
        faces=1,
        frame_bgr=mirrored,
        mirrored=True,
        skeleton_method="mediapipe_pose_lite",
        body_lost=False,
    )
    assert snap.mirrored is True
    assert snap.image_wh == (80, 40)
    assert snap.frame_bgr is not None
    # After horizontal flip, green strip is on the right edge.
    assert int(snap.frame_bgr[0, -1, 1]) == 255
    assert int(snap.frame_bgr[0, 0, 1]) == 0


class _FakeReader:
    def __init__(self, result) -> None:
        self.result = result

    def read(self):
        return self.result


def _fake_capture(result) -> CameraCapture:
    capture = object.__new__(CameraCapture)
    capture._reader = _FakeReader(result)
    capture._cap = None
    capture.last_health = "starting"
    capture.last_error = None
    return capture


def test_camera_capture_reports_runtime_black_and_stall() -> None:
    black = np.zeros((24, 32, 3), dtype=np.uint8)
    capture = _fake_capture((True, black))
    ok, frame = capture.read()
    assert ok and frame is not None
    assert capture.last_health == "black"

    capture = _fake_capture((False, None))
    ok, frame = capture.read()
    assert not ok and frame is None
    assert capture.last_health == "stalled"


def test_capture_heartbeat_keeps_preview_but_drops_stale_pose() -> None:
    tracker = LivePoserTracker()
    tracker._capture_session = 3
    frame = np.full((24, 32, 3), 50, dtype=np.uint8)
    pose = np.ones((37, 4), dtype=np.float32)
    tracker._latest = TrackerSnapshot(
        t=1.0,
        keypoints_px=pose.copy(),
        keypoints_norm=pose.copy(),
        image_wh=(32, 24),
        bridge=None,
        rel=None,
        fps=10.0,
        faces=1,
        frame_bgr=frame,
    )
    tracker._publish_capture_heartbeat(
        health="reconnecting",
        reason="Camera stalled — reconnecting",
        fail_streak=3,
    )
    snap = tracker.latest_snapshot()
    assert snap is not None
    assert snap.capture_ok is False
    assert snap.capture_health == "reconnecting"
    assert snap.capture_session == 3
    assert snap.keypoints_norm is None
    assert snap.frame_bgr is not None
    np.testing.assert_array_equal(snap.frame_bgr, frame)
