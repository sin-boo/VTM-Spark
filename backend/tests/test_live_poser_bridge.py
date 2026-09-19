"""Synthetic / held body fallback unit tests (no camera)."""

from __future__ import annotations

import numpy as np

from iris_tracker import CustomIrisTracker
from skeleton import SkeletonHold, resolve_body7, synth_upper_body


def _fake_pts28() -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    # Rough face in pixel coords.
    pts[0] = [100, 200, 1]
    pts[1] = [140, 260, 1]
    pts[2] = [200, 300, 1]  # chin
    pts[3] = [260, 260, 1]
    pts[4] = [300, 200, 1]
    for i in range(5, 11):
        pts[i] = [120 + (i - 5) * 20, 120, 1]
    pts[11] = [130, 160, 1]
    pts[12] = [150, 155, 1]
    pts[13] = [170, 160, 1]
    pts[14] = [180, 190, 1]
    pts[15] = [200, 200, 1]
    pts[16] = [220, 190, 1]
    pts[17] = [230, 160, 1]
    pts[18] = [250, 155, 1]
    pts[19] = [270, 160, 1]
    for i in range(20, 28):
        pts[i] = [160 + (i - 20) * 8, 240, 1]
    return pts


def _iris_detection(x: float, y: float = 158.0) -> dict:
    return {
        "bbox": (int(x - 15), int(y - 10), int(x + 15), int(y + 10)),
        "pupil": (x, y),
        "score": 0.9,
        "visible": True,
        "cx": x,
        "cy": y,
    }


def test_iris_matching_respects_eye_regions() -> None:
    tracker = object.__new__(CustomIrisTracker)
    tracker.detect = lambda *args, **kwargs: [
        _iris_detection(148.0),
        _iris_detection(252.0),
    ]
    right, left = tracker.match_to_face(
        np.zeros((320, 400, 3), dtype=np.uint8),
        _fake_pts28(),
    )
    assert right.visible and left.visible
    assert abs(right.x - 252.0) < 1e-5
    assert abs(left.x - 148.0) < 1e-5


def test_iris_detection_is_not_stolen_by_other_eye() -> None:
    tracker = object.__new__(CustomIrisTracker)
    tracker.detect = lambda *args, **kwargs: [_iris_detection(148.0)]
    right, left = tracker.match_to_face(
        np.zeros((320, 400, 3), dtype=np.uint8),
        _fake_pts28(),
    )
    assert not right.visible
    assert left.visible


def test_resolve_body7_synthetic_when_no_tracker() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    body, method = resolve_body7(
        frame,
        _fake_pts28(),
        None,
        allow_synth_fallback=True,
    )
    assert method == "synthetic_from_face"
    assert body is not None
    assert body.shape == (7, 4)
    assert int(np.sum(body[:, 3] >= 0.5)) >= 5


def test_skeleton_hold_keeps_last_pose() -> None:
    hold = SkeletonHold()
    body = synth_upper_body(_fake_pts28())
    kept, lost, method = hold.update(body, "synthetic_from_face")
    assert not lost
    held, lost2, method2 = hold.update(None, "none")
    assert lost2
    assert method2.endswith("_held")
    np.testing.assert_allclose(held[:, :2], kept[:, :2])


def test_synth_upper_body_nose_near_face() -> None:
    pts = _fake_pts28()
    body = synth_upper_body(pts)
    # Slot 0 of body7 is body nose — should be near face nose tip pts[15].
    assert abs(float(body[0, 0]) - float(pts[15, 0])) < 5.0
    assert abs(float(body[0, 1]) - float(pts[15, 1])) < 5.0


def test_osf_mouth_map_overrides_slot_25() -> None:
    from label_schema import DEFAULT_MOUTH_OSF, normalize_mouth_osf_map, osf_to_label28

    mapped = normalize_mouth_osf_map({"25": 55, "99": 1})
    assert mapped[25] == 55
    assert mapped[21] == DEFAULT_MOUTH_OSF[21]
    lms = np.zeros((66, 3), dtype=np.float32)
    for i in range(66):
        # OSF stores (y, x, conf).
        lms[i] = [float(i), float(100 + i), 1.0]
    pts, _ = osf_to_label28(lms, mouth_osf={25: 55})
    assert float(pts[25, 0]) == 155.0
    assert float(pts[25, 1]) == 55.0
    assert float(pts[21, 0]) == 160.0


def test_osf_to_label28_does_not_collapse_small_mouth_gap() -> None:
    from label_schema import osf_to_label28

    lms = np.zeros((66, 3), dtype=np.float32)
    for i in range(66):
        lms[i] = [200.0, 300.0, 1.0]
    # OSF is (y, x, conf). Inner upper mid 60 / lower mid 64 with a small gap.
    lms[60] = [180.0, 300.0, 1.0]
    lms[64] = [186.0, 300.0, 1.0]
    lms[62] = [183.0, 260.0, 1.0]
    lms[58] = [183.0, 340.0, 1.0]
    pts, _ = osf_to_label28(lms)
    assert abs(float(pts[25, 1]) - float(pts[21, 1])) == 6.0
