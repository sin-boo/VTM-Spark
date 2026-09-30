"""A closed eye must close the character's eye: all the way, at once."""

from __future__ import annotations

import math

import numpy as np

from backend.iris import retarget
from backend.lids import LidFilter, shape_blink
from harness.pack import pack_keypoints
from harness.protocol import LEFT_IRIS, RIGHT_IRIS


def _eyes() -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    pts[11] = [100.0, 100.0, 1.0]
    pts[12] = [120.0, 70.0, 1.0]
    pts[13] = [140.0, 100.0, 1.0]
    pts[17] = [200.0, 100.0, 1.0]
    pts[18] = [220.0, 70.0, 1.0]
    pts[19] = [240.0, 100.0, 1.0]
    return pts


def _chord_gap(pts: np.ndarray, a: int, lid: int, b: int) -> float:
    """Signed distance of the lid from its corner chord (positive = open)."""
    ax, ay = pts[a, :2]
    bx, by = pts[b, :2]
    px, py = pts[lid, :2]
    cx, cy = bx - ax, by - ay
    nx, ny = (cy, -cx) if cx > 0 else (-cy, cx)
    return float(((px - ax) * nx + (py - ay) * ny) / math.hypot(cx, cy))


def test_a_real_closed_eye_reads_fully_shut() -> None:
    # ARKit rarely goes past ~0.7 on a closed eye; linear, the lid stopped short.
    assert shape_blink({"l": 0.7, "r": 0.62}, "ifm") == {"l": 1.0, "r": 1.0}
    assert shape_blink({"l": 0.8, "r": 0.7}, "osf") == {"l": 1.0, "r": 1.0}
    # An open eye's resting value and noise stay open.
    assert shape_blink({"l": 0.1, "r": 0.0}, "ifm") == {"l": 0.0, "r": 0.0}
    # Looking down lowers real lids part way; that stays part way.
    down = shape_blink({"l": 0.4, "r": 0.4}, "ifm")
    assert 0.3 < down["l"] < 0.8


def test_lid_filter_closes_at_once_and_reopens_softly() -> None:
    lids = LidFilter()
    assert lids.update({"l": 0.0, "r": 0.0}, now=1.0) == {"l": 0.0, "r": 0.0}
    # One 60 Hz frame later the lid is (all but) shut.
    shut = lids.update({"l": 1.0, "r": 0.0}, now=1.0 + 1 / 60)
    assert shut["l"] > 0.7
    shut = lids.update({"l": 1.0, "r": 0.0}, now=1.0 + 2 / 60)
    assert shut["l"] > 0.95
    # Reopening eases over a few frames instead of popping.
    opening = lids.update({"l": 0.0, "r": 0.0}, now=1.0 + 3 / 60)
    assert 0.2 < opening["l"] < 0.9
    # A stall is not a blink: jump straight to the new value.
    assert lids.update({"l": 1.0, "r": 1.0}, now=5.0) == {"l": 1.0, "r": 1.0}


def test_fallback_pupil_is_dropped_on_a_shut_eye() -> None:
    """With the eyes closed the pupil detector finds nothing and retarget
    falls back to the eye centre. That pupil drew the eye open."""
    rows, _method = retarget(_eyes(), blink={"l": 1.0, "r": 0.0})
    assert {int(row["id"]) for row in rows} == {LEFT_IRIS}
    rows, method = retarget(_eyes(), blink={"l": 1.0, "r": 1.0})
    assert rows == [] and method == "none"


def test_packet_does_not_refill_a_shut_eyes_iris() -> None:
    face = [[float(x), float(y), float(s)] for x, y, s in _eyes()]
    both = pack_keypoints(face, [], [], {"l": 0.0, "r": 0.0})
    assert both[RIGHT_IRIS, 3] >= 0.5 and both[LEFT_IRIS, 3] >= 0.5
    wink = pack_keypoints(face, [], [], {"l": 1.0, "r": 0.0})
    assert wink[RIGHT_IRIS, 3] < 0.5
    assert wink[LEFT_IRIS, 3] >= 0.5
    # Even a pupil the lab sent is dropped once the eye is shut.
    sent = [{"id": RIGHT_IRIS, "x": 120.0, "y": 90.0, "score": 1.0, "visible": True}]
    assert pack_keypoints(face, [], sent, {"l": 0.9, "r": 0.0})[RIGHT_IRIS, 3] < 0.5


def test_iphone_blink_shuts_the_character_eye_through_heavy_smooth(monkeypatch) -> None:
    """End to end: eyeBlink 70 % is a closed eye on most faces. With Smooth
    high the lid used to crawl and stop part-open; now it is shut within two
    60 Hz frames and its pupil is gone from the packet."""
    from types import SimpleNamespace

    from backend import lids as lids_mod
    from backend.face import FaceBench
    from backend.feel import feel
    from backend.ifm import blink_of, brow_of, look_of, parse_packet, weights_from_arkit
    from backend.osf_cam import OsfFrame
    from backend.presets import book
    from backend.test_ifm import _anime_rest
    from harness.pack import frame_from_bench

    clock = {"t": 100.0}

    def tick() -> float:
        clock["t"] += 1.0 / 60.0
        return clock["t"]

    monkeypatch.setattr(lids_mod, "time", SimpleNamespace(perf_counter=tick))
    rest = _anime_rest()
    # No authored eye shapes: Eye closed is the drafted lid on its corner line.
    monkeypatch.setattr(book, "shapes", {"rest": rest.copy()})
    monkeypatch.setattr(book, "save", lambda: None)
    bench = FaceBench(rest_pts=rest.copy())

    def frame(raw: str) -> OsfFrame:
        packet = parse_packet(raw)
        assert packet is not None
        return OsfFrame(
            weights=weights_from_arkit(packet),
            blink=blink_of(packet),
            look=look_of(packet),
            brow=brow_of(packet),
            head={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            pose={"cx": 0.0, "cy": 0.0, "bx": 0.0, "by": 0.0, "scale": 1.0, "tz": 0.0, "tilt": 0.0, "ok": 1.0},
            faces=1,
            source="ifm",
        )

    prev = feel.payload()
    try:
        feel.update({"smoothing": 1.0, "use_visemes": 0.0})
        bench._on_osf(frame("eyeBlink_L-0|eyeBlink_R-0|=head#0,0,0,0,0,0"))
        for _ in range(2):
            bench._on_osf(frame("eyeBlink_L-70|eyeBlink_R-70|=head#0,0,0,0,0,0"))
        posed = bench._live_pts
        assert posed is not None
        for a, lid, b in ((11, 12, 13), (17, 18, 19)):
            assert _chord_gap(posed, a, lid, b) < 0.1 * _chord_gap(rest, a, lid, b)
        packet = frame_from_bench(bench)
        rows = {row["i"]: row for row in packet["keypoints"]}
        assert rows[RIGHT_IRIS]["visible"] is False
        assert rows[LEFT_IRIS]["visible"] is False
        # Eyes open again: pupils come back.
        for _ in range(6):
            bench._on_osf(frame("eyeBlink_L-0|eyeBlink_R-0|=head#0,0,0,0,0,0"))
        rows = {row["i"]: row for row in frame_from_bench(bench)["keypoints"]}
        assert rows[RIGHT_IRIS]["visible"] is True
        assert rows[LEFT_IRIS]["visible"] is True
    finally:
        feel.update(prev)


def test_a_shut_eye_lands_on_the_authored_eye_closed(monkeypatch) -> None:
    """Blink blends to the Eye closed shape, placed by the same head as the face."""
    from types import SimpleNamespace

    from backend import lids as lids_mod
    from backend.face import FaceBench
    from backend.ifm import blink_of, brow_of, look_of, parse_packet, weights_from_arkit
    from backend.osf_cam import OsfFrame
    from backend.presets import EYE_SLOTS, book
    from backend.test_ifm import _anime_rest

    clock = {"t": 100.0}

    def tick() -> float:
        clock["t"] += 1.0 / 60.0
        return clock["t"]

    monkeypatch.setattr(lids_mod, "time", SimpleNamespace(perf_counter=tick))
    rest = _anime_rest()
    closed = rest.copy()
    closed[[12, 18], 1] += 6.0
    closed[[11, 13, 17, 19], 1] += 1.5
    monkeypatch.setattr(book, "shapes", {"rest": rest.copy(), "eye_closed": closed})
    monkeypatch.setattr(book, "save", lambda: None)
    bench = FaceBench(rest_pts=rest.copy())

    def frame(raw: str) -> OsfFrame:
        packet = parse_packet(raw)
        assert packet is not None
        return OsfFrame(
            weights=weights_from_arkit(packet),
            blink=blink_of(packet),
            look=look_of(packet),
            brow=brow_of(packet),
            head={"pitch": 0.0, "yaw": 0.0, "roll": 12.0},
            pose={"cx": 0.0, "cy": 0.0, "bx": 0.0, "by": 0.0, "scale": 1.0, "tz": 0.0, "tilt": 0.0, "ok": 1.0},
            faces=1,
            source="ifm",
        )

    bench._on_osf(frame("eyeBlink_L-0|eyeBlink_R-0|=head#0,0,0,0,0,0"))
    for _ in range(4):
        bench._on_osf(frame("eyeBlink_L-90|eyeBlink_R-90|=head#0,0,0,0,0,0"))
    posed = bench._live_pts
    mesh = bench._live_pose[2]
    assert posed is not None and mesh is not None
    want = mesh.copy()
    slots = list(EYE_SLOTS)
    want[slots, :2] += closed[slots, :2] - rest[slots, :2]
    assert np.allclose(posed[slots, :2], bench._rig.project_face(want)[slots, :2], atol=1e-2)
