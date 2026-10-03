"""Live bench behaviour around the pupils and a stopped session."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from backend.face import FaceBench
from backend.osf_cam import OsfFrame
from backend.test_ifm import _anime_rest


def _bench() -> FaceBench:
    return FaceBench(rest_pts=_anime_rest())


def _quiet_bench(monkeypatch, tmp_path) -> FaceBench:
    """A bench that writes nothing under output/ and opens no device."""
    from backend import face as face_mod

    monkeypatch.setattr(face_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(face_mod, "PARTS_PATH", tmp_path / "overlay_parts.json")
    monkeypatch.setattr(face_mod, "IFM_PATH", tmp_path / "ifm.json")
    for name in ("_load_parts", "_load_ifm", "_save_parts", "_save_ifm"):
        monkeypatch.setattr(FaceBench, name, lambda self: None)
    monkeypatch.setattr(FaceBench, "_publish", lambda self, *a, **k: None)
    monkeypatch.setattr(FaceBench, "_ensure_source", lambda self, **k: None)
    monkeypatch.setattr(FaceBench, "_camera_fields", lambda self: {"camera_index": 0, "cameras": []})
    bench = FaceBench(rest_pts=_anime_rest())
    monkeypatch.setattr(bench._ifm, "payload", lambda: {})
    monkeypatch.setattr(bench._ifm, "stop", lambda: None)
    monkeypatch.setattr(bench._osf, "stop", lambda: None)
    return bench


def _ifm_frame(raw: str = "jawOpen-0|=head#0,0,0,0,0,0") -> OsfFrame:
    from backend.ifm import blink_of, brow_of, look_of, parse_packet, weights_from_arkit

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


def test_pupils_hold_with_the_face_on_a_lost_frame() -> None:
    """The face holds its last pose on a lost frame (or an iPhone stall). The
    pupils used to snap to the still's rest pixels, outside a moved head."""
    bench = _bench()
    bench._on_osf(_ifm_frame())
    held = [dict(row) for row in bench._iris]
    assert held
    bench._on_osf(OsfFrame(faces=0, source="ifm"))
    assert bench._iris == held


def test_pupils_ease_inside_their_eye_not_on_screen() -> None:
    bench = _bench()
    posed = _anime_rest()
    # Pupils a little right of each eye's corner midpoint.
    rows = [
        {"id": 28, "x": 33.0, "y": 36.0, "score": 1.0, "visible": True},
        {"id": 29, "x": 69.0, "y": 36.0, "score": 1.0, "visible": True},
    ]
    bench._ease_pupils(rows, posed, 0.3)
    # The whole head moves 40 px; the pupils sit where they were in the eye.
    moved = posed.copy()
    moved[:, 0] += 40.0
    shifted = [dict(row, x=float(row["x"]) + 40.0) for row in rows]
    out = bench._ease_pupils(shifted, moved, 0.3)
    assert [round(float(row["x"]), 6) for row in out] == [73.0, 109.0]
    # Only iris 29 seen now: it keeps its own history, not 28's.
    out = bench._ease_pupils([shifted[1]], moved, 0.3)
    assert round(float(out[0]["x"]), 6) == 109.0


def test_limiter_change_after_stop_does_not_replay_the_last_frame() -> None:
    """Replaying the held frame after Stop re-locked the reset rig on it and
    brought back its preview, meters and blink."""
    bench = _bench()
    bench._on_osf(_ifm_frame("eyeBlink_L-90|eyeBlink_R-90|=head#0,0,0,0,0,0"))
    assert bench._live_pose is not None
    bench._live_pts = None
    bench._blink = {"l": 0.0, "r": 0.0}
    bench._replay_live_pose()
    assert bench._live_pts is None
    assert bench._blink == {"l": 0.0, "r": 0.0}


def _moved_frame() -> OsfFrame:
    """The head turned and slid away from ``_ifm_frame()``'s pose."""
    frame = _ifm_frame()
    return replace(
        frame,
        head={"pitch": 6.0, "yaw": 25.0, "roll": 8.0},
        pose=dict(frame.pose, cx=0.3, cy=-0.2),
    )


def test_each_live_pose_gets_the_next_seq_and_frames_carry_it(monkeypatch, tmp_path) -> None:
    """Camera and command threads both pack frames. Without a number on the
    pose, the one packed first could land last and put an older pose back."""
    from harness.pack import frame_from_bench

    bench = _quiet_bench(monkeypatch, tmp_path)
    start = bench._seq
    bench._on_osf(_ifm_frame())
    first = bench._seq
    bench._on_osf(_moved_frame())
    assert start < first < bench._seq
    frame = frame_from_bench(bench)
    assert frame["session"] == bench.session and frame["session"]
    assert frame["seq"] == bench._seq
    assert frame["pose_t"] > 0.0
    held = bench._seq
    # Stop drops the live pose: a live frame packed before it is older now.
    bench.stop_live()
    assert bench._seq > held
    assert frame_from_bench(bench)["seq"] == bench._seq


def test_each_tracker_process_has_its_own_session(monkeypatch, tmp_path) -> None:
    one = _quiet_bench(monkeypatch, tmp_path)
    two = _quiet_bench(monkeypatch, tmp_path)
    assert one.session and two.session and one.session != two.session


def test_nudge_is_measured_on_the_frame_the_user_saw(monkeypatch, tmp_path) -> None:
    """The user lines a point up on a preview a few frames old. Measured on
    the newest pose, that latency was saved into the offset for good."""
    bench = _quiet_bench(monkeypatch, tmp_path)
    monkeypatch.setattr(bench._ifm, "_running", True)
    bench._on_osf(_ifm_frame())
    seen_seq = bench._seq
    seen = bench._live_pts.copy()
    # The head eases over the next frames while the preview still shows ``seen``.
    for _ in range(20):
        bench._on_osf(_moved_frame())
    now = bench._live_pts.copy()
    moved = np.abs(now[:28, :2] - seen[:28, :2]).max(axis=1)
    idx = int(np.argmax(moved))
    assert float(moved[idx]) > 0.5
    target = (float(seen[idx, 0]) + 7.0, float(seen[idx, 1]) - 3.0)
    out = bench.set_point(
        {"id": idx, "x": target[0], "y": target[1], "seq": seen_seq, "session": bench.session}
    )
    assert not out["error"]
    dx, dy = bench._point_offsets[idx]
    assert abs(dx - 7.0) < 1e-3 and abs(dy + 3.0) < 1e-3


def test_nudge_without_a_seq_uses_the_current_pose(monkeypatch, tmp_path) -> None:
    """The lab's own page and an older desk send no seq: nothing changes for them."""
    bench = _quiet_bench(monkeypatch, tmp_path)
    monkeypatch.setattr(bench._ifm, "_running", True)
    bench._on_osf(_ifm_frame())
    bench._on_osf(_moved_frame())
    now = bench._live_pts.copy()
    bench.set_point({"id": 21, "x": float(now[21, 0]) + 5.0, "y": float(now[21, 1])})
    dx, dy = bench._point_offsets[21]
    assert abs(dx - 5.0) < 1e-3 and abs(dy) < 1e-3
    # A seq no longer kept falls back the same way.
    bench.set_point(
        {"id": 21, "x": float(now[21, 0]) - 4.0, "y": float(now[21, 1]), "seq": -99, "session": bench.session}
    )
    dx, _dy = bench._point_offsets[21]
    assert abs(dx + 4.0) < 1e-3


def test_nudge_from_before_a_restart_is_refused(monkeypatch, tmp_path) -> None:
    """A set_point replayed onto a new tracker was measured against its rest,
    not the live pose it was made on. Refuse it; nothing is saved."""
    bench = _quiet_bench(monkeypatch, tmp_path)
    saved: list[bool] = []
    monkeypatch.setattr(FaceBench, "_save_parts", lambda self: saved.append(True))
    out = bench.set_point({"id": 21, "x": 10.0, "y": 10.0, "seq": 3, "session": "an-old-tracker"})
    assert "restarted" in str(out["error"])
    assert bench._point_offsets == {}
    assert saved == []


def test_pose_history_is_bounded(monkeypatch, tmp_path) -> None:
    from backend import face as face_mod

    bench = _quiet_bench(monkeypatch, tmp_path)
    monkeypatch.setattr(face_mod, "POSE_HISTORY", 3)
    for _ in range(5):
        bench._on_osf(_ifm_frame())
    assert list(bench._pose_history) == [bench._seq - 2, bench._seq - 1, bench._seq]
