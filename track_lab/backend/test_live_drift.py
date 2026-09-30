"""Live tracking must not drift: with frame rate, off a first frame, or by
feeding its own output back in.

Every bench here writes to tmp_path only (parts, iFacialMocap settings,
mouth shapes); no real camera, phone or calibration is touched.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from backend import face as face_mod
from backend.face import FaceBench
from backend.feel import feel
from backend.iris import IrisHit, hits_payload
from backend.osf_cam import OsfFrame
from backend.test_ifm import _anime_rest
from backend.travel_box import default_travel_box

_POSE = {"cx": 0.0, "cy": 0.0, "bx": 0.0, "by": 0.0, "scale": 1.0, "tz": 0.0, "tilt": 0.0, "ok": 1.0}
_CAMS = [{"index": 0, "name": "Desk Cam"}, {"index": 1, "name": "Side Cam"}]


def _bench(monkeypatch, tmp_path, *, parts: dict | None = None) -> FaceBench:
    from backend import presets as presets_mod
    from backend.presets import MouthBook

    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    monkeypatch.setattr(face_mod, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(face_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(face_mod, "PARTS_PATH", tmp_path / "overlay_parts.json")
    monkeypatch.setattr(face_mod, "IFM_PATH", tmp_path / "ifm.json")
    monkeypatch.setattr(face_mod, "list_cameras", lambda: [dict(cam) for cam in _CAMS])
    monkeypatch.setattr(face_mod, "load_camera_choice", lambda: (None, ""))
    monkeypatch.setattr(face_mod, "save_camera_index", lambda *a, **k: None)
    monkeypatch.setattr(face_mod, "travel", SimpleNamespace(payload=default_travel_box))
    monkeypatch.setattr(FaceBench, "_publish", lambda self, *a, **k: None)
    if parts is not None:
        (tmp_path / "overlay_parts.json").write_text(json.dumps(parts), encoding="utf-8")
    rest = _anime_rest()
    book = MouthBook()
    book.seed_rest(rest)
    monkeypatch.setattr(face_mod, "book", book)
    bench = FaceBench(rest_pts=rest.copy())
    monkeypatch.setattr(bench._osf, "start", lambda _cb, index=0: None)
    return bench


def _saved(tmp_path) -> dict:
    return json.loads((tmp_path / "overlay_parts.json").read_text(encoding="utf-8"))


# --- A: mouth / brow / pupil eases are timed, and eased once --------------


def test_expression_ease_trail_holds_when_the_frame_rate_drops(monkeypatch, tmp_path) -> None:
    """The mouth / brow ease counted frames: at 15 fps it trailed twice as
    long as at 30, while the head's (timed) did not."""
    bench = _bench(monkeypatch, tmp_path)
    clock = {"t": 0.0}
    monkeypatch.setattr(
        face_mod, "time", SimpleNamespace(perf_counter=lambda: clock["t"], time=lambda: clock["t"])
    )
    rest = bench.rest_pts.copy()
    opened = rest.copy()
    opened[25, 1] += 12.0

    def gap(fps: float, source: str = "osf", seconds: float = 0.2) -> float:
        bench._smooth_mesh = None
        clock["t"] = 0.0
        bench._ease_mesh(rest, source)
        out = rest
        for i in range(1, round(seconds * fps) + 1):
            clock["t"] = i / fps
            out = bench._ease_mesh(opened, source)
        return float(opened[25, 1] - out[25, 1])

    alpha = feel.alpha()
    # At the webcam's usual 30 fps a frame still eases by Smooth's share...
    assert abs(gap(30.0, seconds=1.0 / 30.0) - 12.0 * (1.0 - alpha)) < 1e-3
    # ...and 15 fps lands where 30 does, not twice as far behind.
    assert gap(30.0) > 0.05
    assert abs(gap(15.0) - gap(30.0)) < 1e-3
    # The iPhone keeps its feel at its ~60 packets a second, and holds it at 30.
    assert abs(gap(60.0, "ifm", seconds=1.0 / 60.0) - 12.0 * (1.0 - alpha)) < 1e-3
    assert abs(gap(30.0, "ifm") - gap(60.0, "ifm")) < 1e-3


def test_pupil_ease_trail_holds_when_the_frame_rate_drops(monkeypatch, tmp_path) -> None:
    bench = _bench(monkeypatch, tmp_path)
    clock = {"t": 0.0}
    monkeypatch.setattr(
        face_mod, "time", SimpleNamespace(perf_counter=lambda: clock["t"], time=lambda: clock["t"])
    )
    rest = bench.rest_pts.copy()
    cx = 0.5 * (float(rest[11, 0]) + float(rest[13, 0]))
    cy = 0.5 * (float(rest[11, 1]) + float(rest[13, 1]))
    look = {"x": cx}
    monkeypatch.setattr(
        face_mod,
        "retarget_iris",
        lambda posed, **kw: ([{"id": 28, "x": look["x"], "y": cy, "score": 1.0, "visible": True}], "iris_pose"),
    )
    frame = OsfFrame(pose=dict(_POSE), faces=1, source="osf")

    def gap(fps: float) -> float:
        bench._iris_ease = {}
        clock["t"] = 0.0
        look["x"] = cx
        bench._finish_live(frame, rest.copy())
        look["x"] = cx + 2.0
        for i in range(1, round(0.2 * fps) + 1):
            clock["t"] = i / fps
            bench._finish_live(frame, rest.copy())
        return 2.0 - bench._iris_ease[28][0]

    assert gap(30.0) > 0.01
    assert abs(gap(15.0) - gap(30.0)) < 1e-6


def test_webcam_hands_raw_mouth_weights_to_the_bench(monkeypatch) -> None:
    """osf_cam eased the weights once a frame, then the bench eased the mesh
    again: two lags on lip sync (the iPhone lost its extra one in 9dcab08)."""
    from backend import osf_cam as osf_cam_mod
    from backend.osf_cam import OsfCam
    from backend.presets import empty_weights

    opened = empty_weights()
    key = next(iter(opened))
    opened[key] = 1.0
    calls = {"n": 0}

    def weights(face, pose=None):
        calls["n"] += 1
        return empty_weights() if calls["n"] == 1 else dict(opened)

    class FakeCap:
        def read(self):
            time.sleep(0.005)
            return True, np.zeros((12, 12, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    class FakeTracker:
        def predict(self, frame):
            return [SimpleNamespace()]

        def close(self) -> None:
            return None

    monkeypatch.setattr(osf_cam_mod, "viseme_weights", weights)
    monkeypatch.setattr(osf_cam_mod, "_encode_jpeg", lambda img: b"x")
    cam = OsfCam()
    cam._cap = FakeCap()
    cam._tracker = FakeTracker()
    cam._running = True
    snaps: list[OsfFrame] = []
    done = threading.Event()

    def on_frame(snap: OsfFrame) -> None:
        snaps.append(snap)
        if len(snaps) >= 2:
            done.set()

    thread = threading.Thread(target=cam._loop, args=(on_frame,), daemon=True)
    cam._thread = thread
    thread.start()
    done.wait(timeout=3.0)
    cam.stop()
    thread.join(timeout=2.0)
    assert len(snaps) >= 2
    assert snaps[1].weights[key] == 1.0


# --- B: the gaze zero -------------------------------------------------------


def _cam_frame() -> OsfFrame:
    hit = {"x": 10.0, "y": 10.0, "score": 1.0, "visible": True, "method": "osf_gaze"}
    return OsfFrame(
        pose=dict(_POSE),
        faces=1,
        lms_xy=np.zeros((68, 3), dtype=np.float32),
        iris_cam=hits_payload(IrisHit(side="r", **hit), IrisHit(side="l", **hit)),
        source="osf",
    )


def _pupils(monkeypatch, reads: list[tuple[float, float]]) -> None:
    """Each frame's pupils, as fractions of their eye box, both eyes alike."""
    queue = list(reads)

    def fake(lms, right, left):
        nx, ny = queue.pop(0)
        return {"r": {"nx": nx, "ny": ny}, "l": {"nx": nx, "ny": ny}}

    monkeypatch.setattr(face_mod, "rest_look_from_cam", fake)


def _rest_looks(monkeypatch) -> list[dict]:
    """The gaze zero each frame's iris retarget is handed."""
    seen: list[dict] = []

    def fake(posed, **kw):
        seen.append(dict(kw.get("rest_look") or {}))
        return [], "none"

    monkeypatch.setattr(face_mod, "retarget_iris", fake)
    return seen


def test_webcam_gaze_rest_waits_for_a_steady_centred_look(monkeypatch, tmp_path) -> None:
    """The first frame with pupils was the gaze zero, saved for good: a
    glance aside there held both eyes off to one side until Set Rest."""
    bench = _bench(monkeypatch, tmp_path)
    seen = _rest_looks(monkeypatch)
    steady = [(0.02 if i % 2 else 0.03, -0.05) for i in range(10)]
    _pupils(monkeypatch, [(0.30, 0.0)] + steady)
    posed = bench.rest_pts.copy()
    bench._finish_live(_cam_frame(), posed)
    assert "r" not in seen[-1]
    for _ in range(9):
        bench._finish_live(_cam_frame(), posed)
    assert "r" not in seen[-1]
    bench._finish_live(_cam_frame(), posed)
    assert seen[-1]["r"] == {"nx": 0.025, "ny": -0.05}
    assert seen[-1]["l"] == {"nx": 0.025, "ny": -0.05}
    # It is this session's: nothing was saved.
    assert not (tmp_path / "overlay_parts.json").exists()


def test_webcam_gaze_rest_skips_a_wandering_look(monkeypatch, tmp_path) -> None:
    bench = _bench(monkeypatch, tmp_path)
    seen = _rest_looks(monkeypatch)
    wander = [(0.0 if i % 2 else 0.12, 0.0) for i in range(12)]
    _pupils(monkeypatch, wander)
    for _ in wander:
        bench._finish_live(_cam_frame(), bench.rest_pts.copy())
    assert "r" not in seen[-1]


def test_iphone_gaze_rest_waits_for_a_steady_quiet_look(monkeypatch, tmp_path) -> None:
    """The first quiet frame (up to 0.2 off) was the iPhone's zero for good."""
    bench = _bench(monkeypatch, tmp_path)
    seen = _rest_looks(monkeypatch)

    def frame(x: float) -> OsfFrame:
        return OsfFrame(look={"x": x, "y": 0.0}, pose=dict(_POSE, sway=1.0), faces=1, source="ifm")

    bench._finish_live(frame(0.18), bench.rest_pts.copy())
    assert "x" not in seen[-1]
    for _ in range(10):
        bench._finish_live(frame(0.0), bench.rest_pts.copy())
    assert seen[-1]["x"] == 0.0 and seen[-1]["y"] == 0.0
    assert not (tmp_path / "overlay_parts.json").exists()


def test_learnt_gaze_rest_lasts_one_session(monkeypatch, tmp_path) -> None:
    bench = _bench(monkeypatch, tmp_path)
    _rest_looks(monkeypatch)
    _pupils(monkeypatch, [(0.05, 0.1)] * 10)
    for _ in range(10):
        bench._finish_live(_cam_frame(), bench.rest_pts.copy())
    with bench._lock:
        assert "r" in bench._rest_look()
    bench.stop_live()
    with bench._lock:
        assert bench._rest_look() == {}


def test_set_rest_gaze_is_dropped_on_another_camera(monkeypatch, tmp_path) -> None:
    bench = _bench(monkeypatch, tmp_path)
    bench.start_live(camera=0)
    _pupils(monkeypatch, [(0.08, 0.1)])
    bench._capture_rest_look(_cam_frame())
    saved = _saved(tmp_path)["look_rest"]
    assert saved["camera"] == "Desk Cam"
    assert saved["r"] == {"nx": 0.08, "ny": 0.1}
    bench.stop_live()
    # Back on the same camera: Set Rest's zero stays.
    bench.start_live(camera=0)
    assert bench._look_rest["r"] == {"nx": 0.08, "ny": 0.1}
    bench.stop_live()
    # Another camera sees the same look elsewhere in the eye: dropped.
    bench.start_live(camera=1)
    assert "r" not in bench._look_rest and "l" not in bench._look_rest
    saved = _saved(tmp_path)["look_rest"]
    assert "r" not in saved and "camera" not in saved


def test_gaze_rest_saved_before_cameras_were_recorded_still_loads(monkeypatch, tmp_path) -> None:
    old = {
        "hair": [],
        "skeleton": [],
        "iris": [],
        "iris_method": "none",
        "look_rest": {"x": 0.05, "y": -0.02, "r": {"nx": 0.04, "ny": 0.12}, "l": {"nx": 0.03, "ny": 0.1}},
        "point_offsets": [],
    }
    bench = _bench(monkeypatch, tmp_path, parts=old)
    assert bench._look_rest == old["look_rest"]
    # Taken as the first camera's...
    bench.start_live(camera=0)
    with bench._lock:
        assert bench._rest_look() == old["look_rest"]
    assert _saved(tmp_path)["look_rest"]["camera"] == "Desk Cam"
    bench.stop_live()
    # ...so a switch drops its pupils; the iPhone's look zero is not a camera's.
    bench.start_live(camera=1)
    assert bench._look_rest == {"x": 0.05, "y": -0.02}


# --- C: hair with no rig ----------------------------------------------------


def test_hair_without_a_rig_is_not_re_clamped_every_frame(monkeypatch, tmp_path) -> None:
    """With no hair rig, last frame's clamped hair was clamped again each
    frame the head sat at a wall, so it slid away, and could be saved so."""
    # A class this build does not rig: build_hair_rig gives None.
    hat = [{"class": "hat", "polygon": [[20.0, 0.0], [80.0, 0.0], [50.0, 15.0]]}]
    bench = _bench(monkeypatch, tmp_path, parts={"hair": hat})
    assert bench._hair_rig is None and bench._hair == hat
    posed = bench.rest_pts.copy()
    posed[:, 0] += 300.0  # far past the head's wall
    frame = OsfFrame(pose=dict(_POSE), faces=1, source="osf")
    bench._finish_live(frame, posed.copy())
    first = [dict(seg) for seg in bench._hair]
    assert first[0]["polygon"] != hat[0]["polygon"]  # the limiter did move it
    for _ in range(5):
        bench._finish_live(frame, posed.copy())
    assert bench._hair == first
    bench._save_parts()
    assert _saved(tmp_path)["hair"] == hat
    bench.stop_live()
    assert bench._hair == hat


# --- D: OpenSeeFace's blink scale -------------------------------------------


def test_osf_tracker_is_built_to_stop_learning_the_blink_scale(monkeypatch, tmp_path) -> None:
    from backend import osf_cam as osf_cam_mod

    seen: dict = {}
    fake = ModuleType("tracker")

    class Tracker:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

    fake.Tracker = Tracker
    monkeypatch.setitem(sys.modules, "tracker", fake)
    (tmp_path / "lm_model3_opt.onnx").write_bytes(b"")
    monkeypatch.setattr(osf_cam_mod, "MODELS_DIR", tmp_path)
    osf_cam_mod._make_tracker(640, 480)
    assert seen["max_feature_updates"] == osf_cam_mod._FEATURE_LEARN_S > 0


def test_osf_feature_scale_holds_after_its_learning_window() -> None:
    """What max_feature_updates is in OpenSeeFace: seconds after the face is
    first seen, past which its median / min / max stop moving. 0 never stops."""
    from backend import osf_cam as osf_cam_mod

    tracker = pytest.importorskip("tracker")
    learn = osf_cam_mod._FEATURE_LEARN_S
    held = tracker.Feature(max_feature_updates=learn)
    forever = tracker.Feature()
    t = 0.0
    # Open eyes (0.3) and a blink (0.1) every 3 s, through the window.
    for i in range(int(learn * 30)):
        x = 0.1 if i % 90 == 45 else 0.3
        held.update(x, now=t)
        forever.update(x, now=t)
        t += 1.0 / 30.0
    scale = (held.current_median, held.min, held.max)
    # Then a long stretch of narrower eyes (a squint, the light changing).
    for _ in range(3000):
        held.update(0.22, now=t)
        forever.update(0.22, now=t)
        t += 1.0 / 30.0
    assert (held.current_median, held.min, held.max) == scale
    assert abs(forever.current_median - 0.22) < 1e-6


# --- E: the head's size and Set Rest zero -----------------------------------


def _rig_rest() -> np.ndarray:
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    return rest


_RIG_POSE = {"cx": 200.0, "cy": 200.0, "bx": 200.0, "by": 200.0, "scale": 100.0, "tz": 500.0, "tilt": 0.0, "ok": 1.0}
_STILL = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}


def test_first_clean_solve_keeps_the_sealed_size() -> None:
    """The stand-in lock (no clean solve yet) seals size from a 5-frame
    median. The first clean solve re-locked the head and took its one
    frame's distance as the size zero for the rest of the session."""
    from backend.rig import _BAD_SOLVE_LOCK, FaceRig

    feel.values.update({"smoothing": 0.0, "max_size": 1.0})
    rest = _rig_rest()
    rig = FaceRig()
    bad = dict(_RIG_POSE, head_ok=0.0)
    for _ in range(_BAD_SOLVE_LOCK + 12):
        rig.apply(rest, rest, _STILL, bad)
    assert rig.locked and rig._provisional and rig._size_ready
    sealed = (rig._cam_tz, rig._cam_scale)
    # The first clean solve comes on a lean toward the camera.
    rig.apply(rest, rest, _STILL, dict(_RIG_POSE, head_ok=1.0, tz=400.0, scale=125.0))
    assert not rig._provisional
    assert (rig._cam_tz, rig._cam_scale) == sealed
    assert rig._s > 1.1
    rig.apply(rest, rest, _STILL, dict(_RIG_POSE, head_ok=1.0))
    assert abs(rig._s - 1.0) < 0.02


def test_set_rest_locks_the_head_on_its_capture_window(monkeypatch) -> None:
    """Set Rest's mouth zero is the mean of its ~1 s window; the head's was
    the one frame the window ended on, twitch and all."""
    from backend import rig as rig_mod
    from backend.rig import FaceRig
    from backend.visemes import _rest

    clock = {"t": 0.0}
    monkeypatch.setattr(rig_mod, "time", SimpleNamespace(perf_counter=lambda: clock["t"]))
    capture = SimpleNamespace(capturing="")
    monkeypatch.setattr(rig_mod, "calibrator", capture)
    feel.values.update({"smoothing": 0.0, "max_size": 1.0})
    rest = _rig_rest()
    pose = dict(_RIG_POSE, tz=0.0)
    saved = _rest.snapshot()
    _rest.reset()
    try:
        rig = FaceRig()
        rig.apply(rest, rest, _STILL, pose)
        capture.capturing = "rest"
        for i in range(36):
            clock["t"] = (i + 1) / 30.0
            yaw = 10.0 + (0.4 if i % 2 else -0.4)
            rig.apply(rest, rest, {"pitch": 0.0, "yaw": yaw, "roll": 0.0}, pose)
        # The capture ends: its mouth zero lands with this frame, a twitch.
        capture.capturing = ""
        _rest.use_snapshot({"open": 0.12, "width": 0.4, "corner": 0.0})
        clock["t"] += 1.0 / 30.0
        rig.apply(rest, rest, {"pitch": 0.0, "yaw": 18.0, "roll": 0.0}, dict(pose, scale=130.0))
        assert abs(rig._yaw - 10.0) < 0.5
        assert rig._cam_scale == 100.0
        # No capture window (a zero loaded from disk): the frame in hand, as before.
        _rest.use_snapshot({"open": 0.10, "width": 0.4, "corner": 0.0})
        rig.apply(rest, rest, {"pitch": 0.0, "yaw": 6.0, "roll": 0.0}, pose)
        assert rig._yaw == 6.0
    finally:
        _rest.reset()
        if saved:
            _rest.use_snapshot(saved)
