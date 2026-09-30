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
