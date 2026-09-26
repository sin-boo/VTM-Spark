"""Camera picks follow the device name, not a DirectShow index that shifts."""

from __future__ import annotations

import numpy as np

from backend import face as face_mod
from backend.cameras import pick_default
from backend.face import FaceBench

CAMS = [
    {"index": 0, "name": "nizima LIVE Virtual Camera"},
    {"index": 1, "name": "DroidCam Video"},
    {"index": 2, "name": "OBS Virtual Camera"},
]


def _bench(monkeypatch, devices: list[list[dict[str, object]]], saved=(None, "")) -> tuple[FaceBench, dict]:
    """devices: what each successive list_cameras call returns (last one repeats)."""
    seen = {"lists": 0, "saved": None, "opened": None}

    def listing() -> list[dict[str, object]]:
        i = min(seen["lists"], len(devices) - 1)
        seen["lists"] += 1
        return [dict(cam) for cam in devices[i]]

    def save(index: int, name: str | None = None) -> None:
        seen["saved"] = (index, name)

    monkeypatch.setattr(face_mod, "list_cameras", listing)
    monkeypatch.setattr(face_mod, "load_camera_choice", lambda: saved)
    monkeypatch.setattr(face_mod, "save_camera_index", save)
    monkeypatch.setattr(face_mod.book, "template", lambda rest: rest)
    monkeypatch.setattr(FaceBench, "_load_parts", lambda self: None)
    monkeypatch.setattr(FaceBench, "_load_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_save_ifm", lambda self: None)
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    bench = FaceBench(rest_pts=rest)
    monkeypatch.setattr(bench, "_ensure_source", lambda *a, **k: None)
    monkeypatch.setattr(bench, "_capture_parts", lambda *a, **k: None)

    def start(_cb, index: int) -> None:
        seen["opened"] = index

    monkeypatch.setattr(bench._osf, "start", start)
    return bench, seen


def test_missing_named_pick_does_not_reuse_its_old_index() -> None:
    cams = [{"index": 0, "name": "OBS Virtual Camera"}, {"index": 1, "name": "WarudoCam"}]
    # DroidCam was index 1 and is gone; index 1 is WarudoCam now. Use the preference list.
    assert pick_default(cams, 1, saved_name="DroidCam Video") == 0
    assert pick_default(cams, 1, saved_name="warudocam") == 1
    # No saved name (old camera.json): the index still counts.
    assert pick_default(cams, 1) == 1


def test_empty_camera_list_is_listed_once(monkeypatch) -> None:
    bench, seen = _bench(monkeypatch, [[]])
    bench.status()
    bench.status()
    bench.status()
    assert seen["lists"] == 1


def test_set_camera_rejects_unlisted_index(monkeypatch) -> None:
    bench, seen = _bench(monkeypatch, [CAMS])
    out = bench.set_camera(9)
    assert "No camera at index 9" in str(out.get("error"))
    assert seen["saved"] is None
    out = bench.set_camera(2)
    assert str(out.get("error") or "") == ""
    assert out["camera_index"] == 2
    assert seen["saved"] == (2, "OBS Virtual Camera")


def test_start_finds_the_pick_after_devices_shift(monkeypatch) -> None:
    # A new device lands at index 0 after the list was cached.
    shifted = [{"index": 0, "name": "USB Webcam"}] + [
        {"index": int(cam["index"]) + 1, "name": cam["name"]} for cam in CAMS
    ]
    bench, seen = _bench(monkeypatch, [CAMS, shifted])
    bench.status()
    out = bench.start_live(camera=1, source="camera")
    assert str(out.get("error") or "") == ""
    assert seen["opened"] == 2
    assert out["camera_index"] == 2
    assert [cam["name"] for cam in out["cameras"]][0] == "USB Webcam"


def test_start_with_unlisted_camera_does_not_open(monkeypatch) -> None:
    bench, seen = _bench(monkeypatch, [CAMS])
    out = bench.start_live(camera=7, source="camera")
    assert "No camera at index 7" in str(out.get("error"))
    assert seen["opened"] is None
