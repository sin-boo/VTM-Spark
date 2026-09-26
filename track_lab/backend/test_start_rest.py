"""Start live must fit a rest mesh before iFacialMocap / camera go on."""

from __future__ import annotations

import numpy as np

from backend.face import FaceBench
from backend import face as face_mod


def test_start_live_tracks_when_rest_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(face_mod, "list_cameras", lambda: [])
    monkeypatch.setattr(face_mod, "pick_default", lambda cameras, saved, **_kwargs: 0)
    monkeypatch.setattr(face_mod.book, "template", lambda rest: rest)
    monkeypatch.setattr(FaceBench, "_load_parts", lambda self: None)
    monkeypatch.setattr(FaceBench, "_load_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_save_ifm", lambda self: None)

    bench = FaceBench(rest_pts=None)
    rest = np.zeros((28, 3), dtype=np.float32)
    calls = {"n": 0}

    def fake_track() -> dict[str, object]:
        calls["n"] += 1
        bench.rest_pts = rest
        bench.last_error = ""
        return {"error": ""}

    bench.track = fake_track  # type: ignore[method-assign]
    monkeypatch.setattr(bench, "_ensure_source", lambda *a, **k: None)
    monkeypatch.setattr(bench, "_capture_parts", lambda *a, **k: None)
    monkeypatch.setattr(bench._ifm, "start", lambda *a, **k: None)
    monkeypatch.setattr(
        bench._osf,
        "start",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("osf should not start")),
    )

    out = bench.start_live(source="ifm")
    assert calls["n"] == 1
    assert bench.last_error == ""
    assert str(out.get("error") or "") == ""
    assert bench.last_tracker == "ifm"


def _mesh(x: float, y: float) -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    for i in range(28):
        pts[i, 0] = x + (i % 7) * 4.0
        pts[i, 1] = y + (i // 7) * 4.0
    return pts


def _bench_for_track(monkeypatch, tmp_path, rest: np.ndarray):
    from backend.presets import MouthBook
    from backend import presets as presets_mod

    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    monkeypatch.setattr(face_mod, "PARTS_PATH", tmp_path / "overlay_parts.json")
    monkeypatch.setattr(face_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(face_mod, "list_cameras", lambda: [])
    monkeypatch.setattr(face_mod, "pick_default", lambda cameras, saved, **_kwargs: 0)
    book = MouthBook()
    monkeypatch.setattr(face_mod, "book", book)
    monkeypatch.setattr(FaceBench, "_load_parts", lambda self: None)
    monkeypatch.setattr(FaceBench, "_load_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_save_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_publish", lambda self, *a, **k: None)
    monkeypatch.setattr(FaceBench, "_camera_fields", lambda self: {"camera_index": 0, "cameras": []})
    monkeypatch.setattr(FaceBench, "_apply_still_iris", lambda self, *a, **k: None)
    monkeypatch.setattr(FaceBench, "_capture_parts", lambda self, *a, **k: None)
    monkeypatch.setattr(FaceBench, "stop_live", lambda self: None)
    monkeypatch.setattr(FaceBench, "_ensure_source", lambda self, **k: None)
    monkeypatch.setattr(face_mod, "draw_label28", lambda frame, pts: frame.copy())
    still = np.zeros((1254, 1254, 3), dtype=np.uint8)
    book.seed_rest(rest)
    bench = FaceBench(rest_pts=rest.copy(), source_bgr=still)
    return bench, book


def test_track_refits_when_old_rest_sits_on_the_hair(tmp_path, monkeypatch) -> None:
    hair = _mesh(300.0, 330.0)
    face = _mesh(620.0, 400.0)
    bench, book = _bench_for_track(monkeypatch, tmp_path, hair)
    book.set_mouth(
        "smile",
        {str(i): [float(hair[i, 0]), float(hair[i, 1]), 1.0] for i in range(20, 28)},
        hair,
    )
    box = np.array([430.0, 280.0, 820.0, 720.0, 0.8], dtype=np.float32)
    monkeypatch.setattr(face_mod, "fit_mesh", lambda frame: (face.copy(), box))
    out = bench.track()
    assert str(out.get("error") or "") == ""
    assert abs(float(bench.rest_pts[0, 0]) - 620.0) < 1e-3
    assert "smile" in book.shapes
    assert abs(float(book.shapes["smile"][20, 0]) - float(face[20, 0])) < 1e-2
    assert abs(float(book.shapes["smile"][20, 1]) - float(face[20, 1])) < 1e-2


def _jaw(x0: float, x1: float) -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    pts[0, 0], pts[4, 0] = x0, x1
    pts[:, 1] = 400.0
    pts[2, 1] = 520.0
    pts[15, 1] = 390.0
    return pts


def test_start_live_refits_when_rest_is_too_small(tmp_path, monkeypatch) -> None:
    tiny = _jaw(252.0, 523.0)
    face = _jaw(430.0, 820.0)
    bench, book = _bench_for_track(monkeypatch, tmp_path, tiny)
    box = np.array([430.0, 220.0, 820.0, 720.0, 0.8], dtype=np.float32)
    monkeypatch.setattr(face_mod, "fit_mesh", lambda frame: (face.copy(), box))
    monkeypatch.setattr(bench._ifm, "start", lambda *a, **k: None)
    monkeypatch.setattr(
        bench._osf,
        "start",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("osf should not start")),
    )
    out = bench.start_live(source="ifm")
    assert str(out.get("error") or "") == ""
    assert abs(float(bench.rest_pts[0, 0]) - 430.0) < 1e-3
    assert abs(float(bench.rest_pts[4, 0]) - 820.0) < 1e-3
    assert bench.last_tracker == "ifm"


def test_track_keeps_visemes_when_rest_stays_put(tmp_path, monkeypatch) -> None:
    rest = _mesh(620.0, 400.0)
    jitter = _mesh(624.0, 403.0)
    bench, book = _bench_for_track(monkeypatch, tmp_path, rest)
    book.set_mouth(
        "A",
        {str(i): [float(rest[i, 0]), float(rest[i, 1] + 6.0), 1.0] for i in range(20, 28)},
        rest,
    )
    box = np.array([430.0, 280.0, 820.0, 720.0, 0.8], dtype=np.float32)
    monkeypatch.setattr(face_mod, "fit_mesh", lambda frame: (jitter.copy(), box))
    out = bench.track()
    assert str(out.get("error") or "") == ""
    assert "A" in book.shapes
    assert abs(float(bench.rest_pts[0, 0]) - 624.0) < 1e-3


def _mirror_bench(monkeypatch) -> FaceBench:
    monkeypatch.setattr(face_mod, "list_cameras", lambda: [])
    monkeypatch.setattr(face_mod, "pick_default", lambda cameras, saved, **_kwargs: 0)
    monkeypatch.setattr(face_mod.book, "template", lambda rest: rest)
    monkeypatch.setattr(FaceBench, "_load_parts", lambda self: None)
    monkeypatch.setattr(FaceBench, "_load_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_save_ifm", lambda self: None)
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    bench = FaceBench(rest_pts=rest)
    monkeypatch.setattr(bench, "_ensure_source", lambda *a, **k: None)
    monkeypatch.setattr(bench, "_capture_parts", lambda *a, **k: None)
    monkeypatch.setattr(bench._ifm, "start", lambda *a, **k: None)
    monkeypatch.setattr(bench._osf, "start", lambda *a, **k: None)
    return bench


def test_mirror_off_is_selfie_everywhere(monkeypatch) -> None:
    """Default (Mirror OFF) is the reflection: one rule reaches rig, expr,
    and the camera preview. Mirror ON turns it off in all of them."""
    bench = _mirror_bench(monkeypatch)
    assert bench.selfie is True
    assert bench._rig.selfie is True
    assert bench._expr.selfie is True
    assert bench._osf.preview_flip is True
    out = bench.set_mirror(True)
    assert bool(out.get("mirror")) is True
    assert bench.selfie is False
    assert bench._rig.selfie is False
    assert bench._expr.selfie is False
    assert bench._osf.preview_flip is False
    started = bench.start_live(source="ifm", mirror=False)
    assert bool(started.get("mirror")) is False
    assert bench._rig.selfie is True


def test_set_mirror_does_not_reset_rig(monkeypatch) -> None:
    bench = _mirror_bench(monkeypatch)
    hits: list[str] = []
    bench._rig.reset = lambda: hits.append("rig")  # type: ignore[method-assign]
    bench._expr.reset = lambda: hits.append("expr")  # type: ignore[method-assign]
    bench._osf._running = True
    bench.set_mirror(True)
    bench.set_mirror(False)
    assert hits == []


def test_set_feel_has_no_side_flags(monkeypatch) -> None:
    from backend.feel import DEFAULTS, feel

    bench = _mirror_bench(monkeypatch)
    prev = feel.payload()
    try:
        out = bench.set_feel({"invert_look": 1, "invert_yaw": 1, "invert_pitch": 1})
        for key in ("invert", "invert_look", "invert_yaw", "invert_pitch"):
            assert key not in DEFAULTS
            assert key not in out["feel"]
        assert bench.selfie is True
    finally:
        feel.update(prev)


def test_mirror_persists_in_ifm_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(face_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(face_mod, "IFM_PATH", tmp_path / "ifm.json")
    monkeypatch.setattr(face_mod, "list_cameras", lambda: [])
    monkeypatch.setattr(face_mod, "pick_default", lambda cameras, saved, **_kwargs: 0)
    monkeypatch.setattr(FaceBench, "_load_parts", lambda self: None)
    first = FaceBench()
    first.set_mirror(True)
    second = FaceBench()
    assert second.selfie is False
    assert bool(second.status(publish=False).get("mirror")) is True

