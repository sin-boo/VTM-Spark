from __future__ import annotations

import math

import numpy as np

from . import presets as presets_mod
from .presets import MouthBook, PRESET_IDS, apply_mouth, draft_mouth, empty_weights


def _rest() -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    for i in range(28):
        pts[i, 0] = 10.0 + i
        pts[i, 1] = 20.0 + i
    return pts


def test_select_does_not_create_a_saved_copy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _rest()
    book.seed_rest(rest)
    try:
        book.apply("smile", rest)
        raise AssertionError("empty smile should not auto-save")
    except ValueError as exc:
        assert "No saved Smile" in str(exc)
    assert "smile" not in book.shapes
    assert book.shapes["rest"][20, 1] == rest[20, 1]


def test_draft_mouth_moves_lips_off_rest() -> None:
    rest = _rest()
    rest[23, :2] = (40.0, 80.0)
    rest[26, :2] = (80.0, 80.0)
    rest[21, :2] = (60.0, 78.0)
    rest[25, :2] = (60.0, 82.0)
    ah = draft_mouth("A", rest)
    ee = draft_mouth("I", rest)
    oo = draft_mouth("U", rest)
    smile = draft_mouth("smile", rest)
    assert ah[25, 1] > rest[25, 1]
    assert ah[21, 1] < rest[21, 1]
    width_rest = float(rest[26, 0] - rest[23, 0])
    width_ee = float(ee[26, 0] - ee[23, 0])
    width_oo = float(oo[26, 0] - oo[23, 0])
    assert width_ee > width_rest
    assert width_oo < width_rest
    assert smile[23, 1] < rest[23, 1]
    assert smile[26, 1] < rest[26, 1]


def test_retarget_mouth_scales_lips_onto_a_new_rest() -> None:
    from .presets import retarget_mouth

    old = _rest()
    old[23, :2] = (40.0, 80.0)
    old[26, :2] = (80.0, 80.0)
    old[21, :2] = (60.0, 78.0)
    old[25, :2] = (60.0, 82.0)
    smile = old.copy()
    smile[23, 0] -= 4.0
    smile[23, 1] -= 6.0
    smile[26, 0] += 4.0
    smile[26, 1] -= 6.0
    new = old.copy()
    new[:, 0] += 200.0
    new[:, 1] += 100.0
    new[23, 0] -= 20.0
    new[26, 0] += 20.0
    out = retarget_mouth(smile, old, new)
    assert abs(float(out[0, 0]) - float(new[0, 0])) < 1e-3
    assert float(new[26, 0] - new[23, 0]) > float(old[26, 0] - old[23, 0]) + 8.0
    assert float(out[23, 1]) < float(new[23, 1]) - 5.0
    assert abs(float(out[23, 0]) - float(out[26, 0])) > abs(float(new[23, 0]) - float(new[26, 0]))
    assert abs(float(out[21, 1]) - float(new[21, 1])) < 1e-3


def test_open_offset_splits_a_rest_slit() -> None:
    from .presets import apply_open_offset, lip_gap

    rest = _rest()
    rest[23, :2] = (40.0, 80.0)
    rest[26, :2] = (80.0, 80.0)
    rest[20, :2] = (50.0, 79.0)
    rest[21, :2] = (60.0, 78.5)
    rest[22, :2] = (70.0, 79.0)
    rest[24, :2] = (50.0, 80.5)
    rest[25, :2] = (60.0, 81.0)
    rest[27, :2] = (70.0, 80.5)
    parked = rest.copy()
    apply_open_offset(parked, rest, 0.0)
    assert abs(lip_gap(parked) - lip_gap(rest)) < 1e-6
    opened = rest.copy()
    apply_open_offset(opened, rest, 1.0)
    assert lip_gap(opened) > lip_gap(rest) + 8.0
    assert float(opened[21, 1]) < float(rest[21, 1])
    assert float(opened[25, 1]) > float(rest[25, 1])


def test_preview_drafts_without_saving(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _rest()
    rest[23, :2] = (40.0, 80.0)
    rest[26, :2] = (80.0, 80.0)
    book.seed_rest(rest)
    previewed = book.preview("A", rest)
    assert "A" not in book.shapes
    assert previewed[25, 1] > rest[25, 1]
    assert book.preview("rest", rest)[20, 1] == rest[20, 1]


def test_paste_mouth_keeps_source_pixel_size(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _rest()
    rest[23, :2] = (40.0, 80.0)
    rest[26, :2] = (80.0, 80.0)
    rest[21, :2] = (60.0, 76.0)
    rest[25, :2] = (60.0, 88.0)
    book.seed_rest(rest)
    edited = rest.copy()
    edited[21, 1] = 70.0
    edited[25, 1] = 96.0
    mouth = {str(i): [float(edited[i, 0]), float(edited[i, 1]), 1.0] for i in range(20, 28)}
    pasted = book.set_mouth("A", mouth, rest)
    assert "A" in book.shapes
    assert abs(float(pasted[21, 1]) - 70.0) < 1e-4
    assert abs(float(pasted[25, 1]) - 96.0) < 1e-4
    assert abs(float(pasted[23, 0]) - 40.0) < 1e-4
    again = book.apply("A", rest)
    assert abs(float(again[25, 1]) - 96.0) < 1e-4


def test_apply_mouth_writes_absolute_slots() -> None:
    rest = _rest()
    mouth = {"21": [60.0, 70.0, 1.0], "25": [60.0, 90.0, 1.0]}
    out = apply_mouth(rest, mouth)
    assert out[21, 0] == 60.0
    assert out[25, 1] == 90.0
    assert out[0, 0] == rest[0, 0]


def test_apply_writes_only_via_set_mouth(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _rest()
    book.seed_rest(rest)
    mouth = {str(i): [float(rest[i, 0]), float(rest[i, 1] + 8.0), 1.0] for i in range(20, 28)}
    pts = book.set_mouth("smile", mouth, rest)
    assert "smile" in book.shapes
    assert abs(float(pts[20, 1]) - float(rest[20, 1] + 8.0)) < 1e-4
    loaded = book.apply("smile", rest)
    assert abs(float(loaded[20, 1]) - float(rest[20, 1] + 8.0)) < 1e-4
    for name in PRESET_IDS:
        if name not in {"rest", "smile"}:
            assert name not in book.shapes


def test_set_mouth_status_keeps_authored_points(tmp_path, monkeypatch) -> None:
    """Apply must not slam leftover nudges back onto the still overlay."""
    from backend import face as face_mod
    from backend.face import FaceBench
    from backend.offsets import nudge

    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    monkeypatch.setattr(face_mod, "PARTS_PATH", tmp_path / "overlay_parts.json")
    book = MouthBook()
    monkeypatch.setattr(face_mod, "book", book)
    monkeypatch.setattr(FaceBench, "_load_parts", lambda self: None)
    monkeypatch.setattr(FaceBench, "_load_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_save_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_ensure_source", lambda self, **k: None)
    monkeypatch.setattr(FaceBench, "_paint", lambda self, pts: None)
    monkeypatch.setattr(FaceBench, "_publish", lambda self, *a, **k: None)
    monkeypatch.setattr(FaceBench, "_camera_fields", lambda self: {"camera_index": 0, "cameras": []})

    rest = _rest()
    rest[2] = (2.0, 20.0, 1.0)
    rest[21] = (60.0, 80.0, 1.0)
    book.seed_rest(rest)
    bench = FaceBench(rest_pts=rest.copy())
    bench._point_offsets = nudge({}, 21, (60.0, 80.0), (60.0, 96.0))
    bench._point_offsets = nudge(bench._point_offsets, 2, (2.0, 20.0), (22.0, 8.0))

    mouth = {str(i): [float(rest[i, 0]), float(rest[i, 1] + (16.0 if i == 21 else 0.0)), 1.0] for i in range(20, 28)}
    out = bench.set_mouth("A", mouth)
    pts = out["points"]
    assert abs(float(pts[21][1]) - 96.0) < 1e-3
    assert abs(float(pts[2][0]) - 2.0) < 1e-3
    assert abs(float(out["shapes"]["A"][21][1]) - 96.0) < 1e-3
    offsets = {int(row["id"]): row for row in out.get("point_offsets") or []}
    assert 21 not in offsets
    assert 2 in offsets


def test_same_still_bgr_matches_pixels() -> None:
    from backend.face import same_still_bgr

    a = np.zeros((4, 4, 3), dtype=np.uint8)
    a[:] = (8, 16, 24)
    b = a.copy()
    c = a.copy()
    c[0, 0] = (9, 16, 24)
    assert same_still_bgr(a, b) is True
    assert same_still_bgr(a, c) is False
    assert same_still_bgr(a, None) is False


def test_set_source_keeps_book_when_still_matches(tmp_path, monkeypatch) -> None:
    import cv2

    from backend import face as face_mod
    from backend.face import FaceBench

    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    monkeypatch.setattr(face_mod, "INPUT_DIR", tmp_path)
    monkeypatch.setattr(face_mod, "PARTS_PATH", tmp_path / "overlay_parts.json")
    book = MouthBook()
    monkeypatch.setattr(face_mod, "book", book)
    monkeypatch.setattr(FaceBench, "_load_parts", lambda self: None)
    monkeypatch.setattr(FaceBench, "_load_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_save_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_ensure_source", lambda self, **k: None)
    monkeypatch.setattr(FaceBench, "_paint", lambda self, pts: None)
    monkeypatch.setattr(FaceBench, "_publish", lambda self, *a, **k: None)
    monkeypatch.setattr(FaceBench, "_camera_fields", lambda self: {"camera_index": 0, "cameras": []})

    still = np.zeros((8, 8, 3), dtype=np.uint8)
    still[:] = (12, 40, 80)
    ok, buf = cv2.imencode(".png", still)
    assert ok
    rest = _rest()
    book.seed_rest(rest)
    mouth = {str(i): [float(rest[i, 0]), float(rest[i, 1] + 6.0), 1.0] for i in range(20, 28)}
    book.set_mouth("A", mouth, rest)
    bench = FaceBench(rest_pts=rest.copy(), source_bgr=still.copy())
    bench.set_source(bytes(buf), "source.png")
    assert "A" in book.shapes
    assert abs(float(book.shapes["A"][21, 1]) - float(rest[21, 1] + 6.0)) < 1e-3
    assert bench.rest_pts is not None


def test_set_source_clears_book_on_new_still(tmp_path, monkeypatch) -> None:
    import cv2

    from backend import face as face_mod
    from backend.face import FaceBench

    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    monkeypatch.setattr(face_mod, "INPUT_DIR", tmp_path)
    monkeypatch.setattr(face_mod, "PARTS_PATH", tmp_path / "overlay_parts.json")
    book = MouthBook()
    monkeypatch.setattr(face_mod, "book", book)
    monkeypatch.setattr(FaceBench, "_load_parts", lambda self: None)
    monkeypatch.setattr(FaceBench, "_load_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_save_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_ensure_source", lambda self, **k: None)
    monkeypatch.setattr(FaceBench, "_paint", lambda self, pts: None)
    monkeypatch.setattr(FaceBench, "_publish", lambda self, *a, **k: None)
    monkeypatch.setattr(FaceBench, "_camera_fields", lambda self: {"camera_index": 0, "cameras": []})

    old = np.zeros((8, 8, 3), dtype=np.uint8)
    old[:] = (12, 40, 80)
    nxt = np.zeros((8, 8, 3), dtype=np.uint8)
    nxt[:] = (200, 10, 10)
    ok, buf = cv2.imencode(".png", nxt)
    assert ok
    rest = _rest()
    book.seed_rest(rest)
    book.set_mouth("smile", {str(i): [float(rest[i, 0]), float(rest[i, 1]), 1.0] for i in range(20, 28)}, rest)
    bench = FaceBench(rest_pts=rest.copy(), source_bgr=old)
    bench.set_source(bytes(buf), "source.png")
    assert book.shapes == {}
    assert bench.rest_pts is None


def _circle_rest() -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    for i in range(20):
        pts[i, 0] = 40.0 + i
        pts[i, 1] = 30.0 + 0.5 * i
    pts[0, 0], pts[4, 0] = 20.0, 100.0
    pts[2] = [60.0, 110.0, 1.0]
    pts[15] = [60.0, 55.0, 1.0]
    ring = (20, 21, 22, 26, 27, 25, 24, 23)
    for i, slot in enumerate(ring):
        ang = i * (math.pi * 2.0 / 8.0) - math.pi / 2.0
        pts[slot, 0] = 60.0 + 10.0 * math.cos(ang)
        pts[slot, 1] = 80.0 + 10.0 * math.sin(ang)
    return pts


def _mouth_gap(pts: np.ndarray) -> float:
    return float(pts[25, 1] - pts[21, 1])


def test_mix_keeps_open_circle_as_rest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _circle_rest()
    book.seed_rest(rest)
    gap = _mouth_gap(rest)
    assert gap > 8.0
    parked = book.mix(empty_weights())
    assert parked is not None
    assert abs(_mouth_gap(parked) - gap) < 1e-3
    np.testing.assert_allclose(parked[20:28, :2], rest[20:28, :2], atol=1e-4)
    weights = empty_weights()
    weights["A"] = 1.0
    opened = book.mix(weights)
    assert opened is not None
    assert _mouth_gap(opened) > gap + 1.0


def test_mix_ignores_rest_leftover_vowels(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _circle_rest()
    book.seed_rest(rest)
    ah = rest.copy()
    ah[21, 1] -= 20.0
    ah[25, 1] += 24.0
    book.shapes["A"] = ah
    book.shapes["I"] = ah.copy()
    book.shapes["U"] = ah.copy()
    weights = empty_weights()
    weights.update({"A": 0.03, "I": 0.058, "U": 0.086, "sad": 0.039})
    mixed = book.mix(weights)
    assert mixed is not None
    assert abs(_mouth_gap(mixed) - _mouth_gap(rest)) < 0.5
