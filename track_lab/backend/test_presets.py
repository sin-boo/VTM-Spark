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


def _hair_bench(tmp_path, monkeypatch, still):
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
    rest = _rest()
    book.seed_rest(rest)
    return FaceBench(rest_pts=rest.copy(), source_bgr=still.copy())


def test_same_still_keeps_painted_hair_over_green(tmp_path, monkeypatch) -> None:
    """A green prop inside painted hair is not green screen. Re-push must not carve it."""
    import cv2

    still = np.zeros((64, 64, 3), dtype=np.uint8)
    still[:] = (0, 255, 0)
    bench = _hair_bench(tmp_path, monkeypatch, still)
    painted = [{"class": "hair_left", "polygon": [[4.0, 4.0], [40.0, 4.0], [40.0, 40.0], [4.0, 40.0]]}]
    bench.set_hair({"hair": painted})
    ok, buf = cv2.imencode(".png", still)
    assert ok
    bench.set_source(bytes(buf), "source.png")
    assert [seg["polygon"] for seg in bench._hair] == [painted[0]["polygon"]]


def test_save_parts_writes_rest_hair_while_posed(tmp_path, monkeypatch) -> None:
    import json

    from backend import face as face_mod

    still = np.zeros((64, 64, 3), dtype=np.uint8)
    bench = _hair_bench(tmp_path, monkeypatch, still)
    painted = [{"class": "hair_middle", "polygon": [[10.0, 5.0], [30.0, 5.0], [30.0, 25.0], [10.0, 25.0]]}]
    bench.set_hair({"hair": painted})
    bench._hair = [{"class": "hair_middle", "polygon": [[22.0, 9.0], [42.0, 9.0], [42.0, 29.0], [22.0, 29.0]]}]
    bench._save_parts()
    saved = json.loads(face_mod.PARTS_PATH.read_text(encoding="utf-8"))
    assert saved["hair"][0]["polygon"] == painted[0]["polygon"]


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


def test_pair_ids_cover_every_mouth_combination() -> None:
    from itertools import combinations

    from .presets import EYE_IDS, pair_ends, pair_id

    mouths = [name for name in PRESET_IDS if name not in EYE_IDS]
    pairs = [pair_id(a, b) for a, b in combinations(mouths, 2)]
    # Seven shapes (O folded into U) -> 7 choose 2.
    assert len(pairs) == 21
    assert len(set(pairs)) == 21
    assert pair_id("eye_closed", "eye_open") == "eye_open+eye_closed"
    assert pair_id("smile", "rest") == "rest+smile"
    assert pair_ends("smile+rest") is None
    assert pair_ends("rest+smile") == ("rest", "smile")


def test_saved_midpoint_bends_the_rest_blend(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _rest()
    book.seed_rest(rest)
    smile = {
        str(i): [float(rest[i, 0]), float(rest[i, 1] + (10.0 if i == 21 else 0.0)), 1.0]
        for i in range(20, 28)
    }
    book.set_mouth("smile", smile, rest)
    mid = {
        str(i): [float(rest[i, 0]), float(rest[i, 1] + (8.0 if i == 21 else 0.0)), 1.0]
        for i in range(20, 28)
    }
    book.set_mouth("rest+smile", mid, rest)
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "plain.json")
    straight = MouthBook()
    straight.seed_rest(rest)
    straight.set_mouth("smile", smile, rest)
    bent = book.mix({"smile": 0.55})
    line = straight.mix({"smile": 0.55})
    assert bent is not None and line is not None
    assert abs(float(bent[21, 1]) - (float(rest[21, 1]) + 8.0)) < 0.05
    assert abs(float(line[21, 1]) - (float(rest[21, 1]) + 5.0)) < 0.2
    assert "rest+smile" in book.payload(rest)["mids"]


def test_several_stops_bend_the_blend_and_can_slide(tmp_path, monkeypatch) -> None:
    from .presets import key_id, key_t, pair_ends

    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _rest()
    book.seed_rest(rest)
    smile = {
        str(i): [float(rest[i, 0]), float(rest[i, 1] + (10.0 if i == 21 else 0.0)), 1.0]
        for i in range(20, 28)
    }
    book.set_mouth("smile", smile, rest)
    early = {
        str(i): [float(rest[i, 0]), float(rest[i, 1] + (8.0 if i == 21 else 0.0)), 1.0]
        for i in range(20, 28)
    }
    late = {
        str(i): [float(rest[i, 0]), float(rest[i, 1] + (2.0 if i == 21 else 0.0)), 1.0]
        for i in range(20, 28)
    }
    book.set_mouth(key_id("rest", "smile", 0.25), early, rest)
    book.set_mouth(key_id("smile", "rest", 0.75), late, rest)
    assert pair_ends("rest+smile@250") == ("rest", "smile")
    assert pair_ends("rest+smile@0") is None
    assert key_t("rest+smile") == 0.5
    assert key_t("rest+smile@250") == 0.25
    assert key_id("rest", "smile", 0.5) == "rest+smile"
    # weight 0.325 -> amount 0.25, which lands on the early stop (+8)
    at_early = book.mix({"smile": 0.325})
    # weight 0.55 -> amount 0.5, halfway from +8 to +2
    at_mid = book.mix({"smile": 0.55})
    assert at_early is not None and at_mid is not None
    assert abs(float(at_early[21, 1]) - (float(rest[21, 1]) + 8.0)) < 0.05
    assert abs(float(at_mid[21, 1]) - (float(rest[21, 1]) + 5.0)) < 0.05
    moved = book.move_key("rest+smile@250", 0.4)
    assert moved == "rest+smile@400"
    assert "rest+smile@250" not in book.shapes
    assert "rest+smile@750" in book.payload(rest)["mids"]
    again = MouthBook()
    assert "rest+smile@400" in again.shapes
    assert "rest+smile@750" in again.shapes
    book.drop_key("rest+smile@400")
    assert "rest+smile@400" not in book.shapes
    assert "rest+smile@750" in book.shapes


def _eye_rest() -> np.ndarray:
    pts = _rest()
    pts[11, :2] = [100.0, 100.0]
    pts[12, :2] = [120.0, 80.0]
    pts[13, :2] = [140.0, 100.0]
    pts[17, :2] = [200.0, 100.0]
    pts[18, :2] = [220.0, 80.0]
    pts[19, :2] = [240.0, 100.0]
    return pts


def test_eye_shapes_are_chips_that_own_only_the_eyes(tmp_path, monkeypatch) -> None:
    from .presets import EYE_SLOTS, pair_ends, shape_slots

    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _eye_rest()
    book.seed_rest(rest)
    ids = [row["id"] for row in book.payload(rest)["presets"]]
    assert ids[-2:] == ["eye_open", "eye_closed"]
    assert shape_slots("eye_closed") == EYE_SLOTS
    assert shape_slots("eye_open+eye_closed@300") == EYE_SLOTS
    assert pair_ends("eye_open+eye_closed") == ("eye_open", "eye_closed")
    # Eyes pair with eyes only.
    assert pair_ends("rest+eye_closed") is None
    assert pair_ends("A+eye_open") is None
    saved = book.set_mouth("eye_closed", {"12": [120.0, 99.0], "20": [0.0, 0.0]}, rest)
    assert saved[12, 1] == 99.0
    assert np.array_equal(saved[20], rest[20])


def test_blink_moves_each_eye_along_its_shapes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _eye_rest()
    book.seed_rest(rest)
    # Unsaved: the lid lies on its corner line.
    drafted = book.blink(rest, rest, {"l": 1.0, "r": 0.0})
    assert drafted is not None
    assert abs(float(drafted[12, 1]) - 100.0) < 1e-4
    assert np.array_equal(drafted[[17, 18, 19]], rest[[17, 18, 19]])
    closed = rest.copy()
    closed[[12, 18], 1] = 104.0
    closed[[11, 13, 17, 19], 1] = 102.0
    book.shapes["eye_closed"] = closed
    # "l" drives 11-13, "r" drives 17-19; half a blink is halfway.
    out = book.blink(rest, rest, {"l": 1.0, "r": 0.5})
    assert out is not None
    assert np.allclose(out[[11, 12, 13], 1], [102.0, 104.0, 102.0])
    assert abs(float(out[18, 1]) - 92.0) < 1e-4
    # A saved in-between is honoured.
    mid = rest.copy()
    mid[[12, 18], 1] = 100.0
    book.shapes["eye_open+eye_closed@250"] = mid
    early = book.blink(rest, rest, {"l": 0.25, "r": 0.0})
    assert early is not None and abs(float(early[12, 1]) - 100.0) < 1e-4
    # The move rides on top of whatever else drove the mesh.
    moved = rest.copy()
    moved[12, 0] += 3.0
    shifted = book.blink(moved, rest, {"l": 1.0, "r": 0.0})
    assert shifted is not None and abs(float(shifted[12, 0]) - 123.0) < 1e-4


def test_eye_open_is_where_an_open_eye_sits(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _eye_rest()
    book.seed_rest(rest)
    wide = rest.copy()
    wide[[12, 18], 1] = 76.0
    book.shapes["eye_open"] = wide
    out = book.blink(rest, rest, {"l": 0.0, "r": 0.0})
    assert out is not None
    assert np.allclose(out[[12, 18], 1], 76.0)


def test_rebase_carries_eye_shapes_with_the_eyes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    rest = _eye_rest()
    book.seed_rest(rest)
    closed = rest.copy()
    closed[12, 1] = 100.0
    book.set_mouth("eye_closed", {"12": [120.0, 100.0]}, rest)
    moved = rest.copy()
    moved[:, 0] += 10.0
    book.rebase(moved)
    assert abs(float(book.shapes["eye_closed"][12, 0]) - 130.0) < 1e-3
    assert abs(float(book.shapes["eye_closed"][12, 1]) - 100.0) < 1e-3


def _char_rest(dx: float, dy: float, mouth_w: float, eye_w: float) -> np.ndarray:
    """A character rest: jaw / brows / nose, both eyes, and a lip ring."""
    from .presets import EYE_L, EYE_R

    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 0.9137
    for i in range(20):
        pts[i, 0] = 400.0 + dx + 17.3 * i
        pts[i, 1] = 300.0 + dy + 2.9 * i
    for (a, lid, b), x0 in ((EYE_L, 480.0), (EYE_R, 590.0)):
        pts[a, :2] = [x0 + dx, 350.0 + dy]
        pts[lid, :2] = [x0 + dx + 0.5 * eye_w, 336.0 + dy]
        pts[b, :2] = [x0 + dx + eye_w, 351.0 + dy]
    ring = {
        23: (-1.0, 0.0),
        20: (-0.5, -0.3),
        21: (0.0, -0.35),
        22: (0.5, -0.3),
        26: (1.0, 0.02),
        27: (0.5, 0.3),
        25: (0.0, 0.4),
        24: (-0.5, 0.3),
    }
    for slot, (u, v) in ring.items():
        pts[slot, 0] = 560.0 + dx + u * 0.5 * mouth_w
        pts[slot, 1] = 450.0 + dy + v * 0.5 * mouth_w
    return pts


def _lips(rest: np.ndarray, moves: dict[int, tuple[float, float]]) -> dict[str, list[float]]:
    return {
        str(s): [
            float(rest[s, 0]) + moves.get(s, (0.0, 0.0))[0],
            float(rest[s, 1]) + moves.get(s, (0.0, 0.0))[1],
            1.0,
        ]
        for s in range(20, 28)
    }


def _author_plan(book: MouthBook, rest: np.ndarray) -> None:
    book.seed_rest(rest)
    book.set_mouth("smile", _lips(rest, {23: (-6.1, -5.3), 26: (6.1, -5.3)}), rest)
    book.set_mouth("A", _lips(rest, {21: (0.0, -7.7), 25: (0.0, 11.3), 24: (0.4, 9.1)}), rest)
    book.set_mouth("rest+smile@300", _lips(rest, {23: (-1.9, -1.7), 26: (1.9, -1.7)}), rest)
    closed = {
        "12": [float(rest[12, 0]), float(rest[12, 1]) + 13.9],
        "18": [float(rest[18, 0]), float(rest[18, 1]) + 14.2],
    }
    book.set_mouth("eye_closed", closed, rest)


def test_rebase_back_to_the_authored_rest_is_exact(tmp_path, monkeypatch) -> None:
    from .presets import json_to_pts, pts_to_json

    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    home = _char_rest(0.123, 0.456, 96.7, 42.1)
    narrow = _char_rest(31.789, -17.25, 41.3, 30.7)
    wide = _char_rest(-12.5, 8.75, 130.2, 51.9)
    book = MouthBook()
    _author_plan(book, home)
    authored = {name: pts.copy() for name, pts in book.shapes.items()}
    for _ in range(10):
        for other in (narrow, wide):
            book.rebase(other)
            book.rebase(home)
    assert set(book.shapes) == set(authored)
    for name, pts in authored.items():
        assert np.array_equal(book.shapes[name], pts), name
    # Across restarts (0.001 px save rounding) the shapes never walk off either.
    stored = {name: json_to_pts(pts_to_json(pts)) for name, pts in authored.items()}
    for _ in range(10):
        for other in (narrow, wide):
            book.rebase(other)
            book = MouthBook()
            book.rebase(home)
            book = MouthBook()
    for name, pts in stored.items():
        assert np.array_equal(book.shapes[name], pts), name


def test_rebase_skips_an_unchanged_rest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    home = _char_rest(0.123, 0.456, 96.7, 42.1)
    book = MouthBook()
    _author_plan(book, home)
    book.apply("rest", home)
    before = {name: pts.copy() for name, pts in book.shapes.items()}
    writes: list[int] = []
    monkeypatch.setattr(MouthBook, "save", lambda self: writes.append(1))
    # The same face at another precision (a .vtm rest, a re-push).
    again = home.copy()
    again[:, :2] = np.round(home[:, :2].astype(np.float64) + 0.003, 2)
    book.rebase(again)
    assert not writes
    assert book.active == "rest"
    for name, pts in before.items():
        assert np.array_equal(book.shapes[name], pts), name


def test_preset_file_without_authored_copy_loads_as_before(tmp_path, monkeypatch) -> None:
    import json

    from .presets import json_to_pts, pts_to_json, retarget_mouth

    path = tmp_path / "mouth_presets.json"
    monkeypatch.setattr(presets_mod, "PRESET_PATH", path)
    home = _char_rest(0.123, 0.456, 96.7, 42.1)
    other = _char_rest(31.789, -17.25, 41.3, 30.7)
    smile = home.copy()
    smile[23, :2] += (-6.1, -5.3)
    smile[26, :2] += (6.1, -5.3)
    old_file = {"active": "smile", "shapes": {"rest": pts_to_json(home), "smile": pts_to_json(smile)}}
    path.write_text(json.dumps(old_file), encoding="utf-8")
    book = MouthBook()
    loaded = {"rest": json_to_pts(pts_to_json(home)), "smile": json_to_pts(pts_to_json(smile))}
    assert book.active == "smile"
    for name, pts in loaded.items():
        assert np.array_equal(book.shapes[name], pts)
    book.rebase(other)
    assert np.array_equal(book.shapes["smile"], retarget_mouth(loaded["smile"], loaded["rest"], other))
    # The desk reads absolute points, rest included, in the same rows.
    shapes = book.payload(other)["shapes"]
    assert shapes["rest"] == pts_to_json(other)
    assert shapes["smile"] == pts_to_json(book.shapes["smile"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["active"] == "rest"
    assert saved["shapes"] == shapes
    again = MouthBook()
    again.rebase(home)
    for name, pts in loaded.items():
        assert np.array_equal(again.shapes[name], pts)
