"""Limiters fitted to a character still: framing and drawn pose."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from backend.travel_box import DEFAULT_TRAVEL_BOX, face_height
from backend.travel_fit import drawn_pose, fit_travel_box, green_screen_mask

# Rest overlay (character pixels) of a green-screen bust, 768 x 768, whose
# limiters were tuned by hand. Slot 30 is not a point.
_BUST_XY = [
    [239.0, 378.5], [284.4, 466.6], [388.5, 525.4], [490.0, 458.6],
    [535.4, 370.5], [271.1, 298.5], [303.1, 295.8], [337.8, 306.5],
    [420.6, 303.8], [455.3, 293.1], [492.6, 293.1], [255.1, 370.5],
    [292.4, 351.8], [335.1, 370.5], [355.0, 414.0], [380.5, 421.3],
    [406.1, 414.0], [436.6, 365.2], [479.3, 346.5], [514.0, 362.5],
    [368.5, 460.0], [383.2, 464.0], [400.5, 460.0], [353.8, 456.0],
    [368.5, 460.0], [383.2, 464.0], [417.9, 456.0], [400.5, 460.0],
    [302.1, 371.9], [464.3, 367.9], [0.0, 0.0], [385.6, 609.7],
    [105.2, 660.0], [89.4, 748.0], [660.8, 660.0], [676.6, 748.0],
    [387.2, 699.6],
]
# What the hand tuning settled on for that bust.
_TUNED_ROOM = {
    "left": 0.68, "right": 0.65, "up": 0.29, "down": 0.57,
    "body_left": 0.54, "body_right": 0.48, "body_up": 0.1, "body_down": 0.12,
}
SIZE = 768


def _bust() -> np.ndarray:
    k = np.zeros((37, 4), dtype=np.float32)
    k[:, :2] = np.asarray(_BUST_XY, dtype=np.float32)
    k[:, 2] = 1.0
    k[:, 3] = 1.0
    k[30] = 0.0
    return k


def _green_still(top: int = 5, bottom_cut: bool = True) -> np.ndarray:
    """BGR green screen with the character's shape: hair from ``top`` down,
    torso to the bottom edge (or stopping short of it)."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img[:, :] = (0, 255, 0)
    img[top:560, 60:710] = (120, 170, 220)
    img[560 : SIZE if bottom_cut else 740, 60:710] = (60, 60, 60)
    return img


def _shifted(k: np.ndarray, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> np.ndarray:
    out = k.copy()
    out[:, 0] = (k[:, 0] - SIZE / 2) * scale + SIZE / 2 + dx
    out[:, 1] = (k[:, 1] - SIZE / 2) * scale + SIZE / 2 + dy
    out[30] = 0.0
    return out


def test_fit_matches_the_hand_tuned_bust() -> None:
    box = fit_travel_box(_bust(), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX, image_bgr=_green_still())
    for key, want in _TUNED_ROOM.items():
        assert box[key] == pytest.approx(want, abs=0.06), key


def test_room_follows_where_the_character_sits() -> None:
    base = fit_travel_box(_bust(), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    left = fit_travel_box(_shifted(_bust(), dx=-120.0), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    assert left["left"] < base["left"] and left["right"] > base["right"]
    low = fit_travel_box(_shifted(_bust(), dy=100.0), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    assert low["up"] > base["up"] and low["down"] < base["down"]


def test_small_character_gets_more_room_and_a_close_up_less() -> None:
    base = fit_travel_box(_bust(), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    small = fit_travel_box(_shifted(_bust(), scale=0.5), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    close = fit_travel_box(_shifted(_bust(), scale=1.8), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    for key in ("left", "right", "up", "down"):
        assert small[key] > base[key] or small[key] == 1.2
        # Never frozen, even when the face nearly fills the picture.
        assert close[key] == pytest.approx(0.1)


def test_torso_never_walks_further_than_the_head() -> None:
    """A close-up's shoulders are off the picture: only the neck is left to
    measure, which read as room to walk the torso the whole way across."""
    k = _shifted(_bust(), scale=1.8)
    for slot in (32, 33, 34, 35, 36):
        k[slot, 3] = 0.0
    box = fit_travel_box(k, SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    for side in ("left", "right", "up", "down"):
        assert box[f"body_{side}"] <= box[side]


def test_torso_cut_by_the_bottom_edge_barely_rises() -> None:
    cut = fit_travel_box(_bust(), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX, image_bgr=_green_still())
    assert cut["body_up"] == pytest.approx(0.1)
    # The whole character in frame: the torso may rise with the head.
    small = _shifted(_bust(), scale=0.5)
    whole = fit_travel_box(small, SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    assert whole["body_up"] == pytest.approx(whole["up"])


def test_head_top_comes_from_the_green_screen() -> None:
    """Tall hair or a hat keeps the head from rising out of the picture."""
    tall = fit_travel_box(_bust(), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX, image_bgr=_green_still(top=5))
    room = fit_travel_box(_bust(), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX, image_bgr=_green_still(top=120))
    fh = face_height(_bust())
    assert tall["up"] == pytest.approx((5 + 0.3 * fh) / fh, abs=0.01)
    assert room["up"] == pytest.approx((120 + 0.3 * fh) / fh, abs=0.01)


def test_green_screen_is_read_from_the_border() -> None:
    assert green_screen_mask(_green_still()) is not None
    photo = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
    assert green_screen_mask(photo) is None
    assert green_screen_mask(None) is None


def _rotate(k: np.ndarray, deg: float) -> np.ndarray:
    out = k.copy()
    c = k[:37, :2].mean(axis=0)
    a = math.radians(deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
    out[:, :2] = (k[:, :2] - c) @ rot.T + c
    out[30] = 0.0
    return out


def test_frontal_still_gets_even_turn_and_tilt() -> None:
    """Hand-tuned sides (turn 21 / 22, tilt 22 / 12) even out on a still
    drawn straight on."""
    pose = drawn_pose(_bust())
    assert abs(pose["yaw"]) < 3.0 and abs(pose["roll"]) < 3.0
    box = fit_travel_box(_bust(), SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    assert box["turn_left"] == box["turn_right"] == pytest.approx(21.5)
    assert box["tilt_left"] == box["tilt_right"] == pytest.approx(17.0)


def test_tilt_is_centred_on_a_tilted_still() -> None:
    """Crown drawn 10 deg toward screen-right: 10 less to tilt that way."""
    k = _rotate(_bust(), 10.0)
    assert drawn_pose(k)["roll"] == pytest.approx(10.0 + drawn_pose(_bust())["roll"], abs=0.5)
    box = fit_travel_box(k, SIZE, SIZE, base={**DEFAULT_TRAVEL_BOX, "tilt_left": 17, "tilt_right": 17})
    assert box["tilt_right"] < box["tilt_left"]
    assert box["tilt_left"] - box["tilt_right"] == pytest.approx(2 * drawn_pose(k)["roll"], abs=0.3)
    assert abs(drawn_pose(k)["yaw"]) < 3.0


def test_turn_is_centred_on_a_turned_still() -> None:
    """Chin and mouth drawn toward screen-right of the cheeks: the still
    faces right, so it has less room to turn right."""
    k = _bust()
    span = float(k[4, 0] - k[0, 0])
    lean = 0.5 * span * math.sin(math.radians(12.0))
    for slot in (2, *range(20, 28)):
        k[slot, 0] += lean
    pose = drawn_pose(k)
    assert pose["yaw"] > 8.0
    box = fit_travel_box(k, SIZE, SIZE, base={**DEFAULT_TRAVEL_BOX, "turn_left": 22, "turn_right": 22})
    assert box["turn_right"] == pytest.approx(22 - pose["yaw"], abs=0.1)
    assert box["turn_left"] == pytest.approx(22 + pose["yaw"], abs=0.1)


def test_fitting_again_does_not_ratchet_a_side() -> None:
    """Each Fit press re-averaged the box the last one clamped: tilt 22 / 12
    on a still tilted 14 deg went 31 / 5, 32 / 5, 32.5 / 5, 33 / 5 ..."""
    k = _rotate(_bust(), 14.0)
    first = fit_travel_box(k, SIZE, SIZE, base=DEFAULT_TRAVEL_BOX)
    assert first["tilt_right"] == pytest.approx(5.0)
    box = first
    for _ in range(5):
        box = fit_travel_box(k, SIZE, SIZE, base=box)
        for key in ("turn_left", "turn_right", "tilt_left", "tilt_right"):
            assert box[key] == first[key], key
    # A turned still whose far side is held at the minimum stays put too.
    turned = _bust()
    span = float(turned[4, 0] - turned[0, 0])
    for slot in (2, *range(20, 28)):
        turned[slot, 0] += 0.5 * span * math.sin(math.radians(20.0))
    base = {**DEFAULT_TRAVEL_BOX, "turn_left": 20.0, "turn_right": 20.0}
    once = fit_travel_box(turned, SIZE, SIZE, base=base)
    assert once["turn_right"] == pytest.approx(5.0)
    assert fit_travel_box(turned, SIZE, SIZE, base=once) == once


def test_a_hand_set_box_is_still_recentred() -> None:
    """A side is read back only when the drawn pose explains it sitting at the
    minimum; a box set by hand otherwise is centred from its mean, as before."""
    k = _rotate(_bust(), 14.0)
    roll = drawn_pose(k)["roll"]
    box = fit_travel_box(k, SIZE, SIZE, base={**DEFAULT_TRAVEL_BOX, "tilt_left": 35.0, "tilt_right": 5.0})
    assert box["tilt_left"] == pytest.approx(20.0 + roll, abs=0.1)
    assert box["tilt_right"] == pytest.approx(20.0 - roll, abs=0.1)


def test_look_eyes_size_and_on_come_from_base() -> None:
    base = {**DEFAULT_TRAVEL_BOX, "pitch_up": 30.0, "pitch_down": 9.0, "eye": 0.3, "size": 0.2, "enabled": False}
    box = fit_travel_box(_bust(), SIZE, SIZE, base=base)
    for key in ("pitch_up", "pitch_down", "eye", "size", "enabled"):
        assert box[key] == base[key]


def test_bench_fits_the_loaded_still(tmp_path: Path, monkeypatch) -> None:
    from backend.face import FaceBench
    from backend.feel import feel
    from backend.travel_box import default_travel_box, travel

    monkeypatch.setattr("backend.travel_box.TRAVEL_PATH", tmp_path / "travel_box.json")
    saved_box = travel.payload()
    saved_feel = feel.payload()
    k = _bust()
    bench = FaceBench(rest_pts=np.c_[k[:28, :2], np.ones(28, dtype=np.float32)])
    skeleton = [{"id": i, "x": float(k[i, 0]), "y": float(k[i, 1]), "score": 1.0} for i in range(31, 37)]
    bench._skeleton_rest = skeleton
    monkeypatch.setattr(bench, "_ensure_source", lambda **_kw: None)
    try:
        travel.values = {**default_travel_box(), "pitch_down": 9.0}
        bench.source_bgr = None
        missing = bench.fit_travel({})
        assert missing["error"]
        bench.source_bgr = _green_still()
        # Keeps look / eyes / size unless asked to start from the built-in box.
        kept = bench.fit_travel({})
        assert kept["error"] == ""
        assert kept["travel_box"]["pitch_down"] == 9.0
        assert kept["travel_box"]["body_up"] == pytest.approx(0.1)
        fresh = bench.fit_travel({"from": "default"})
        assert fresh["travel_box"]["pitch_down"] == DEFAULT_TRAVEL_BOX["pitch_down"]
        assert feel.max_yaw()[0] == pytest.approx(fresh["travel_box"]["turn_left"] / 80.0)
    finally:
        travel.values = saved_box
        feel.update(saved_feel)
