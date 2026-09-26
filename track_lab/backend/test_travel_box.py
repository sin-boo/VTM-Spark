"""Focused tests for Track Lab travel / limiter box."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.travel_box import (
    DEFAULT_TRAVEL_BOX,
    TRAVEL_VERSION,
    TravelStore,
    apply_limits,
    boxes_equal,
    default_travel_box,
    face_height,
    head_mesh_rect_px,
    lab_feel_caps,
    limiter_rects_px,
    merge_travel_box,
    normalize_travel_box,
    pack_overlay,
)
from harness.dispatch import handle
from harness.pack import pack_frame, pack_status
from harness.protocol import COMMANDS, parse_command


def _rest_face() -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    # Simple upright face: jaw 0..4, brows, eyes, nose, mouth.
    pts[0] = [80, 140, 1]
    pts[1] = [100, 160, 1]
    pts[2] = [120, 180, 1]  # chin
    pts[3] = [140, 160, 1]
    pts[4] = [160, 140, 1]
    for i, x in enumerate((90, 100, 110, 130, 140, 150)):
        pts[5 + i] = [x, 90, 1]
    pts[11] = [95, 110, 1]
    pts[12] = [105, 105, 1]
    pts[13] = [115, 110, 1]
    pts[14] = [110, 130, 1]
    pts[15] = [120, 135, 1]
    pts[16] = [130, 130, 1]
    pts[17] = [125, 110, 1]
    pts[18] = [135, 105, 1]
    pts[19] = [145, 110, 1]
    for i, x in enumerate((100, 120, 140, 95, 110, 145, 130)):
        pts[20 + i] = [x, 155 + (i % 3) * 4, 1]
    return pts


def _rest_skeleton() -> list[dict[str, object]]:
    return [
        {"id": 31, "x": 120.0, "y": 200.0, "score": 1.0},
        {"id": 32, "x": 90.0, "y": 220.0, "score": 1.0},
        {"id": 33, "x": 80.0, "y": 260.0, "score": 1.0},
        {"id": 34, "x": 150.0, "y": 220.0, "score": 1.0},
        {"id": 35, "x": 160.0, "y": 260.0, "score": 1.0},
        {"id": 36, "x": 120.0, "y": 240.0, "score": 1.0},
    ]


def test_normalize_and_defaults() -> None:
    assert DEFAULT_TRAVEL_BOX["enabled"] is True
    assert set(DEFAULT_TRAVEL_BOX) == {
        "version", "enabled", "left", "right", "up", "down",
        "body_left", "body_right", "body_up", "body_down",
        "turn_left", "turn_right", "tilt_left", "tilt_right",
        "pitch_up", "pitch_down", "eye", "size",
    }
    out = normalize_travel_box(
        {"version": TRAVEL_VERSION, "left": -2, "turn_left": 900, "tilt_right": -5, "eye": 3, "enabled": "off"}
    )
    assert out["left"] == 0.0
    assert out["turn_left"] == 80.0
    assert out["tilt_right"] == 0.0
    assert out["eye"] == 1.0
    assert out["enabled"] is False


def test_old_saved_box_drops_its_room_and_keeps_angles() -> None:
    """Old boxes saved up/down = 0 for a wall that never held. Do not lock the character."""
    old = {"left": 0.0, "up": 0.0, "body_up": 0.0, "yaw": 40, "eye_x": 0.5, "pad_px": 50, "side": True}
    out = normalize_travel_box(old)
    assert out["version"] == TRAVEL_VERSION
    assert out["up"] == DEFAULT_TRAVEL_BOX["up"]
    assert out["body_up"] == DEFAULT_TRAVEL_BOX["body_up"]
    assert out["turn_left"] == out["turn_right"] == 40.0
    assert out["eye"] == 0.5
    assert "pad_px" not in out and "side" not in out and "yaw" not in out


def test_one_turn_or_tilt_cap_fills_both_sides() -> None:
    """Boxes saved before the split kept one ``yaw`` / ``roll`` for both sides."""
    out = normalize_travel_box({"version": TRAVEL_VERSION, "yaw": 30, "roll": 7})
    assert out["turn_left"] == out["turn_right"] == 30.0
    assert out["tilt_left"] == out["tilt_right"] == 7.0
    sided = merge_travel_box(default_travel_box(), {"turn_left": 5, "tilt_right": 9})
    assert sided["turn_left"] == 5.0 and sided["turn_right"] == DEFAULT_TRAVEL_BOX["turn_right"]
    assert sided["tilt_right"] == 9.0 and sided["tilt_left"] == DEFAULT_TRAVEL_BOX["tilt_left"]
    both = merge_travel_box(sided, {"yaw": 33})
    assert both["turn_left"] == both["turn_right"] == 33.0


def test_merge_travel_box() -> None:
    out = merge_travel_box(default_travel_box(), {"left": 0.2, "body_up": 0.5})
    assert abs(out["left"] - 0.2) < 1e-6
    assert abs(out["body_up"] - 0.5) < 1e-6
    assert out["right"] == default_travel_box()["right"]
    again = merge_travel_box(out, {"yaw": 20})
    assert abs(again["left"] - 0.2) < 1e-6


def test_lab_feel_caps_yaw_half() -> None:
    caps = lab_feel_caps(
        {"turn_left": 40, "turn_right": 20, "tilt_left": 8, "tilt_right": 16, "eye": 0.4, "size": 0.35, "enabled": True}
    )
    assert abs(caps["max_yaw_left"] - 0.5) < 1e-6
    assert abs(caps["max_yaw_right"] - 0.25) < 1e-6
    assert abs(caps["max_roll_left"] - 0.1) < 1e-6
    assert abs(caps["max_roll_right"] - 0.2) < 1e-6
    assert abs(caps["max_size"] - 0.5) < 1e-6
    assert abs(lab_feel_caps({"size": 0.35, "enabled": False})["max_size"] - 1.0) < 1e-6
    assert abs(caps["max_look_x"] - 0.4) < 1e-6
    assert abs(caps["max_look_y"] - 0.4) < 1e-6
    off = lab_feel_caps({"turn_left": 40, "enabled": False})
    assert abs(off["max_yaw_left"] - 1.0) < 1e-6
    assert abs(off["max_look_x"] - 1.0) < 1e-6


def test_set_travel_store_merge_persist_noop(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "travel_box.json"
    monkeypatch.setattr("backend.travel_box.TRAVEL_PATH", path)
    store = TravelStore()
    first, changed = store.update({"turn_left": 40, "left": 0.15})
    assert changed is True
    assert abs(first["turn_left"] - 40.0) < 1e-6
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert abs(saved["turn_left"] - 40.0) < 1e-6
    second, again = store.update({"turn_left": 40, "left": 0.15})
    assert again is False
    assert boxes_equal(first, second)


def test_status_and_frame_include_travel_box() -> None:
    box = default_travel_box()
    box["turn_left"] = 40.0
    status = pack_status({"ok": True, "live": False, "feel": {}, "travel_box": box})
    assert status["travel_box"]["turn_left"] == 40.0
    assert "set_travel" in status["commands"]
    frame = pack_frame(
        {
            "live": True,
            "tracker": "osf",
            "faces": 1,
            "points": [[float(i), float(i), 1.0] for i in range(28)],
            "skeleton": _rest_skeleton(),
            "feel": {},
            "travel_box": box,
            "weights": {},
            "head": {"pitch": 0, "yaw": 0, "roll": 0},
            "blink": {"l": 0, "r": 0},
        },
        image_wh=(200, 300),
    )
    assert frame["travel_box"]["turn_left"] == 40.0
    assert "set_travel" in COMMANDS
    msg = parse_command({"op": "set_travel", "body": {"yaw": 20}})
    assert msg["op"] == "set_travel"
    assert msg["body"]["yaw"] == 20


def test_dispatch_set_travel_updates_feel(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "travel_box.json"
    monkeypatch.setattr("backend.travel_box.TRAVEL_PATH", path)
    from backend.travel_box import default_travel_box, travel as store

    store.values = default_travel_box()

    class Bench:
        def __init__(self) -> None:
            self.feel_caps: dict[str, float] = {}
            self.published = 0

        def status(self, *, publish: bool = False) -> dict[str, object]:
            if publish:
                self.published += 1
            return {
                "ok": True,
                "live": False,
                "error": "",
                "feel": dict(self.feel_caps) if self.feel_caps else {},
                "travel_box": store.payload(),
            }

        def live_status(self) -> dict[str, object]:
            return {
                "ok": True,
                "live": False,
                "tracker": "",
                "faces": 0,
                "ms": 0,
                "error": "",
                "points": [],
                "skeleton": [],
                "hair": [],
                "weights": {},
                "head": {"pitch": 0, "yaw": 0, "roll": 0},
                "blink": {"l": 0, "r": 0},
                "feel": {},
                "travel_box": store.payload(),
            }

        def set_travel(self, body: object) -> dict[str, object]:
            from backend.travel_box import lab_feel_caps

            box, changed = store.update(body if isinstance(body, dict) else {})
            if changed:
                self.feel_caps = lab_feel_caps(box)
            return self.status(publish=changed)

    bench = Bench()
    reply = handle(bench, {"op": "set_travel", "id": "t1", "body": {"turn_left": 40}})
    assert reply["ok"] is True
    assert abs(float(reply["status"]["travel_box"]["turn_left"]) - 40.0) < 1e-6
    assert abs(float(bench.feel_caps["max_yaw_left"]) - 0.5) < 1e-6
    assert "travel_box" in reply["frame"]
    # Repeat same box → no-op (no feel rewrite / no bench publish).
    before = bench.published
    caps_before = dict(bench.feel_caps)
    again = handle(bench, {"op": "set_travel", "body": {"turn_left": 40}})
    assert again["ok"] is True
    assert bench.published == before
    assert bench.feel_caps == caps_before


def test_soft_barrier_eases_past_the_wall_then_stops() -> None:
    from backend.travel_box import _BARRIER_GIVE, soft_barrier

    assert soft_barrier(4.0, -10.0, 10.0, 0.0) == 4.0
    near = soft_barrier(11.0, -10.0, 10.0, 0.0)
    assert 10.7 < near < 11.0
    far = soft_barrier(80.0, -10.0, 10.0, 0.0)
    further = soft_barrier(400.0, -10.0, 10.0, 0.0)
    ceiling = 10.0 + 10.0 * _BARRIER_GIVE
    assert 10.0 < far < ceiling + 1e-6
    assert further - far < 0.05
    assert soft_barrier(40.0, 0.0, 0.0, 0.0) == 0.0


def _rest() -> np.ndarray:
    return pack_overlay(_rest_face(), _rest_skeleton())


def _neck_offset(k: np.ndarray) -> np.ndarray:
    return k[31, :2] - k[15, :2]


def test_moving_up_stops_head_and_torso_at_their_own_walls() -> None:
    rest = _rest()
    fh = face_height(rest)
    live = rest.copy()
    live[:, 1] -= 3.0 * fh
    box = merge_travel_box(default_travel_box(), {"up": 0.2, "body_up": 0.15})
    out, _ = apply_limits(live, rest, box)
    assert abs(float(out[15, 1] - rest[15, 1]) + 0.2 * fh) < 1e-3
    assert abs(float(out[36, 1] - rest[36, 1]) + 0.15 * fh) < 1e-3
    # Equal rooms keep the neck exactly where it was under the head.
    same = merge_travel_box(default_travel_box(), {"up": 0.15, "body_up": 0.15})
    out, _ = apply_limits(live, rest, same)
    np.testing.assert_allclose(_neck_offset(out), _neck_offset(rest), atol=1e-4)


def test_torso_wall_is_fixed_to_rest_not_the_live_head() -> None:
    """The old torso limiter was re-attached to the head afterwards, so it followed you."""
    rest = _rest()
    fh = face_height(rest)
    live = rest.copy()
    live[:, 1] -= 2.0 * fh
    box = merge_travel_box(default_travel_box(), {"up": 1.2, "body_up": 0.1})
    out, _ = apply_limits(live, rest, box)
    assert float(out[36, 1]) >= float(rest[36, 1]) - 0.1 * fh - 1e-3
    assert float(out[31, 1]) >= float(rest[31, 1]) - 0.1 * fh - 1e-3
    # A tight torso wall does not hold the head back.
    assert abs(float(out[15, 1] - rest[15, 1]) + 1.2 * fh) < 1e-3


def test_head_dragging_the_torso_stops_at_the_body_wall() -> None:
    """Only the head moved, and it dragged the torso along; the torso must stop."""
    rest = _rest()
    fh = face_height(rest)
    live = rest.copy()
    live[:, 0] += 0.25 * fh
    box = merge_travel_box(default_travel_box(), {"right": 0.3, "body_right": 0.05})
    out, _ = apply_limits(live, rest, box)
    np.testing.assert_allclose(out[:28, :2], live[:28, :2], atol=1e-4)
    for slot in (31, 32, 34, 36):
        assert float(out[slot, 0] - rest[slot, 0]) <= 0.05 * fh + 1e-3


def test_every_torso_point_stays_in_its_room_when_the_torso_tilts() -> None:
    """A tilt barely moves the torso's center; the old center-only wall let the chest swing out."""
    rest = _rest()
    fh = face_height(rest)
    live = rest.copy()
    neck = rest[31, :2].copy()
    ang = np.radians(20.0)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
    for slot in (32, 33, 34, 35, 36):
        live[slot, :2] = neck + rot @ (rest[slot, :2] - neck)
    room = 0.17
    box = merge_travel_box(
        default_travel_box(),
        {"body_left": room, "body_right": room, "body_up": room, "body_down": room},
    )
    assert abs(float(live[36, 0] - rest[36, 0])) > room * fh
    out, _ = apply_limits(live, rest, box)
    for slot in (31, 32, 34, 36):
        moved = out[slot, :2] - rest[slot, :2]
        assert np.all(np.abs(moved) <= room * fh + 1e-3), slot
    np.testing.assert_allclose(out[:28, :2], live[:28, :2], atol=1e-4)


def test_moving_inside_the_walls_is_untouched() -> None:
    rest = _rest()
    fh = face_height(rest)
    live = rest.copy()
    live[:, 0] += 0.1 * fh
    out, _ = apply_limits(live, rest, default_travel_box())
    np.testing.assert_allclose(out, live, atol=1e-5)


def test_talking_and_arm_swing_at_the_wall_do_not_push() -> None:
    rest = _rest()
    fh = face_height(rest)
    box = merge_travel_box(default_travel_box(), {"down": 0.2, "body_down": 0.2})
    live = rest.copy()
    live[:, 1] += 0.2 * fh
    live[2, 1] += 25.0  # jaw drops
    live[20:28, 1] += 18.0  # mouth opens
    live[33, 1] += 40.0  # elbow swings
    out, _ = apply_limits(live, rest, box)
    np.testing.assert_allclose(out[:, :2], live[:, :2], atol=1e-4)


def test_hair_moves_with_the_character() -> None:
    rest = _rest()
    fh = face_height(rest)
    live = rest.copy()
    live[:, 0] -= 2.0 * fh
    hair = [{"class": "hair_middle", "polygon": [[100.0, 60.0], [140.0, 60.0], [120.0, 90.0]]}]
    moved = [{"class": "hair_middle", "polygon": [[x - 2.0 * fh, y] for x, y in hair[0]["polygon"]]}]
    out, hair_out = apply_limits(live, rest, default_travel_box(), hair=moved)
    dx = float(out[15, 0] - live[15, 0])
    assert dx > 0.0
    assert hair_out is not None
    assert abs(hair_out[0]["polygon"][0][0] - (moved[0]["polygon"][0][0] + dx)) < 1e-4


def test_limiter_rects_come_from_rest_only() -> None:
    rest = _rest()
    fh = face_height(rest)
    box = merge_travel_box(default_travel_box(), {"left": 0.5, "up": 0.25})
    rects = limiter_rects_px(rest, box)
    head = head_mesh_rect_px(rest)
    assert head is not None and rects["head"] is not None and rects["head_wall"] is not None
    assert abs(rects["head_wall"][0] - (head[0] - 0.5 * fh)) < 0.05
    assert abs(rects["head_wall"][1] - (head[1] - 0.25 * fh)) < 0.05
    assert rects["body_wall"] is not None
    assert "size" not in rects


def test_travel_disabled_skips_clamp() -> None:
    rest = _rest()
    live = rest.copy()
    live[:28, 0] -= 90.0
    out, _ = apply_limits(live, rest, {"enabled": False})
    assert abs(float(out[0, 0]) - float(live[0, 0])) < 1e-5
