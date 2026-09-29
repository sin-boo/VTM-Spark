"""Limiters: head and body walls fixed to the rest pose; the character moves as one piece."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.engine import neutral_keypoints
from backend.pose_controller import CHEST, FACE_SLOTS, NECK, face_center, face_height
from backend.travel_box import (
    DEFAULT_TRAVEL_BOX,
    TRAVEL_VERSION,
    apply_travel_box,
    body_mesh_rect_norm,
    body_rect_norm,
    changed_preview_axis,
    default_travel_box,
    draw_travel_box,
    head_mesh_rect_norm,
    head_rect_norm,
    lab_feel_caps,
    merge_travel_box,
    motion_caps,
    normalize_travel_box,
    preview_travel_pose,
)


def _box(**room: float) -> dict:
    return merge_travel_box(default_travel_box(), room)


def test_look_up_default_stops_short_of_a_ceiling_nod() -> None:
    box = default_travel_box()
    assert box["pitch_up"] == 14.0
    assert box["pitch_up"] < 50.0


def test_only_the_simple_limiters_remain() -> None:
    assert set(DEFAULT_TRAVEL_BOX) == {
        "version", "enabled", "left", "right", "up", "down",
        "body_left", "body_right", "body_up", "body_down",
        "turn_left", "turn_right", "tilt_left", "tilt_right",
        "pitch_up", "pitch_down", "eye", "size",
    }


def test_normalize_clips_and_fills() -> None:
    out = normalize_travel_box(
        {"version": TRAVEL_VERSION, "left": -2, "right": 9, "enabled": "off", "turn_right": 900}
    )
    assert out["left"] == 0.0
    assert out["right"] == 1.2
    assert out["enabled"] is False
    assert out["turn_right"] == 80.0


def test_old_saved_box_drops_its_room_and_keeps_angles() -> None:
    old = {"left": 0.0, "up": 0.0, "body_up": 0.0, "yaw": 40, "eye_x": 0.5, "pad_px": 50, "body_rotate": True}
    out = normalize_travel_box(old)
    assert out["version"] == TRAVEL_VERSION
    assert out["up"] == DEFAULT_TRAVEL_BOX["up"]
    assert out["body_up"] == DEFAULT_TRAVEL_BOX["body_up"]
    assert out["turn_left"] == out["turn_right"] == 40.0
    assert out["eye"] == 0.5
    assert "pad_px" not in out and "body_rotate" not in out


def test_merge_partial_keeps_the_rest() -> None:
    out = _box(left=0.05)
    assert out["left"] == 0.05
    assert out["right"] == DEFAULT_TRAVEL_BOX["right"]
    assert out["turn_left"] == DEFAULT_TRAVEL_BOX["turn_left"]


def test_motion_and_feel_caps() -> None:
    caps = motion_caps(_box(turn_left=30, turn_right=5, tilt_left=6, tilt_right=20))
    assert (caps["turn_left"], caps["turn_right"], caps["tilt_left"], caps["tilt_right"]) == (30.0, 5.0, 6.0, 20.0)
    off = motion_caps({"enabled": False, "turn_left": 3, "pitch_down": 2})
    assert off["turn_left"] == off["turn_right"] == 80.0
    assert off["pitch_down"] == 32.0
    feel = lab_feel_caps({"yaw": 40, "roll": 20, "pitch_up": 25, "pitch_down": 16, "eye": 0.4})
    assert feel["max_yaw_left"] == feel["max_yaw_right"] == 0.5
    assert feel["max_roll_left"] == feel["max_roll_right"] == 0.25
    assert feel["max_pitch_up"] == 0.5
    assert feel["max_pitch_down"] == 0.5
    assert abs(feel["max_look_x"] - 0.4) < 1e-6
    assert abs(feel["max_look_y"] - 0.4) < 1e-6


def test_moving_up_stops_head_and_torso_together() -> None:
    rest = neutral_keypoints()
    fh = face_height(rest)
    live = rest.copy()
    live[:, 1] -= 3.0 * fh
    out, _, scale = apply_travel_box(live, rest, _box(up=0.2, body_up=0.15))
    assert scale == 1.0
    assert abs(float(face_center(out)[1] - face_center(rest)[1]) + 0.15 * fh) < 1e-3
    assert abs(float(out[CHEST, 1] - rest[CHEST, 1]) + 0.15 * fh) < 1e-3
    np.testing.assert_allclose(
        out[NECK, :2] - face_center(out), rest[NECK, :2] - face_center(rest), atol=1e-4
    )


def test_sideways_stops_at_the_head_wall() -> None:
    rest = neutral_keypoints()
    fh = face_height(rest)
    live = rest.copy()
    live[:, 0] += 2.0
    out, _, _ = apply_travel_box(live, rest, _box(right=0.18, body_right=1.2))
    assert abs(float(face_center(out)[0] - face_center(rest)[0]) - 0.18 * fh) < 1e-3


def test_inside_the_walls_is_unchanged() -> None:
    rest = neutral_keypoints()
    fh = face_height(rest)
    live = rest.copy()
    live[:, 0] += 0.1 * fh
    live[21, 1] += 0.02
    out, _, _ = apply_travel_box(live, rest, default_travel_box())
    np.testing.assert_allclose(out, live, atol=1e-6)


def test_disabled_passthrough() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[:, 0] += 2.0
    out, _, _ = apply_travel_box(live, rest, {"enabled": False})
    np.testing.assert_allclose(out, live, atol=1e-6)


def test_hair_follows_the_held_character() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[:, 0] += 2.0
    hair = [{"class": "hair_middle", "polygon": [[0.0, -0.5], [0.2, -0.5], [0.1, -0.3]]}]
    out, hair_out, _ = apply_travel_box(live, rest, default_travel_box(), hair=hair)
    dx = float(out[0, 0] - live[0, 0])
    assert dx < 0.0
    assert hair_out is not None
    assert abs(hair_out[0]["polygon"][0][0] - (0.0 + dx)) < 1e-5


def test_eye_limit_pulls_iris_into_the_eye() -> None:
    from backend.pose_controller import RIGHT_IRIS

    rest = neutral_keypoints()
    live = rest.copy()
    live[RIGHT_IRIS, 0] += 0.4
    out, _, _ = apply_travel_box(live, rest, _box(eye=0.5))
    assert float(out[RIGHT_IRIS, 0]) < float(live[RIGHT_IRIS, 0])


def test_walls_are_the_rest_boxes_plus_room() -> None:
    rest = neutral_keypoints()
    fh = face_height(rest)
    tight = head_mesh_rect_norm(rest)
    wall = head_rect_norm(rest, _box(left=0.5, right=0.1, up=0.25, down=0.0))
    assert tight is not None and wall is not None
    np.testing.assert_allclose(
        wall,
        (tight[0] - 0.5 * fh, tight[1] - 0.25 * fh, tight[2] + 0.1 * fh, tight[3]),
        atol=1e-5,
    )
    body = body_mesh_rect_norm(rest)
    body_wall = body_rect_norm(rest, _box(body_left=0.2))
    assert body is not None and body_wall is not None
    assert abs(body_wall[0] - (body[0] - 0.2 * fh)) < 1e-5
    assert head_rect_norm(rest, _box(body_left=1.0)) == head_rect_norm(rest, _box(body_left=0.0))


def test_preview_slides_the_whole_character() -> None:
    rest = neutral_keypoints()
    fh = face_height(rest)
    out = preview_travel_pose(rest, _box(left=0.4), "left")
    assert abs(float(face_center(out)[0] - face_center(rest)[0]) + 0.4 * fh) < 1e-4
    assert abs(float(out[CHEST, 0] - rest[CHEST, 0]) + 0.4 * fh) < 1e-4
    nod = preview_travel_pose(rest, _box(pitch_up=50.0), "pitch_up")
    assert np.linalg.norm(nod[list(FACE_SLOTS), :2] - rest[list(FACE_SLOTS), :2]) > 1e-3
    # Look down shows a nod (it used to tilt the face sideways): the face
    # drops, nothing moves across, and the torso stays where it is.
    down = preview_travel_pose(rest, _box(pitch_down=12.0), "pitch_down")
    assert float(down[15, 1]) > float(rest[15, 1])
    np.testing.assert_allclose(down[list(FACE_SLOTS), 0], rest[list(FACE_SLOTS), 0])
    np.testing.assert_allclose(down[[NECK, CHEST]], rest[[NECK, CHEST]])
    assert changed_preview_axis(default_travel_box(), _box(left=0.2)) == "left"
    assert changed_preview_axis(default_travel_box(), _box(left=0.2, right=0.1)) is None


def test_draw_comes_from_rest_and_follows_room() -> None:
    from PIL import Image

    rest = neutral_keypoints()
    img = Image.new("RGB", (160, 160), (20, 20, 20))
    tight = np.asarray(draw_travel_box(img, rest, _box(left=0.0)))
    wide = np.asarray(draw_travel_box(img, rest, _box(left=0.6)))
    assert not np.array_equal(tight, wide)
    off = np.asarray(draw_travel_box(img, rest, {"enabled": False}))
    np.testing.assert_array_equal(off, np.asarray(img))


def _payload_runtime(status: dict) -> object:
    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._status = status
    rt.status = lambda: dict(rt._status)
    rt._body_lost = False
    rt._mouth_snapped = False
    rt.engine = type("E", (), {"_ref_keypoints": None})()
    return rt


def test_slider_preview_draws_moved_mesh(monkeypatch) -> None:
    from PIL import Image

    from backend.stream import StreamRuntime

    seen: list = []
    monkeypatch.setattr(
        "backend.stream.draw_keypoint_mesh",
        lambda image, kps, **kwargs: seen.append(np.asarray(kps).copy()) or image,
    )
    rest = neutral_keypoints()
    rt = _payload_runtime({"show_mesh": True, "show_hair": False, "travel_box": default_travel_box()})
    rt._last_overlay_kps = rest.copy()
    rt._travel_preview_kps = None
    rt._tracking = False
    rt.engine = type("E", (), {"_ref_keypoints": rest})()
    StreamRuntime._refresh_travel_preview(rt, default_travel_box(), _box(left=0.5))
    StreamRuntime._frame_payload(rt, Image.new("RGB", (8, 8), (12, 34, 56)), rest)
    assert seen
    assert float(face_center(seen[0])[0]) < float(face_center(rest)[0]) - 0.02
    np.testing.assert_allclose(rt._last_overlay_kps, rest, atol=1e-6)


def test_shown_limiters_stay_on_the_rest_pose(monkeypatch) -> None:
    """The Show toggle used to box the live pose, so the walls followed you."""
    from PIL import Image

    from backend.stream import StreamRuntime

    drawn: list = []
    monkeypatch.setattr(
        "backend.stream.draw_travel_box",
        lambda image, rest, box: drawn.append(np.asarray(rest).copy()) or image,
    )
    rest = neutral_keypoints()
    moved = rest.copy()
    moved[:, 1] -= 0.5
    rt = _payload_runtime(
        {"show_mesh": False, "show_hair": False, "show_limiters": True, "travel_box": default_travel_box()}
    )
    rt._travel_preview_kps = None
    rt._last_overlay_kps = moved
    rt.engine = type("E", (), {"_ref_keypoints": rest})()
    StreamRuntime._frame_payload(rt, Image.new("RGB", (8, 8)), moved)
    assert drawn
    np.testing.assert_allclose(drawn[0], rest, atol=1e-6)


def test_limiter_panel_has_the_simple_set() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "ui" / "src" / "components" / "TravelBox.tsx").read_text(encoding="utf-8")
    for gone in ("pad_px", "body_yaw", "body_rotate", "eye_x", "Size pad", "Skel rotate"):
        assert gone not in text
    for kept in ("body_left", "body_up", "pitch_up", "'travel.head'", "'travel.body'", "'travel.eyes'", "'common.show'"):
        assert kept in text
