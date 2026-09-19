"""Travel box: selectable walls for character pose, not overlay-only."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from backend.engine import neutral_keypoints
from backend.pose_controller import CHEST, L_SHOULDER, MOUTH, NECK, OVERLAY_BODY_SLOTS, FACE_SLOTS, face_center, face_height
from backend.travel_box import (
    apply_travel_box,
    apply_walk_box,
    body_rect_norm,
    box_rect_norm,
    character_mask,
    changed_preview_axis,
    default_travel_box,
    draw_travel_box,
    hard_silhouette,
    lab_feel_caps,
    merge_travel_box,
    motion_caps,
    normalize_travel_box,
    preview_travel_pose,
    silhouette_rect_norm,
    size_rect_norm,
    head_mesh_rect_norm,
    head_rect_norm,
    SIZE_COLOR,
    BOX_COLOR,
    HEAD_SLOTS,
)


def test_normalize_clips_and_fills() -> None:
    gb = normalize_travel_box({"left": -2, "right": 9, "enabled": "off", "pad_px": 900})
    assert gb["left"] == 0.0
    assert gb["right"] == 1.2
    assert gb["enabled"] is False
    assert gb["side"] is True
    assert gb["rotate"] is True
    assert gb["look_up"] is True
    assert gb["body"] is True
    assert gb["up"] == default_travel_box()["up"]
    assert gb["yaw"] == default_travel_box()["yaw"]
    assert gb["pad_px"] == 200
    assert default_travel_box()["pad_px"] == 50
    assert default_travel_box()["yaw"] == 80.0
    assert default_travel_box()["body_rotate"] is True
    assert default_travel_box()["body_yaw"] == 80.0
    assert default_travel_box()["body_left"] == 0.0
    assert default_travel_box()["eyes"] is True
    assert abs(default_travel_box()["eye_x"] - 0.78) < 1e-6
    assert gb["body_rotate"] is True


def test_merge_partial() -> None:
    out = merge_travel_box(default_travel_box(), {"left": 0.05, "look_up": False})
    assert out["left"] == 0.05
    assert out["right"] == default_travel_box()["right"]
    assert out["look_up"] is False
    assert out["side"] is True
    assert out["rotate"] is True
    assert out["yaw"] == default_travel_box()["yaw"]


def test_head_alias_maps_to_side() -> None:
    out = normalize_travel_box({"head": False, "yaw": 12})
    assert out["side"] is False
    assert out["yaw"] == 12.0


def test_motion_caps_follow_toggles() -> None:
    full = motion_caps(default_travel_box())
    assert full["yaw"] == 80.0
    assert full["pitch_up"] == 50.0
    tight = motion_caps({"rotate": True, "yaw": 10, "roll": 4, "look_up": True, "pitch_up": 8})
    assert tight["yaw"] == 10.0
    assert tight["roll"] == 4.0
    assert tight["pitch_up"] == 8.0
    off = motion_caps({"enabled": False, "yaw": 3, "pitch_down": 2})
    assert off["yaw"] == 80.0
    assert off["pitch_down"] == 32.0
    rot_off = motion_caps({"rotate": False, "yaw": 6})
    assert rot_off["yaw"] == 80.0
    feel = lab_feel_caps({"rotate": True, "yaw": 40, "roll": 20, "pitch_up": 25, "pitch_down": 16})
    assert feel["max_yaw"] == 0.5
    assert feel["max_roll"] == 0.25
    assert feel["max_pitch_up"] == 0.5
    assert feel["max_pitch_down"] == 0.5
    assert abs(feel["max_look_x"] - 0.78) < 1e-6
    assert abs(feel["max_look_y"] - 0.78) < 1e-6
    full_look = lab_feel_caps({"enabled": True, "eyes": False, "eye_x": 0.2})
    assert full_look["max_look_x"] == 1.0
    tight_look = lab_feel_caps({"enabled": True, "eyes": True, "eye_x": 0.4, "eye_y": 0.5})
    assert tight_look["max_look_x"] == 0.4
    assert tight_look["max_look_y"] == 0.5


def test_sideways_head_stops_at_wall() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[:, 0] += 0.85
    box = {"enabled": True, "left": 0.18, "right": 0.18, "up": 0.5, "down": 0.5}
    out, _, scale = apply_travel_box(live, rest, box)
    assert scale < 0.35
    fh = face_height(rest)
    dx = float(face_center(out)[0] - face_center(rest)[0])
    assert dx <= 0.18 * fh + 1e-3
    assert dx > 0.02
    rect = box_rect_norm(rest, box)
    assert rect is not None
    x0, _y0, x1, _y1 = rect
    for i in (0, 4):
        assert float(out[i, 0]) <= x1 + 1e-4
        assert float(out[i, 0]) >= x0 - 1e-4


def test_inside_box_is_unchanged() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[21, 1] += 0.02
    box = default_travel_box()
    out, _, scale = apply_travel_box(live, rest, box)
    assert scale == 1.0
    np.testing.assert_allclose(out, live, atol=1e-6)


def test_disabled_passthrough() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[:, 0] += 0.7
    box = merge_travel_box(default_travel_box(), {"enabled": False})
    out, _, scale = apply_travel_box(live, rest, box)
    assert scale == 1.0
    np.testing.assert_allclose(out, live)


def test_side_off_allows_lateral_travel() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[:, 0] += 0.55
    capped = apply_travel_box(live, rest, default_travel_box())[0]
    free = apply_travel_box(live, rest, {"side": False})[0]
    assert abs(float(free[4, 0] - rest[4, 0])) > abs(float(capped[4, 0] - rest[4, 0])) + 0.2


def test_look_toggles_do_not_y_clamp_face() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[:, 1] -= 0.45
    capped = apply_travel_box(
        live, rest, {"look_up": True, "side": False, "body": False, "body_rotate": False}
    )[0]
    free = apply_travel_box(
        live, rest, {"look_up": False, "side": False, "body": False, "body_rotate": False}
    )[0]
    np.testing.assert_allclose(capped, free, atol=1e-6)


def test_body_on_caps_shoulder_swing() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[L_SHOULDER, 0] += 0.8
    tight = apply_travel_box(
        live, rest, {"body": True, "body_left": 0.12, "body_right": 0.12}
    )[0]
    loose = apply_travel_box(live, rest, {"body": False})[0]
    dx_tight = abs(float(tight[L_SHOULDER, 0] - rest[L_SHOULDER, 0]))
    dx_loose = abs(float(loose[L_SHOULDER, 0] - rest[L_SHOULDER, 0]))
    assert dx_loose > dx_tight + 0.15


def test_head_room_does_not_expand_skeleton_box() -> None:
    rest = neutral_keypoints()
    head_extra = body_rect_norm(rest, {"body": True, "left": 0.5, "right": 0.5})
    none = body_rect_norm(rest, {"body": True, "left": 0.0, "right": 0.0})
    skel_extra = body_rect_norm(rest, {"body": True, "body_left": 0.25, "body_right": 0.25})
    assert head_extra is not None and none is not None and skel_extra is not None
    np.testing.assert_allclose(head_extra, none, atol=1e-6)
    fh = face_height(rest)
    np.testing.assert_allclose(skel_extra[0], none[0] - 0.25 * fh, atol=1e-5)
    np.testing.assert_allclose(skel_extra[2], none[2] + 0.25 * fh, atol=1e-5)
    head = box_rect_norm(rest, {"left": 0.0, "right": 0.0, "body_left": 0.5})
    head_plain = box_rect_norm(rest, {"left": 0.0, "right": 0.0, "body_left": 0.0})
    assert head is not None and head_plain is not None
    np.testing.assert_allclose(head, head_plain, atol=1e-6)


def test_body_rotation_caps_torso_turn() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    ox, oy = float(rest[NECK, 0]), float(rest[NECK, 1])
    rad = math.radians(40.0)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    for i in OVERLAY_BODY_SLOTS:
        x = float(live[i, 0]) - ox
        y = float(live[i, 1]) - oy
        live[i, 0] = ox + cos_a * x - sin_a * y
        live[i, 1] = oy + sin_a * x + cos_a * y

    def torso_delta(k: np.ndarray) -> float:
        live_a = math.atan2(float(k[CHEST, 1]) - float(k[NECK, 1]), float(k[CHEST, 0]) - float(k[NECK, 0]))
        rest_a = math.atan2(float(rest[CHEST, 1]) - float(rest[NECK, 1]), float(rest[CHEST, 0]) - float(rest[NECK, 0]))
        d = live_a - rest_a
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return abs(math.degrees(d))

    free = apply_travel_box(
        live, rest, {"body": False, "body_rotate": False, "side": False, "look_up": False, "look_down": False}
    )[0]
    capped = apply_travel_box(
        live,
        rest,
        {
            "body": False,
            "body_rotate": True,
            "body_yaw": 8.0,
            "body_roll": 80.0,
            "side": False,
            "look_up": False,
            "look_down": False,
        },
    )[0]
    assert torso_delta(free) > 30.0
    assert torso_delta(capped) <= 8.0 + 1.5


def test_travel_box_ui_drops_room_up_down() -> None:
    src = Path(__file__).resolve().parents[2] / "ui" / "src" / "components" / "TravelBox.tsx"
    text = src.read_text(encoding="utf-8")
    assert "Room up" not in text
    assert "Room down" not in text
    assert "body_yaw" in text
    assert "body_left" in text
    assert "Skel rotate" in text
    assert "eye_x" in text
    assert "Eyes" in text
    assert "travel-legend" in text
    assert "travel-kind" in text
    assert "Scale" in text
    assert "Form" in text
    assert "Rotation" in text
    assert "is-look" not in text
    assert "is-turn" not in text
    assert "hair included" in text
    css = Path(__file__).resolve().parents[2].joinpath("ui", "src", "App.css").read_text(encoding="utf-8")
    assert "is-look" not in css
    assert "is-turn" not in css
    assert "Head / look" not in text
    assert "travel-block-head" in text
    assert "travel-section" not in text
    assert "travel-section" not in css


def test_preview_left_slides_head_not_skeleton() -> None:
    rest = neutral_keypoints()
    out = preview_travel_pose(rest, {"left": 0.4, "side": True}, "left")
    assert float(face_center(out)[0]) < float(face_center(rest)[0]) - 0.02
    assert abs(float(out[CHEST, 0]) - float(rest[CHEST, 0])) < 1e-6
    stay = preview_travel_pose(rest, {"left": 0.4, "side": False}, "left")
    np.testing.assert_allclose(stay, rest, atol=1e-6)


def test_preview_look_up_nods_head() -> None:
    rest = neutral_keypoints()
    out = preview_travel_pose(rest, {"pitch_up": 50.0, "look_up": True}, "pitch_up")
    assert np.linalg.norm(out[list(FACE_SLOTS), :2] - rest[list(FACE_SLOTS), :2]) > 1e-3
    assert changed_preview_axis(default_travel_box(), {"left": 0.2}) == "left"
    assert changed_preview_axis(default_travel_box(), {"left": 0.2, "right": 0.3}) is None


def test_slider_preview_draws_moved_mesh(monkeypatch) -> None:
    from PIL import Image

    from backend.stream import StreamRuntime

    seen: list = []
    monkeypatch.setattr(
        "backend.stream.draw_keypoint_mesh",
        lambda image, kps, **kwargs: seen.append(np.asarray(kps).copy()) or image,
    )
    rest = neutral_keypoints()
    rt = _payload_runtime(
        {
            "show_mesh": True,
            "show_hair": False,
            "travel_box": default_travel_box(),
        }
    )
    rt._last_overlay_kps = rest.copy()
    rt._travel_preview_kps = None
    rt._tracking = False
    rt.engine = type("E", (), {"_ref_keypoints": rest})()
    StreamRuntime._refresh_travel_preview(rt, default_travel_box(), {"left": 0.5, "side": True})
    image = Image.new("RGB", (8, 8), (12, 34, 56))
    StreamRuntime._frame_payload(rt, image, rest)
    assert seen
    assert float(face_center(seen[0])[0]) < float(face_center(rest)[0]) - 0.02
    np.testing.assert_allclose(rt._last_overlay_kps, rest, atol=1e-6)


def test_eye_limiter_pulls_iris_off_the_rim() -> None:
    from backend.pose_controller import RIGHT_IRIS

    rest = neutral_keypoints()
    rest[11:14, 3] = 1.0
    rest[RIGHT_IRIS, 3] = 1.0
    live = rest.copy()
    cx = 0.5 * (float(rest[11, 0]) + float(rest[13, 0]))
    live[RIGHT_IRIS, 0] = float(rest[13, 0])
    out, _, _ = apply_travel_box(live, rest, default_travel_box())
    assert abs(float(out[RIGHT_IRIS, 0]) - cx) < abs(float(live[RIGHT_IRIS, 0]) - cx) - 1e-4
    free, _, _ = apply_travel_box(live, rest, {"eyes": False})
    assert abs(float(free[RIGHT_IRIS, 0]) - float(live[RIGHT_IRIS, 0])) < 1e-5


def test_walk_box_slides_back_without_squashing_look() -> None:
    from backend.pose_controller import RIGHT_IRIS

    rest = neutral_keypoints()
    rest[11:14, 3] = 1.0
    rest[RIGHT_IRIS, 3] = 1.0
    live = rest.copy()
    live[:, 0] += 0.7
    look = 0.04
    live[RIGHT_IRIS, 0] = float(live[13, 0]) + look
    box = {"enabled": True, "left": 0.0, "right": 0.0, "side": True}
    out, _, _ = apply_walk_box(live, rest, box)
    assert float(out[4, 0]) < float(live[4, 0]) - 0.2
    iris_rel = float(out[RIGHT_IRIS, 0] - out[13, 0])
    assert abs(iris_rel - look) < 1e-5


def test_walk_box_moves_hair_with_the_head() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[:, 0] += 0.7
    hair = [{"class": "hair_middle", "polygon": [[0.8, -0.4], [0.9, -0.4], [0.85, -0.2]]}]
    out, hair_out, _ = apply_walk_box(live, rest, default_travel_box(), hair=hair)
    assert hair_out is not None
    dx_head = float(out[4, 0] - live[4, 0])
    dx_hair = float(hair_out[0]["polygon"][0][0] - hair[0]["polygon"][0][0])
    assert abs(dx_head - dx_hair) < 1e-5
    assert dx_head < -0.2


def test_hair_follows_clamped_head() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[:, 0] += 0.7
    hair = [{"class": "hair_middle", "polygon": [[0.8, -0.4], [0.9, -0.4], [0.85, -0.2]]}]
    _out, hair_out, scale = apply_travel_box(live, rest, default_travel_box(), hair=hair)
    assert scale < 1.0
    assert hair_out is not None
    dx = float(hair_out[0]["polygon"][0][0] - hair[0]["polygon"][0][0])
    assert dx < -0.2


def _green_ref(size: int = 128) -> np.ndarray:
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:] = (8, 240, 20)
    rgb[20:100, 30:90] = (200, 40, 80)
    return rgb


def test_hard_silhouette_keys_green() -> None:
    rgb = _green_ref()
    mask = character_mask(rgb)
    bw = hard_silhouette(rgb)
    assert not bool(mask[0, 0])
    assert bool(mask[50, 50])
    assert int(bw[0, 0]) == 0
    assert int(bw[50, 50]) == 255
    assert set(np.unique(bw).tolist()) <= {0, 255}


def test_silhouette_rect_expands_and_clamps() -> None:
    rgb = _green_ref(128)

    def n(px: float) -> float:
        return px / 128.0 * 2.0 - 1.0

    tight = silhouette_rect_norm(rgb, pad_px=0)
    padded = silhouette_rect_norm(rgb, pad_px=10)
    assert tight is not None and padded is not None
    np.testing.assert_allclose(tight, (n(30), n(20), n(90), n(100)), atol=1e-6)
    np.testing.assert_allclose(padded, (n(20), n(10), n(100), n(110)), atol=1e-6)
    huge = silhouette_rect_norm(rgb, pad_px=400)
    assert huge == (-1.0, -1.0, 1.0, 1.0)


def test_walk_uses_silhouette_crop() -> None:
    rest = neutral_keypoints()
    live = rest.copy()
    live[:, 0] += 0.85
    rgb = _green_ref(128)
    sil = silhouette_rect_norm(rgb, pad_px=10)
    box = {"enabled": True, "left": 0.0, "right": 0.0, "up": 0.0, "down": 0.0}
    out, _, scale = apply_walk_box(live, rest, box, silhouette=sil)
    assert sil is not None
    assert scale == 1.0
    x0, _y0, x1, _y1 = size_rect_norm(rest, box, silhouette=sil)
    assert x0 is not None
    for i in (0, 4):
        assert float(out[i, 0]) <= x1 + 1e-4
        assert float(out[i, 0]) >= x0 - 1e-4


def test_head_room_expands_face_box_not_size() -> None:
    rest = neutral_keypoints()
    rgb = _green_ref(128)
    sil = silhouette_rect_norm(rgb, pad_px=10)
    zero = {"left": 0.0, "right": 0.0}
    extra = {"left": 0.2, "right": 0.2}
    size_a = size_rect_norm(rest, zero, silhouette=sil)
    size_b = size_rect_norm(rest, extra, silhouette=sil)
    a = head_rect_norm(rest, zero)
    b = head_rect_norm(rest, extra)
    mesh = head_mesh_rect_norm(rest, extra)
    assert size_a is not None and size_b is not None and a is not None and b is not None
    assert mesh is not None
    np.testing.assert_allclose(size_a, size_b, atol=1e-6)
    fh = face_height(rest)
    np.testing.assert_allclose(b[0], a[0] - 0.2 * fh, atol=1e-5)
    np.testing.assert_allclose(b[2], a[2] + 0.2 * fh, atol=1e-5)
    np.testing.assert_allclose(mesh, a, atol=1e-6)
    assert a != size_a


def test_stream_keeps_original_preview_silhouette() -> None:
    from PIL import Image

    from backend.stream import StreamRuntime

    rgb = _green_ref(128)
    rt = StreamRuntime.__new__(StreamRuntime)
    rt._status = {"travel_box": default_travel_box()}
    rt._travel_ref_rgb = None
    rt._travel_silhouette = None
    StreamRuntime._set_travel_reference(rt, Image.fromarray(rgb))
    expected = silhouette_rect_norm(rgb, pad_px=50)
    assert rt._travel_silhouette is not None and expected is not None
    np.testing.assert_allclose(rt._travel_silhouette, expected, atol=1e-6)
    tight = silhouette_rect_norm(rgb, pad_px=0)
    assert rt._travel_silhouette_tight is not None and tight is not None
    np.testing.assert_allclose(rt._travel_silhouette_tight, tight, atol=1e-6)
    rt._status["travel_box"] = merge_travel_box(rt._status["travel_box"], {"pad_px": 10})
    StreamRuntime._refresh_travel_silhouette(rt)
    expected10 = silhouette_rect_norm(rgb, pad_px=10)
    assert expected10 is not None
    np.testing.assert_allclose(rt._travel_silhouette, expected10, atol=1e-6)


def _payload_runtime(status: dict) -> object:
    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    rt._status = status
    rt.status = lambda: dict(rt._status)
    rt._body_lost = False
    rt._mouth_snapped = False
    rt._travel_silhouette = None
    rt._travel_silhouette_tight = None
    rt.engine = type("E", (), {"_ref_keypoints": None})()
    return rt


def test_overlay_does_not_draw_limiter_lines() -> None:
    import inspect

    from backend.stream import StreamRuntime

    src = inspect.getsource(StreamRuntime._frame_payload)
    assert "draw_travel_box" not in src


def test_head_box_follows_overlay_face_points() -> None:
    rest = neutral_keypoints()
    tight = {"left": 0.0, "right": 0.0, "side": True}
    a = head_rect_norm(rest, tight)
    moved = rest.copy()
    moved[:, 0] += 0.25
    b = head_rect_norm(moved, tight)
    assert a is not None and b is not None
    np.testing.assert_allclose((b[0] - a[0], b[2] - a[2]), (0.25, 0.25), atol=1e-5)
    pts = np.stack([rest[i, :2] for i in HEAD_SLOTS if float(rest[i, 3]) >= 0.5], axis=0)
    np.testing.assert_allclose(a[0], float(pts[:, 0].min()), atol=1e-5)
    np.testing.assert_allclose(a[2], float(pts[:, 0].max()), atol=1e-5)
    np.testing.assert_allclose(a[1], float(pts[:, 1].min()), atol=1e-5)
    np.testing.assert_allclose(a[3], float(pts[:, 1].max()), atol=1e-5)


def _color_bbox(image, color: tuple[int, int, int]) -> tuple[int, int, int, int]:
    arr = np.asarray(image)
    ys, xs = np.where((arr == color).all(axis=2))
    assert len(xs) > 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_draw_head_walls_follow_room() -> None:
    from PIL import Image

    rest = neutral_keypoints()
    rgb = _green_ref(128)
    sil = silhouette_rect_norm(rgb, pad_px=10)
    quiet = {
        "look_up": False,
        "look_down": False,
        "body": False,
        "eyes": False,
        "side": True,
    }
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    tight = draw_travel_box(img, rest, {**quiet, "left": 0.0, "right": 0.0}, silhouette=sil)
    lean = draw_travel_box(img, rest, {**quiet, "left": 0.43, "right": 0.0}, silhouette=sil)
    a = _color_bbox(tight, BOX_COLOR)
    b = _color_bbox(lean, BOX_COLOR)
    assert b[0] < a[0]


def test_draw_look_walls_follow_pitch() -> None:
    from PIL import Image

    rest = neutral_keypoints()
    rgb = _green_ref(128)
    sil = silhouette_rect_norm(rgb, pad_px=10)
    quiet = {
        "side": True,
        "body": False,
        "eyes": False,
        "look_down": False,
        "look_up": True,
    }
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    still = draw_travel_box(img, rest, {**quiet, "pitch_up": 0.0}, silhouette=sil)
    tilted = draw_travel_box(img, rest, {**quiet, "pitch_up": 50.0}, silhouette=sil)
    a = _color_bbox(still, BOX_COLOR)
    b = _color_bbox(tilted, BOX_COLOR)
    assert (b[0], b[1], b[2], b[3]) != (a[0], a[1], a[2], a[3])


def test_head_limiter_is_face_mesh_not_hair_or_body() -> None:
    assert set(HEAD_SLOTS).issubset(set(FACE_SLOTS))
    assert not set(HEAD_SLOTS) & set(OVERLAY_BODY_SLOTS)
    assert not set(HEAD_SLOTS) & set(MOUTH)


def test_size_box_ignores_head_room() -> None:
    rest = neutral_keypoints()
    rgb = _green_ref(128)
    sil = silhouette_rect_norm(rgb, pad_px=10)
    tight = {"left": 0.0, "right": 0.0}
    wide = {"left": 0.4, "right": 0.4}
    a = size_rect_norm(rest, tight, silhouette=sil)
    b = size_rect_norm(rest, wide, silhouette=sil)
    assert a is not None and b is not None
    np.testing.assert_allclose(a, b, atol=1e-6)
    head = head_rect_norm(rest, wide)
    assert head is not None
    assert head != a


def test_look_sliders_do_not_move_boxes() -> None:
    rest = neutral_keypoints()
    rgb = _green_ref(128)
    sil = silhouette_rect_norm(rgb, pad_px=10)
    a = {
        "pitch_up": 10.0,
        "pitch_down": 8.0,
        "look_up": True,
        "look_down": True,
    }
    b = {
        "pitch_up": 50.0,
        "pitch_down": 32.0,
        "look_up": True,
        "look_down": True,
    }
    np.testing.assert_allclose(
        size_rect_norm(rest, a, silhouette=sil),
        size_rect_norm(rest, b, silhouette=sil),
        atol=1e-6,
    )
    np.testing.assert_allclose(head_rect_norm(rest, a), head_rect_norm(rest, b), atol=1e-6)


def _has_color(image, color: tuple[int, int, int]) -> bool:
    arr = np.asarray(image)
    return bool((arr == color).all(axis=2).any())


def test_draw_shows_size_head() -> None:
    from PIL import Image

    rest = neutral_keypoints()
    rgb = _green_ref(128)
    sil = silhouette_rect_norm(rgb, pad_px=10)
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    out = draw_travel_box(img, rest, default_travel_box(), silhouette=sil)
    assert _has_color(out, SIZE_COLOR)
    assert _has_color(out, BOX_COLOR)
    assert not _has_color(out, (80, 200, 196))
    assert not _has_color(out, (232, 168, 64))
    assert not _has_color(out, (255, 196, 90))


def test_draw_hides_look_ring() -> None:
    from PIL import Image

    rest = neutral_keypoints()
    rgb = _green_ref(128)
    sil = silhouette_rect_norm(rgb, pad_px=10)
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    out = draw_travel_box(
        img, rest, {"look_up": True, "look_down": True, "rotate": True}, silhouette=sil
    )
    assert _has_color(out, SIZE_COLOR)
    assert not _has_color(out, (80, 200, 196))
    assert not _has_color(out, (232, 168, 64))
    assert not _has_color(out, (255, 196, 90))
