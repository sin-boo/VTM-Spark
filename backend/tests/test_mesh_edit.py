"""Manual mesh drag bakes into the character rest pose and sidecar."""

from __future__ import annotations

import numpy as np

from backend.engine import (
    apply_overlay_drag_to_rest,
    find_sidecar_keypoints,
    save_sidecar_keypoints,
    sidecar_keypoints_npy_path,
)
from backend.engine import _as_keypoints37, load_keypoints_file, neutral_keypoints


def test_drag_offset_bakes_into_rest_slot() -> None:
    rest = neutral_keypoints()
    base = rest.copy()
    edited = rest.copy()
    edited[21, 0] += 0.05
    edited[21, 1] += 0.08
    out = apply_overlay_drag_to_rest(rest, base, edited, [21])
    assert abs(float(out[21, 0]) - float(rest[21, 0]) - 0.05) < 1e-6
    assert abs(float(out[21, 1]) - float(rest[21, 1]) - 0.08) < 1e-6
    # Other slots stay put.
    np.testing.assert_allclose(out[15, :2], rest[15, :2], atol=1e-6)


def test_drag_offset_on_driven_overlay_does_not_eat_expression() -> None:
    rest = neutral_keypoints()
    # Overlay at press already includes a live smile (jaw down).
    base = rest.copy()
    base[25, 1] += 0.10
    edited = base.copy()
    edited[25, 0] += 0.04
    out = apply_overlay_drag_to_rest(rest, base, edited, [25])
    assert abs(float(out[25, 0]) - float(rest[25, 0]) - 0.04) < 1e-6
    assert abs(float(out[25, 1]) - float(rest[25, 1])) < 1e-6


def test_sidecar_roundtrip(tmp_path) -> None:
    img = tmp_path / "upload.png"
    img.write_bytes(b"\x89PNG")
    kps = neutral_keypoints()
    kps[21, 1] += 0.11
    saved = save_sidecar_keypoints(img, kps)
    assert saved == sidecar_keypoints_npy_path(img)
    assert saved.is_file()
    found = find_sidecar_keypoints(img)
    assert found == saved
    loaded = _as_keypoints37(load_keypoints_file(saved))
    np.testing.assert_allclose(loaded[21, 1], kps[21, 1], atol=1e-5)


def test_lab_mesh_release_sends_set_point(tmp_path) -> None:
    from PIL import Image

    from backend.tests.test_pose_keys import _bare_runtime

    rest = neutral_keypoints()
    rt = _bare_runtime(tmp_path, rest)
    rt._lab_drive = True
    rt._last_image = Image.new("RGB", (100, 50), (0, 0, 0))
    rt._emit = lambda _payload: None
    rt._frame_payload = lambda _image, _kps: {}
    calls: list[tuple[str, dict]] = []
    rt._lab_ack = lambda op, body=None: calls.append((op, dict(body or {}))) or {"ok": True}
    rt._lab_overlay_keypoints = lambda: rest.copy()
    edited = rest.copy()
    edited[28, 0] += 0.2
    rt._last_overlay_kps = edited
    rt._drag_slots = {28}
    rt._drag_base_kps = rest.copy()
    rt._drag_xy = (60.0, 20.0)
    rt.mesh_release()
    assert calls == [("set_point", {"id": 28, "x": 60.0, "y": 20.0})]
    assert rt._mesh_edited is False


def test_lab_mesh_reset_clears_offsets(tmp_path) -> None:
    from backend.tests.test_pose_keys import _bare_runtime

    rest = neutral_keypoints()
    rt = _bare_runtime(tmp_path, rest)
    rt._lab_drive = True
    rt._emit = lambda _payload: None
    rt._frame_payload = lambda _image, _kps: {}
    calls: list[tuple[str, dict]] = []
    rt._lab_ack = lambda op, body=None: calls.append((op, dict(body or {}))) or {"ok": True}
    rt._lab_overlay_keypoints = lambda: rest.copy()
    rt.mesh_reset()
    assert calls == [("reset_points", {})]


def test_lab_mesh_press_does_not_freeze_overlay(tmp_path) -> None:
    from PIL import Image

    from backend.pose_controller import normalized_to_pixels
    from backend.tests.test_pose_keys import _bare_runtime

    rest = neutral_keypoints()
    rt = _bare_runtime(tmp_path, rest)
    rt._lab_drive = True
    rt._pose_frozen = False
    rt._last_image = Image.new("RGB", (100, 50), (0, 0, 0))
    rt.status = lambda: {"show_mesh": True}
    pix = normalized_to_pixels(rest, 100, 50)
    rt.mesh_press(float(pix[21, 0]), float(pix[21, 1]))
    assert rt._mesh_edited is False
    assert 21 in rt._drag_slots


def test_lab_mesh_drag_keeps_tracking(tmp_path) -> None:
    from PIL import Image

    from backend.pose_controller import pixels_to_normalized
    from backend.tests.test_pose_keys import _bare_runtime

    rest = neutral_keypoints()
    rt = _bare_runtime(tmp_path, rest)
    rt._lab_drive = True
    rt._tracking = True
    rt._last_image = Image.new("RGB", (100, 50), (0, 0, 0))
    rt._drag_slots = {21}
    rt._drag_xy = (10.0, 10.0)
    rt._emit = lambda _payload: None
    rt._frame_payload = lambda _image, _kps: {}
    calls: list[tuple[str, dict]] = []
    rt._lab_ack = lambda op, body=None: calls.append((op, dict(body or {}))) or {"ok": True}
    rt.mesh_drag(40.0, 22.0)
    assert rt._mesh_edited is False
    assert rt._drag_xy == (40.0, 22.0)
    nx, ny = pixels_to_normalized(40.0, 22.0, 100, 50)
    assert abs(float(rt._last_overlay_kps[21, 0]) - nx) < 1e-5
    assert abs(float(rt._last_overlay_kps[21, 1]) - ny) < 1e-5
    np.testing.assert_allclose(rt._last_overlay_kps[0, :2], rest[0, :2], atol=1e-6)
    assert calls == []


def test_nudge_lab_point_scales_desk_pixels_to_lab(tmp_path) -> None:
    from PIL import Image

    from backend.engine import neutral_keypoints
    from backend.tests.test_pose_keys import _bare_runtime

    rest = neutral_keypoints()
    rt = _bare_runtime(tmp_path, rest)
    rt._last_image = Image.new("RGB", (200, 100), (0, 0, 0))
    rt._lab_image_wh = (100, 50)
    calls: list[tuple[str, dict]] = []
    rt._lab_ack = lambda op, body=None: calls.append((op, dict(body or {}))) or {"ok": True}
    rt._nudge_lab_point(21, 100.0, 50.0)
    assert calls == [("set_point", {"id": 21, "x": 50.0, "y": 25.0})]


def test_lab_mesh_release_converts_desk_pixels_to_lab(tmp_path) -> None:
    from PIL import Image

    from backend.engine import neutral_keypoints
    from backend.tests.test_pose_keys import _bare_runtime

    rest = neutral_keypoints()
    rt = _bare_runtime(tmp_path, rest)
    rt._lab_drive = True
    rt._last_image = Image.new("RGB", (200, 100), (0, 0, 0))
    rt._lab_image_wh = (100, 50)
    rt._emit = lambda _payload: None
    rt._frame_payload = lambda _image, _kps: {}
    calls: list[tuple[str, dict]] = []
    rt._lab_ack = lambda op, body=None: calls.append((op, dict(body or {}))) or {"ok": True}
    rt._lab_overlay_keypoints = lambda frame=None: rest.copy()
    edited = rest.copy()
    edited[28, 0] += 0.2
    rt._last_overlay_kps = edited
    rt._drag_slots = {28}
    rt._drag_base_kps = rest.copy()
    rt._drag_xy = (120.0, 40.0)
    rt.mesh_release()
    assert calls == [("set_point", {"id": 28, "x": 60.0, "y": 20.0})]


def test_lab_drag_pins_slot_on_live_overlay(tmp_path) -> None:
    from PIL import Image

    from backend.pose_controller import pixels_to_normalized
    from backend.tests.test_pose_keys import _bare_runtime

    rest = neutral_keypoints()
    rt = _bare_runtime(tmp_path, rest)
    rt._lab_drive = True
    rt._tracking = True
    rt._last_image = Image.new("RGB", (100, 50), (0, 0, 0))
    rt._drag_slots = {28}
    rt._drag_xy = (80.0, 10.0)
    live = rest.copy()
    live[0, 0] += 0.25
    out = rt._apply_drag_to_overlay(live)
    nx, ny = pixels_to_normalized(80.0, 10.0, 100, 50)
    assert abs(float(out[28, 0]) - nx) < 1e-5
    assert abs(float(out[28, 1]) - ny) < 1e-5
    np.testing.assert_allclose(out[0, :2], live[0, :2], atol=1e-6)


def test_lab_drag_does_not_freeze_current_keypoints(tmp_path) -> None:
    from backend.tests.test_pose_keys import _bare_runtime

    rest = neutral_keypoints()
    rt = _bare_runtime(tmp_path, rest)
    rt._lab_drive = True
    rt._tracking = True
    rt._mesh_edited = False
    rt._drag_slots = {21}
    rt._drag_xy = (40.0, 20.0)
    rt.status = lambda: {"drive_pose": True}
    calls: list[int] = []

    def overlay():
        calls.append(1)
        live = rest.copy()
        live[0, 0] += 0.2
        return rt._apply_drag_to_overlay(live)

    rt._lab_overlay_keypoints = overlay
    out = rt._current_keypoints()
    assert calls == [1]
    assert out is not None
    np.testing.assert_allclose(out[0, 0], rest[0, 0] + 0.2, atol=1e-6)
