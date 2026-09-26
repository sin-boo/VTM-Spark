from __future__ import annotations

import cv2
import numpy as np

from .presets import MOUTH_SLOTS, MouthBook, empty_weights
from .visemes import _heuristic, _rest


def _set_rest() -> dict[str, float]:
    row = {"open": 0.15, "width": 0.40, "inner": 0.22, "corner": 0.50}
    _rest.reset()
    _rest.use_snapshot(row)
    return row


def test_closed_mouth_is_neutral() -> None:
    weights = _heuristic(_set_rest())
    assert all(abs(value) < 1e-6 for value in weights.values())


def test_tracking_waits_for_set_rest() -> None:
    _rest.reset()
    weights = _heuristic({"open": 0.70, "width": 0.45, "corner": 0.50})
    assert all(abs(value) < 1e-6 for value in weights.values())


def test_open_mouth_reaches_a_without_overshoot() -> None:
    rest = _set_rest()
    weights = _heuristic({**rest, "open": 0.60})
    assert weights["A"] > 0.8
    assert 0.0 <= sum(weights[name] for name in ("A", "I", "U", "E", "O")) <= 1.0


def test_vowels_share_one_bounded_budget() -> None:
    rest = _set_rest()
    spread = _heuristic({**rest, "open": 0.45, "width": 0.52})
    rounded = _heuristic({**rest, "open": 0.45, "width": 0.30})
    assert spread["I"] + spread["E"] > 0.0
    assert rounded["U"] + rounded["O"] > 0.0
    for weights in (spread, rounded):
        assert sum(weights[name] for name in ("A", "I", "U", "E", "O")) <= 1.000001


def test_blank_face_jitter_is_not_a_smile() -> None:
    rest = _set_rest()
    weights = _heuristic({**rest, "corner": rest["corner"] + 0.015})
    assert weights["smile"] < 0.05
    assert weights["sad"] < 0.05


def test_real_smile_moves_smile() -> None:
    rest = _set_rest()
    weights = _heuristic({**rest, "corner": rest["corner"] + 0.14, "width": rest["width"] * 1.10})
    assert weights["smile"] > 0.2


def test_stale_absolute_corner_does_not_force_a_smile() -> None:
    _rest.reset()
    _rest.use_snapshot({"open": 0.15, "width": 0.40, "corner": -2.97251})
    weights = _heuristic({"open": 0.15, "width": 0.40, "corner": -0.55})
    assert weights["smile"] < 0.05
    assert weights["A"] < 0.05


def test_session_closed_mouth_becomes_rest() -> None:
    _rest.reset()
    feat = {"open": 0.18, "width": 0.38, "corner": -0.55}
    for _ in range(12):
        _rest.observe(feat["open"], feat["width"], feat["corner"], True)
    weights = _heuristic(feat)
    assert weights["smile"] < 0.05
    assert weights["A"] < 0.05
    assert _rest.locked


def test_moving_mouth_does_not_become_rest() -> None:
    _rest.reset()
    for i in range(24):
        bump = 0.04 if i % 2 else 0.0
        _rest.observe(0.18 + bump, 0.38 + bump, -0.55 + bump, True)
    assert not _rest.locked
    assert _rest.width_rest is None


def test_saved_poses_wait_for_session_rest() -> None:
    from .calibrate import Calibrator

    cal = Calibrator()
    cal.samples = {
        "rest": {"open": 0.15, "width": 0.40, "corner": 0.50, "lift": 0.0},
        "smile": {"open": 0.16, "width": 0.55, "corner": 0.92, "lift": 0.0},
    }
    _rest.reset()
    weights = cal.weights(
        {"open": 0.16, "width": 0.55, "corner": 0.92, "lift": 0.0},
        _heuristic,
    )
    assert weights["smile"] < 0.05
    assert weights["A"] < 0.05


def test_new_session_blank_face_is_not_old_pose() -> None:
    from .calibrate import Calibrator

    cal = Calibrator()
    cal.samples = {
        "rest": {"open": 0.15, "width": 0.40, "corner": 0.50, "lift": 0.0},
        "smile": {"open": 0.16, "width": 0.55, "corner": 0.92, "lift": 0.0},
        "sad": {"open": 0.16, "width": 0.38, "corner": 0.18, "lift": 0.0},
    }
    session = {"open": 0.18, "width": 0.38, "corner": -0.55, "lift": 0.55}
    _rest.reset()
    _rest.use_snapshot(session)
    weights = cal.weights(session, _heuristic, origin=_rest.snapshot())
    assert weights["smile"] < 0.05
    assert weights["sad"] < 0.05
    assert weights["A"] < 0.05


def test_small_ee_reaches_i_without_a_jaw() -> None:
    rest = _set_rest()
    weights = _heuristic({**rest, "open": rest["open"] + 0.07, "width": rest["width"] * 1.14})
    assert weights["I"] > 0.2
    assert weights["I"] > weights["A"]


def test_small_oo_reaches_u() -> None:
    rest = _set_rest()
    weights = _heuristic({**rest, "open": rest["open"] + 0.07, "width": rest["width"] * 0.82})
    assert weights["U"] > 0.2
    assert weights["U"] > weights["A"]


def test_smile_and_ee_can_run_together() -> None:
    rest = _set_rest()
    weights = _heuristic(
        {
            **rest,
            "open": rest["open"] + 0.08,
            "width": rest["width"] * 1.16,
            "corner": rest["corner"] + 0.14,
        }
    )
    assert weights["smile"] > 0.15
    assert weights["I"] > 0.15


def test_banks_keep_small_vowels_when_smile_is_huge() -> None:
    from .calibrate import Calibrator

    cal = Calibrator()
    cal.samples = {
        "rest": {"open": 0.15, "width": 0.40, "corner": 0.50, "lift": 0.0},
        "smile": {"open": 0.16, "width": 0.55, "corner": 0.92, "lift": 0.0},
        "sad": {"open": 0.16, "width": 0.38, "corner": 0.18, "lift": 0.0},
        "I": {"open": 0.22, "width": 0.46, "corner": 0.52, "lift": 0.0},
        "E": {"open": 0.30, "width": 0.48, "corner": 0.52, "lift": 0.0},
        "U": {"open": 0.22, "width": 0.32, "corner": 0.48, "lift": 0.0},
        "O": {"open": 0.34, "width": 0.30, "corner": 0.48, "lift": 0.0},
    }
    near_i = cal.weights(
        {"open": 0.22, "width": 0.455, "corner": 0.51, "lift": 0.0},
        _heuristic,
        origin=cal.samples["rest"],
    )
    assert near_i["I"] > 0.35
    assert near_i["I"] > near_i["smile"]
    assert near_i["I"] > near_i["U"]
    near_u = cal.weights(
        {"open": 0.22, "width": 0.325, "corner": 0.49, "lift": 0.0},
        _heuristic,
        origin=cal.samples["rest"],
    )
    assert near_u["U"] > 0.35
    assert near_u["U"] > near_u["I"]
    grin = cal.weights(
        {"open": 0.16, "width": 0.54, "corner": 0.90, "lift": 0.0},
        _heuristic,
        origin=cal.samples["rest"],
    )
    assert grin["smile"] > 0.35
    assert grin["I"] < 0.25


def test_mesh_mix_normalizes_bad_external_weights() -> None:
    book = MouthBook()
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    book.shapes["rest"] = rest
    for name in ("A", "I", "U", "E", "O"):
        shape = rest.copy()
        shape[list(MOUTH_SLOTS), 1] = 10.0
        book.shapes[name] = shape
    weights = empty_weights()
    weights.update({"A": 1.0, "I": 1.0, "U": 1.0, "E": 1.0, "O": 1.0})
    mixed = book.mix(weights)
    assert mixed is not None
    assert float(mixed[list(MOUTH_SLOTS), 1].max()) <= 10.000001


def test_live_mouth_reads_mapped_corners() -> None:
    from .mouth_bits import DEFAULT_ON, DEFAULT_TO, bits
    from .visemes import mouth_features

    prev, prev_to = bits.snapshot()
    pts = np.zeros((66, 3), dtype=np.float32)
    pts[0, 0], pts[16, 0] = 1.0, -1.0
    pts[1, 0], pts[15, 0] = 0.9, -0.9
    pts[27, 1], pts[28, 1], pts[29, 1], pts[30, 1] = 1.2, 0.8, 0.4, 0.0
    pts[58] = [-0.40, -0.50, 1.0]
    pts[60] = [0.00, -0.42, 1.0]
    pts[62] = [0.40, -0.50, 1.0]
    pts[64] = [0.00, -0.58, 1.0]
    # Other on lips must not change open/width.
    pts[59] = [-0.90, 2.0, 1.0]
    pts[61] = [0.90, 2.0, 1.0]
    pts[63] = [0.90, 2.0, 1.0]
    pts[65] = [0.00, 2.0, 1.0]
    lms = np.ones((66, 3), dtype=np.float32)
    face = type("Face", (), {"pnp_error": 1.0, "lms": lms, "pts_3d": pts})()
    _rest.reset()
    try:
        bits.restore(frozenset(DEFAULT_ON), dict(DEFAULT_TO))
        feat = mouth_features(face)
        assert feat is not None
        assert abs(feat["open"] - 0.20) < 0.02
        assert "inner" not in feat
    finally:
        bits.restore(prev, prev_to)


def test_stuck_inner_lips_still_read_outer_open() -> None:
    from .mouth_bits import DEFAULT_ON, DEFAULT_TO, bits
    from .visemes import mouth_features

    prev, prev_to = bits.snapshot()
    pts = np.zeros((66, 3), dtype=np.float32)
    pts[0, 0], pts[16, 0] = 1.0, -1.0
    pts[1, 0], pts[15, 0] = 0.9, -0.9
    pts[27, 1], pts[28, 1], pts[29, 1], pts[30, 1] = 1.2, 0.8, 0.4, 0.0
    # Inner ring stays a slit. Outer 51/57 split like a real open mouth.
    pts[58] = [-0.40, -0.50, 1.0]
    pts[60] = [0.00, -0.50, 1.0]
    pts[62] = [0.40, -0.50, 1.0]
    pts[64] = [0.00, -0.51, 1.0]
    pts[51] = [0.00, -0.32, 1.0]
    pts[57] = [0.00, -0.72, 1.0]
    lms = np.ones((66, 3), dtype=np.float32)
    face = type("Face", (), {"pnp_error": 1.0, "lms": lms, "pts_3d": pts})()
    _rest.reset()
    try:
        bits.restore(frozenset(DEFAULT_ON), dict(DEFAULT_TO))
        feat = mouth_features(face)
        assert feat is not None
        assert feat["open"] > 0.35
    finally:
        bits.restore(prev, prev_to)


def test_head_rig_holds_without_camera_pose() -> None:
    from .rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[15] = [50.0, 40.0, 1.0]
    pts = rest.copy()
    pts[20:28, 1] = 8.0
    rig = FaceRig()
    pose = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "ok": 0.0}
    head = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    out = rig.apply(pts, rest, head, pose)
    assert out is not None
    assert np.allclose(out[:, :2], pts[:, :2])


def test_head_rig_yaws_in_place() -> None:
    from .feel import feel
    from .rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[11] = [30.0, 35.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[18] = [70.0, 35.0, 1.0]
    rest[21] = [50.0, 70.0, 1.0]
    pts = rest.copy()
    rig = FaceRig()
    origin = {
        "cx": 200.0,
        "cy": 200.0,
        "bx": 200.0,
        "by": 200.0,
        "scale": 100.0,
        "tilt": 0.0,
        "ok": 1.0,
    }
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_yaw": 1.0})
    try:
        rig.apply(pts, rest, head0, origin)
        # Nose slide from a look must not walk the character.
        shifted = {
            "cx": 260.0,
            "cy": 200.0,
            "bx": 200.0,
            "by": 200.0,
            "scale": 100.0,
            "tilt": 0.0,
            "ok": 1.0,
        }
        out = pts
        for _ in range(12):
            out = rig.apply(pts, rest, head0, shifted)
        assert out is not None
        assert abs(float(out[15, 0]) - float(rest[15, 0])) < 1.0
        assert abs(float(out[15, 1]) - float(rest[15, 1])) < 1.0
        look = FaceRig()
        look.apply(pts, rest, head0, origin)
        turned = pts
        for _ in range(12):
            turned = look.apply(
                pts,
                rest,
                {"pitch": 0.0, "yaw": 35.0, "roll": 0.0},
                origin,
            )
        assert turned is not None
        assert abs(float(turned[15, 0]) - float(rest[15, 0])) < 1.5
        assert abs(float(turned[15, 1]) - float(rest[15, 1])) < 1.5
        d_left = float(turned[0, 0]) - float(rest[0, 0])
        d_right = float(turned[4, 0]) - float(rest[4, 0])
        assert abs(d_left) > 4.0
        assert abs(d_right) > 4.0
        assert abs(d_left - d_right) > 0.5
        rest[21] = [50.0, 70.0, 1.0]
        rest[25] = [50.0, 78.0, 1.0]
        gap0 = abs(float(rest[25, 1] - rest[21, 1]))
        gap_look = look.apply(rest, rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}, origin)
        assert gap_look is not None
        gap1 = abs(float(gap_look[25, 1] - gap_look[21, 1]))
        assert abs(gap1 - gap0) < 1.5
        # Mouth points must retain the same yaw projection as the rest of
        # the face instead of being replaced by a roll-only stamp.
        rest[20] = [38.0, 70.0, 1.0]
        rest[24] = [62.0, 70.0, 1.0]
        mouth_turned = look.apply(
            rest, rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}, origin
        )
        assert mouth_turned is not None
        mouth_left = float(mouth_turned[20, 0]) - float(rest[20, 0])
        mouth_right = float(mouth_turned[24, 0]) - float(rest[24, 0])
        assert abs(mouth_left) > 0.25
        assert abs(mouth_left - mouth_right) > 0.25
    finally:
        feel.update(prev)


def test_project_xy_identity_at_rest() -> None:
    from .rig import project_xy

    xs = np.array([-20.0, 0.0, 22.0, -80.0, 80.0], dtype=np.float64)
    ys = np.array([-12.0, 0.0, 18.0, -10.0, 10.0], dtype=np.float64)
    px, py = project_xy(xs, ys, 0.0, 0.0, 0.0, 50.0)
    np.testing.assert_allclose(px, xs, atol=1e-6)
    np.testing.assert_allclose(py, ys, atol=1e-6)


def test_project_xy_yaw_near_side_grows() -> None:
    from .rig import project_xy

    # Image-left (x < 0) comes closer on +yaw. Height is not foreshortened
    # by the turn, so perspective makes that side larger than rest.
    xs = np.array([-24.0, -24.0, 24.0, 24.0], dtype=np.float64)
    ys = np.array([-8.0, 8.0, -8.0, 8.0], dtype=np.float64)
    px, py = project_xy(xs, ys, np.radians(35.0), 0.0, 0.0, 50.0)
    rest_h = 16.0
    near_h = abs(float(py[1] - py[0]))
    far_h = abs(float(py[3] - py[2]))
    assert near_h > rest_h
    assert far_h < rest_h
    assert near_h > far_h


def test_project_xy_hair_does_not_balloon() -> None:
    from .rig import project_xy

    # Spikes sit well outside the face disk. A turn must tuck them, not
    # perspective-scale the raw offset from the nose.
    xs = np.array([-90.0, 90.0], dtype=np.float64)
    ys = np.array([0.0, 0.0], dtype=np.float64)
    px, _py = project_xy(xs, ys, np.radians(35.0), 0.0, 0.0, 50.0)
    near = abs(float(px[0]))
    far = abs(float(px[1]))
    assert near < 90.0 * 1.20
    assert far < 90.0
    assert far < near


def test_plane_xy_identity_at_rest() -> None:
    from .rig import plane_xy

    xs = np.array([-40.0, 0.0, 55.0, -90.0], dtype=np.float64)
    ys = np.array([-12.0, 8.0, 20.0, -70.0], dtype=np.float64)
    px, py = plane_xy(xs, ys, 0.0, 0.0, 0.0, 50.0)
    np.testing.assert_allclose(px, xs, atol=1e-6)
    np.testing.assert_allclose(py, ys, atol=1e-6)


def test_plane_xy_does_not_fold_a_silhouette_bump() -> None:
    from .rig import plane_xy, project_xy

    # Tip sits farther from the nose than its neighbors. Sphere-projecting
    # each vertex independently folds that bump into a needle.
    xs = np.array([173.1, 194.1, 188.4], dtype=np.float64)
    ys = np.array([56.2, 73.3, 22.1], dtype=np.float64)
    spx, spy = project_xy(xs, ys, np.radians(25.0), 0.0, 0.0, 220.0)
    sphere = np.stack([spx, spy], axis=1)

    def _ang(p: np.ndarray) -> float:
        v1 = p[0] - p[1]
        v2 = p[2] - p[1]
        e1 = float(np.linalg.norm(v1))
        e2 = float(np.linalg.norm(v2))
        return float(
            np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (e1 * e2), -1.0, 1.0)))
        )

    rest = np.stack([xs, ys], axis=1)
    assert _ang(sphere) < 8.0
    px, py = plane_xy(xs, ys, np.radians(25.0), 0.0, 0.0, 220.0)
    card = np.stack([px, py], axis=1)
    assert _ang(card) > _ang(rest) * 0.6
    assert _ang(card) > 20.0


def test_project_xy_look_up_chin_grows() -> None:
    from .rig import project_xy

    # OSF pitch+ is look-down: chin tucks, brow is more exposed.
    xs = np.array([0.0, 0.0], dtype=np.float64)
    ys = np.array([-20.0, 24.0], dtype=np.float64)
    _down_x, down_y = project_xy(xs, ys, 0.0, np.radians(28.0), 0.0, 50.0)
    assert abs(float(down_y[1])) < 24.0
    assert abs(float(down_y[0])) > 20.0
    _up_x, up_y = project_xy(xs, ys, 0.0, np.radians(-28.0), 0.0, 50.0)
    assert abs(float(up_y[1])) > 24.0
    assert abs(float(up_y[0])) < 20.0


def _hold_distance(rig, pts, rest, head, pose, n: int = 12):
    """Repeat one distance until the rig accepts it as rest size."""
    out = pts
    for _ in range(n):
        out = rig.apply(pts, rest, head, pose)
    return out


def _drive_head(
    rest: np.ndarray, head: dict[str, float], *, selfie: bool = False
) -> np.ndarray:
    from .feel import feel
    from .rig import FaceRig

    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0})
    try:
        rig = FaceRig()
        rig.selfie = selfie
        rig.apply(rest, rest, head0, origin)
        out = rest
        for _ in range(12):
            out = rig.apply(rest, rest, head, origin)
    finally:
        feel.update(prev)
    assert out is not None
    return out


def test_head_rig_yaw_keeps_both_eyes() -> None:
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0] = [10.0, 32.0, 1.0]
    rest[4] = [90.0, 32.0, 1.0]
    rest[11] = [26.0, 38.0, 1.0]
    rest[12] = [30.0, 28.0, 1.0]
    rest[13] = [34.0, 38.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[17] = [66.0, 38.0, 1.0]
    rest[18] = [70.0, 28.0, 1.0]
    rest[19] = [74.0, 38.0, 1.0]
    turned = _drive_head(rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0})
    assert abs(float(turned[15, 0]) - 50.0) < 1.5
    rest_left = abs(float(rest[12, 1] - rest[11, 1]))
    rest_right = abs(float(rest[18, 1] - rest[17, 1]))
    near = abs(float(turned[12, 1] - turned[11, 1]))
    far = abs(float(turned[18, 1] - turned[17, 1]))
    # A turn may show which side is nearer. It must not resize an eye
    # away from the drawing.
    assert near > far
    assert abs(near - rest_left) < 0.25 * rest_left
    assert abs(far - rest_right) < 0.25 * rest_right


def test_ceiling_look_stops_at_the_look_up_cap() -> None:
    """A look at the ceiling is a large euler. The look-up limiter holds it."""
    import math

    from .feel import feel
    from .rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[5] = [50.0, 12.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_pitch_up": 12.0 / 50.0, "max_pitch_down": 1.0})
    try:
        rig = FaceRig()
        rig.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        held = rest
        for _ in range(8):
            held = rig.apply(rest, rest, {"pitch": -70.0, "yaw": 0.0, "roll": 0.0}, origin)
        assert held is not None
        assert abs(abs(math.degrees(rig._pitch_r)) - 12.0) < 0.05
        assert float(held[2, 1]) > float(held[15, 1])
    finally:
        feel.update(prev)


def test_head_rig_look_keeps_the_jaw() -> None:
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[5] = [50.0, 12.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    down = _drive_head(rest, {"pitch": 28.0, "yaw": 0.0, "roll": 0.0})
    rest_chin = abs(float(rest[2, 1] - rest[15, 1]))
    rest_brow = abs(float(rest[15, 1] - rest[5, 1]))
    chin_down = abs(float(down[2, 1] - down[15, 1]))
    brow_down = abs(float(down[15, 1] - down[5, 1]))
    # A nod foreshortens. It must not cave the chin into the mouth.
    assert chin_down < rest_chin
    assert chin_down > 0.75 * rest_chin
    assert brow_down > 0.75 * rest_brow
    assert float(down[5, 1]) < float(down[15, 1]) < float(down[2, 1])
    up = _drive_head(rest, {"pitch": -28.0, "yaw": 0.0, "roll": 0.0})
    chin_up = abs(float(up[2, 1] - up[15, 1]))
    brow_up = abs(float(up[15, 1] - up[5, 1]))
    assert 0.75 * rest_chin < chin_up < 1.15 * rest_chin
    assert 0.75 * rest_brow < brow_up < 1.15 * rest_brow
    assert float(up[5, 1]) < float(up[15, 1]) < float(up[2, 1])


def test_stepping_back_stops_at_the_size_limiter() -> None:
    """Walking away from the camera shrank the face to 62% and off the art."""
    from .feel import feel
    from .rig import FaceRig
    from .travel_box import lab_feel_caps

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[15] = [50.0, 40.0, 1.0]
    near = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tz": 50.0, "tilt": 0.0, "ok": 1.0}
    far = dict(near, tz=90.0, scale=55.0)
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, **lab_feel_caps({"size": 0.1, "enabled": True})})
    try:
        rig = FaceRig()
        _hold_distance(rig, rest, rest, head0, near, n=20)
        assert abs(rig._s - 1.0) < 1e-6
        for _ in range(4):
            rig.apply(rest, rest, head0, far)
        held = rig._s
        feel.update(lab_feel_caps({"enabled": False}))
        rig.apply(rest, rest, head0, far)
        free = rig._s
    finally:
        feel.update(prev)
    assert held > 0.85
    assert held < 0.9
    assert free < 0.62 + 1e-6


def test_look_down_does_not_fold_the_arm_into_the_head() -> None:
    """The look-down limiter used to sphere-project the torso. The arm vanished."""
    import math

    from .feel import feel
    from .rig import FaceRig
    from .skeleton import follow_skeleton, skeleton_from_face

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0] = [10.0, 32.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    rest[4] = [90.0, 32.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[21] = [50.0, 62.0, 1.0]
    body = skeleton_from_face(rest)
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_pitch_down": 1.0})
    try:
        rig = FaceRig()
        rig.apply(rest, rest, head0, origin)
        parked = {int(j["id"]): j for j in follow_skeleton(body, rest, rig=rig)}
        for _ in range(12):
            rig.apply(rest, rest, {"pitch": 32.0, "yaw": 0.0, "roll": 0.0}, origin)
        turned = {int(j["id"]): j for j in follow_skeleton(body, rest, rig=rig)}
    finally:
        feel.update(prev)

    def _len(pose: dict, a: int, b: int) -> float:
        return math.hypot(pose[a]["x"] - pose[b]["x"], pose[a]["y"] - pose[b]["y"])

    rest_arm = _len(parked, 32, 33)
    live_arm = _len(turned, 32, 33)
    assert rest_arm > 10.0
    assert live_arm > 0.7 * rest_arm


def test_selfie_flips_turn() -> None:
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0] = [10.0, 32.0, 1.0]
    rest[4] = [90.0, 32.0, 1.0]
    rest[11] = [26.0, 38.0, 1.0]
    rest[12] = [30.0, 28.0, 1.0]
    rest[13] = [34.0, 38.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[17] = [66.0, 38.0, 1.0]
    rest[18] = [70.0, 28.0, 1.0]
    rest[19] = [74.0, 38.0, 1.0]
    natural = _drive_head(rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0})
    flipped = _drive_head(rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}, selfie=True)
    near = abs(float(natural[12, 1] - natural[11, 1]))
    far = abs(float(natural[18, 1] - natural[17, 1]))
    flip_near = abs(float(flipped[12, 1] - flipped[11, 1]))
    flip_far = abs(float(flipped[18, 1] - flipped[17, 1]))
    assert near > far
    assert flip_far > flip_near


def test_toggle_selfie_flips_live_turn_without_reset() -> None:
    from .feel import feel
    from .rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0] = [10.0, 32.0, 1.0]
    rest[4] = [90.0, 32.0, 1.0]
    rest[11] = [26.0, 38.0, 1.0]
    rest[12] = [30.0, 28.0, 1.0]
    rest[13] = [34.0, 38.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[17] = [66.0, 38.0, 1.0]
    rest[18] = [70.0, 28.0, 1.0]
    rest[19] = [74.0, 38.0, 1.0]
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    head = {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}
    prev = feel.payload()
    try:
        feel.update({"smoothing": 0.0})
        rig = FaceRig()
        rig.apply(rest, rest, head0, origin)
        left = rest
        for _ in range(12):
            left = rig.apply(rest, rest, head, origin)
        assert left is not None
        rig.selfie = True
        right = rig.apply(rest, rest, head, origin)
        assert right is not None
        assert rig.locked
    finally:
        feel.update(prev)
    left_l = abs(float(left[12, 1] - left[11, 1]))
    left_r = abs(float(left[18, 1] - left[17, 1]))
    right_l = abs(float(right[12, 1] - right[11, 1]))
    right_r = abs(float(right[18, 1] - right[17, 1]))
    assert left_l > left_r
    assert right_r > right_l


def test_selfie_keeps_pitch() -> None:
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[5] = [50.0, 12.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    natural = _drive_head(rest, {"pitch": -28.0, "yaw": 0.0, "roll": 0.0})
    flipped = _drive_head(rest, {"pitch": -28.0, "yaw": 0.0, "roll": 0.0}, selfie=True)
    assert np.allclose(natural[:, :2], flipped[:, :2], atol=1e-4)


def test_feel_max_yaw_blocks_turn() -> None:
    from .feel import feel

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    rest[11] = [20.0, 28.0, 1.0]
    rest[18] = [80.0, 28.0, 1.0]
    prev = feel.payload()
    try:
        free = _drive_head(rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0})
        feel.update({"max_yaw": 0.0})
        held = _drive_head(rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0})
    finally:
        feel.update(prev)
    free_span = abs(float(free[18, 0] - free[11, 0]) - 60.0)
    held_span = abs(float(held[18, 0] - held[11, 0]) - 60.0)
    assert free_span > held_span
    assert free_span > 0.5
    assert held_span < 0.2


def test_turn_and_tilt_stop_each_side_on_its_own() -> None:
    """Right is positive yaw / roll: a closed right side must not hold the left."""
    import math

    from .feel import feel
    from .rig import FaceRig
    from .travel_box import lab_feel_caps

    rest, _osf = _toy_face()
    origin = {"cx": 200.0, "cy": 200.0, "bx": 200.0, "by": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    box = {"turn_left": 40.0, "turn_right": 0.0, "tilt_left": 0.0, "tilt_right": 40.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, **lab_feel_caps(box)})

    def drive(yaw: float, roll: float) -> tuple[float, float]:
        rig = FaceRig()
        rig.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        moved = dict(origin, tilt=roll)
        for _ in range(8):
            rig.apply(rest, rest, {"pitch": 0.0, "yaw": yaw, "roll": roll}, moved)
        return math.degrees(rig._yaw_r), math.degrees(rig._roll_r)

    try:
        right_yaw, left_roll = drive(20.0, -20.0)
        left_yaw, right_roll = drive(-20.0, 20.0)
    finally:
        feel.update(prev)
    assert abs(right_yaw) < 0.5 and abs(left_roll) < 0.5
    assert abs(left_yaw + 20.0) < 1.0 and abs(right_roll - 20.0) < 1.0


def test_old_max_yaw_sets_both_sides() -> None:
    from .feel import feel

    prev = feel.payload()
    try:
        feel.update({"max_yaw": 0.25, "max_roll": 0.5})
        assert feel.max_yaw() == (0.25, 0.25)
        assert feel.max_roll() == (0.5, 0.5)
    finally:
        feel.update(prev)


def test_face_place_moves_and_scales_the_character() -> None:
    from .feel import feel
    from .rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    pts = rest.copy()
    origin = {
        "cx": 200.0,
        "cy": 200.0,
        "bx": 200.0,
        "by": 200.0,
        "scale": 100.0,
        "tilt": 0.0,
        "ok": 1.0,
    }
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_size": 1.0})
    try:
        walk = FaceRig()
        walk.apply(pts, rest, head0, origin)
        slid = dict(origin)
        slid["bx"] = 260.0
        out = pts
        for _ in range(12):
            out = walk.apply(pts, rest, head0, slid)
        assert out is not None
        # A step in the camera carries the chin with the nose. The jaw stays one piece.
        nose_dx = float(out[15, 0]) - float(rest[15, 0])
        chin_dx = float(out[2, 0]) - float(rest[2, 0])
        nose_dy = float(out[15, 1]) - float(rest[15, 1])
        chin_dy = float(out[2, 1]) - float(rest[2, 1])
        assert abs(nose_dx) > 5.0
        assert abs(chin_dx - nose_dx) < 1.5
        assert abs(chin_dy - nose_dy) < 1.5
        zoom = FaceRig()
        _hold_distance(zoom, pts, rest, head0, origin)
        closer = dict(origin)
        closer["scale"] = 140.0
        grown = pts
        for _ in range(12):
            grown = zoom.apply(pts, rest, head0, closer)
        assert grown is not None
        rest_h = float(rest[2, 1] - rest[15, 1])
        live_h = float(grown[2, 1] - grown[15, 1])
        assert live_h > rest_h * 1.15
        # With PnP distance, a turn that shrinks the 2D box is not a zoom.
        far = dict(origin)
        far["tz"] = 500.0
        steady = FaceRig()
        _hold_distance(steady, pts, rest, head0, far)
        profile = dict(far)
        profile["scale"] = 60.0
        held = pts
        for _ in range(12):
            held = steady.apply(pts, rest, {"pitch": 0.0, "yaw": 40.0, "roll": 0.0}, profile)
        assert held is not None
        assert abs(float(steady._s) - 1.0) < 0.02
        nearer = dict(far)
        nearer["tz"] = 500.0 / 1.3
        for _ in range(12):
            steady.apply(pts, rest, head0, nearer)
        assert steady._s > 1.2
    finally:
        feel.update(prev)


def test_opening_distance_does_not_inflate_the_overlay() -> None:
    from .feel import feel
    from .rig import FaceRig
    from .visemes import _rest

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    head = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    far = {
        "cx": 200.0,
        "cy": 200.0,
        "bx": 200.0,
        "by": 200.0,
        "scale": 40.0,
        "tz": 900.0,
        "tilt": 0.0,
        "ok": 1.0,
    }
    home = dict(far)
    home["scale"] = 100.0
    home["tz"] = 500.0
    prev = feel.payload()
    saved = _rest.snapshot()
    feel.update({"smoothing": 0.0, "max_size": 1.0})
    _rest.reset()
    try:
        rig = FaceRig()
        rig.apply(rest, rest, head, far)
        assert rig._s == 1.0
        for _ in range(12):
            rig.apply(rest, rest, head, home)
        assert rig._size_ready
        assert abs(rig._s - 1.0) < 0.02
        closer = dict(home)
        closer["tz"] = 500.0 / 1.4
        for _ in range(4):
            rig.apply(rest, rest, head, closer)
        assert 1.2 < rig._s < 1.65
    finally:
        feel.update(prev)
        _rest.reset()
        if saved:
            _rest.use_snapshot(saved)


def test_set_rest_seals_size_on_the_face_in_hand() -> None:
    from .feel import feel
    from .rig import FaceRig
    from .visemes import _rest

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    head = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    far = {
        "cx": 200.0,
        "cy": 200.0,
        "bx": 200.0,
        "by": 200.0,
        "scale": 40.0,
        "tilt": 0.0,
        "ok": 1.0,
    }
    home = dict(far)
    home["scale"] = 100.0
    prev = feel.payload()
    saved = _rest.snapshot()
    feel.update({"smoothing": 0.0, "max_size": 1.0})
    _rest.reset()
    try:
        rig = FaceRig()
        rig.apply(rest, rest, head, far)
        assert not rig._size_ready
        _rest.use_snapshot({"open": 0.12, "width": 0.4, "corner": 0.0})
        rig.apply(rest, rest, head, home)
        assert rig._size_ready
        assert abs(rig._s - 1.0) < 0.02
        closer = dict(home)
        closer["scale"] = 140.0
        rig.apply(rest, rest, head, closer)
        assert rig._s > 1.2
    finally:
        feel.update(prev)
        _rest.reset()
        if saved:
            _rest.use_snapshot(saved)


def test_head_rig_look_down_keeps_hair_above_nose() -> None:
    from .feel import feel
    from .rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[5] = [50.0, 8.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0})
    try:
        look = FaceRig()
        look.apply(rest, rest, head0, origin)
        down = rest
        for _ in range(12):
            down = look.apply(rest, rest, {"pitch": 55.0, "yaw": 0.0, "roll": 0.0}, origin)
        assert down is not None
        assert float(down[5, 1]) < float(down[15, 1]) - 6.0
        assert float(down[15, 1]) < float(down[2, 1])
        flipped = FaceRig()
        flipped.apply(rest, rest, head0, origin)
        held = rest
        for _ in range(8):
            held = flipped.apply(rest, rest, {"pitch": 8.0, "yaw": 0.0, "roll": 0.0}, origin)
        for _ in range(8):
            held = flipped.apply(rest, rest, {"pitch": 150.0, "yaw": 0.0, "roll": 0.0}, origin)
        assert held is not None
        assert float(held[5, 1]) < float(held[15, 1])
    finally:
        feel.update(prev)


def test_back_at_center_rezeros_drifted_open() -> None:
    _rest.reset()
    home = dict(opened=0.18, width=0.40, corner=0.0)
    for _ in range(14):
        _rest.observe(home["opened"], home["width"], home["corner"], True, yaw=0.0, cx=200.0, scale=100.0)
    assert _rest.locked
    rest_o = float(_rest.open_rest or 0.0)
    for _ in range(10):
        _rest.observe(0.30, 0.40, 0.0, True, yaw=40.0, cx=80.0, scale=100.0)
    assert abs(float(_rest.open_rest or 0.0) - rest_o) < 0.02
    drifted = {**home, "opened": 0.26}
    for _ in range(10):
        _rest.observe(drifted["opened"], drifted["width"], drifted["corner"], True, yaw=0.0, cx=200.0, scale=100.0)
    weights = _heuristic(
        {"open": drifted["opened"], "width": drifted["width"], "corner": drifted["corner"]}
    )
    assert weights["A"] < 0.15


def _toy_face() -> tuple[np.ndarray, np.ndarray]:
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[2] = [50.0, 90.0, 1.0]
    rest[5] = [18.0, 22.0, 1.0]
    rest[6] = [30.0, 20.0, 1.0]
    rest[7] = [42.0, 21.0, 1.0]
    rest[8] = [58.0, 21.0, 1.0]
    rest[9] = [70.0, 20.0, 1.0]
    rest[10] = [82.0, 22.0, 1.0]
    rest[11] = [28.0, 36.0, 1.0]
    rest[12] = [32.0, 34.0, 1.0]
    rest[13] = [36.0, 36.0, 1.0]
    rest[14] = [43.0, 50.0, 1.0]
    rest[15] = [50.0, 48.0, 1.0]
    rest[16] = [57.0, 50.0, 1.0]
    rest[17] = [64.0, 36.0, 1.0]
    rest[18] = [68.0, 34.0, 1.0]
    rest[19] = [72.0, 36.0, 1.0]
    rest[21] = [50.0, 70.0, 1.0]
    rest[25] = [50.0, 78.0, 1.0]
    # dlib order on an unflipped frame: 0 / 17-21 / 36-41 / 31 are image-left.
    osf = np.zeros((66, 3), dtype=np.float32)
    osf[0, 0], osf[16, 0] = -1.0, 1.0
    osf[4] = [-0.7, 0.8, 0.0]
    osf[12] = [0.7, 0.8, 0.0]
    osf[8] = [0.0, 1.2, 0.0]
    osf[17] = [-0.4, -1.0, 0.0]
    osf[18] = [-0.3, -1.05, 0.0]
    osf[19] = [-0.2, -1.1, 0.0]
    osf[20] = [-0.12, -1.05, 0.0]
    osf[21] = [-0.05, -1.0, 0.0]
    osf[22] = [0.05, -1.0, 0.0]
    osf[23] = [0.12, -1.05, 0.0]
    osf[24] = [0.2, -1.1, 0.0]
    osf[25] = [0.30, -1.05, 0.0]
    osf[26] = [0.4, -1.0, 0.0]
    osf[30] = [0.0, 0.0, 0.0]
    osf[31] = [-0.15, 0.05, 0.0]
    osf[35] = [0.15, 0.05, 0.0]
    osf[36], osf[39] = [-0.45, -0.35, 0.0], [-0.25, -0.35, 0.0]
    osf[37], osf[38] = [-0.40, -0.42, 0.0], [-0.30, -0.42, 0.0]
    osf[41], osf[40] = [-0.40, -0.28, 0.0], [-0.30, -0.28, 0.0]
    osf[42], osf[45] = [0.25, -0.35, 0.0], [0.45, -0.35, 0.0]
    osf[43], osf[44] = [0.30, -0.42, 0.0], [0.40, -0.42, 0.0]
    osf[47], osf[46] = [0.30, -0.28, 0.0], [0.40, -0.28, 0.0]
    return rest, osf


def test_face_expr_still_pose_does_not_move_mesh() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    mixed = rest.copy()
    mixed[21, 1] = 74.0
    expr = FaceExpr()
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        out = expr.apply(mixed, rest, osf, {"l": 0.0, "r": 0.0})
        out = expr.apply(mixed, rest, osf, {"l": 0.0, "r": 0.0})
        assert out is not None
        assert np.allclose(out[:20, :2], rest[:20, :2], atol=0.05)
        assert abs(float(out[21, 1]) - float(rest[21, 1])) < 1e-5
    finally:
        feel.update(prev)


def test_face_expr_brow_and_blink_move_eyes_not_mouth() -> None:
    from .eye_bits import bits as eyes
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    live = osf.copy()
    live[[17, 19, 21, 22, 24, 26], 1] -= 0.35
    expr = FaceExpr()
    prev = feel.payload()
    eye_snap = eyes.snapshot()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        eyes.restore(frozenset(), {})
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0})
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, live, {"l": 0.8, "r": 0.0})
        assert out is not None
        assert float(out[6, 1]) < float(rest[6, 1]) - 1.0
        assert float(out[9, 1]) < float(rest[9, 1]) - 1.0
        assert float(out[12, 1]) > float(rest[12, 1])
        assert np.allclose(out[20:28, :2], rest[20:28, :2], atol=0.05)
    finally:
        feel.update(prev)
        eyes.restore(*eye_snap)


def _camera_from_osf(osf: np.ndarray) -> np.ndarray:
    camera = np.zeros((68, 2), dtype=np.float32)
    n = min(66, len(osf))
    camera[:n] = osf[:n, :2]
    if n > 38:
        camera[66] = 0.5 * (camera[37] + camera[38])
    if n > 44:
        camera[67] = 0.5 * (camera[43] + camera[44])
    return camera


def test_camera_brow_down_moves_character_brows_down() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    camera = _camera_from_osf(osf)
    live_osf = osf.copy()
    live_osf[list(range(17, 27)), 1] += 0.22
    live_cam = _camera_from_osf(live_osf)
    expr = FaceExpr()
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0}, mouth_pts=camera)
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, live_osf, {"l": 0.0, "r": 0.0}, mouth_pts=live_cam)
        assert out is not None
        assert float(out[6, 1]) > float(rest[6, 1]) + 1.0
        assert float(out[9, 1]) > float(rest[9, 1]) + 1.0
    finally:
        feel.update(prev)


def test_head_turn_in_camera_does_not_move_brows() -> None:
    """Image brows foreshorten on a yaw. Pose-free 3D brows do not. Brows must not move."""
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    camera = _camera_from_osf(osf)
    turned_cam = camera.copy()
    turned_cam[:, 0] *= 0.55
    expr = FaceExpr()
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0}, mouth_pts=camera)
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0}, mouth_pts=turned_cam)
        assert out is not None
        assert np.allclose(out[5:11, :2], rest[5:11, :2], atol=0.3)
        assert np.allclose(out[11:14, :2], rest[11:14, :2], atol=0.3)
        assert np.allclose(out[17:20, :2], rest[17:20, :2], atol=0.3)
    finally:
        feel.update(prev)


def test_osf_y_up_frown_moves_character_brows_down() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    osf = osf.copy()
    osf[:, 1] *= -1.0
    live = osf.copy()
    live[list(range(17, 27)), 1] -= 0.35
    expr = FaceExpr()
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0})
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, live, {"l": 0.0, "r": 0.0})
        assert out is not None
        assert float(out[6, 1]) > float(rest[6, 1]) + 1.0
        assert float(out[9, 1]) > float(rest[9, 1]) + 1.0
    finally:
        feel.update(prev)


def test_camera_left_brow_drives_character_left_brow() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    osf = osf.copy()
    osf[:, 0] *= -1.0
    camera = _camera_from_osf(osf)
    live_osf = osf.copy()
    live_osf[list(range(17, 22)), 1] += 0.24
    live_cam = _camera_from_osf(live_osf)
    expr = FaceExpr()
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0}, mouth_pts=camera)
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, live_osf, {"l": 0.0, "r": 0.0}, mouth_pts=live_cam)
        assert out is not None
        assert float(out[6, 1]) > float(rest[6, 1]) + 1.0
        assert float(out[9, 1]) < float(rest[9, 1]) + 0.4
    finally:
        feel.update(prev)


def test_head_pitch_does_not_invert_camera_brow_down() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    camera = _camera_from_osf(osf)
    live_osf = osf.copy()
    live_osf[list(range(17, 27)), 1] += 0.22
    live_cam = _camera_from_osf(live_osf)
    posed = rest.copy()
    posed[5:11, 1] -= 8.0
    posed[11:14, 1] -= 8.0
    posed[17:20, 1] -= 8.0
    expr = FaceExpr()
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0}, mouth_pts=camera)
        out = posed
        for _ in range(8):
            expr.apply(rest, rest, live_osf, {"l": 0.0, "r": 0.0}, mouth_pts=live_cam)
            out = expr.place_brows(posed.copy(), rest)
        assert out is not None
        assert float(out[6, 1]) > float(posed[6, 1]) + 1.0
        assert float(out[9, 1]) > float(posed[9, 1]) + 1.0
    finally:
        feel.update(prev)


def test_osf_lids_drive_anime_12_and_18() -> None:
    from .eye_bits import DEFAULT_ON, DEFAULT_TO, LID_MID_L, LID_MID_R, bits as eyes
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    camera = np.zeros((68, 2), dtype=np.float32)
    camera[:66] = osf[:, :2]
    camera[LID_MID_R] = 0.5 * (osf[37, :2] + osf[38, :2])
    camera[LID_MID_L] = 0.5 * (osf[43, :2] + osf[44, :2])
    live_osf = osf.copy()
    live_osf[43, 1] += 0.20
    live_osf[44, 1] += 0.20
    live_osf[37, 1] += 0.20
    live_osf[38, 1] += 0.20
    live_osf[36, 0] += 0.15
    live_osf[39, 0] -= 0.15
    live_cam = np.zeros((68, 2), dtype=np.float32)
    live_cam[:66] = live_osf[:, :2]
    live_cam[LID_MID_R] = 0.5 * (live_osf[37, :2] + live_osf[38, :2])
    live_cam[LID_MID_L] = 0.5 * (live_osf[43, :2] + live_osf[44, :2])
    expr = FaceExpr()
    prev = feel.payload()
    snap = eyes.snapshot()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        eyes.restore(frozenset(DEFAULT_ON), dict(DEFAULT_TO))
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0}, mouth_pts=camera)
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, live_osf, {"l": 0.0, "r": 0.0}, mouth_pts=live_cam)
        assert out is not None
        assert float(out[12, 1]) > float(rest[12, 1]) + 0.4
        assert float(out[18, 1]) > float(rest[18, 1]) + 0.4
        # Canonical: image-left eye outer 36 / inner 39 drive slots 11 / 13.
        assert float(out[11, 0]) > float(rest[11, 0]) + 0.2
        assert float(out[13, 0]) < float(rest[13, 0]) - 0.2
    finally:
        feel.update(prev)
        eyes.restore(*snap)


def test_crossed_eye_maps_still_drive_dest_slots() -> None:
    from .eye_bits import LID_MID_L, LID_MID_R, bits as eyes
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    camera = np.zeros((68, 2), dtype=np.float32)
    camera[:66] = osf[:, :2]
    camera[LID_MID_R] = 0.5 * (osf[37, :2] + osf[38, :2])
    camera[LID_MID_L] = 0.5 * (osf[43, :2] + osf[44, :2])
    live_osf = osf.copy()
    live_osf[37, 1] += 0.22
    live_osf[38, 1] += 0.22
    live_osf[43, 1] += 0.22
    live_osf[44, 1] += 0.22
    live_osf[36, 0] += 0.18
    live_cam = np.zeros((68, 2), dtype=np.float32)
    live_cam[:66] = live_osf[:, :2]
    live_cam[LID_MID_R] = 0.5 * (live_osf[37, :2] + live_osf[38, :2])
    live_cam[LID_MID_L] = 0.5 * (live_osf[43, :2] + live_osf[44, :2])
    expr = FaceExpr()
    prev = feel.payload()
    snap = eyes.snapshot()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        eyes.restore(
            frozenset((36, 39, 42, 45, LID_MID_R, LID_MID_L)),
            {36: 11, 39: 13, LID_MID_R: 12, 42: 17, 45: 19, LID_MID_L: 18},
        )
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0}, mouth_pts=camera)
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, live_osf, {"l": 0.0, "r": 0.0}, mouth_pts=live_cam)
        assert out is not None
        assert float(out[12, 1]) > float(rest[12, 1]) + 0.4
        assert float(out[18, 1]) > float(rest[18, 1]) + 0.4
        assert float(out[11, 0]) > float(rest[11, 0]) + 0.15
    finally:
        feel.update(prev)
        eyes.restore(*snap)


def test_osf_58_drives_anime_23() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    rest[23] = [40.0, 72.0, 1.0]
    osf[58] = [-0.25, 0.55, 0.0]
    live = osf.copy()
    live[58, 1] += 0.40
    mixed = rest.copy()
    mixed[21, 1] = 74.0
    expr = FaceExpr()
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    from .mouth_bits import bits

    snap = bits.snapshot()
    try:
        bits.set_on(58, True)
        bits.set_to(58, 23)
        expr.apply(mixed, rest, osf, {"l": 0.0, "r": 0.0})
        out = mixed
        for _ in range(8):
            out = expr.apply(mixed, rest, live, {"l": 0.0, "r": 0.0})
        assert out is not None
        assert float(out[23, 1]) < float(rest[23, 1]) - 1.0
    finally:
        feel.update(prev)
        bits.restore(*snap)


def test_mapped_lips_use_visible_2d_motion() -> None:
    from .feel import feel
    from .mouth_bits import bits
    from .retarget import FaceExpr, dest_mouth_width, fit_mouth_box

    rest, osf = _toy_face()
    rest[23] = [40.0, 72.0, 1.0]
    rest[26] = [60.0, 72.0, 1.0]
    camera_rest = np.zeros((66, 3), dtype=np.float32)
    camera_rest[:, 2] = 1.0
    camera_rest[58] = [-0.4, 0.2, 1.0]
    camera_rest[62] = [0.6, 0.2, 1.0]
    camera_live = camera_rest.copy()
    camera_live[58] += [0.1, 0.3, 0.0]
    camera_live[62] += [-0.1, -0.1, 0.0]
    src = np.array([camera_rest[58, :2], camera_rest[62, :2]], dtype=np.float32)
    src_c = 0.5 * (src.min(axis=0) + src.max(axis=0))
    expr = FaceExpr()
    prev_feel = feel.payload()
    prev_bits = bits.snapshot()
    feel.update(
        {
            "smoothing": 0.0,
            "response": 1.0,
            "mouth": 0.5,
            "mouth_x": 1.0,
            "mouth_y": 1.0,
            "mouth_z": 0.5,
            "box_x": 0.5,
            "box_y": 0.5,
        }
    )
    dest_c, _dw, _dh, scale = fit_mouth_box(
        rest,
        float(np.ptp(src[:, 0])),
        max(float(np.ptp(src[:, 1])), 1e-6),
        dest_mouth_width(rest),
    )
    try:
        bits.restore(frozenset((58, 62)), {58: 23, 62: 26})
        expr.apply(rest, rest, osf, mouth_pts=camera_rest)
        out = expr.apply(rest, rest, osf, mouth_pts=camera_live)
        assert out is not None
        assert np.allclose(out[23, :2], dest_c + (camera_live[58, :2] - src_c) * scale)
        assert np.allclose(out[26, :2], dest_c + (camera_live[62, :2] - src_c) * scale)
        box = expr.mouth_box()
        assert box is not None
        assert box[2] > 1.0 and box[3] > 0.0
    finally:
        feel.update(prev_feel)
        bits.restore(*prev_bits)


def test_mapped_lips_keep_camera_shape() -> None:
    from .feel import feel
    from .mouth_bits import bits
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    camera_rest = np.zeros((66, 3), dtype=np.float32)
    camera_rest[:, 2] = 1.0
    camera_rest[58] = [-0.4, 0.2, 1.0]
    camera_rest[60] = [0.0, 0.05, 1.0]
    camera_rest[62] = [0.6, 0.2, 1.0]
    camera_live = camera_rest.copy()
    camera_live[58] += [0.05, 0.25, 0.0]
    camera_live[60] += [0.0, 0.40, 0.0]
    camera_live[62] += [-0.05, 0.20, 0.0]
    expr = FaceExpr()
    prev_feel = feel.payload()
    prev_bits = bits.snapshot()
    feel.update(
        {
            "smoothing": 0.0,
            "response": 1.0,
            "mouth": 0.5,
            "mouth_x": 1.0,
            "mouth_y": 1.0,
            "mouth_z": 0.5,
        }
    )
    try:
        bits.restore(frozenset((58, 60, 62)), {58: 23, 60: 21, 62: 26})
        parked = expr.apply(rest, rest, osf, mouth_pts=camera_rest)
        out = expr.apply(rest, rest, osf, mouth_pts=camera_live)
        assert parked is not None and out is not None
        cam = camera_live[[58, 60, 62], :2] - camera_rest[[58, 60, 62], :2]
        moved = out[[23, 21, 26], :2] - parked[[23, 21, 26], :2]
        scales = moved / np.where(np.abs(cam) < 1e-8, 1.0, cam)
        usable = np.abs(cam) >= 1e-8
        assert np.allclose(scales[usable], scales[usable][0])
        cam_span = camera_live[62, :2] - camera_live[58, :2]
        out_span = out[26, :2] - out[23, :2]
        assert np.allclose(out_span / cam_span, scales[usable][0])
    finally:
        feel.update(prev_feel)
        bits.restore(*prev_bits)


def test_mouth_auto_fits_saved_rest_and_keeps_lips_inside() -> None:
    from .feel import feel
    from .mouth_bits import bits
    from .retarget import FaceExpr, dest_mouth_width, fit_mouth_box

    rest, osf = _toy_face()
    rest[20] = [42.0, 69.0, 1.0]
    rest[21] = [50.0, 67.0, 1.0]
    rest[22] = [58.0, 69.0, 1.0]
    rest[23] = [40.0, 72.0, 1.0]
    rest[24] = [44.0, 75.0, 1.0]
    rest[25] = [50.0, 77.0, 1.0]
    rest[26] = [60.0, 72.0, 1.0]
    rest[27] = [56.0, 75.0, 1.0]
    prev = feel.payload()
    prev_bits = bits.snapshot()
    try:
        feel.update({"mouth": 0.5})
        dest_c, dest_w, dest_h, _scale = fit_mouth_box(
            rest, 1.0, 0.45, dest_mouth_width(rest), rest
        )
        assert np.allclose(dest_c, [50.0, 72.0], atol=0.5)
        assert dest_w >= 20.0
        assert dest_h >= 10.0
        camera = np.zeros((66, 3), dtype=np.float32)
        camera[:, 2] = 1.0
        camera[58] = [-8.0, -8.0, 1.0]
        camera[62] = [8.0, 8.0, 1.0]
        bits.restore(frozenset((58, 62)), {58: 23, 62: 26})
        expr = FaceExpr()
        out = expr.apply(rest, rest, osf, mouth_pts=camera)
        assert out is not None
        box = expr.mouth_box()
        assert box is not None
        x, y, w, h = box
        for slot in (23, 26):
            assert x - 1e-3 <= float(out[slot, 0]) <= x + w + 1e-3
            assert y - 1e-3 <= float(out[slot, 1]) <= y + h + 1e-3
    finally:
        feel.update(prev)
        bits.restore(*prev_bits)


def test_mouth_box_stays_inside_face_cage() -> None:
    from .feel import feel
    from .mouth_bits import bits
    from .retarget import FaceExpr, dest_mouth_cage

    rest, osf = _toy_face()
    camera = np.zeros((66, 3), dtype=np.float32)
    camera[:, 2] = 1.0
    camera[58] = [-0.8, 0.05, 1.0]
    camera[62] = [0.8, 0.85, 1.0]
    expr = FaceExpr()
    prev_feel = feel.payload()
    prev_bits = bits.snapshot()
    feel.update(
        {
            "smoothing": 0.0,
            "response": 1.0,
            "mouth": 0.5,
            "mouth_x": 1.0,
            "mouth_y": 1.0,
            "mouth_z": 0.5,
        }
    )
    try:
        bits.restore(frozenset((58, 62)), {58: 23, 62: 26})
        out = expr.apply(rest, rest, osf, mouth_pts=camera)
        assert out is not None
        left, top, right, bottom = dest_mouth_cage(rest)
        box = expr.mouth_box()
        assert box is not None
        x, y, w, h = box
        assert x >= left - 1e-3
        assert y >= top - 1e-3
        assert x + w <= right + 1e-3
        assert y + h <= bottom + 1e-3
        for slot in (23, 26):
            assert left - 1e-3 <= float(out[slot, 0]) <= right + 1e-3
            assert top - 1e-3 <= float(out[slot, 1]) <= bottom + 1e-3
    finally:
        feel.update(prev_feel)
        bits.restore(*prev_bits)


def test_toggling_a_lip_does_not_jump_the_others() -> None:
    from .feel import feel
    from .mouth_bits import bits
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    camera_rest = np.zeros((66, 3), dtype=np.float32)
    camera_rest[:, 2] = 1.0
    camera_rest[58] = [-0.4, 0.2, 1.0]
    camera_rest[60] = [0.0, 0.05, 1.0]
    camera_rest[62] = [0.6, 0.2, 1.0]
    camera_live = camera_rest.copy()
    camera_live[58] += [0.04, 0.18, 0.0]
    camera_live[60] += [0.0, 0.28, 0.0]
    camera_live[62] += [-0.04, 0.12, 0.0]
    expr = FaceExpr()
    prev_feel = feel.payload()
    prev_bits = bits.snapshot()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        bits.restore(frozenset((58, 60, 62)), {58: 23, 60: 21, 62: 26})
        expr.apply(rest, rest, osf, mouth_pts=camera_rest)
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, osf, mouth_pts=camera_live)
        assert out is not None
        before_23 = out[23, :2].copy()
        before_26 = out[26, :2].copy()
        bits.set_on(60, False)
        jumped = expr.apply(rest, rest, osf, mouth_pts=camera_live)
        assert jumped is not None
        assert np.allclose(jumped[23, :2], before_23, atol=0.45)
        assert np.allclose(jumped[26, :2], before_26, atol=0.45)
    finally:
        feel.update(prev_feel)
        bits.restore(*prev_bits)


def test_mouth_gain_expands_preset_points() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    rest[20] = [42.0, 69.0, 1.0]
    rest[21] = [50.0, 67.0, 1.0]
    rest[22] = [58.0, 69.0, 1.0]
    rest[23] = [40.0, 72.0, 1.0]
    rest[24] = [44.0, 75.0, 1.0]
    rest[25] = [50.0, 77.0, 1.0]
    rest[26] = [60.0, 72.0, 1.0]
    rest[27] = [56.0, 75.0, 1.0]
    mixed = rest.copy()
    mixed[20, 0] = 38.0
    mixed[22, 0] = 62.0
    prev = feel.payload()
    try:
        feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
        natural = FaceExpr().apply(mixed, rest, osf, keep_mouth=True)
        feel.update({"mouth": 1.0})
        doubled = FaceExpr().apply(mixed, rest, osf, keep_mouth=True)
        assert natural is not None and doubled is not None
        slots = list(range(20, 28))
        nat = natural[slots, :2]
        big = doubled[slots, :2]
        nat_c = np.mean(nat, axis=0)
        big_c = np.mean(big, axis=0)
        assert np.allclose(nat_c, big_c, atol=0.4)
        nat_span = float(np.max(np.linalg.norm(nat - nat_c, axis=1)))
        big_span = float(np.max(np.linalg.norm(big - big_c, axis=1)))
        assert big_span / max(nat_span, 1e-6) > 1.85
    finally:
        feel.update(prev)


def test_keep_mouth_uses_viseme_mix_inside_the_box() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    rest[21] = [50.0, 70.0, 1.0]
    mixed = rest.copy()
    mixed[21, 1] = 74.0
    camera = np.zeros((66, 3), dtype=np.float32)
    camera[:, 2] = 1.0
    camera[60] = [0.0, 8.0, 1.0]
    expr = FaceExpr()
    prev = feel.payload()
    try:
        feel.update(
            {
                "smoothing": 0.0,
                "response": 1.0,
                "mouth": 0.5,
                "mouth_x": 1.0,
                "mouth_y": 1.0,
                "mouth_z": 0.5,
            }
        )
        out = expr.apply(mixed, rest, osf, mouth_pts=camera, keep_mouth=True)
        assert out is not None
        assert abs(float(out[21, 1]) - 74.0) < 1e-3
        box = expr.mouth_box()
        assert box is not None
        x, y, w, h = box
        assert x - 1e-3 <= float(out[21, 0]) <= x + w + 1e-3
        assert y - 1e-3 <= float(out[21, 1]) <= y + h + 1e-3
    finally:
        feel.update(prev)


def test_keep_mouth_does_not_clip_authored_a() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    for slot, xy in (
        (20, (42.0, 71.0)),
        (21, (50.0, 70.0)),
        (22, (58.0, 71.0)),
        (23, (40.0, 72.0)),
        (24, (44.0, 73.0)),
        (25, (50.0, 74.0)),
        (26, (60.0, 72.0)),
        (27, (56.0, 73.0)),
    ):
        rest[slot] = [xy[0], xy[1], 1.0]
    mixed = rest.copy()
    mixed[21, 1] = 58.0
    mixed[25, 1] = 88.0
    prev = feel.payload()
    try:
        feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
        out = FaceExpr().apply(mixed, rest, osf, keep_mouth=True)
        assert out is not None
        assert float(out[25, 1]) > 84.0
        assert float(out[21, 1]) < 62.0
    finally:
        feel.update(prev)


def test_keep_mouth_camera_open_adds_lip_split() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    for slot, xy in (
        (20, (42.0, 71.0)),
        (21, (50.0, 70.0)),
        (22, (58.0, 71.0)),
        (23, (40.0, 72.0)),
        (24, (44.0, 73.0)),
        (25, (50.0, 74.0)),
        (26, (60.0, 72.0)),
        (27, (56.0, 73.0)),
    ):
        rest[slot] = [xy[0], xy[1], 1.0]
    closed = np.zeros((66, 3), dtype=np.float32)
    closed[:, 2] = 1.0
    closed[51] = [0.0, 0.00, 1.0]
    closed[57] = [0.0, 0.02, 1.0]
    closed[58] = [-0.30, 0.01, 1.0]
    closed[60] = [0.0, 0.00, 1.0]
    closed[62] = [0.30, 0.01, 1.0]
    closed[64] = [0.0, 0.02, 1.0]
    opened = closed.copy()
    opened[51, 1] = -0.20
    opened[57, 1] = 0.40
    opened[60, 1] = -0.20
    opened[64, 1] = 0.40
    expr = FaceExpr()
    prev = feel.payload()
    try:
        feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
        expr.apply(rest, rest, osf, mouth_pts=closed, keep_mouth=True)
        out = expr.apply(rest, rest, osf, mouth_pts=opened, keep_mouth=True)
        assert out is not None
        assert float(out[25, 1]) > float(rest[25, 1]) + 3.0
        assert float(out[21, 1]) < float(rest[21, 1]) - 2.0
    finally:
        feel.update(prev)


def test_mapped_lips_keep_viseme_mix() -> None:
    from .feel import feel
    from .mouth_bits import DEFAULT_ON, DEFAULT_TO, bits
    from .retarget import FaceExpr

    prev, prev_to = bits.snapshot()
    rest, osf = _toy_face()
    rest[21] = [50.0, 70.0, 1.0]
    mixed = rest.copy()
    mixed[21, 1] = 74.0
    live = osf.copy()
    live[60, 1] += 0.40
    expr = FaceExpr()
    feel_prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        bits.restore(frozenset(DEFAULT_ON), dict(DEFAULT_TO))
        expr.apply(mixed, rest, osf, {"l": 0.0, "r": 0.0})
        still = expr.apply(mixed, rest, osf, {"l": 0.0, "r": 0.0})
        assert still is not None
        assert abs(float(still[21, 1]) - float(rest[21, 1])) < 1e-5
        out = mixed
        for _ in range(8):
            out = expr.apply(mixed, rest, live, {"l": 0.0, "r": 0.0})
        assert out is not None
        assert float(out[21, 1]) < float(rest[21, 1])
    finally:
        feel.update(feel_prev)
        bits.restore(prev, prev_to)


def _restore_bits(
    prev: frozenset[int],
    to: dict[int, int] | None = None,
) -> None:
    from .mouth_bits import bits

    bits.restore(prev, to if to is not None else bits.maps())


def test_default_lip_maps_match_roster() -> None:
    from .mouth_bits import DEFAULT_ON, DEFAULT_ROLES, DEFAULT_TO, bits

    prev, prev_to = bits.snapshot()
    try:
        bits.restore(frozenset(DEFAULT_ON), dict(DEFAULT_TO))
        assert DEFAULT_ON == frozenset((58, 59, 60, 61, 62, 63, 64, 65))
        assert DEFAULT_TO == {
            58: 23,
            59: 20,
            60: 21,
            61: 22,
            62: 26,
            63: 27,
            64: 25,
            65: 24,
        }
        assert bits.roles() == DEFAULT_ROLES
        assert bits.roles() == (58, 60, 62, 64)
    finally:
        bits.restore(prev, prev_to)


def test_mouth_bits_toggle_changes_roles() -> None:
    from .mouth_bits import DEFAULT_ON, DEFAULT_TO, bits

    prev, prev_to = bits.snapshot()
    try:
        bits.restore(frozenset(DEFAULT_ON), dict(DEFAULT_TO))
        assert bits.roles() == (58, 60, 62, 64)
        bits.set_on(60, False)
        roles = bits.roles()
        assert roles is not None
        assert roles[1] == 61
        bits.set_on(50, True)
        bits.set_to(50, 21)
        roles = bits.roles()
        assert roles is not None
        assert roles[1] == 50
    finally:
        bits.restore(prev, prev_to)


def test_changing_lip_map_eases_instead_of_snapping() -> None:
    from .feel import feel
    from .mouth_bits import bits
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    rest[21] = [50.0, 70.0, 1.0]
    rest[23] = [40.0, 72.0, 1.0]
    rest[26] = [60.0, 72.0, 1.0]
    camera = np.zeros((66, 3), dtype=np.float32)
    camera[:, 2] = 1.0
    camera[58] = [-0.4, 0.35, 1.0]
    camera[62] = [0.4, 0.20, 1.0]
    expr = FaceExpr()
    prev_feel = feel.payload()
    prev_bits = bits.snapshot()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        bits.restore(frozenset((58, 62)), {58: 23, 62: 26})
        before = rest
        for _ in range(8):
            before = expr.apply(rest, rest, osf, mouth_pts=camera)
        bits.set_to(58, 21)
        first = expr.apply(rest, rest, osf, mouth_pts=camera)
        later = first
        for _ in range(40):
            later = expr.apply(rest, rest, osf, mouth_pts=camera)
        assert before is not None and first is not None and later is not None
        assert float(np.linalg.norm(first[23, :2] - before[23, :2])) < 0.35
        assert float(np.linalg.norm(first[21, :2] - before[21, :2])) < 0.35
        assert float(np.linalg.norm(later[21, :2] - before[21, :2])) > 1.0
    finally:
        feel.update(prev_feel)
        bits.restore(*prev_bits)


def test_mouth_map_54_drives_anime_21() -> None:
    from .feel import feel
    from .mouth_bits import bits
    from .retarget import FaceExpr

    prev, prev_to = bits.snapshot()
    rest, osf = _toy_face()
    rest[21] = [50.0, 70.0, 1.0]
    osf[54] = [0.30, 0.40, 0.0]
    live = osf.copy()
    live[54, 1] += 0.35
    expr = FaceExpr()
    feel_prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        bits.set_on(54, True)
        bits.set_to(54, 21)
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0})
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, live, {"l": 0.0, "r": 0.0})
        assert out is not None
        assert float(out[21, 1]) < float(rest[21, 1]) - 1.0
    finally:
        feel.update(feel_prev)
        bits.restore(prev, prev_to)


def test_character_skeleton_comes_from_the_face() -> None:
    from .skeleton import follow_skeleton, merge_skeleton, skeleton_from_face

    rest, _osf = _toy_face()
    body = skeleton_from_face(rest)
    ids = {int(j["id"]) for j in body}
    assert ids == {31, 32, 33, 34, 35, 36}
    filled = merge_skeleton(
        [
            {"id": 30, "name": "nose", "x": 1.0, "y": 2.0, "score": 0.9},
            {"id": 31, "name": "neck", "x": 1.0, "y": 22.0, "score": 0.9},
        ],
        rest,
    )
    assert all(int(j["id"]) != 30 for j in filled)
    neck = next(j for j in filled if j["id"] == 31)
    assert float(neck["y"]) > float(rest[15, 1]) + 20.0
    live = rest.copy()
    live[15] = [62.0, 51.0, 1.0]
    moved = follow_skeleton(filled, live)
    moved_neck = next(j for j in moved if j["id"] == 31)
    rest_sh = next(j for j in filled if j["id"] == 32)
    moved_sh = next(j for j in moved if j["id"] == 32)
    assert abs(float(moved_neck["x"]) - float(neck["x"])) < 0.15
    assert abs(float(moved_neck["y"]) - float(neck["y"])) < 0.15
    assert abs(float(moved_sh["x"]) - float(rest_sh["x"])) < 0.15
    assert abs(float(moved_sh["y"]) - float(rest_sh["y"])) < 0.15


def test_chin_stays_when_osf_jaw_slides() -> None:
    from .feel import feel
    from .retarget import FaceExpr

    rest, osf = _toy_face()
    live = osf.copy()
    live[8] = [0.8, 1.6, 0.0]
    live[0] = [-0.4, 0.4, 0.0]
    expr = FaceExpr()
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
    try:
        expr.apply(rest, rest, osf, {"l": 0.0, "r": 0.0})
        out = rest
        for _ in range(8):
            out = expr.apply(rest, rest, live, {"l": 0.0, "r": 0.0})
        assert out is not None
        assert abs(float(out[2, 0]) - float(rest[2, 0])) < 0.2
        assert abs(float(out[2, 1]) - float(rest[2, 1])) < 0.2
    finally:
        feel.update(prev)


def test_head_rig_holds_when_camera_drops() -> None:
    from .feel import feel
    from .rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[11] = [30.0, 35.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[18] = [70.0, 35.0, 1.0]
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    lost = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_yaw": 1.0})
    try:
        rig = FaceRig()
        rig.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        turned = rest
        for _ in range(12):
            turned = rig.apply(rest, rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}, origin)
        held = rig.apply(rest, rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}, lost)
        assert turned is not None and held is not None
        assert abs(float(held[0, 0]) - float(turned[0, 0])) < 0.2
        assert abs(float(held[0, 0]) - float(rest[0, 0])) > 4.0
    finally:
        feel.update(prev)


def test_head_turn_moves_the_torso_with_the_face() -> None:
    from .feel import feel
    from .rig import FaceRig
    from .skeleton import follow_skeleton, skeleton_from_face

    rest, _osf = _toy_face()
    body = skeleton_from_face(rest)
    origin = {
        "cx": 200.0,
        "cy": 200.0,
        "bx": 200.0,
        "by": 200.0,
        "scale": 100.0,
        "tilt": 0.0,
        "ok": 1.0,
    }
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_yaw": 1.0, "max_pitch_up": 1.0})
    try:
        rig = FaceRig()
        rig.apply(rest, rest, head0, origin)
        parked_face = rig.apply(rest, rest, head0, origin)
        parked = {int(j["id"]): j for j in follow_skeleton(body, rest, rig=rig)}
        turned_face = rest
        for _ in range(12):
            turned_face = rig.apply(
                rest, rest, {"pitch": -24.0, "yaw": 35.0, "roll": 0.0}, origin
            )
        turned = {int(j["id"]): j for j in follow_skeleton(body, rest, rig=rig)}
    finally:
        feel.update(prev)
    assert parked_face is not None and turned_face is not None
    chin_dx = float(turned_face[2, 0]) - float(parked_face[2, 0])
    chin_dy = float(turned_face[2, 1]) - float(parked_face[2, 1])
    assert abs(chin_dx) > 1.0 or abs(chin_dy) > 1.0
    neck_dx = float(turned[31]["x"]) - float(parked[31]["x"])
    neck_dy = float(turned[31]["y"]) - float(parked[31]["y"])
    assert abs(neck_dx) > 1.0 or abs(neck_dy) > 1.0
    assert abs(neck_dx - chin_dx) < 12.0
    assert abs(neck_dy - chin_dy) < 12.0


def test_big_head_turn_does_not_turn_the_torso() -> None:
    """Turning the head folded the shoulders into a line and swung them round the nose."""
    import math

    from .feel import feel
    from .rig import FaceRig
    from .skeleton import follow_skeleton, skeleton_from_face

    rest, _osf = _toy_face()
    body = skeleton_from_face(rest)
    origin = {"cx": 200.0, "cy": 200.0, "bx": 200.0, "by": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_yaw": 1.0, "max_pitch_up": 1.0, "max_roll": 1.0})
    try:
        rig = FaceRig()
        rig.apply(rest, rest, head0, origin)
        parked = {int(j["id"]): j for j in follow_skeleton(body, rest, rig=rig)}
        for _ in range(12):
            rig.apply(rest, rest, {"pitch": -30.0, "yaw": 80.0, "roll": 0.0}, origin)
        turned = {int(j["id"]): j for j in follow_skeleton(body, rest, rig=rig)}
    finally:
        feel.update(prev)

    def _vec(pose: dict, a: int, b: int) -> tuple[float, float]:
        return pose[b]["x"] - pose[a]["x"], pose[b]["y"] - pose[a]["y"]

    # Shoulders keep their width and the torso keeps its shape around the neck.
    for a, b in ((32, 34), (31, 36), (31, 32), (32, 33)):
        rx, ry = _vec(parked, a, b)
        tx, ty = _vec(turned, a, b)
        assert math.hypot(tx - rx, ty - ry) < 1.0, (a, b)


def test_manual_skeleton_ignores_camera_body() -> None:
    from .skeleton import follow_skeleton, skeleton_from_face

    rest, _osf = _toy_face()
    body = skeleton_from_face(rest)
    cam_rest = [
        {"id": 31, "x": 100.0, "y": 80.0},
        {"id": 32, "x": 70.0, "y": 90.0},
        {"id": 33, "x": 60.0, "y": 130.0},
        {"id": 34, "x": 130.0, "y": 90.0},
        {"id": 35, "x": 140.0, "y": 130.0},
    ]
    cam_live = [dict(j) for j in cam_rest]
    cam_live[2] = {"id": 33, "x": 110.0, "y": 125.0}
    moved = follow_skeleton(body, rest, cam_rest, cam_live)
    rest_chest = next(j for j in body if j["id"] == 36)
    live_chest = next(j for j in moved if j["id"] == 36)
    rest_sh = next(j for j in body if j["id"] == 32)
    live_sh = next(j for j in moved if j["id"] == 32)
    rest_el = next(j for j in body if j["id"] == 33)
    live_el = next(j for j in moved if j["id"] == 33)
    assert abs(float(live_chest["x"]) - float(rest_chest["x"])) < 0.15
    assert abs(float(live_sh["x"]) - float(rest_sh["x"])) < 0.15
    assert abs(float(live_el["x"]) - float(rest_el["x"])) < 0.15


def test_neck_sits_between_chin_and_shoulders() -> None:
    from .skeleton import skeleton_from_face

    rest, _osf = _toy_face()
    rest[2] = [50.0, 95.0, 1.0]
    body = {int(j["id"]): j for j in skeleton_from_face(rest)}
    chin_y = float(rest[2, 1])
    neck_y = float(body[31]["y"])
    sh_y = float(body[32]["y"])
    assert neck_y > chin_y
    assert sh_y > neck_y
    assert (sh_y - neck_y) < (neck_y - chin_y)
    assert abs(float(body[32]["y"]) - float(body[34]["y"])) < 0.2


def test_shoulders_stay_when_chin_drops() -> None:
    from .skeleton import skeleton_from_face

    rest, _osf = _toy_face()
    high = {int(j["id"]): j for j in skeleton_from_face(rest)}
    rest[2] = [50.0, 120.0, 1.0]
    low = {int(j["id"]): j for j in skeleton_from_face(rest)}
    assert abs(float(high[32]["y"]) - float(low[32]["y"])) < 0.2
    assert abs(float(high[36]["y"]) - float(low[36]["y"])) < 0.2


def test_face_skeleton_stacks_torso_below_the_chin() -> None:
    from .skeleton import skeleton_from_face

    rest, _osf = _toy_face()
    body = {int(j["id"]): j for j in skeleton_from_face(rest)}
    assert float(body[31]["y"]) < float(body[32]["y"]) < float(body[36]["y"]) < float(body[33]["y"])
    assert float(body[33]["y"]) - float(body[32]["y"]) > 40.0


def test_chroma_torso_follows_the_bust_not_the_face() -> None:
    from .skeleton import skeleton_from_still

    rest, _osf = _toy_face()
    rest[0] = [140.0, 80.0, 1.0]
    rest[2] = [200.0, 130.0, 1.0]
    rest[4] = [260.0, 80.0, 1.0]
    rest[21] = [200.0, 100.0, 1.0]
    still = np.zeros((400, 400, 3), dtype=np.uint8)
    still[:] = (40, 220, 40)
    still[145:175, 175:225] = (90, 90, 200)
    still[175:400, 70:330] = (160, 170, 210)
    still[300:400, 40:360] = (150, 160, 200)
    body = {int(j["id"]): j for j in skeleton_from_still(rest, still)}
    guess = {int(j["id"]): j for j in skeleton_from_still(rest, None)}
    assert float(body[32]["y"]) > float(guess[32]["y"]) + 8.0
    assert float(body[33]["y"]) > 300.0
    assert float(body[32]["x"]) < 120.0
    assert float(body[34]["x"]) > 280.0
    assert float(body[36]["y"]) > float(body[32]["y"])
    assert float(body[31]["y"]) < float(body[32]["y"])


def test_wide_collar_still_uses_the_torso() -> None:
    """A neck row already near the body width must not fall back to face size."""
    from .skeleton import skeleton_from_face, skeleton_from_still

    rest, _osf = _toy_face()
    rest[0] = [160.0, 70.0, 1.0]
    rest[2] = [200.0, 120.0, 1.0]
    rest[4] = [240.0, 70.0, 1.0]
    rest[21] = [200.0, 100.0, 1.0]
    still = np.zeros((400, 400, 3), dtype=np.uint8)
    still[:] = (40, 220, 40)
    still[130:400, 40:360] = (160, 170, 210)
    body = {int(j["id"]): j for j in skeleton_from_still(rest, still)}
    guess = {int(j["id"]): j for j in skeleton_from_face(rest)}
    assert float(body[34]["x"]) - float(body[32]["x"]) > (
        float(guess[34]["x"]) - float(guess[32]["x"])
    ) + 40.0
    assert float(body[32]["x"]) < 80.0
    assert float(body[34]["x"]) > 320.0


def test_skeleton_translates_with_the_head() -> None:
    from .skeleton import follow_skeleton, skeleton_from_face

    rest, _osf = _toy_face()
    body = skeleton_from_face(rest)
    parked = follow_skeleton(body, rest)
    moved = follow_skeleton(
        body,
        rest,
        place={"dx": 24.0, "dy": -30.0, "scale": 1.0, "cx": float(rest[15, 0]), "cy": float(rest[15, 1])},
    )
    park_neck = next(j for j in parked if j["id"] == 31)
    move_neck = next(j for j in moved if j["id"] == 31)
    park_sh = next(j for j in parked if j["id"] == 32)
    move_sh = next(j for j in moved if j["id"] == 32)
    assert abs(float(move_neck["x"]) - float(park_neck["x"]) - 24.0) < 0.3
    assert abs(float(move_neck["y"]) - float(park_neck["y"]) + 30.0) < 0.3
    assert abs(float(move_sh["y"]) - float(park_sh["y"]) + 30.0) < 0.3
    grown = follow_skeleton(
        body,
        rest,
        place={"dx": 0.0, "dy": 0.0, "scale": 1.3, "cx": float(rest[15, 0]), "cy": float(rest[15, 1])},
    )
    grow_sh = next(j for j in grown if j["id"] == 32)
    assert abs(float(grow_sh["x"]) - float(rest[15, 0])) > abs(float(park_sh["x"]) - float(rest[15, 0])) + 2.0


def test_global_smoothing_blends_skeleton_and_hair_points() -> None:
    from .face import FaceBench

    skeleton = FaceBench._smooth_records(
        [{"id": 31, "x": 0.0, "y": 10.0}],
        [{"id": 31, "x": 10.0, "y": 30.0}],
        0.25,
    )
    assert skeleton[0]["x"] == 2.5
    assert skeleton[0]["y"] == 15.0
    hair = FaceBench._smooth_records(
        [{"class": "hair", "polygon": [[0.0, 0.0], [10.0, 10.0]]}],
        [{"class": "hair", "polygon": [[8.0, 4.0], [14.0, 18.0]]}],
        0.5,
        polygons=True,
    )
    assert hair[0]["polygon"] == [[4.0, 2.0], [12.0, 14.0]]


def test_hair_stays_planted_when_camera_slides() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[40.0, 8.0], [60.0, 8.0], [55.0, 22.0], [45.0, 22.0]],
        }
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    parked = follow_hair(hrig, rest)
    face = FaceRig()
    origin = {
        "cx": 200.0,
        "cy": 200.0,
        "bx": 200.0,
        "by": 200.0,
        "scale": 100.0,
        "tilt": 0.0,
        "ok": 1.0,
    }
    shifted = {
        "cx": 260.0,
        "cy": 200.0,
        "bx": 200.0,
        "by": 200.0,
        "scale": 100.0,
        "tilt": 0.0,
        "ok": 1.0,
    }
    head = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0})
    try:
        face.apply(rest, rest, head, origin)
        for _ in range(12):
            face.apply(rest, rest, head, shifted)
        moved = follow_hair(hrig, rest, face)
    finally:
        feel.update(prev)
    assert parked and moved
    assert abs(moved[0]["polygon"][0][0] - parked[0]["polygon"][0][0]) < 1.0


def test_hair_scales_with_the_face_and_returns() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[40.0, 8.0], [60.0, 8.0], [55.0, 22.0], [45.0, 22.0]],
        }
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    origin = {
        "cx": 200.0,
        "cy": 200.0,
        "bx": 200.0,
        "by": 200.0,
        "scale": 100.0,
        "tilt": 0.0,
        "ok": 1.0,
    }
    head = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_size": 1.0})
    try:
        face = FaceRig()
        _hold_distance(face, rest, rest, head, origin)
        parked = follow_hair(hrig, rest, face)
        closer = dict(origin)
        closer["scale"] = 140.0
        for _ in range(12):
            face.apply(rest, rest, head, closer)
        grown = follow_hair(hrig, rest, face)
        for _ in range(12):
            face.apply(rest, rest, head, origin)
        home = follow_hair(hrig, rest, face)
    finally:
        feel.update(prev)

    def _span(parts):
        xs = [p[0] for p in parts[0]["polygon"]]
        ys = [p[1] for p in parts[0]["polygon"]]
        return max(xs) - min(xs), max(ys) - min(ys)

    park_w, park_h = _span(parked)
    grow_w, grow_h = _span(grown)
    home_w, home_h = _span(home)
    assert grow_w > park_w * 1.15
    assert grow_h > park_h * 1.15
    assert abs(home_w - park_w) < 1.0
    assert abs(home_h - park_h) < 1.0


def test_hair_yaws_instead_of_sliding() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {"class": "hair_left", "polygon": [[8.0, 10.0], [28.0, 10.0], [28.0, 40.0], [8.0, 40.0]]},
        {"class": "hair_right", "polygon": [[72.0, 10.0], [92.0, 10.0], [92.0, 40.0], [72.0, 40.0]]},
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    face = FaceRig()
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head0 = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0})
    try:
        face.apply(rest, rest, head0, origin)
        parked = follow_hair(hrig, rest, face)
        turned = rest
        for _ in range(12):
            turned = face.apply(rest, rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}, origin)
        moved = follow_hair(hrig, turned, face)
    finally:
        feel.update(prev)

    def _mean_x(parts, cls):
        poly = next(p["polygon"] for p in parts if p["class"] == cls)
        return sum(p[0] for p in poly) / len(poly)

    d_left = _mean_x(moved, "hair_left") - _mean_x(parked, "hair_left")
    d_right = _mean_x(moved, "hair_right") - _mean_x(parked, "hair_right")
    assert abs(d_left - d_right) > 0.5

    def _span_x(parts, cls):
        xs = [p[0] for p in next(p["polygon"] for p in parts if p["class"] == cls)]
        return max(xs) - min(xs)

    near_w = _span_x(moved, "hair_left")
    far_w = _span_x(moved, "hair_right")
    rest_w = _span_x(parked, "hair_left")
    assert near_w > rest_w
    assert far_w < rest_w
    assert near_w > far_w


def test_hair_follow_does_not_collapse_a_bump() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {
            "class": "hair_middle",
            "polygon": [
                [50.0 + 173.1, 48.0 + 56.2],
                [50.0 + 194.1, 48.0 + 73.3],
                [50.0 + 188.4, 48.0 + 22.1],
            ],
        }
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    face = FaceRig()
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "hair_pin": 0.7})

    def _ang(poly: list) -> float:
        p = np.asarray(poly, dtype=np.float64)
        v1 = p[0] - p[1]
        v2 = p[2] - p[1]
        e1 = float(np.linalg.norm(v1))
        e2 = float(np.linalg.norm(v2))
        return float(
            np.degrees(np.arccos(np.clip(np.dot(v1, v2) / max(e1 * e2, 1e-8), -1.0, 1.0)))
        )

    try:
        face.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        parked = follow_hair(hrig, rest, face)
        turned = rest
        for _ in range(12):
            turned = face.apply(
                rest, rest, {"pitch": 0.0, "yaw": 25.0, "roll": 0.0}, origin
            )
        moved = follow_hair(hrig, turned, face)
    finally:
        feel.update(prev)

    rest_ang = _ang(parked[0]["polygon"])
    live_ang = _ang(moved[0]["polygon"])
    assert rest_ang > 20.0
    assert live_ang > rest_ang * 0.6
    assert live_ang > 20.0


def _hair_pin_segs() -> list[dict]:
    return [
        {
            "class": "hair_middle",
            "polygon": [
                [46.0, 36.0],
                [54.0, 36.0],
                [50.0, -90.0],
            ],
        }
    ]


def _poly_xy(parts, cls: str) -> list[list[float]]:
    return next(p["polygon"] for p in parts if p["class"] == cls)


def test_hair_silhouette_stays_put_on_big_turn() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    hrig = build_hair_rig(_hair_pin_segs(), rest)
    assert hrig is not None
    face = FaceRig()
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "hair_pin": 1.0})
    try:
        face.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        parked = follow_hair(hrig, rest, face)
        turned = rest
        for _ in range(12):
            turned = face.apply(
                rest, rest, {"pitch": 0.0, "yaw": 60.0, "roll": 0.0}, origin
            )
        moved = follow_hair(hrig, turned, face)
    finally:
        feel.update(prev)

    park = _poly_xy(parked, "hair_middle")
    live = _poly_xy(moved, "hair_middle")
    span = float(hrig.rest_ms)
    outer_dx = abs(live[2][0] - park[2][0])
    assert outer_dx < 0.15 * span


def test_hair_pin_zero_matches_old_follow() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    hrig = build_hair_rig(_hair_pin_segs(), rest)
    assert hrig is not None
    face = FaceRig()
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "hair_pin": 0.0, "hair_width": 0.0})
    try:
        turned = rest
        for _ in range(12):
            turned = face.apply(
                rest, rest, {"pitch": 0.0, "yaw": 50.0, "roll": 0.0}, origin
            )
        moved = follow_hair(hrig, turned, face)
        xs, ys = face.map_plane(hrig.parts[0].local[:, 0], hrig.parts[0].local[:, 1])
        expect = [[round(float(x), 1), round(float(y), 1)] for x, y in zip(xs, ys)]
        assert moved[0]["polygon"] == expect
    finally:
        feel.update(prev)


def test_hair_pin_ignores_pitch_at_silhouette() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    hrig = build_hair_rig(_hair_pin_segs(), rest)
    assert hrig is not None
    face = FaceRig()
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "hair_pin": 1.0})
    try:
        face.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        parked = follow_hair(hrig, rest, face)
        for pitch in (40.0, -40.0):
            nod = rest
            for _ in range(12):
                nod = face.apply(
                    rest, rest, {"pitch": pitch, "yaw": 0.0, "roll": 0.0}, origin
                )
            moved = follow_hair(hrig, nod, face)
            park = _poly_xy(parked, "hair_middle")
            live = _poly_xy(moved, "hair_middle")
            assert abs(live[2][1] - park[2][1]) < 0.15 * float(hrig.rest_ms)
    finally:
        feel.update(prev)


def _edge_lengths(poly: list) -> list[float]:
    pts = np.asarray(poly, dtype=np.float64)
    n = len(pts)
    return [
        float(np.linalg.norm(pts[(i + 1) % n] - pts[i]))
        for i in range(n)
    ]


def test_hair_rigid_keeps_edge_ratios_on_yaw() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {
            "class": "hair_middle",
            "polygon": [
                [42.0, 6.0],
                [58.0, 10.0],
                [54.0, 32.0],
                [46.0, 26.0],
            ],
        }
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    face = FaceRig()
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "hair_pin": 1.0})
    try:
        face.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        parked = follow_hair(hrig, rest, face)
        turned = rest
        for _ in range(12):
            turned = face.apply(
                rest, rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}, origin
            )
        moved = follow_hair(hrig, turned, face)
    finally:
        feel.update(prev)
    rest_e = np.asarray(_edge_lengths(parked[0]["polygon"]))
    live_e = np.asarray(_edge_lengths(moved[0]["polygon"]))
    rest_r = rest_e / max(float(np.mean(rest_e)), 1e-6)
    live_r = live_e / max(float(np.mean(live_e)), 1e-6)
    assert np.allclose(rest_r, live_r, atol=0.08)


def test_hair_middle_width_stays_on_yaw() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[42.0, 8.0], [58.0, 8.0], [56.0, 36.0], [50.0, 40.0], [44.0, 36.0]],
        }
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    face = FaceRig()
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "hair_pin": 1.0})
    try:
        face.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        parked = follow_hair(hrig, rest, face)
        turned = rest
        for _ in range(12):
            turned = face.apply(
                rest, rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}, origin
            )
        moved = follow_hair(hrig, turned, face)
    finally:
        feel.update(prev)

    def _span_x(parts):
        xs = [p[0] for p in parts[0]["polygon"]]
        return max(xs) - min(xs)

    assert abs(_span_x(moved) - _span_x(parked)) < 1.0


def test_hair_middle_parents_to_hairline_not_tips() -> None:
    from .hair import build_hair_rig

    rest, _osf = _toy_face()
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[42.0, 8.0], [58.0, 8.0], [56.0, 36.0], [50.0, 40.0], [44.0, 36.0]],
        }
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    # Nose is local (0, 0). Brows sit around y=-27; bangs tips around y=-8.
    assert float(hrig.parts[0].anchor[1]) < -15.0
    assert hrig.parts[0].cls == "hair_middle"


def test_hair_middle_stays_welded_to_sides_on_yaw() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {"class": "hair_left", "polygon": [[20.0, 10.0], [40.0, 10.0], [40.0, 40.0], [20.0, 40.0]]},
        {"class": "hair_middle", "polygon": [[40.0, 10.0], [60.0, 10.0], [60.0, 40.0], [40.0, 40.0]]},
        {"class": "hair_right", "polygon": [[60.0, 10.0], [80.0, 10.0], [80.0, 40.0], [60.0, 40.0]]},
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    assert hrig.welds
    face = FaceRig()
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "hair_pin": 1.0, "hair_width": 1.0})
    try:
        face.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        for _ in range(12):
            face.apply(rest, rest, {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}, origin)
        moved = follow_hair(hrig, rest, face)
    finally:
        feel.update(prev)

    def _poly(cls: str) -> set[tuple[float, float]]:
        return {tuple(p) for p in next(r["polygon"] for r in moved if r["class"] == cls)}

    left = _poly("hair_left")
    mid = _poly("hair_middle")
    right = _poly("hair_right")
    assert left & mid
    assert right & mid


def test_hair_selfie_swaps_near_lock_without_flipping_still() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {"class": "hair_left", "polygon": [[8.0, 10.0], [28.0, 10.0], [28.0, 40.0], [8.0, 40.0]]},
        {"class": "hair_right", "polygon": [[72.0, 10.0], [92.0, 10.0], [92.0, 40.0], [72.0, 40.0]]},
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head = {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "hair_pin": 1.0})

    def _span_x(parts, cls):
        xs = [p[0] for p in next(p["polygon"] for p in parts if p["class"] == cls)]
        return max(xs) - min(xs)

    try:
        face = FaceRig()
        face.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        for _ in range(12):
            face.apply(rest, rest, head, origin)
        natural = follow_hair(hrig, rest, face)
        face.selfie = True
        face.apply(rest, rest, head, origin)
        flipped = follow_hair(hrig, rest, face)
    finally:
        feel.update(prev)

    assert _span_x(natural, "hair_left") > _span_x(natural, "hair_right")
    assert _span_x(flipped, "hair_right") > _span_x(flipped, "hair_left")
    left_rest = hrig.parts[0].local.copy()
    assert left_rest[0, 0] < 0.0


def test_hair_reports_width_and_gain_scales_it() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {"class": "hair_left", "polygon": [[8.0, 10.0], [28.0, 10.0], [28.0, 40.0], [8.0, 40.0]]},
        {"class": "hair_middle", "polygon": [[42.0, 8.0], [58.0, 8.0], [56.0, 36.0], [50.0, 40.0], [44.0, 36.0]]},
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head = {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}
    prev = feel.payload()

    def _rec(parts, cls):
        return next(p for p in parts if p["class"] == cls)

    try:
        feel.update({"smoothing": 0.0, "hair_pin": 1.0, "hair_width": 1.0})
        face = FaceRig()
        face.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        for _ in range(12):
            face.apply(rest, rest, head, origin)
        one = follow_hair(hrig, rest, face)
        feel.update({"hair_width": 2.0})
        two = follow_hair(hrig, rest, face)
        feel.update({"hair_width": 0.0})
        none = follow_hair(hrig, rest, face)
    finally:
        feel.update(prev)

    left1 = _rec(one, "hair_left")
    assert left1["side"] == "l"
    assert left1["pin"] == 1.0
    assert left1["width"] > 1.0
    assert _rec(one, "hair_middle")["side"] == "mid"
    assert _rec(one, "hair_middle")["width"] == 1.0
    assert _rec(two, "hair_left")["width"] > left1["width"]
    assert _rec(none, "hair_left")["width"] == 1.0


def test_hair_width_applies_with_pin_zero() -> None:
    from .feel import feel
    from .hair import build_hair_rig, follow_hair
    from .rig import FaceRig

    rest, _osf = _toy_face()
    segs = [
        {"class": "hair_left", "polygon": [[8.0, 10.0], [28.0, 10.0], [28.0, 40.0], [8.0, 40.0]]},
        {"class": "hair_right", "polygon": [[72.0, 10.0], [92.0, 10.0], [92.0, 40.0], [72.0, 40.0]]},
    ]
    hrig = build_hair_rig(segs, rest)
    assert hrig is not None
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    head = {"pitch": 0.0, "yaw": 35.0, "roll": 0.0}
    prev = feel.payload()

    def _span_x(parts, cls):
        xs = [p[0] for p in next(p["polygon"] for p in parts if p["class"] == cls)]
        return max(xs) - min(xs)

    try:
        face = FaceRig()
        face.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        for _ in range(12):
            face.apply(rest, rest, head, origin)
        feel.update({"smoothing": 0.0, "hair_pin": 0.0, "hair_width": 0.0})
        flat = follow_hair(hrig, rest, face)
        feel.update({"hair_width": 1.0})
        wide = follow_hair(hrig, rest, face)
    finally:
        feel.update(prev)

    left = next(p for p in wide if p["class"] == "hair_left")
    assert left["pin"] == 0.0
    assert left["width"] > 1.0
    assert _span_x(wide, "hair_left") > _span_x(flat, "hair_left")
    assert _span_x(wide, "hair_right") < _span_x(flat, "hair_right")


def test_refine_hair_keeps_detector_mask() -> None:
    from .hair import refine_hair

    rest, _osf = _toy_face()
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[10.0, 4.0], [90.0, 4.0], [90.0, 80.0], [10.0, 80.0]],
            "area": 6000.0,
            "score": 1.0,
        }
    ]
    out = refine_hair(None, segs, rest)
    assert out
    layer = np.zeros((100, 100), dtype=np.uint8)
    for seg in out:
        pts = np.round(np.asarray(seg["polygon"], dtype=np.float32)).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(layer, [pts], 255)
    assert int(layer[10, 50]) > 0
    assert int(layer[55, 50]) > 0


def test_refine_hair_keeps_side_lock() -> None:
    from .hair import refine_hair

    rest, _osf = _toy_face()
    segs = [
        {
            "class": "hair_left",
            "polygon": [[2.0, 18.0], [16.0, 18.0], [16.0, 88.0], [2.0, 88.0]],
            "area": 1000.0,
            "score": 1.0,
        }
    ]
    out = refine_hair(None, segs, rest)
    assert out and out[0]["class"] == "hair_left"
    ys = [p[1] for p in out[0]["polygon"]]
    assert max(ys) > 70.0


def test_face_keepout_opens_under_chin_not_brows() -> None:
    from .hair import _face_keepout

    rest, _osf = _toy_face()
    keep = _face_keepout(rest, 100, 100)
    assert keep is not None
    # Temple / side-lock column at eye height must stay hair.
    assert int(keep[34, 8]) == 0
    assert int(keep[55, 50]) > 0
    # Brow row is not a solid jaw-width shell.
    assert int(keep[21, 8]) == 0
    # Forehead / bangs between brows and eyes stay hair.
    assert int(keep[28, 50]) == 0
    # Neck below the chin may open so a hole does not refill.
    assert int(keep[95, 50]) > 0


def test_face_keepout_ignores_outer_jaw_box() -> None:
    from .hair import _face_keepout

    rest, _osf = _toy_face()
    rest[0] = [5.0, 55.0, 1.0]
    rest[4] = [95.0, 55.0, 1.0]
    keep = _face_keepout(rest, 100, 100)
    assert keep is not None
    assert int(keep[55, 8]) == 0
    assert int(keep[55, 92]) == 0
    assert int(keep[55, 50]) > 0


def test_refine_hair_keeps_forehead_bangs() -> None:
    from .hair import refine_hair

    rest, _osf = _toy_face()
    segs = [
        {
            "class": "hair_middle",
            "polygon": [[30.0, 8.0], [70.0, 8.0], [70.0, 32.0], [30.0, 32.0]],
            "area": 1200.0,
            "score": 1.0,
        }
    ]
    out = refine_hair(None, segs, rest)
    assert out
    layer = np.zeros((100, 100), dtype=np.uint8)
    for seg in out:
        pts = np.round(np.asarray(seg["polygon"], dtype=np.float32)).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(layer, [pts], 255)
    assert int(layer[12, 50]) > 0
    assert int(layer[28, 50]) > 0


def test_pose_osf_keeps_rest_mouth_without_end_shapes(tmp_path, monkeypatch) -> None:
    from . import face as face_mod
    from . import presets as presets_mod
    from .face import FaceBench
    from .feel import feel
    from .osf_cam import OsfFrame
    from .presets import MouthBook, empty_weights

    monkeypatch.setattr(presets_mod, "PRESET_PATH", tmp_path / "mouth_presets.json")
    book = MouthBook()
    monkeypatch.setattr(face_mod, "book", book)
    monkeypatch.setattr(FaceBench, "_load_parts", lambda self: None)
    monkeypatch.setattr(FaceBench, "_load_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_save_ifm", lambda self: None)
    monkeypatch.setattr(FaceBench, "_camera_fields", lambda self: {"camera_index": 0, "cameras": []})

    rest, osf = _toy_face()
    rest[20] = [42.0, 70.0, 1.0]
    rest[21] = [50.0, 68.0, 1.0]
    rest[22] = [58.0, 70.0, 1.0]
    rest[23] = [40.0, 76.0, 1.0]
    rest[24] = [44.0, 82.0, 1.0]
    rest[25] = [50.0, 84.0, 1.0]
    rest[26] = [60.0, 76.0, 1.0]
    rest[27] = [56.0, 82.0, 1.0]
    book.seed_rest(rest)
    closed = osf.copy()
    closed[:, 2] = 1.0
    closed[51] = [0.0, 0.00, 1.0]
    closed[57] = [0.0, 0.02, 1.0]
    closed[58] = [-0.30, 0.01, 1.0]
    closed[60] = [0.0, 0.00, 1.0]
    closed[62] = [0.30, 0.01, 1.0]
    closed[64] = [0.0, 0.02, 1.0]
    live = closed.copy()
    live[51, 1] = -0.20
    live[57, 1] = 0.40
    live[60, 1] -= 0.20
    live[64, 1] += 0.38
    bench = FaceBench(rest_pts=rest.copy())
    closed_frame = OsfFrame(
        weights=empty_weights(),
        pts_3d=closed,
        mouth_2d=closed,
        faces=1,
        source="osf",
    )
    frame = OsfFrame(
        weights=empty_weights(),
        pts_3d=live,
        mouth_2d=live,
        faces=1,
        source="osf",
    )
    prev = feel.payload()
    try:
        feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5, "use_visemes": 1.0})
        bench._pose_osf(closed_frame)
        posed = bench._pose_osf(frame)
    finally:
        feel.update(prev)
    assert posed is not None
    assert not book.has_visemes()
    assert float(posed[25, 1]) > float(rest[25, 1]) + 3.0
    assert float(posed[21, 1]) < float(rest[21, 1]) - 2.0
