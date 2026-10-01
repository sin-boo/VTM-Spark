"""The torso turns with the head round the neck base; arms hang; hair turns
with the face; head_sway reaches the webcam. Nothing here touches output/."""

from __future__ import annotations

import json
import math

import numpy as np

from backend import feel as feel_mod
from backend.feel import DEFAULTS, Feel
from backend.skeleton import (
    ARM_EASE_S,
    ArmEase,
    body_oval,
    follow_skeleton,
    skeleton_from_face,
    turn_on_oval,
)


class _Rig:
    """A locked rig holding still at one turn (radians) and place."""

    locked = True

    def __init__(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> None:
        self._turn = {"yaw": yaw, "pitch": pitch, "roll": roll}

    def place(self) -> dict[str, float]:
        return {"cx": 50.0, "cy": 50.0, "scale": 1.0, "body_dx": 0.0, "body_dy": 0.0}

    def turn(self) -> dict[str, float]:
        return dict(self._turn)


def _face() -> np.ndarray:
    pts = np.zeros((28, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    pts[0] = [10.0, 32.0, 1.0]
    pts[2] = [50.0, 78.0, 1.0]
    pts[4] = [90.0, 32.0, 1.0]
    pts[15] = [50.0, 40.0, 1.0]
    pts[21] = [50.0, 62.0, 1.0]
    return pts


def _posed(rig: _Rig, share: float = 1.5) -> dict[int, np.ndarray]:
    body = skeleton_from_face(_face())
    return {int(j["id"]): np.array([j["x"], j["y"]]) for j in follow_skeleton(body, None, rig=rig, share=share)}


def test_rest_pose_is_the_rest_skeleton() -> None:
    body = {int(j["id"]): np.array([j["x"], j["y"]]) for j in skeleton_from_face(_face())}
    posed = _posed(_Rig())
    for idx, xy in body.items():
        assert np.allclose(posed[idx], xy, atol=0.11)  # outputs round to 0.1 px, idx


def test_turn_on_oval_keeps_the_neck_and_narrows_the_shoulders() -> None:
    half, depth = 100.0, 40.0
    turn = math.radians(30.0)
    assert turn_on_oval(0.0, half, depth, turn) == 0.0
    near, far = turn_on_oval(60.0, half, depth, turn), turn_on_oval(-60.0, half, depth, turn)
    # Both shoulders draw in, and shift away from the turn together.
    assert near - far < 120.0
    assert near + far < 0.0


def test_a_head_turn_turns_the_torso_round_a_fixed_neck() -> None:
    rest = _posed(_Rig())
    turned = _posed(_Rig(yaw=math.radians(20.0)))
    assert np.allclose(turned[31], rest[31], atol=0.11)  # outputs round to 0.1 px
    span_rest = np.linalg.norm(rest[32] - rest[34])
    span = np.linalg.norm(turned[32] - turned[34])
    # 1.5 x 20 deg = 30 deg of body turn: the shoulders narrow to about cos 30.
    assert 0.8 * span_rest < span < 0.95 * span_rest


def test_a_tilt_leans_the_shoulders_and_the_arms_still_hang() -> None:
    rest = _posed(_Rig())
    tilted = _posed(_Rig(roll=math.radians(15.0)))
    slope = math.degrees(math.atan2(tilted[34][1] - tilted[32][1], tilted[34][0] - tilted[32][0]))
    assert 15.0 < slope < 30.0  # 1.5 x 15 deg, round the neck base
    for sh, el in ((32, 33), (34, 35)):
        arm = tilted[el] - tilted[sh]
        hang = rest[el] - rest[sh]
        gap = math.degrees(abs(math.atan2(arm[0], arm[1]) - math.atan2(hang[0], hang[1])))
        # Rigid, the arm would turn the full 22.5 deg with the shoulders.
        assert gap < 7.0, el
        assert np.linalg.norm(arm) > 0.95 * np.linalg.norm(hang)


def test_body_turn_zero_only_walks_and_sizes() -> None:
    rest = _posed(_Rig(), share=0.0)
    turned = _posed(_Rig(yaw=0.5, pitch=0.3, roll=0.2), share=0.0)
    for idx, xy in rest.items():
        assert np.allclose(turned[idx], xy, atol=0.11)  # outputs round to 0.1 px, idx


def test_arms_ease_down_softly() -> None:
    clock = {"t": 0.0}
    arms = ArmEase(clock=lambda: clock["t"])
    down = np.array([0.0, 1.0])
    side = np.array([1.0, 0.0])
    assert np.allclose(arms.step({33: side})[33], side)
    clock["t"] = ARM_EASE_S
    mid = arms.step({33: down})[33]
    # About 63% of the way after one time constant, not a snap.
    angle = math.degrees(math.atan2(mid[0], mid[1]))
    assert 20.0 < angle < 45.0
    for _ in range(20):
        clock["t"] += 0.05
        settled = arms.step({33: down})[33]
    assert np.allclose(settled, down, atol=1e-3)
    arms.reset()
    assert np.allclose(arms.step({33: side})[33], side)


def test_body_oval_spans_every_joint() -> None:
    rest_by = {int(j["id"]): j for j in skeleton_from_face(_face())}
    half, depth = body_oval(rest_by)
    reach = max(abs(j["x"] - rest_by[31]["x"]) for j in rest_by.values())
    assert half > reach
    assert math.isclose(depth / half, 0.4)


def test_hair_turns_through_the_face_lens_and_pinned_anchors_keep_the_wide_one() -> None:
    from backend.rig import FaceRig, face_xy, plane_xy

    face = FaceRig()
    face._rest_ms = 100.0
    face._s = 1.0
    face._yaw_r = math.radians(25.0)
    face._pitch_r = math.radians(10.0)
    face._roll_r = 0.0
    radius = 105.0
    xs = np.array([0.0, 0.0, 200.0])
    ys = np.array([0.0, -40.0, -90.0])
    origin = np.array([face._rest_cx + face._dx, face._rest_cy + face._dy])
    for wide, lens in ((False, face_xy), (True, plane_xy)):
        x2, y2 = face.map_hair(xs, ys, wide=wide)
        want_x, want_y = lens(xs, ys, face._yaw_r, face._pitch_r, 0.0, radius, 1.0)
        assert np.allclose(x2 - origin[0], want_x) and np.allclose(y2 - origin[1], want_y), wide


def test_old_head_sway_default_moves_to_the_new_one(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tracking_feel.json"
    monkeypatch.setattr(feel_mod, "FEEL_PATH", path)
    path.write_text(json.dumps({"head_sway": 1.0, "smoothing": 0.2}), encoding="utf-8")
    moved = Feel()
    assert moved.head_sway() == DEFAULTS["head_sway"]
    assert moved.payload()["smoothing"] == 0.2
    assert moved.body_turn() == DEFAULTS["body_turn"]
    moved.save()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["version"] == feel_mod.FEEL_VERSION
    # Chosen on purpose since: kept.
    path.write_text(json.dumps({"head_sway": 1.0, "version": feel_mod.FEEL_VERSION}), encoding="utf-8")
    assert Feel().head_sway() == 1.0
    path.write_text(json.dumps({"head_sway": 0.8}), encoding="utf-8")
    assert Feel().head_sway() == 0.8
