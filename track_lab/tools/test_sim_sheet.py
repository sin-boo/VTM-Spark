"""The simulated iPhone and webcam: same head, same drawing.

Each device hands Track Lab only what the real one does (sim_sheet): the
iFacialMocap text, and OpenSeeFace's landmarks, solve and 3D points.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.test_ifm import _anime_rest
from tools.sim_sheet import IPhone, Webcam

TO_THE_LIMITS = [
    {"turn": 15},
    {"turn": -35},
    {"nod": 10},
    {"nod": 30},
    {"nod": -25},
    {"tilt": 10},
    {"turn": 30, "nod": 15},
]


def _ease_in(device, move: dict[str, float], steps: int = 6):
    frame = None
    for i in range(steps):
        frame = device.frame({key: value * (i + 1) / steps for key, value in move.items()})
    return frame


def _rig_after(device, move: dict[str, float]):
    from backend.rig import FaceRig

    rest = _anime_rest()
    rig = FaceRig()
    still = device.frame({})
    rig.apply(rest, rest, still.head, still.pose)
    frame = _ease_in(device, move)
    rig.apply(rest, rest, frame.head, frame.pose)
    return rig


@pytest.mark.parametrize("move", TO_THE_LIMITS, ids=str)
def test_webcam_and_iphone_draw_the_same_head_and_leave_the_torso(move: dict[str, float]) -> None:
    from backend.feel import feel
    from backend.travel_box import DEFAULT_TRAVEL_BOX, lab_feel_caps

    feel.update(lab_feel_caps(DEFAULT_TRAVEL_BOX))
    phone = _rig_after(IPhone(), move)
    cam = _rig_after(Webcam(), move)
    for key in ("yaw", "pitch", "roll"):
        assert math.degrees(abs(phone.turn()[key] - cam.turn()[key])) < 1.5, key
    width = float(phone._rest_ms)
    a, b = phone.place(), cam.place()
    # The webcam neck swings a little unlike the rig's; the head may differ
    # by that much, the torso not at all.
    assert abs(a["dx"] - b["dx"]) < 0.06 * width and abs(a["dy"] - b["dy"]) < 0.06 * width
    for place in (a, b):
        assert abs(place["body_dx"]) < 1e-6 and abs(place["body_dy"]) < 1e-6


def _expr_after(move: dict[str, float]):
    from backend.retarget import FaceExpr

    rest = _anime_rest()
    cam = Webcam()
    expr = FaceExpr()
    still = cam.frame({})
    expr.apply(rest, rest, still.pts_3d, mouth_pts=still.mouth_2d, keep_mouth=True)
    frame = _ease_in(cam, move)
    out = expr.apply(rest, rest, frame.pts_3d, mouth_pts=frame.mouth_2d, keep_mouth=True)
    return expr, out, rest


@pytest.mark.parametrize("move", [{"turn": 50}, {"turn": -45}, {"nod": -30}, {"turn": 30, "nod": -15}], ids=str)
def test_a_turn_or_look_up_does_not_open_a_shut_mouth(move: dict[str, float]) -> None:
    """The open amount read the image lips, divided by the face's width on
    screen: a 50 deg turn opened a shut mouth halfway."""
    expr, _out, _rest = _expr_after(move)
    assert expr._camera_open() < 0.02


@pytest.mark.parametrize("move", [{"open": 0.6}, {"open": 0.6, "turn": 45}, {"open": 0.6, "nod": -20}], ids=str)
def test_a_real_open_still_reads_whichever_way_you_face(move: dict[str, float]) -> None:
    expr, _out, _rest = _expr_after(move)
    assert expr._camera_open() > 0.1


@pytest.mark.parametrize("move", [{"nod": 25}, {"nod": -25}, {"turn": 40}], ids=str)
def test_webcam_nose_keeps_its_shape_before_the_rig(move: dict[str, float]) -> None:
    """The nose wings copied the image landmarks, which carry the head's own
    nod: a look-down folded the nose into a V, then the rig nodded it again."""
    _expr, out, rest = _expr_after(move)
    assert out is not None
    for wing in (14, 16):
        drawn = out[wing, :2] - out[15, :2]
        home = rest[wing, :2] - rest[15, :2]
        assert float(np.hypot(*(drawn - home))) < 1.0
