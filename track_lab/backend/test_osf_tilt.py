import math
from types import SimpleNamespace

import cv2
import numpy as np

from backend.osf_cam import _SWAP, _head, _pnp_head


def _rot(axis: str, deg: float) -> np.ndarray:
    vec = np.zeros(3)
    vec["xyz".index(axis)] = math.radians(deg)
    return cv2.Rodrigues(vec)[0]


def _face(h: np.ndarray) -> SimpleNamespace:
    # What OSF solves for head turn H: PnP on (y, x) points, model facing +z.
    osf = -_SWAP @ h
    rvec = cv2.Rodrigues(osf)[0]
    return SimpleNamespace(success=True, rotation=rvec, euler=cv2.RQDecomp3x3(osf)[0])


_CAM = _rot("x", -23.0)  # webcam below / above the face


def test_turn_with_camera_off_level_stays_a_turn() -> None:
    for cam in (np.eye(3), _CAM, _rot("x", 23.0)):
        front = _pnp_head(_face(cam))
        for deg in (-60.0, -45.0, 30.0, 60.0):
            got = _pnp_head(_face(cam @ _rot("y", deg)))
            assert got is not None and front is not None
            assert abs(got["roll"] - front["roll"]) < 0.5
            assert abs(got["pitch"] - front["pitch"]) < 0.5
            assert abs((got["yaw"] - front["yaw"]) + deg) < 0.5


def test_osf_euler_would_have_leaked_the_turn_into_roll() -> None:
    front = _face(_CAM).euler
    turned = _face(_CAM @ _rot("y", 50.0)).euler
    assert abs(turned[2] - front[2]) > 15.0


def test_tilt_and_nod_read_on_their_own_axes() -> None:
    # Tilt is on the screen plane, outside the camera offset and the turn.
    tilt = _pnp_head(_face(_rot("z", 15.0) @ _CAM @ _rot("y", 45.0)))
    assert tilt is not None and abs(tilt["roll"] - 15.0) < 0.5
    nod = _pnp_head(_face(_rot("x", 20.0)))
    assert nod is not None and abs(nod["pitch"] - 20.0) < 0.5 and abs(nod["roll"]) < 0.5


def test_osf_yaw_sign_is_kept_near_front() -> None:
    small = _face(_CAM @ _rot("y", 10.0))
    front = _face(_CAM)
    osf_delta = float(small.euler[1] - front.euler[1])
    ours = _head(small)["yaw"] - _head(front)["yaw"]
    assert osf_delta * ours > 0.0


def test_no_rotation_falls_back_to_osf_euler() -> None:
    face = SimpleNamespace(success=False, rotation=None, euler=[1.0, 2.0, 3.0])
    assert _pnp_head(face) is None
    assert _head(face) == {"pitch": 1.0, "yaw": 2.0, "roll": 3.0}
