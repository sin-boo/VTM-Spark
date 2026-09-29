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


def _pose_face(eye_tilt_deg: float, h: np.ndarray) -> SimpleNamespace:
    """68 OSF-ish landmarks with the eye line at ``eye_tilt_deg``, solved as ``h``."""
    lms = np.zeros((68, 3), dtype=np.float32)
    lms[:, 2] = 1.0
    a = math.radians(eye_tilt_deg)
    def put(i: int, x: float, y: float) -> None:
        # OSF rows are (y, x, conf).
        lms[i, 0], lms[i, 1] = 200 + y, 200 + x
    for i, off in ((36, -60), (39, -30), (42, 30), (45, 60)):
        put(i, off * math.cos(a), off * math.sin(a))
    put(30, 0.0, 40.0)
    put(0, -90.0, 0.0)
    put(16, 90.0, 0.0)
    face = _face(h)
    face.lms = lms
    return face


def test_half_turn_solve_is_turned_back_not_refused() -> None:
    """On a real DroidCam feed the solve sat half a turn round the view axis
    (roll ~176 on an upright face). Refused as a bad solve, it froze the turn
    for good; turned back, turn and nod read as they are."""
    from backend.osf_cam import _face_pose

    good = _face_pose(_pose_face(3.0, _rot("z", 3.0)))
    assert good["head_ok"] == 1.0 and abs(good["tilt"] - 3.0) < 1.0
    turned = _rot("x", -19.0) @ _rot("y", -27.0)
    upright = _pnp_head(_face(turned))
    flipped = _pnp_head(_face(_rot("z", 180.0) @ turned))
    assert upright is not None and flipped is not None
    for key in ("yaw", "pitch", "roll"):
        assert abs(flipped[key] - upright[key]) < 1e-6
    pose = _face_pose(_pose_face(3.0, _rot("z", 180.0)))
    assert pose["head_ok"] == 1.0 and abs(pose["tilt"]) < 1.0


def test_solve_off_the_eye_line_is_still_refused() -> None:
    """Off by something other than a half turn (roll 123 while the eyes read
    3): a bad solve. Locked as rest, it tipped the hair to the roll limit."""
    from backend.osf_cam import _face_pose

    bad = _face_pose(_pose_face(3.0, _rot("z", 110.0)))
    assert bad["head_ok"] == 0.0
    assert abs(bad["tilt"] - bad["tilt_eyes"]) < 1e-6


def test_rig_never_locks_rest_on_a_bad_solve() -> None:
    from backend.rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 0] = np.linspace(-50, 50, 28)
    rest[:, 1] = np.linspace(-40, 40, 28)
    rest[:, 2] = 1.0
    pose = {"ok": 1.0, "cx": 0.0, "cy": 0.0, "bx": 0.0, "by": 0.0, "scale": 100.0, "tz": 0.0}
    rig = FaceRig()
    bad = dict(pose, tilt=-179.0, head_ok=0.0)
    assert rig._sync(rest, {"yaw": 170.0, "pitch": 0.0}, bad) is False
    assert rig.locked is False
    rig._sync(rest, {"yaw": 5.0, "pitch": 0.0}, dict(pose, tilt=3.0, head_ok=1.0))
    assert rig.locked is True
    # A later bad frame keeps the last good turn instead of jumping.
    rig._sync(rest, {"yaw": -175.0, "pitch": 0.0}, dict(pose, tilt=3.0, head_ok=0.0))
    assert abs(rig.turn()["yaw"]) < 1e-6
    assert abs(rig.turn()["roll"]) < 1e-6


def test_rig_locks_on_eye_line_when_solves_never_come_good() -> None:
    from backend.rig import _BAD_SOLVE_LOCK, FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 0] = np.linspace(-50, 50, 28)
    rest[:, 1] = np.linspace(-40, 40, 28)
    rest[:, 2] = 1.0
    pose = {"ok": 1.0, "cx": 0.0, "cy": 0.0, "bx": 0.0, "by": 0.0, "scale": 100.0, "tz": 0.0}
    rig = FaceRig()
    bad = dict(pose, tilt=4.0, head_ok=0.0)
    for _ in range(_BAD_SOLVE_LOCK - 1):
        rig._sync(rest, {"yaw": 170.0, "pitch": 80.0}, bad)
    assert rig.locked is False
    rig._sync(rest, {"yaw": 170.0, "pitch": 80.0}, bad)
    assert rig.locked is True
    # Locked straight on at the eye-line tilt, not on the bad solve.
    assert abs(rig.turn()["yaw"]) < 1e-6 and abs(rig.turn()["roll"]) < 1e-6


def test_turn_plus_nod_is_not_mistaken_for_a_flip() -> None:
    """A correct solve at 60 turn / 30 nod slants the eye line ~41 deg."""
    from backend.osf_cam import _PNP_TILT_TRUST

    slant = math.degrees(math.atan(math.sin(math.radians(30)) * math.tan(math.radians(60))))
    assert slant < _PNP_TILT_TRUST < 165.0


def _rig_rest() -> tuple[np.ndarray, dict]:
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 0] = np.linspace(-50, 50, 28)
    rest[:, 1] = np.linspace(-40, 40, 28)
    rest[:, 2] = 1.0
    pose = {"ok": 1.0, "cx": 0.0, "cy": 0.0, "bx": 0.0, "by": 0.0, "scale": 100.0, "tz": 0.0}
    return rest, pose


def test_eye_line_lock_gives_way_to_the_first_clean_solve() -> None:
    """Kept on the stand-in zero, a camera 23 deg below eye level read as a
    permanent nod pinned at the pitch stop."""
    from backend.rig import _BAD_SOLVE_LOCK, FaceRig

    rest, pose = _rig_rest()
    rig = FaceRig()
    for _ in range(_BAD_SOLVE_LOCK):
        rig._sync(rest, {"yaw": 170.0, "pitch": 80.0}, dict(pose, tilt=0.0, head_ok=0.0))
    assert rig.locked is True
    level = {"yaw": 0.0, "pitch": 23.0}
    rig._sync(rest, level, dict(pose, tilt=0.0, head_ok=1.0))
    rig._sync(rest, level, dict(pose, tilt=0.0, head_ok=1.0))
    assert abs(rig.turn()["pitch"]) < 1e-6
    # From then on it is a normal lock: a real nod reads as one.
    rig._sync(rest, {"yaw": 0.0, "pitch": 26.0}, dict(pose, tilt=0.0, head_ok=1.0))
    assert rig.turn()["pitch"] > 0.04


def test_bad_frame_keeps_the_tilt_continuous() -> None:
    """A bad frame's tilt is the bare eye line; mid-turn that sat ~10 deg off
    the solved roll and twitched the head for a frame."""
    from backend.rig import FaceRig

    rest, pose = _rig_rest()
    rig = FaceRig()
    rig._sync(rest, {"yaw": 0.0, "pitch": 0.0}, dict(pose, tilt=0.0, tilt_eyes=0.0, head_ok=1.0))
    # Turned: solved roll 12, eye line 2.
    rig._sync(rest, {"yaw": 30.0, "pitch": 0.0}, dict(pose, tilt=12.0, tilt_eyes=2.0, head_ok=1.0))
    good = rig.turn()["roll"]
    # Next frame's solve flips; the eye line barely moved.
    rig._sync(rest, {"yaw": -150.0, "pitch": 0.0}, dict(pose, tilt=2.5, tilt_eyes=2.5, head_ok=0.0))
    assert abs(rig.turn()["roll"] - good) < math.radians(1.0)
