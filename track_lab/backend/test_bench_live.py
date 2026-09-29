"""Live bench behaviour around the pupils and a stopped session."""

from __future__ import annotations

from backend.face import FaceBench
from backend.osf_cam import OsfFrame
from backend.test_ifm import _anime_rest


def _bench() -> FaceBench:
    return FaceBench(rest_pts=_anime_rest())


def _ifm_frame(raw: str = "jawOpen-0|=head#0,0,0,0,0,0") -> OsfFrame:
    from backend.ifm import blink_of, brow_of, look_of, parse_packet, weights_from_arkit

    packet = parse_packet(raw)
    assert packet is not None
    return OsfFrame(
        weights=weights_from_arkit(packet),
        blink=blink_of(packet),
        look=look_of(packet),
        brow=brow_of(packet),
        head={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        pose={"cx": 0.0, "cy": 0.0, "bx": 0.0, "by": 0.0, "scale": 1.0, "tz": 0.0, "tilt": 0.0, "ok": 1.0},
        faces=1,
        source="ifm",
    )


def test_pupils_hold_with_the_face_on_a_lost_frame() -> None:
    """The face holds its last pose on a lost frame (or an iPhone stall). The
    pupils used to snap to the still's rest pixels, outside a moved head."""
    bench = _bench()
    bench._on_osf(_ifm_frame())
    held = [dict(row) for row in bench._iris]
    assert held
    bench._on_osf(OsfFrame(faces=0, source="ifm"))
    assert bench._iris == held


def test_pupils_ease_inside_their_eye_not_on_screen() -> None:
    bench = _bench()
    posed = _anime_rest()
    # Pupils a little right of each eye's corner midpoint.
    rows = [
        {"id": 28, "x": 33.0, "y": 36.0, "score": 1.0, "visible": True},
        {"id": 29, "x": 69.0, "y": 36.0, "score": 1.0, "visible": True},
    ]
    bench._ease_pupils(rows, posed, 0.3)
    # The whole head moves 40 px; the pupils sit where they were in the eye.
    moved = posed.copy()
    moved[:, 0] += 40.0
    shifted = [dict(row, x=float(row["x"]) + 40.0) for row in rows]
    out = bench._ease_pupils(shifted, moved, 0.3)
    assert [round(float(row["x"]), 6) for row in out] == [73.0, 109.0]
    # Only iris 29 seen now: it keeps its own history, not 28's.
    out = bench._ease_pupils([shifted[1]], moved, 0.3)
    assert round(float(out[0]["x"]), 6) == 109.0


def test_limiter_change_after_stop_does_not_replay_the_last_frame() -> None:
    """Replaying the held frame after Stop re-locked the reset rig on it and
    brought back its preview, meters and blink."""
    bench = _bench()
    bench._on_osf(_ifm_frame("eyeBlink_L-90|eyeBlink_R-90|=head#0,0,0,0,0,0"))
    assert bench._live_pose is not None
    bench._live_pts = None
    bench._blink = {"l": 0.0, "r": 0.0}
    bench._replay_live_pose()
    assert bench._live_pts is None
    assert bench._blink == {"l": 0.0, "r": 0.0}
