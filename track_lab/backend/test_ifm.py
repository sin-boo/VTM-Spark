from __future__ import annotations

import math

import numpy as np

from .ifm import IfmPacket, apply_shapes, parse_packet, rest_landmarks


def test_parses_v1_hyphen_packet() -> None:
    packet = parse_packet(
        "mouthSmile_L-40|jawOpen-25|eyeBlink_R-10"
        "|=head#-12.5,8.0,3.0,0.1,0.2,0.3|rightEye#1,2,3|leftEye#4,5,6"
    )
    assert packet is not None
    assert packet.get("mouthSmile_L") == 0.4
    assert packet.get("jawOpen") == 0.25
    assert packet.head["yaw"] == 8.0
    assert packet.position["z"] == 0.3
    assert packet.left_eye[0] == 4.0
    assert packet.has_right_eye is True
    assert packet.has_left_eye is True


def test_look_of_uses_ifm_eye_vectors() -> None:
    from .ifm import look_of

    packet = IfmPacket()
    packet.right_eye = (0.0, -25.0, 0.0)
    packet.left_eye = (0.0, -25.0, 0.0)
    packet.has_right_eye = True
    packet.has_left_eye = True
    look = look_of(packet)
    assert look["x"] > 0.5
    assert abs(look["y"]) < 0.05


def test_look_of_falls_back_to_blendshapes() -> None:
    from .ifm import look_of

    packet = parse_packet("eyeLookOut_R-100|eyeLookIn_L-100")
    assert packet is not None
    look = look_of(packet)
    assert look["x"] > 0.5
    assert packet.has_right_eye is False
    assert packet.has_left_eye is False


def test_look_of_look_up_beats_rest_eye_bones() -> None:
    from .ifm import look_of

    packet = parse_packet(
        "eyeLookUp_L-80|eyeLookUp_R-80|eyeLookDown_L-0|eyeLookDown_R-0"
        "|rightEye#6.0,2.0,0.2|leftEye#6.0,-1.6,0.1"
    )
    assert packet is not None
    look = look_of(packet)
    rest_eyes = parse_packet("eyeLookUp_L-0|eyeLookUp_R-0|rightEye#6,2,0|leftEye#6,-1,0")
    assert rest_eyes is not None
    assert look["y"] < -0.5
    assert look_of(rest_eyes)["y"] > look["y"]


def test_look_of_look_up_ignores_bone_yaw() -> None:
    from .ifm import look_of

    packet = parse_packet(
        "eyeLookUp_L-100|eyeLookUp_R-100|eyeLookDown_L-0|eyeLookDown_R-0"
        "|rightEye#8.0,18.0,0|leftEye#8.0,16.0,0"
    )
    assert packet is not None
    look = look_of(packet)
    assert look["y"] < -0.7
    assert abs(look["x"]) < 0.2


def test_look_quiet_rejects_a_look_up() -> None:
    from .ifm import look_quiet

    assert look_quiet({"x": 0.0, "y": 0.05})
    assert not look_quiet({"x": 0.0, "y": -0.8})


def test_apply_shapes_does_not_slide_lids_with_gaze() -> None:
    rest = rest_landmarks()
    zero = parse_packet("jawOpen-0|=head#0,0,0,0,0,0")
    packet = parse_packet(
        "eyeLookUp_L-100|eyeLookUp_R-100|eyeLookOut_R-100|eyeLookIn_L-100"
        "|rightEye#0,-25,0|leftEye#0,-25,0"
    )
    assert zero is not None and packet is not None
    rest_pts = apply_shapes(rest, zero)
    live = apply_shapes(rest, packet)
    assert np.allclose(live[36:48], rest_pts[36:48], atol=1e-5)


def test_head_of_reads_ifm_pitch_as_look_down() -> None:
    """A real nod down drew as a look up. iFacialMocap +pitch is look-down
    like the rig's; roll flips with it (the pair is one mirror)."""
    import pytest

    from .ifm import head_of

    for raw, want in (("20,0,0", (20.0, 0.0, 0.0)), ("0,8,0", (0.0, 8.0, 0.0)), ("0,0,3", (0.0, 0.0, -3.0))):
        packet = parse_packet(f"jawOpen-0|=head#{raw},0,0,0")
        assert packet is not None
        head = head_of(packet)
        assert (head["pitch"], head["yaw"], head["roll"]) == pytest.approx(want, abs=1e-3)


def _ifm_angles(rot: np.ndarray) -> str:
    """iFacialMocap's head#pitch,yaw,roll for a rig-frame rotation (yaw outermost)."""
    a = math.degrees(math.atan2(rot[0, 2], rot[2, 2]))
    b = math.degrees(math.asin(-rot[1, 2]))
    c = math.degrees(math.atan2(rot[1, 0], rot[1, 1]))
    return f"{b:.6f},{-a:.6f},{-c:.6f}"


def test_turn_from_an_off_level_rest_is_just_a_turn() -> None:
    """Phone below the face: rest reads ~15 deg looking up and a little rolled.
    Taken angle by angle, a plain turn then also nodded (up to ~12 deg on a
    real phone) and rolled ~0.35 deg per degree of turn."""
    import pytest

    from .feel import feel
    from .ifm import head_of, pose_of
    from .rig import FaceRig, _rot_y, head_matrix_yaw_outer

    rest_rot = head_matrix_yaw_outer(5.0, -15.0, 5.0)
    rest = parse_packet(f"jawOpen-0|=head#{_ifm_angles(rest_rot)},0,0,0")
    turned = parse_packet(f"jawOpen-0|=head#{_ifm_angles(rest_rot @ _rot_y(math.radians(-40.0)))},0,0,0")
    assert rest is not None and turned is not None
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_yaw_left": 1.0, "max_yaw_right": 1.0, "max_pitch_up": 1.0, "max_pitch_down": 1.0, "max_roll_left": 1.0, "max_roll_right": 1.0})
    try:
        rig = FaceRig()
        rig.apply(_anime_rest(), _anime_rest(), head_of(rest), pose_of(rest))
        rig.apply(_anime_rest(), _anime_rest(), head_of(turned), pose_of(turned))
        turn = rig.turn()
    finally:
        feel.update(prev)
    assert math.degrees(turn["yaw"]) == pytest.approx(40.0, abs=0.5)
    assert math.degrees(turn["pitch"]) == pytest.approx(0.0, abs=0.05)
    assert math.degrees(turn["roll"]) == pytest.approx(0.0, abs=0.05)


def test_iris_of_slides_inside_the_eye_box() -> None:
    from .ifm import iris_of, look_of

    rest = rest_landmarks()
    zero = iris_of(rest, {"x": 0.0, "y": 0.0})
    right = iris_of(rest, {"x": 1.0, "y": 0.0})
    down = iris_of(rest, {"x": 0.0, "y": 1.0})
    assert float(right[0, 0]) > float(zero[0, 0])
    assert float(right[1, 0]) > float(zero[1, 0])
    assert float(down[0, 1]) > float(zero[0, 1])
    packet = IfmPacket()
    packet.right_eye = (25.0, 0.0, 0.0)
    packet.left_eye = (25.0, 0.0, 0.0)
    packet.has_right_eye = True
    packet.has_left_eye = True
    look = look_of(packet)
    assert look["y"] < -0.5
    up = iris_of(rest, look)
    assert float(up[0, 1]) < float(zero[0, 1])


def test_parses_v2_ampersand_packet() -> None:
    packet = parse_packet("mouthSmile_R&55|jawOpen&0|=head#0,-15,0,0,0,0")
    assert packet is not None
    assert packet.get("mouthSmile_R") == 0.55
    assert packet.head["yaw"] == -15.0


def test_normalizes_arkit_left_right_names() -> None:
    packet = parse_packet("eyeBlinkLeft&80|mouthSmileRight-30")
    assert packet is not None
    assert packet.get("eyeBlink_L") == 0.8
    assert packet.get("mouthSmile_R") == 0.3


def test_weights_from_arkit_closed_is_zero() -> None:
    from .ifm import weights_from_arkit

    packet = parse_packet("jawOpen-0|mouthSmile_L-0|mouthSmile_R-0|=head#0,0,0,0,0,0")
    assert packet is not None
    weights = weights_from_arkit(packet)
    assert weights["A"] < 0.05
    assert weights["smile"] < 0.05
    assert weights["sad"] < 0.05
    assert weights["I"] < 0.05
    assert weights["U"] < 0.05


def test_weights_from_arkit_jaw_is_a() -> None:
    from .ifm import weights_from_arkit

    packet = parse_packet("jawOpen-80|=head#0,0,0,0,0,0")
    assert packet is not None
    weights = weights_from_arkit(packet)
    assert weights["A"] > 0.2


def test_weights_from_arkit_smile() -> None:
    from .ifm import weights_from_arkit

    packet = parse_packet("mouthSmile_L-70|mouthSmile_R-70")
    assert packet is not None
    weights = weights_from_arkit(packet)
    assert weights["smile"] > 0.3
    assert weights["sad"] == 0.0


def test_jaw_open_feeds_the_same_a_mix() -> None:
    from .ifm import weights_from_arkit

    closed = parse_packet("jawOpen-0|=head#0,0,0,0,0,0")
    opened = parse_packet("jawOpen-80|=head#0,0,0,0,0,0")
    assert closed is not None and opened is not None
    rest = rest_landmarks()
    live = apply_shapes(rest, opened)
    assert abs(float(live[60, 1] - live[64, 1])) > abs(float(rest[60, 1] - rest[64, 1])) + 0.05
    assert weights_from_arkit(opened)["A"] > 0.2
    assert weights_from_arkit(closed)["A"] < 0.05


def test_one_percent_is_not_fully_open() -> None:
    packet = parse_packet("jawOpen-1|mouthClose&1|eyeBlink_L-1")
    assert packet is not None
    assert abs(packet.get("jawOpen") - 0.01) < 1e-9
    assert abs(packet.get("mouthClose") - 0.01) < 1e-9
    assert abs(packet.get("eyeBlink_L") - 0.01) < 1e-9


def test_rest_template_lips_are_shut() -> None:
    rest = rest_landmarks()
    assert abs(float(rest[64, 1] - rest[60, 1])) < 0.02


def test_arkit_closed_stays_closed_after_talking() -> None:
    from .ifm import weights_from_arkit

    opened = parse_packet("jawOpen-70|=head#0,0,0,0,0,0")
    assert opened is not None
    assert weights_from_arkit(opened)["A"] > 0.3
    for raw in ("jawOpen-0", "jawOpen-1", "jawOpen-3"):
        closed = parse_packet(f"{raw}|=head#0,0,0,0,0,0")
        assert closed is not None
        assert weights_from_arkit(closed)["A"] < 0.05, (raw, weights_from_arkit(closed)["A"])


def test_smile_widens_the_spread_corners() -> None:
    rest = rest_landmarks()
    packet = parse_packet("mouthSmile_L-70|mouthSmile_R-70")
    assert packet is not None
    live = apply_shapes(rest, packet)
    assert float(live[62, 0] - live[58, 0]) > float(rest[62, 0] - rest[58, 0])


def test_lip_raise_without_jaw_still_opens_a() -> None:
    from .ifm import weights_from_arkit

    packet = parse_packet(
        "mouthUpperUp_L-60|mouthUpperUp_R-60|"
        "mouthLowerDown_L-70|mouthLowerDown_R-70|=head#0,0,0,0,0,0"
    )
    assert packet is not None
    rest = rest_landmarks()
    live = apply_shapes(rest, packet)
    assert abs(float(live[60, 1] - live[64, 1])) > abs(
        float(rest[60, 1] - rest[64, 1])
    ) + 0.04
    assert weights_from_arkit(packet)["A"] > 0.2
    smile = parse_packet("mouthSmile_L-70|mouthSmile_R-70|=head#0,0,0,0,0,0")
    assert smile is not None
    assert weights_from_arkit(smile)["A"] < 0.05


def test_lan_ipv4_skips_loopback_and_link_local() -> None:
    from .ifm import is_lan_ipv4, subnet_broadcast

    assert is_lan_ipv4("192.168.1.42")
    assert is_lan_ipv4("10.0.0.8")
    assert not is_lan_ipv4("127.0.0.1")
    assert not is_lan_ipv4("169.254.1.1")
    assert not is_lan_ipv4("8")
    assert subnet_broadcast("10.0.0.5") == "10.0.0.255"


def test_handshake_finds_phone_without_typed_ip() -> None:
    from .ifm import handshake_targets

    dests = handshake_targets("", "192.168.1.9", 49983)
    names = [ip for ip, port in dests]
    assert dests[0] == ("192.168.1.9", 49983)
    assert "255.255.255.255" in names
    forced = handshake_targets("192.168.1.7", "", 49983)
    assert forced[0] == ("192.168.1.7", 49983)
    live = handshake_targets("192.168.1.7", "192.168.1.9", 49983, live_ip="192.168.1.9")
    assert live == [("192.168.1.9", 49983)]


def test_handshake_never_targets_this_pc(monkeypatch) -> None:
    from . import ifm
    from .ifm_cam import IfmCam

    monkeypatch.setattr(ifm, "lan_ipv4s", lambda: ["192.168.0.2"])
    dests = [ip for ip, _port in ifm.handshake_targets("192.168.0.2", "", 49983)]
    assert "192.168.0.2" not in dests
    assert "192.168.0.255" in dests
    cam = IfmCam()
    cam.configure(host="192.168.0.2")
    assert cam.host == ""
    cam.configure(host="192.168.0.4")
    assert cam.host == "192.168.0.4"
    cam.configure(host="")
    assert cam.host == ""


def test_ping_is_silent_while_receiving() -> None:
    from .ifm_cam import IfmCam

    class Spy:
        sent: list[tuple[str, int]] = []

        def sendto(self, _payload: bytes, addr: tuple[str, int]) -> None:
            self.sent.append(addr)

    cam = IfmCam()
    sock = Spy()
    cam.receiving = True
    cam._ping(sock, now=100.0, last=0.0)  # type: ignore[arg-type]
    assert sock.sent == []
    cam.receiving = False
    cam._ping(sock, now=100.0, last=0.0)  # type: ignore[arg-type]
    assert sock.sent


def test_hold_packet_keeps_missing_jaw() -> None:
    from .ifm import hold_packet

    full = parse_packet("jawOpen-80|mouthSmile_L-40|=head#2,4,1,0,0,0")
    stub = parse_packet("mouthSmile_L-50")
    assert full is not None and stub is not None
    held = hold_packet(full, stub)
    assert abs(held.get("jawOpen") - 0.8) < 1e-9
    assert abs(held.get("mouthSmile_L") - 0.5) < 1e-9
    assert held.head["yaw"] == 4.0


def test_keep_sender_ignores_other_ip_while_live() -> None:
    from .ifm import keep_sender

    assert keep_sender("", "192.168.0.4", live=False)
    assert keep_sender("192.168.0.4", "192.168.0.4", live=True)
    assert not keep_sender("192.168.0.4", "192.168.0.9", live=True)
    assert keep_sender("192.168.0.4", "192.168.0.9", live=False)


def test_ifm_head_is_not_eased_before_the_bench() -> None:
    """The bench's Smooth is the one ease. A second one here (plus a dead
    band) left the iPhone head trailing the webcam path and moving in steps."""
    from .ifm_cam import _unwrap_head

    import pytest

    prev = {"pitch": 1.0, "yaw": 2.0, "roll": 0.0}
    nxt = {"pitch": 1.2, "yaw": 12.1, "roll": -0.3}
    assert _unwrap_head(prev, nxt) == pytest.approx(nxt)
    assert _unwrap_head(None, nxt) == nxt


def test_draw_ifm_numbers_labels_every_rest_point() -> None:
    from .feel import feel
    from .ifm_cam import _draw_ifm

    rest = rest_landmarks()
    prev = feel.payload()
    info = {"fps": 60.0, "peer": "192.168.0.4:65077", "primary": "10.0.0.1"}
    try:
        feel.update({"show_face": 1, "show_ids": 0})
        plain = _draw_ifm(rest, True, info)
        feel.update({"show_ids": 1})
        numbered = _draw_ifm(rest, True, info)
        assert numbered.shape == plain.shape
        assert int(numbered.sum()) > int(plain.sum())
    finally:
        feel.update(prev)


def _anime_rest() -> np.ndarray:
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
    rest[20] = [42.0, 69.0, 1.0]
    rest[21] = [50.0, 67.0, 1.0]
    rest[22] = [58.0, 69.0, 1.0]
    rest[23] = [40.0, 72.0, 1.0]
    rest[24] = [44.0, 75.0, 1.0]
    rest[25] = [50.0, 77.0, 1.0]
    rest[26] = [60.0, 72.0, 1.0]
    rest[27] = [56.0, 75.0, 1.0]
    return rest


def test_pose_of_tilt_equals_head_roll() -> None:
    from .ifm import pose_of

    from .ifm import head_of

    packet = parse_packet("jawOpen-0|=head#4.0,-12.0,18.5,0,0,0")
    assert packet is not None
    pose = pose_of(packet)
    assert pose["tilt"] == head_of(packet)["roll"]
    assert pose["tz"] == 0.0


def test_ifm_head_unwraps_yaw_instead_of_flipping() -> None:
    from .ifm_cam import _unwrap_head

    prev = {"pitch": 0.0, "yaw": 170.0, "roll": 0.0}
    nxt = {"pitch": 0.0, "yaw": -170.0, "roll": 0.0}
    mixed = _unwrap_head(prev, nxt)
    assert mixed["yaw"] == 190.0


def test_facerig_rolls_when_ifm_tilt_changes() -> None:
    from .feel import feel
    from .rig import FaceRig

    rest = _anime_rest()
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
    feel.update({"smoothing": 0.0})
    try:
        rig = FaceRig()
        rig.apply(rest, rest, head0, origin)
        live = dict(origin)
        live["tilt"] = 25.0
        out = rest
        for _ in range(12):
            out = rig.apply(
                rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 25.0}, live
            )
        assert out is not None
        assert abs(float(out[11, 1]) - float(out[19, 1])) > 1.0
        assert abs(float(out[11, 1]) - float(rest[11, 1])) > 0.8
    finally:
        feel.update(prev)


def test_neck_offset_keeps_each_motion_on_its_axis() -> None:
    from .rig import NECK_FORWARD, NECK_UP, neck_offset

    r = math.radians
    assert neck_offset(0.0, 0.0, 0.0) == (0.0, 0.0)
    # A turn swings the eyes sideways only; which way follows the turn.
    x, y = neck_offset(r(20.0), 0.0, 0.0)
    assert abs(x - NECK_FORWARD * math.sin(r(20.0))) < 1e-9 and y == 0.0
    # A nod down drops them; up raises them.
    assert neck_offset(0.0, r(15.0), 0.0)[1] > 0.1
    assert neck_offset(0.0, r(-15.0), 0.0)[1] < -0.1
    # A tilt swings them sideways round a pivot below, dipping a little.
    x, y = neck_offset(0.0, 0.0, r(10.0))
    assert abs(x - NECK_UP * math.sin(r(10.0))) < 1e-9 and 0.0 < y < 0.01
    # A turn that carries a little tilt (real necks do) never rises.
    for turn, tilt in ((30.0, -5.0), (-30.0, 5.0), (30.0, 5.0), (-30.0, -5.0)):
        assert neck_offset(r(turn), 0.0, r(tilt))[1] >= 0.0


def test_ifm_turn_slides_the_head_like_the_webcam() -> None:
    """The webcam's eye midpoint swings round the neck on a turn, so the
    character slides the way it turns. The iPhone used to only spin in place."""
    from .feel import feel
    from .ifm import head_of, pose_of
    from .rig import FaceRig

    rest = _anime_rest()
    straight = parse_packet("jawOpen-0|=head#0,0,0,0,0,0")
    turned = parse_packet("jawOpen-0|=head#0,20,0,0,0,0")
    assert straight is not None and turned is not None
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_yaw_left": 1.0, "max_yaw_right": 1.0})
    try:
        for selfie in (False, True):
            rig = FaceRig()
            rig.selfie = selfie
            rig.apply(rest, rest, head_of(straight), pose_of(straight, 1.0))
            rig.apply(rest, rest, head_of(turned), pose_of(turned, 1.0))
            dx = rig.place()["dx"]
            face_w = float(rest[4, 0] - rest[0, 0])
            assert abs(abs(dx) - 0.188 * face_w) < 0.01 * face_w
            assert abs(rig.place()["dy"]) < 1e-6
            # Same way the face turns, mirrored or not.
            assert dx * rig.turn()["yaw"] > 0.0
            still = FaceRig()
            still.selfie = selfie
            still.apply(rest, rest, head_of(straight), pose_of(straight, 0.0))
            still.apply(rest, rest, head_of(turned), pose_of(turned, 0.0))
            assert still.place()["dx"] == 0.0
    finally:
        feel.update(prev)


def test_draw_ifm_yaws_without_rotating_pts3d() -> None:
    from .feel import feel
    from .ifm import frame_from_packet
    from .ifm_cam import _draw_ifm
    from .rig import project_head

    packet = parse_packet("jawOpen-0|=head#0,30,0,0,0,0")
    assert packet is not None
    face, _weights, _blink = frame_from_packet(packet)
    rest = rest_landmarks()
    raw = apply_shapes(rest, packet)
    assert np.allclose(face.pts_3d[:, :2], raw[:, :2])
    info = {"fps": 60.0, "peer": "192.168.0.4:1", "primary": "10.0.0.1"}
    prev = feel.payload()
    try:
        feel.update({"show_face": 1, "show_ids": 0})
        flat = _draw_ifm(face.pts_3d, True, info)
        turned = _draw_ifm(face.pts_3d, True, info, packet.head)
        assert turned.shape == flat.shape
        assert int(turned.sum()) != int(flat.sum())
        radius = max(float(np.ptp(face.pts_3d[:17, 0])) * 1.05, 1.0)
        xs, _ys = project_head(face.pts_3d[:, 0], face.pts_3d[:, 1], packet.head, radius)
        assert abs(float(xs[16] - face.pts_3d[16, 0])) > 0.05
    finally:
        feel.update(prev)


def test_ifm_look_up_moves_iris_not_lids() -> None:
    from .ifm import look_of
    from .iris import retarget as retarget_iris

    rest = _anime_rest()
    packet = parse_packet("eyeLookUp_L-100|eyeLookUp_R-100|rightEye#6,2,0|leftEye#6,-1,0")
    assert packet is not None
    look = look_of(packet)
    assert look["y"] < -0.5
    iris_rest, method_rest = retarget_iris(rest, look={"x": 0.0, "y": 0.0})
    iris_up, method_up = retarget_iris(rest, look=look)
    assert method_rest == "look" and method_up == "look"
    rest_y = {int(row["id"]): float(row["y"]) for row in iris_rest}
    up_y = {int(row["id"]): float(row["y"]) for row in iris_up}
    assert up_y[28] < rest_y[28]
    assert up_y[29] < rest_y[29]


def test_ifm_look_up_nods_the_head_rig() -> None:
    from .feel import feel
    from .ifm import head_of
    from .rig import FaceRig

    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[5] = [50.0, 12.0, 1.0]
    rest[15] = [50.0, 40.0, 1.0]
    rest[2] = [50.0, 78.0, 1.0]
    # iFacialMocap +pitch looks down, so a look up comes in negative.
    packet = parse_packet("jawOpen-0|=head#-28.0,0,0,0,0,0")
    assert packet is not None
    head = head_of(packet)
    origin = {"cx": 200.0, "cy": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    prev = feel.payload()
    feel.update({"smoothing": 0.0, "max_pitch_up": 1.0, "max_pitch_down": 1.0})
    try:
        rig = FaceRig()
        rig.apply(rest, rest, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, origin)
        out = rest
        for _ in range(12):
            out = rig.apply(rest, rest, head, origin)
        pitch = rig.turn()["pitch"]
    finally:
        feel.update(prev)
    assert out is not None
    assert pitch < 0.0
    rest_chin = abs(float(rest[2, 1] - rest[15, 1]))
    rest_brow = abs(float(rest[15, 1] - rest[5, 1]))
    chin = abs(float(out[2, 1] - out[15, 1]))
    brow = abs(float(out[15, 1] - out[5, 1]))
    # A look-up tips the face: the nose, in front, rises toward the brows.
    # The chin stays a chin.
    assert brow < 0.9 * rest_brow
    assert 0.75 * rest_chin < chin < 1.15 * rest_chin
    assert float(out[5, 1]) < float(out[15, 1]) < float(out[2, 1])


def test_mapped_blink_is_not_applied_twice() -> None:
    from .eye_bits import DEFAULT_ON, DEFAULT_TO, bits as eyes
    from .feel import feel
    from .retarget import FaceExpr

    rest = _anime_rest()
    template = rest_landmarks()
    closed_pkt = parse_packet("eyeBlink_L-0|eyeBlink_R-0|=head#0,0,0,0,0,0")
    blink_pkt = parse_packet("eyeBlink_L-80|eyeBlink_R-80|=head#0,0,0,0,0,0")
    assert closed_pkt is not None and blink_pkt is not None
    closed = apply_shapes(template, closed_pkt)
    blinked = apply_shapes(template, blink_pkt)
    assert abs(float(blinked[37, 1] - blinked[40, 1])) < abs(
        float(closed[37, 1] - closed[40, 1])
    ) - 0.02
    expr = FaceExpr()
    prev = feel.payload()
    snap = eyes.snapshot()
    feel.update({"smoothing": 0.0, "response": 1.0})
    try:
        eyes.restore(frozenset(DEFAULT_ON), dict(DEFAULT_TO))
        expr.apply(rest, rest, closed, {"l": 0.0, "r": 0.0}, mouth_pts=closed)
        opened = rest
        slammed = rest
        for _ in range(8):
            opened = expr.apply(
                rest, rest, closed, {"l": 0.0, "r": 0.0}, mouth_pts=closed
            )
        expr_s = FaceExpr()
        expr_s.apply(rest, rest, closed, {"l": 0.0, "r": 0.0}, mouth_pts=closed)
        for _ in range(8):
            slammed = expr_s.apply(
                rest, rest, closed, {"l": 0.8, "r": 0.8}, mouth_pts=closed
            )
        assert opened is not None and slammed is not None
        assert np.allclose(opened[[11, 13], :2], slammed[[11, 13], :2], atol=0.15)
        assert np.allclose(opened[[17, 19], :2], slammed[[17, 19], :2], atol=0.15)
        assert float(slammed[12, 1]) > float(opened[12, 1]) + 0.15
        assert float(slammed[18, 1]) > float(opened[18, 1]) + 0.15
        without = rest
        with_blink = rest
        expr_w = FaceExpr()
        expr_w.apply(rest, rest, closed, {"l": 0.0, "r": 0.0}, mouth_pts=closed)
        expr2 = FaceExpr()
        expr2.apply(rest, rest, closed, {"l": 0.0, "r": 0.0}, mouth_pts=closed)
        for _ in range(8):
            without = expr_w.apply(
                rest, rest, blinked, {"l": 0.0, "r": 0.0}, mouth_pts=blinked
            )
        for _ in range(8):
            with_blink = expr2.apply(
                rest, rest, blinked, {"l": 0.8, "r": 0.8}, mouth_pts=blinked
            )
        assert without is not None and with_blink is not None
        assert np.allclose(without[[11, 13], :2], with_blink[[11, 13], :2], atol=0.15)
        assert np.allclose(without[[17, 19], :2], with_blink[[17, 19], :2], atol=0.15)
        assert float(with_blink[12, 1]) >= float(without[12, 1]) - 0.05
        assert float(with_blink[18, 1]) >= float(without[18, 1]) - 0.05
    finally:
        feel.update(prev)
        eyes.restore(*snap)


def test_left_wink_closes_image_left_lid_only() -> None:
    from .eye_bits import DEFAULT_ON, DEFAULT_TO, bits as eyes
    from .feel import feel
    from .retarget import FaceExpr

    rest = _anime_rest()
    template = rest_landmarks()
    closed_pkt = parse_packet("eyeBlink_L-0|eyeBlink_R-0|=head#0,0,0,0,0,0")
    left_pkt = parse_packet("eyeBlink_L-80|eyeBlink_R-0|=head#0,0,0,0,0,0")
    right_pkt = parse_packet("eyeBlink_L-0|eyeBlink_R-80|=head#0,0,0,0,0,0")
    assert closed_pkt is not None and left_pkt is not None and right_pkt is not None
    closed = apply_shapes(template, closed_pkt)
    left_src = apply_shapes(template, left_pkt)
    right_src = apply_shapes(template, right_pkt)
    prev = feel.payload()
    snap = eyes.snapshot()
    feel.update({"smoothing": 0.0, "response": 1.0})
    try:
        eyes.restore(frozenset(DEFAULT_ON), dict(DEFAULT_TO))
        open_expr = FaceExpr()
        opened = rest
        for _ in range(8):
            opened = open_expr.apply(
                rest, rest, closed, {"l": 0.0, "r": 0.0}, mouth_pts=closed
            )
        left_expr = FaceExpr()
        left = rest
        for _ in range(8):
            left = left_expr.apply(
                rest, rest, left_src, {"l": 0.8, "r": 0.0}, mouth_pts=left_src
            )
        right_expr = FaceExpr()
        right = rest
        for _ in range(8):
            right = right_expr.apply(
                rest, rest, right_src, {"l": 0.0, "r": 0.8}, mouth_pts=right_src
            )
        assert opened is not None and left is not None and right is not None
        assert float(left[12, 1]) > float(opened[12, 1]) + 0.15
        assert abs(float(left[18, 1]) - float(opened[18, 1])) < 0.2
        assert float(right[18, 1]) > float(opened[18, 1]) + 0.15
        assert abs(float(right[12, 1]) - float(opened[12, 1])) < 0.2
    finally:
        feel.update(prev)
        eyes.restore(*snap)


def test_jaw_open_keeps_lip_gap_after_retarget() -> None:
    from .feel import feel
    from .mouth_bits import DEFAULT_ON, DEFAULT_TO, bits as mouths
    from .retarget import FaceExpr

    rest = _anime_rest()
    template = rest_landmarks()
    closed_pkt = parse_packet("jawOpen-0|=head#0,0,0,0,0,0")
    opened_pkt = parse_packet("jawOpen-80|=head#0,0,0,0,0,0")
    assert closed_pkt is not None and opened_pkt is not None
    closed = apply_shapes(template, closed_pkt)
    opened = apply_shapes(template, opened_pkt)
    expr = FaceExpr()
    prev = feel.payload()
    snap = mouths.snapshot()
    feel.update(
        {"smoothing": 0.0, "response": 1.0, "mouth": 0.69, "use_visemes": 0.0}
    )
    try:
        mouths.restore(frozenset(DEFAULT_ON), dict(DEFAULT_TO))
        expr.apply(rest, rest, closed, {"l": 0.0, "r": 0.0}, mouth_pts=closed)
        out = rest
        for _ in range(8):
            out = expr.apply(
                rest, rest, opened, {"l": 0.0, "r": 0.0}, mouth_pts=opened
            )
        assert out is not None
        rest_gap = abs(float(rest[25, 1] - rest[21, 1]))
        live_gap = abs(float(out[25, 1] - out[21, 1]))
        assert live_gap > rest_gap + 1.0
    finally:
        feel.update(prev)
        mouths.restore(*snap)


def test_drive_ifm_opens_mouth_and_closes_lids() -> None:
    from .feel import feel
    from .ifm import drive_ifm, weights_from_arkit

    rest = _anime_rest()
    opened = parse_packet("jawOpen-80|eyeBlink_L-80|eyeBlink_R-80|=head#0,0,0,0,0,0")
    assert opened is not None
    prev = feel.payload()
    try:
        feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
        weights = weights_from_arkit(opened)
        out = drive_ifm(rest, weights, {"l": 0.8, "r": 0.8})
    finally:
        feel.update(prev)
    assert out is not None
    assert float(out[25, 1]) > float(rest[25, 1]) + 1.0
    assert float(out[12, 1]) > float(rest[12, 1]) + 0.15
    assert float(out[18, 1]) > float(rest[18, 1]) + 0.15


def test_drive_ifm_smile_spreads_corners() -> None:
    from .feel import feel
    from .ifm import drive_ifm, weights_from_arkit

    rest = _anime_rest()
    packet = parse_packet("mouthSmile_L-70|mouthSmile_R-70")
    assert packet is not None
    prev = feel.payload()
    try:
        feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
        out = drive_ifm(rest, weights_from_arkit(packet), {"l": 0.0, "r": 0.0})
    finally:
        feel.update(prev)
    assert out is not None
    rest_w = float(rest[26, 0] - rest[23, 0])
    live_w = float(out[26, 0] - out[23, 0])
    assert live_w > rest_w + 1.0


def test_ifm_mixer_frame_has_no_pts3d() -> None:
    from .feel import feel
    from .ifm import blink_of, brow_of, drive_ifm, look_of, weights_from_arkit
    from .osf_cam import OsfFrame

    packet = parse_packet("jawOpen-50|mouthSmile_L-40|mouthSmile_R-40|=head#0,0,0,0,0,0")
    assert packet is not None
    snap = OsfFrame(
        weights=weights_from_arkit(packet),
        blink=blink_of(packet),
        look=look_of(packet),
        brow=brow_of(packet),
        faces=1,
        source="ifm",
    )
    assert snap.pts_3d is None
    assert snap.mouth_2d is None
    assert snap.source == "ifm"
    prev = feel.payload()
    try:
        feel.update({"smoothing": 0.0, "mouth": 0.5})
        rest = _anime_rest()
        out = drive_ifm(rest, snap.weights, snap.blink, snap.brow)
    finally:
        feel.update(prev)
    assert out is not None
    assert float(out[25, 1]) > float(rest[25, 1])


def test_pose_ifm_skips_face_expr() -> None:
    from .face import FaceBench
    from .feel import feel
    from .ifm import blink_of, brow_of, look_of, weights_from_arkit
    from .osf_cam import OsfFrame

    rest = _anime_rest()
    packet = parse_packet(
        "jawOpen-80|eyeBlink_L-80|browInnerUp-60|=head#0,0,0,0,0,0"
    )
    assert packet is not None
    bench = FaceBench(rest_pts=rest.copy())
    bench._expr.apply = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("FaceExpr must not run on IFM")
    )
    frame = OsfFrame(
        weights=weights_from_arkit(packet),
        blink=blink_of(packet),
        look=look_of(packet),
        brow=brow_of(packet),
        head={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        faces=1,
        source="ifm",
    )
    prev = feel.payload()
    try:
        feel.update({"smoothing": 0.0, "response": 1.0, "mouth": 0.5})
        posed = bench._pose_ifm(frame)
    finally:
        feel.update(prev)
    assert posed is not None
    assert float(posed[25, 1]) > float(rest[25, 1])


def test_ifm_doodle_has_mouth_features() -> None:
    from .ifm import IfmPacket, frame_from_packet
    from .visemes import mouth_features

    face, _weights, _blink = frame_from_packet(IfmPacket())
    feat = mouth_features(face, None)
    assert feat is not None
    assert feat["open"] >= 0.0
    assert feat["width"] > 0.0


def test_start_rest_retries_after_capture_clock_finishes() -> None:
    import time

    from .calibrate import CAPTURE_SEC, Calibrator

    cal = Calibrator()
    cal.capturing = "rest"
    cal._started = time.perf_counter() - CAPTURE_SEC - 0.05
    cal.start("rest")
    assert cal.capturing == "rest"


def test_start_rest_blocks_while_capture_runs() -> None:
    from .calibrate import Calibrator

    cal = Calibrator()
    cal.start("rest")
    try:
        cal.start("rest")
    except ValueError as exc:
        assert "Already capturing" in str(exc)
    else:
        raise AssertionError("expected Already capturing")
