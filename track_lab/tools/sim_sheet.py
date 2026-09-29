"""Feed simulated tracker data through Track Lab's live path and draw what comes out.

    python -m tools.sim_sheet                 # saved limiters, Look down 12
    python -m tools.sim_sheet --look-down 3   # any limiter override

Each move is a head pose the performer holds, in their own directions ("turn
to your left"). It becomes what each tracker hands Track Lab, and nothing
more: the raw iFacialMocap UDP text, and OpenSeeFace's face result (its 68
landmarks from OSF's own 3D face model, plus the solved rotation and
position). The head turns round a neck behind and below the eyes, the way a
webcam sees it. Frames go through FaceBench._on_osf, the entry the live
trackers use, with the saved limiters, Mirror and character; the sheet draws
the frame Track Lab publishes. Writes output/captures/sim_<time>/ and nothing
else: every save is stubbed and output/ is checked afterwards.

The webcam here is ideal: perfect landmarks and solves. A real one is
noisier and loses the face somewhere past a 45 deg turn.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
CAPTURES = OUTPUT / "captures"

# Simulated neck: the pivot sits this far behind / below the eyes, in face
# widths. The rig assumes 0.55 / 0.5 (rig.NECK_FORWARD / NECK_UP); a real
# neck is not the model, so the webcam here swings a little further.
PIVOT_BACK = 0.6
PIVOT_DOWN = 0.5
# Performer about 90 cm from a 640 x 480 webcam (OSF's tracker size).
DEPTH = 6.0
REST_FRAMES = 20
RAMP_FRAMES = 12
HOLD_FRAMES = 30
# Frames the mouth takes to open, at the end of the hold.
OPEN_FRAMES = 5

TILE = 380
BAR = 64
INSET = 130
HEAD_COLOR = (105, 91, 244)
BODY_COLOR = (244, 164, 91)

# (id, label, pose). Pose: turn + = your left, nod + = down, tilt + = toward
# your left shoulder, lean = body sideways in face widths, + = your left.
SHEETS: list[tuple[str, list[tuple[str, str, dict[str, float]]]]] = [
    (
        "turns",
        [
            ("rest", "hold still", {}),
            ("turn_l10", "turn to your LEFT 10°", {"turn": 10}),
            ("turn_l20", "turn to your LEFT 20°", {"turn": 20}),
            ("turn_l35", "turn to your LEFT 35°", {"turn": 35}),
            ("turn_l50", "turn to your LEFT 50°", {"turn": 50}),
            ("turn_r10", "turn to your RIGHT 10°", {"turn": -10}),
            ("turn_r20", "turn to your RIGHT 20°", {"turn": -20}),
            ("turn_r50", "turn to your RIGHT 50°", {"turn": -50}),
        ],
    ),
    (
        "looks",
        [
            ("down5", "look DOWN 5°", {"nod": 5}),
            ("down10", "look DOWN 10°", {"nod": 10}),
            ("down20", "look DOWN 20°", {"nod": 20}),
            ("down35", "look DOWN 35°", {"nod": 35}),
            ("up10", "look UP 10°", {"nod": -10}),
            ("up20", "look UP 20°", {"nod": -20}),
            ("up35", "look UP 35°", {"nod": -35}),
            ("rest2", "hold still", {}),
        ],
    ),
    (
        "tilts_and_mixes",
        [
            ("tilt_l15", "tilt to your LEFT shoulder 15°", {"tilt": 15}),
            ("tilt_l30", "tilt to your LEFT shoulder 30°", {"tilt": 30}),
            ("tilt_r15", "tilt to your RIGHT shoulder 15°", {"tilt": -15}),
            ("tilt_r30", "tilt to your RIGHT shoulder 30°", {"tilt": -30}),
            ("l30_down15", "LEFT 30° + look DOWN 15°", {"turn": 30, "nod": 15}),
            ("r30_up15", "RIGHT 30° + look UP 15°", {"turn": -30, "nod": -15}),
            ("lean_l", "lean your body LEFT ~8 cm", {"lean": 0.57}),
            ("l20_lean_l", "LEFT 20° + lean LEFT ~8 cm", {"turn": 20, "lean": 0.57}),
        ],
    ),
]
# Every move is run twice: mouth shut, then saying "ah" (jaw this open).
AH = 0.6
ROWS_PER_PAGE = 4
# OSF model rows that drop when the jaw opens: lower lip, then the chin.
_LOWER_LIP = (53, 54, 55, 56, 57, 63, 64, 65)
_CHIN = (6, 7, 8, 9, 10)


# ---------------------------------------------------------------- guarding


def _output_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in OUTPUT.glob("*"):
        if path.is_file():
            out[path.name] = hashlib.sha1(path.read_bytes()).hexdigest()
    return out


def _stub_saves() -> None:
    """Nothing the bench, feel or limiters would save may reach disk.

    Before backend.face is imported: importing it builds a bench, which
    saves the feel.
    """
    from backend import calibrate, eye_bits, mouth_bits
    from backend.feel import feel
    from backend.travel_box import travel

    feel.save = lambda: None  # type: ignore[method-assign]
    travel.save = lambda: None  # type: ignore[method-assign]
    calibrate.calibrator.save = lambda: None  # type: ignore[method-assign]
    for bits in (eye_bits.bits, mouth_bits.bits):
        bits.save = lambda: None  # type: ignore[method-assign]
    from backend.face import FaceBench

    FaceBench._save_parts = lambda self: None  # type: ignore[method-assign]
    FaceBench._save_ifm = lambda self: None  # type: ignore[method-assign]


# ------------------------------------------------------------------- poses


def _rig_angles(move: dict[str, float]) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in the rig's camera-frame signs.

    Unflipped camera: your left is image-right. Yaw + faces image-right,
    pitch + looks down, roll + is clockwise (crown toward image-right, i.e.
    toward your left shoulder).
    """
    return float(move.get("turn", 0.0)), float(move.get("nod", 0.0)), float(move.get("tilt", 0.0))


def _osf_face_model() -> np.ndarray:
    """OSF's own 70-point face model, read from its tracker source."""
    text = (ROOT / "osf" / "tracker.py").read_text(encoding="utf-8")
    block = re.search(r"self\.face_3d = np\.array\((\[.*?\])\s*,\s*np\.float32\)", text, re.S)
    if block is None:
        raise RuntimeError("OSF face model not found in osf/tracker.py")
    rows = re.sub(r"#[^\n]*", "", block.group(1))
    return np.asarray(ast.literal_eval(rows), dtype=np.float64)


class _SimOsfFace:
    """What OpenSeeFace's tracker returns for one face, built exactly."""

    def __init__(self, lms: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, pts_3d: np.ndarray) -> None:
        self.lms = lms
        self.rotation = rvec
        self.translation = tvec
        self.pts_3d = pts_3d
        self.success = True
        self.pnp_error = 0.0
        self.conf = 0.95
        self.euler = np.zeros(3, dtype=np.float32)
        self.eye_blink = [1.0, 1.0]


class Webcam:
    """Head poses → OpenSeeFace face results → the frame osf_cam._loop builds.

    Landmarks come from a real camera (centred lens). Rotation, position and
    3D points come the way OSF gets them: solvePnP on its contour points
    with its own camera matrix (its lens centre sits at image x 240, y 320),
    seeded by the last frame, then its back-projection.
    """

    # FaceInfo.contour_pts for Track Lab's model_type 3.
    CONTOUR = [0, 1, 8, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    def __init__(self) -> None:
        from backend.osf_cam import CAM_H, CAM_W, _SWAP

        self.model = _osf_face_model()
        self.swap = _SWAP
        # OSF lays points out as (image y, image x).
        self.lens = np.array([[CAM_W, 0.0, CAM_H / 2], [0.0, CAM_W, CAM_W / 2], [0.0, 0.0, 1.0]])
        self.osf_cam = np.array([[CAM_W, 0.0, CAM_W / 2], [0.0, CAM_W, CAM_H / 2], [0.0, 0.0, 1.0]])
        self.face_w = abs(float(self.model[16, 0] - self.model[0, 0]))
        eyes = 0.5 * (self.model[66] + self.model[67])
        # Model axes: +x toward image-left, +y up, +z toward the camera.
        self.pivot = eyes + np.array([0.0, -PIVOT_DOWN * self.face_w, -PIVOT_BACK * self.face_w])
        self.rest_rot = -_SWAP
        # Eyes at image (320, 210), DEPTH away.
        want = np.array(
            [(210.0 - self.lens[0, 2]) * DEPTH / CAM_W, (320.0 - self.lens[1, 2]) * DEPTH / CAM_W, DEPTH]
        )
        self.rest_t = want - self.rest_rot @ eyes
        self.guess: tuple[np.ndarray, np.ndarray] | None = None

    def _solve(self, lms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.guess is None:
            rvec, _ = cv2.Rodrigues(self.rest_rot)
            self.guess = (rvec.reshape(3, 1), self.rest_t.reshape(3, 1).copy())
        ok, rvec, tvec = cv2.solvePnP(
            self.model[self.CONTOUR].astype(np.float64),
            lms[self.CONTOUR, :2].astype(np.float64),
            self.osf_cam,
            np.zeros((4, 1)),
            rvec=self.guess[0].copy(),
            tvec=self.guess[1].copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed")
        self.guess = (rvec, tvec)
        return rvec.reshape(3), tvec.reshape(3)

    def _pts_3d(self, lms: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """OSF's estimate_depth: landmarks back to the model frame at the
        solved depth (pose-free, keeps expression)."""
        rmat, _ = cv2.Rodrigues(rvec)
        ref = (self.model @ rmat.T + tvec) @ self.osf_cam.T
        depth = ref[:, 2:3]
        out = np.zeros((70, 3))
        cam = np.c_[lms[:66, :2], np.ones(66)] * depth[:66]
        out[:66] = (cam @ np.linalg.inv(self.osf_cam).T - tvec) @ np.linalg.inv(rmat).T
        out[66:] = self.model[66:]
        return out.astype(np.float32)

    def face(self, move: dict[str, float]) -> _SimOsfFace:
        from backend.rig import head_matrix

        yaw, pitch, roll = _rig_angles(move)
        rot = -self.swap @ head_matrix(yaw, pitch, roll)
        lean = np.array([0.0, float(move.get("lean", 0.0)) * self.face_w, 0.0])
        pivot_cam = self.rest_rot @ self.pivot + self.rest_t + lean
        head = self.model[:68].copy()
        # An open jaw: the lower lip and chin drop (model +y is up).
        drop = float(move.get("open", 0.0)) * 0.12 * self.face_w
        head[list(_LOWER_LIP), 1] -= drop
        head[list(_CHIN), 1] -= 1.2 * drop
        cam_pts = (head - self.pivot) @ rot.T + pivot_cam
        uv = cam_pts @ self.lens.T
        uv = uv[:, :2] / uv[:, 2:3]
        lms = np.c_[uv, np.full(68, 0.95)].astype(np.float32)
        rvec, tvec = self._solve(lms)
        return _SimOsfFace(lms, rvec.astype(np.float32), tvec.astype(np.float32), self._pts_3d(lms, rvec, tvec))

    def frame(self, move: dict[str, float]) -> Any:
        from backend.feel import feel
        from backend.iris import IrisHit, hits_payload, merge_hits, osf_gaze_hits
        from backend.osf_cam import OsfFrame, _blink, _face_local_2d, _face_pose, _head, _lms_xy, _smooth
        from backend.presets import empty_weights
        from backend.visemes import viseme_weights

        face = self.face(move)
        pose = _face_pose(face)
        head = _head(face)
        # osf_cam._loop eases the mouth weights across frames.
        self.weights = _smooth(getattr(self, "weights", None) or empty_weights(), viseme_weights(face, pose), feel.alpha())
        lms_xy = _lms_xy(np.asarray(face.lms, dtype=np.float32))
        gaze = osf_gaze_hits(lms_xy)
        right, left, _method = merge_hits((IrisHit(side="r"), IrisHit(side="l")), gaze)
        return OsfFrame(
            weights=dict(self.weights),
            head=head,
            blink=_blink(face),
            pose=pose,
            faces=1,
            pts_3d=np.asarray(face.pts_3d, dtype=np.float32).copy(),
            mouth_2d=_face_local_2d(face, pose),
            lms_xy=lms_xy,
            iris_cam=hits_payload(right, left),
            source="osf",
        )


# Every ARKit shape iFacialMocap sends; a neutral face sends them all at 0.
_ARKIT = (
    "browDown_L browDown_R browInnerUp browOuterUp_L browOuterUp_R cheekPuff cheekSquint_L "
    "cheekSquint_R eyeBlink_L eyeBlink_R eyeLookDown_L eyeLookDown_R eyeLookIn_L eyeLookIn_R "
    "eyeLookOut_L eyeLookOut_R eyeLookUp_L eyeLookUp_R eyeSquint_L eyeSquint_R eyeWide_L eyeWide_R "
    "jawForward jawLeft jawOpen jawRight mouthClose mouthDimple_L mouthDimple_R mouthFrown_L "
    "mouthFrown_R mouthFunnel mouthLeft mouthLowerDown_L mouthLowerDown_R mouthPress_L mouthPress_R "
    "mouthPucker mouthRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper "
    "mouthSmile_L mouthSmile_R mouthStretch_L mouthStretch_R mouthUpperUp_L mouthUpperUp_R "
    "noseSneer_L noseSneer_R tongueOut"
).split()


def _yaw_outer(rot: np.ndarray) -> tuple[float, float, float]:
    """Angles (a, b, c) with rot = Ry(a) Rx(b) Rz(c), degrees."""
    b = math.asin(max(-1.0, min(1.0, -float(rot[1, 2]))))
    a = math.atan2(float(rot[0, 2]), float(rot[2, 2]))
    c = math.atan2(float(rot[1, 0]), float(rot[1, 1]))
    return math.degrees(a), math.degrees(b), math.degrees(c)


class IPhone:
    """Head poses → iFacialMocap UDP text → the frame ifm_cam._loop builds."""

    def __init__(self) -> None:
        self.held = None
        self.head = None

    @staticmethod
    def packet(move: dict[str, float]) -> str:
        """The text the phone sends. It composes the head yaw outermost
        (Ry Rx Rz, Unity) and FaceRig reads it back through ifm.head_of;
        signs are the ones head_of documents (yaw as sent, +pitch down,
        roll flipped). Position is sent too; Track Lab does not use it."""
        from backend.rig import head_matrix

        a, b, c = _yaw_outer(head_matrix(*_rig_angles(move)))
        yaw, pitch, roll = -a, b, -c
        lean_cm = 14.0 * float(move.get("lean", 0.0))
        # Shapes go out as whole percent.
        values = {"jawOpen": round(100.0 * float(move.get("open", 0.0)))}
        shapes = "|".join(f"{name}-{values.get(name, 0)}" for name in _ARKIT)
        return (
            f"{shapes}|=head#{pitch:.4f},{yaw:.4f},{roll:.4f},{-lean_cm / 100:.4f},0.0000,-0.6400|"
            "rightEye#0,0,0|leftEye#0,0,0|"
        )

    def frame(self, move: dict[str, float]) -> Any:
        from backend.feel import feel
        from backend.ifm import (
            blink_of,
            brow_of,
            head_of,
            hold_packet,
            look_of,
            parse_packet,
            pose_of,
            weights_from_arkit,
        )
        from backend.ifm_cam import _unwrap_head
        from backend.osf_cam import OsfFrame

        packet = parse_packet(self.packet(move))
        assert packet is not None
        packet = hold_packet(self.held, packet)
        self.held = packet
        head = _unwrap_head(self.head, head_of(packet))
        self.head = head
        pose = pose_of(packet, feel.head_sway())
        pose["tilt"] = float(head.get("roll", 0.0))
        return OsfFrame(
            weights=weights_from_arkit(packet),
            head=dict(head),
            blink=blink_of(packet),
            pose=pose,
            faces=1,
            look=look_of(packet),
            source="ifm",
            brow=brow_of(packet),
        )


# --------------------------------------------------------------- the run


def _ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _blend(move: dict[str, float], t: float) -> dict[str, float]:
    return {key: float(value) * t for key, value in move.items()}


def run_move(source: str, move: dict[str, float], box: dict[str, Any]) -> dict[str, Any]:
    """One fresh bench, rest → ease into the move → hold. Returns the frame."""
    from backend.face import FaceBench
    from backend.feel import feel
    from backend.offsets import apply_points
    from backend.presets import pts_to_json
    from backend.travel_box import lab_feel_caps, travel
    from harness.pack import pack_frame

    from backend.visemes import _rest as viseme_rest

    travel.values = dict(box)
    # A fresh session: the webcam's viseme rest is module state.
    viseme_rest.reset()
    bench = FaceBench()
    feel.update(lab_feel_caps(box))
    tracker = IPhone() if source == "ifm" else Webcam()
    # The head eases into the pose and holds; the mouth says "ah" at the end
    # of the hold. A mouth held open while you face the camera is re-zeroed
    # by the webcam's viseme rest, which is lip sync's business, not this.
    head = {key: value for key, value in move.items() if key != "open"}
    frames = [{}] * REST_FRAMES
    frames += [_blend(head, _ease((i + 1) / RAMP_FRAMES)) for i in range(RAMP_FRAMES)]
    frames += [dict(head) for _ in range(HOLD_FRAMES - OPEN_FRAMES)]
    opened = float(move.get("open", 0.0))
    frames += [dict(head, open=opened * _ease((i + 1) / OPEN_FRAMES)) for i in range(OPEN_FRAMES)]
    for pose in frames:
        bench._on_osf(tracker.frame(pose))
    fields = bench._live_fields()
    points = apply_points(pts_to_json(bench._live_pts), dict(bench._point_offsets))
    h, w = bench.source_bgr.shape[:2]
    packet = pack_frame({**fields, "live": True, "points": points, "tracker": source}, image_wh=(w, h))
    rest = bench._rest_overlay()
    return {
        "frame": packet,
        "turn": bench._rig.debug(),
        "rest": rest,
        "rects": bench._travel_rects(),
        "mirror": bool(bench._mirror),
        "image": bench.source_bgr,
    }


# ------------------------------------------------------------------ drawing

def _chain(*ids: int) -> tuple[tuple[int, int], ...]:
    return tuple(zip(ids, ids[1:]))


# As Track Lab and the desk draw them (anime.MOUTH_UPPER / MOUTH_LOWER):
# both lips run corner 23 to corner 26.
_FACE_LINES = (
    _chain(0, 1, 2, 3, 4)
    + _chain(5, 6, 7)
    + _chain(8, 9, 10)
    + _chain(11, 12, 13)
    + _chain(17, 18, 19)
    + _chain(14, 15, 16)
    + _chain(23, 20, 21, 22, 26)
    + _chain(23, 24, 25, 27, 26)
)
_BONES = ((31, 32), (32, 33), (31, 34), (34, 35), (31, 36))


def _dashed(img: np.ndarray, rect: list[float] | None, color: tuple[int, int, int]) -> None:
    if not rect:
        return
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        steps = max(1, int(math.hypot(bx - ax, by - ay) // 12))
        for i in range(0, steps, 2):
            p = (int(ax + (bx - ax) * i / steps), int(ay + (by - ay) * i / steps))
            q = (int(ax + (bx - ax) * min(i + 1, steps) / steps), int(ay + (by - ay) * min(i + 1, steps) / steps))
            cv2.line(img, p, q, color, 2, cv2.LINE_AA)


def _moved(k: np.ndarray, rest: np.ndarray, slots: tuple[int, ...]) -> np.ndarray:
    idx = [i for i in slots if k[i, 3] >= 0.5 and rest[i, 3] >= 0.5]
    return np.mean(k[idx, :2], axis=0) - np.mean(rest[idx, :2], axis=0)


def _side(value: float, left: str, right: str, eps: float = 0.3) -> str:
    return "" if abs(value) < eps else (right if value > 0 else left)


def draw_tile(result: dict[str, Any], box: dict[str, Any], source: str, label: str) -> np.ndarray:
    from backend.travel_box import BODY_ANCHOR, HEAD_ANCHOR, face_height

    img = (result["image"] * 0.42).astype(np.uint8)
    rest = result["rest"]
    k = np.array(
        [[row["x"], row["y"], row["score"], 1.0 if row["visible"] else 0.0] for row in result["frame"]["keypoints"]],
        dtype=np.float32,
    )
    _dashed(img, result["rects"].get("head_wall"), HEAD_COLOR)
    _dashed(img, result["rects"].get("body_wall"), BODY_COLOR)
    for a, b in _FACE_LINES + _BONES:
        cv2.line(img, tuple(int(v) for v in rest[a, :2]), tuple(int(v) for v in rest[b, :2]), (150, 150, 150), 1, cv2.LINE_AA)
    for seg in result["frame"].get("hair") or []:
        pts = np.asarray(seg.get("polygon") or [], dtype=np.int32)
        if len(pts) > 2:
            cv2.polylines(img, [pts], True, (40, 200, 255), 1, cv2.LINE_AA)
    for a, b in _FACE_LINES:
        if k[a, 3] >= 0.5 and k[b, 3] >= 0.5:
            cv2.line(img, tuple(int(v) for v in k[a, :2]), tuple(int(v) for v in k[b, :2]), (90, 255, 90), 3, cv2.LINE_AA)
    for a, b in _BONES:
        if k[a, 3] >= 0.5 and k[b, 3] >= 0.5:
            cv2.line(img, tuple(int(v) for v in k[a, :2]), tuple(int(v) for v in k[b, :2]), BODY_COLOR, 4, cv2.LINE_AA)
    for i in (28, 29):
        if k[i, 3] >= 0.5:
            cv2.circle(img, (int(k[i, 0]), int(k[i, 1])), 6, (255, 255, 255), -1, cv2.LINE_AA)
    tile = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
    # The mouth is a few pixels tall at tile size: a zoomed inset, top-right,
    # on the drawn mouth (grey = rest lips).
    mouth = k[20:28, :2] if (k[20:28, 3] >= 0.5).all() else rest[20:28, :2]
    cx, cy = (float(v) for v in mouth.mean(axis=0))
    half = max(40, int(0.75 * float(np.ptp(rest[20:28, 0]))))
    h, w = img.shape[:2]
    x0, y0 = int(np.clip(cx - half, 0, w - 2 * half)), int(np.clip(cy - half, 0, h - 2 * half))
    inset = cv2.resize(img[y0 : y0 + 2 * half, x0 : x0 + 2 * half], (INSET, INSET), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(inset, (0, 0), (INSET - 1, INSET - 1), (255, 255, 255), 1)
    tile[4 : 4 + INSET, TILE - INSET - 4 : TILE - 4] = inset

    turn = result["turn"]
    fh = face_height(rest)
    head = _moved(k, rest, HEAD_ANCHOR) / fh
    body = _moved(k, rest, BODY_ANCHOR) / fh
    yaw, pitch, roll = float(turn["yaw"]), float(turn["pitch"]), float(turn["roll"])
    drawn = []
    if abs(yaw) >= 1.0:
        cap = box["turn_right"] if yaw > 0 else box["turn_left"]
        drawn.append(f"turn {abs(yaw):.0f}° {_side(yaw, 'scr-L', 'scr-R')} (lim {cap:.0f})")
    if abs(pitch) >= 1.0:
        cap = box["pitch_down"] if pitch > 0 else box["pitch_up"]
        drawn.append(f"{'down' if pitch > 0 else 'up'} {abs(pitch):.0f}° (lim {cap:.0f})")
    if abs(roll) >= 1.0:
        cap = box["tilt_right"] if roll > 0 else box["tilt_left"]
        drawn.append(f"tilt {abs(roll):.0f}° {_side(roll, 'scr-L', 'scr-R')} (lim {cap:.0f})")
    # Upper lip middle (21) to lower lip middle (25), past the rest gap.
    opened = (float(k[25, 1] - k[21, 1]) - float(rest[25, 1] - rest[21, 1])) / fh
    drawn.append(f"mouth open {opened:.2f}" if opened >= 0.02 else "mouth shut")

    def _move_text(v: np.ndarray) -> str:
        parts = [f"{abs(v[0]):.2f}{_side(v[0], '←', '→', 0.005)}", f"{abs(v[1]):.2f}{_side(v[1], '↑', '↓', 0.005)}"]
        return " ".join(p for p in parts if not p.startswith("0.00"))

    lines = [
        ("iPHONE" if source == "ifm" else "WEBCAM") + " · you: " + label,
        "drawn: " + (", ".join(drawn) if drawn else "straight"),
        "head " + (_move_text(head) or "still") + " · torso " + (_move_text(body) or "still") + "  (face heights)",
    ]
    bar = np.full((BAR, TILE, 3), 24, dtype=np.uint8)
    colors = ((255, 255, 255), (170, 230, 170), (200, 200, 200))
    for i, (text, color) in enumerate(zip(lines, colors)):
        _put(bar, text, (6, 18 + 20 * i), color, 0.42 if i == 0 else 0.4)
    return np.vstack([bar, tile])


def _put(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float) -> None:
    # Hershey has no arrows or degree sign; spell them.
    text = text.replace("°", " deg").replace("←", "<").replace("→", ">").replace("↑", "^").replace("↓", "v")
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def sheet(moves: list[tuple[str, str, dict[str, float]]], box: dict[str, Any], header: list[str]) -> np.ndarray:
    """One row per move: iPhone | webcam with the mouth shut, then both
    saying "ah"."""
    gap = np.full((BAR + TILE, 6, 3), 60, dtype=np.uint8)
    rows = []
    for _mid, label, move in moves:
        tiles = []
        for opened, text in ((0.0, label), (AH, f'{label}, "ah"')):
            for source in ("ifm", "osf"):
                pose = dict(move, open=opened) if opened else dict(move)
                tiles.append(draw_tile(run_move(source, pose, box), box, source, text))
        parts = []
        for j, tile in enumerate(tiles):
            parts.append(tile)
            if j < len(tiles) - 1:
                parts.append(gap if j != 1 else np.full_like(gap, 140))
        rows.append(np.hstack(parts))
        rows.append(np.full((6, rows[-1].shape[1], 3), 60, dtype=np.uint8))
    body = np.vstack(rows)
    top = np.full((24 * len(header) + 10, body.shape[1], 3), 12, dtype=np.uint8)
    for i, text in enumerate(header):
        _put(top, text, (8, 22 + 24 * i), (255, 255, 255) if i == 0 else (200, 200, 200), 0.6 if i == 0 else 0.5)
    return np.vstack([top, body])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ("turn-left", "turn-right", "tilt-left", "tilt-right", "look-up", "look-down"):
        parser.add_argument(f"--{key}", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    before = _output_hashes()
    _stub_saves()
    from backend.face import FaceBench
    from backend.travel_box import merge_travel_box, travel

    box = travel.payload()
    patch = {
        "turn_left": args.turn_left,
        "turn_right": args.turn_right,
        "tilt_left": args.tilt_left,
        "tilt_right": args.tilt_right,
        "pitch_up": args.look_up,
        "pitch_down": 12.0 if args.look_down is None else args.look_down,
    }
    box = merge_travel_box(box, {k: v for k, v in patch.items() if v is not None})
    mirror = bool(FaceBench()._mirror)
    out = args.out or CAPTURES / f"sim_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    header = [
        "Simulated tracker data -> Track Lab live path (FaceBench._on_osf) -> published frame.",
        'Each row is one move. Left pair: mouth shut (iPhone | webcam). Right pair: the same move saying "ah" (iPhone | webcam).',
        f"Limiters: turn L{box['turn_left']:.0f} R{box['turn_right']:.0f}  tilt L{box['tilt_left']:.0f} R{box['tilt_right']:.0f}  "
        f"look up {box['pitch_up']:.0f} down {box['pitch_down']:.0f}  head room L{box['left']:.2f} R{box['right']:.2f} "
        f"U{box['up']:.2f} D{box['down']:.2f}  body L{box['body_left']:.2f} R{box['body_right']:.2f} U{box['body_up']:.2f} D{box['body_down']:.2f}",
        "Mirror ON: the character copies you, so your left is its left = screen-right (scr-R)."
        if mirror
        else "Mirror OFF: the character is your reflection, so your left shows on screen-left (scr-L).",
        "Grey = rest pose. Green = face, blue = torso, yellow = hair, dashed = head (red) / body (blue) limiter walls.",
    ]
    for name, moves in SHEETS:
        for page, start in enumerate(range(0, len(moves), ROWS_PER_PAGE), start=1):
            path = out / f"{name}_{page}.png"
            cv2.imwrite(str(path), sheet(moves[start : start + ROWS_PER_PAGE], box, header))
            print(path, flush=True)
    changed = [name for name, digest in _output_hashes().items() if before.get(name) != digest]
    if changed:
        raise SystemExit(f"output/ changed during the simulation: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
