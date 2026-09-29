"""iFacialMocap UDP packets → ARKit weights, blink, look, and head.

The live mixer maps those onto the shared mouth AUs (smile, sad, A I U E O).
apply_shapes still builds a doodle face for the iPhone pip only.
"""

from __future__ import annotations

import math
import socket
import time
from dataclasses import dataclass, field

import numpy as np

from .feel import feel
from .presets import VOWEL_IDS, apply_open_offset, empty_weights, open_amount
from .rig import head_angles, head_matrix_yaw_outer

HANDSHAKE = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719|sendDataVersion=v2"
DEFAULT_PORT = 49983
IFM_W = 640
IFM_H = 480

_ALIASES = (
    ("Left", "_L"),
    ("Right", "_R"),
)


_LAN_TTL = 5.0
_lan_at = 0.0
_lan_ips: list[str] = []


def is_lan_ipv4(ip: str) -> bool:
    parts = str(ip).strip().split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return False
    if any(n < 0 or n > 255 for n in nums):
        return False
    if nums[0] in (0, 127) or nums[0] >= 224:
        return False
    if nums[0] == 169 and nums[1] == 254:
        return False
    return True


def subnet_broadcast(ip: str) -> str:
    a, b, c, _d = str(ip).split(".")
    return f"{a}.{b}.{c}.255"


def lan_ipv4s() -> list[str]:
    global _lan_at, _lan_ips
    now = time.time()
    if _lan_ips and now - _lan_at < _LAN_TTL:
        return list(_lan_ips)
    found: list[str] = []

    def add(ip: object) -> None:
        text = str(ip).strip()
        if is_lan_ipv4(text) and text not in found:
            found.append(text)

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("1.1.1.1", 80))
        add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    try:
        _host, _aliases, ips = socket.gethostbyname_ex(socket.gethostname())
        for ip in ips:
            add(ip)
    except OSError:
        pass
    _lan_at = now
    _lan_ips = found
    return found


def handshake_targets(
    host: str, last_peer: str, port: int, live_ip: str = ""
) -> list[tuple[str, int]]:
    """Unicast phone + LAN broadcast. Empty host means find-the-phone."""
    if str(live_ip).strip():
        return [(str(live_ip).strip(), int(port))]
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    local = lan_ipv4s()

    def add(ip: str) -> None:
        ip = str(ip).strip()
        if not ip or ip in seen or ip in local:
            return
        seen.add(ip)
        out.append((ip, int(port)))

    add(host)
    add(last_peer)
    for ip in local:
        add(subnet_broadcast(ip))
    add("255.255.255.255")
    return out


def keep_sender(lock_ip: str, incoming_ip: str, *, live: bool) -> bool:
    """One phone at a time. Source-port changes on that IP are the same sender."""
    incoming = str(incoming_ip).strip()
    locked = str(lock_ip).strip()
    if not incoming:
        return False
    if not locked or incoming == locked:
        return True
    return not live


def hold_packet(prev: IfmPacket | None, nxt: IfmPacket) -> IfmPacket:
    """Keep the last real blendshape / head when a datagram omits them."""
    if prev is None:
        return nxt
    return IfmPacket(
        shapes={**prev.shapes, **nxt.shapes},
        head=dict(nxt.head if nxt.has_head else prev.head),
        position=dict(nxt.position if nxt.has_head else prev.position),
        right_eye=nxt.right_eye if nxt.has_right_eye else prev.right_eye,
        left_eye=nxt.left_eye if nxt.has_left_eye else prev.left_eye,
        has_head=bool(nxt.has_head or prev.has_head),
        has_right_eye=bool(nxt.has_right_eye or prev.has_right_eye),
        has_left_eye=bool(nxt.has_left_eye or prev.has_left_eye),
    )


def sweep_hosts(skip: set[str] | None = None) -> list[str]:
    ignore = skip or set()
    out: list[str] = []
    for ip in lan_ipv4s():
        a, b, c, _d = ip.split(".")
        for n in range(1, 255):
            host = f"{a}.{b}.{c}.{n}"
            if host in ignore:
                continue
            out.append(host)
    return out


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else float(value)


def _norm_name(raw: str) -> str:
    name = raw.strip()
    if name.startswith("="):
        name = name[1:]
    for old, new in _ALIASES:
        if name.endswith(old):
            name = name[: -len(old)] + new
            break
    return name


def _split_shape(token: str) -> tuple[str, float] | None:
    token = token.strip()
    if not token or token.startswith("="):
        return None
    if "&" in token:
        name, _, raw = token.partition("&")
    elif "-" in token:
        name, _, raw = token.rpartition("-")
    else:
        return None
    name = _norm_name(name)
    if not name:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    # iFacialMocap always sends integer percentages. Guessing the scale from
    # the magnitude read "jawOpen-1" (1%) as fully open.
    value *= 0.01
    return name, float(np.clip(value, -1.0, 1.0))


def _split_hash(token: str, key: str) -> list[float] | None:
    token = token.strip().lstrip("=")
    prefix = f"{key}#"
    if not token.startswith(prefix):
        return None
    parts = token[len(prefix) :].split(",")
    out: list[float] = []
    for part in parts:
        try:
            out.append(float(part))
        except ValueError:
            break
    return out or None


@dataclass
class IfmPacket:
    shapes: dict[str, float] = field(default_factory=dict)
    head: dict[str, float] = field(
        default_factory=lambda: {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    )
    position: dict[str, float] = field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
    )
    right_eye: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_eye: tuple[float, float, float] = (0.0, 0.0, 0.0)
    has_head: bool = False
    has_right_eye: bool = False
    has_left_eye: bool = False

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.shapes.get(name, default))


def parse_packet(raw: str | bytes) -> IfmPacket | None:
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            return None
    else:
        text = raw
    text = text.strip()
    if "___iFacialMocap" in text:
        text = text.split("___iFacialMocap", 1)[0]
    if not text:
        return None
    packet = IfmPacket()
    found = False
    for token in text.split("|"):
        token = token.strip()
        if not token:
            continue
        head = _split_hash(token, "head")
        if head is not None and len(head) >= 3:
            packet.head = {
                "pitch": round(head[0], 3),
                "yaw": round(head[1], 3),
                "roll": round(head[2], 3),
            }
            if len(head) >= 6:
                packet.position = {
                    "x": float(head[3]),
                    "y": float(head[4]),
                    "z": float(head[5]),
                }
            packet.has_head = True
            found = True
            continue
        right = _split_hash(token, "rightEye")
        if right is not None and len(right) >= 3:
            packet.right_eye = (right[0], right[1], right[2])
            packet.has_right_eye = True
            found = True
            continue
        left = _split_hash(token, "leftEye")
        if left is not None and len(left) >= 3:
            packet.left_eye = (left[0], left[1], left[2])
            packet.has_left_eye = True
            found = True
            continue
        shape = _split_shape(token)
        if shape is None:
            continue
        packet.shapes[shape[0]] = shape[1]
        found = True
    if not found:
        return None
    return packet


class IfmFace:
    """Doodle-face landmarks for the iPhone pip. Not the live mixer."""

    def __init__(self, pts: np.ndarray, head: dict[str, float], blink: dict[str, float]):
        self.pts_3d = pts
        self.lms = pts.copy()
        self.lms[:, 2] = 1.0
        self.euler = np.array(
            [head["pitch"], head["yaw"], head["roll"]], dtype=np.float32
        )
        self.pnp_error = 0.0
        # OSF stores 1 = open. Slider is blink amount.
        self.eye_blink = np.array(
            [1.0 - float(blink["r"]), 1.0 - float(blink["l"])], dtype=np.float32
        )


def rest_landmarks() -> np.ndarray:
    """66-point rest, Y-down, jaw width 2. Point 58/60/62/64 = L/U/R/D lips."""
    pts = np.zeros((66, 3), dtype=np.float32)
    pts[:, 2] = 1.0
    jaw_x = np.linspace(-1.0, 1.0, 17)
    pts[0:17, 0] = jaw_x
    pts[0:17, 1] = 0.22 + 0.98 * (1.0 - jaw_x * jaw_x)
    for i, x in enumerate(np.linspace(-0.52, -0.10, 5)):
        pts[17 + i, 0] = x
        pts[17 + i, 1] = -0.98 - 0.05 * math.sin(math.pi * i / 4.0)
    for i, x in enumerate(np.linspace(0.10, 0.52, 5)):
        pts[22 + i, 0] = x
        pts[22 + i, 1] = -0.98 - 0.05 * math.sin(math.pi * i / 4.0)
    pts[27:31, 0] = 0.0
    pts[27:31, 1] = np.linspace(-0.62, 0.08, 4)
    pts[31:36, 0] = np.linspace(-0.22, 0.22, 5)
    pts[31:36, 1] = 0.16 + 0.06 * np.array([1, 0, 0, 0, 1], dtype=np.float32)
    _eye(pts, 36, -0.32)
    _eye(pts, 42, 0.32)
    # Outer mouth 48-57, then the 8-point tracking ring 58-65.
    # Lips start shut: jawOpen 0 must read as a closed mouth, not a gap.
    for i, ang in enumerate(np.linspace(math.pi, 0.0, 7)):
        pts[48 + i, 0] = 0.34 * math.cos(ang)
        pts[48 + i, 1] = 0.58 - 0.08 * math.sin(ang)
    for i, ang in enumerate(np.linspace(-0.45, -math.pi + 0.45, 3)):
        pts[55 + i, 0] = 0.34 * math.cos(ang)
        pts[55 + i, 1] = 0.58 - 0.10 * math.sin(ang)
    ring = (
        (58, -0.28, 0.58),
        (59, -0.16, 0.575),
        (60, 0.00, 0.575),
        (61, 0.16, 0.575),
        (62, 0.28, 0.58),
        (63, 0.16, 0.585),
        (64, 0.00, 0.585),
        (65, -0.16, 0.585),
    )
    for idx, x, y in ring:
        pts[idx, 0] = x
        pts[idx, 1] = y
    return pts


def _eye(pts: np.ndarray, start: int, cx: float) -> None:
    # 0 outer, 1-2 upper, 3 inner, 4-5 lower. Matches dlib 36-41 / 42-47.
    ring = (
        (cx - 0.16, -0.34),
        (cx - 0.07, -0.40),
        (cx + 0.07, -0.40),
        (cx + 0.16, -0.34),
        (cx + 0.07, -0.28),
        (cx - 0.07, -0.28),
    )
    for i, (x, y) in enumerate(ring):
        pts[start + i, 0] = x
        pts[start + i, 1] = y


def apply_shapes(rest: np.ndarray, packet: IfmPacket) -> np.ndarray:
    pts = rest.copy()
    g = packet.get
    jaw = _clip(g("jawOpen"))
    close = _clip(g("mouthClose"))
    open_amt = _clip(jaw * (1.0 - 0.65 * close))
    smile_l = _clip(g("mouthSmile_L"))
    smile_r = _clip(g("mouthSmile_R"))
    frown_l = _clip(g("mouthFrown_L"))
    frown_r = _clip(g("mouthFrown_R"))
    pucker = _clip(g("mouthPucker"))
    funnel = _clip(g("mouthFunnel"))
    stretch_l = _clip(g("mouthStretch_L"))
    stretch_r = _clip(g("mouthStretch_R"))
    lower_l = _clip(g("mouthLowerDown_L"))
    lower_r = _clip(g("mouthLowerDown_R"))
    upper_l = _clip(g("mouthUpperUp_L"))
    upper_r = _clip(g("mouthUpperUp_R"))
    left = _clip(g("mouthLeft")) - _clip(g("mouthRight"))
    brow_in = _clip(g("browInnerUp"))
    brow_down_l = _clip(g("browDown_L"))
    brow_down_r = _clip(g("browDown_R"))
    brow_l = _clip(g("browOuterUp_L")) - brow_down_l
    brow_r = _clip(g("browOuterUp_R")) - brow_down_r
    blink_l = _clip(g("eyeBlink_L"))
    blink_r = _clip(g("eyeBlink_R"))
    wide_l = _clip(g("eyeWide_L"))
    wide_r = _clip(g("eyeWide_R"))
    squint_l = _clip(g("eyeSquint_L"))
    squint_r = _clip(g("eyeSquint_R"))
    roll_lo = _clip(g("mouthRollLower"))
    roll_up = _clip(g("mouthRollUpper"))
    shrug = 0.5 * (_clip(g("mouthShrugLower")) + _clip(g("mouthShrugUpper")))
    puff = _clip(g("cheekPuff"))

    jaw_w = 1.0 - np.abs(np.linspace(-1.0, 1.0, 17))
    pts[0:17, 1] += 0.40 * open_amt * jaw_w
    pts[8, 1] += 0.10 * open_amt
    pts[58:66, 1] += 0.06 * open_amt
    pts[[63, 64, 65], 1] += 0.32 * open_amt
    pts[[59, 60, 61], 1] -= 0.12 * open_amt
    pts[[55, 56, 57], 1] += 0.18 * open_amt
    # Teeth / lip raise without jawOpen still has to split the inner ring.
    lip_lo = 0.5 * (lower_l + lower_r)
    lip_up = 0.5 * (upper_l + upper_r)
    pts[[63, 64, 65], 1] += 0.28 * lip_lo
    pts[[59, 60, 61], 1] -= 0.20 * lip_up
    pts[[55, 56, 57], 1] += 0.16 * lip_lo
    pts[[49, 50, 51], 1] -= 0.12 * lip_up

    pts[58, 0] -= 0.16 * smile_r + 0.10 * stretch_r
    pts[58, 1] -= 0.10 * smile_r
    pts[62, 0] += 0.16 * smile_l + 0.10 * stretch_l
    pts[62, 1] -= 0.10 * smile_l
    pts[48, 0] -= 0.12 * smile_r
    pts[48, 1] -= 0.08 * smile_r
    pts[54, 0] += 0.12 * smile_l
    pts[54, 1] -= 0.08 * smile_l
    pts[58, 1] += 0.10 * frown_r
    pts[62, 1] += 0.10 * frown_l
    pts[58:66, 0] *= 1.0 - 0.22 * pucker
    pts[58:66, 1] = 0.58 + (pts[58:66, 1] - 0.58) * (1.0 + 0.35 * funnel)
    pts[58:66, 0] += 0.12 * left
    shut = 0.58
    pts[[63, 64, 65], 1] += (shut - pts[[63, 64, 65], 1]) * 0.55 * roll_lo
    pts[[59, 60, 61], 1] += (shut - pts[[59, 60, 61], 1]) * 0.55 * roll_up
    pts[48:66, 1] -= 0.08 * shrug
    pts[0:5, 0] -= 0.06 * puff
    pts[12:17, 0] += 0.06 * puff

    pts[21, 1] -= 0.10 * brow_in
    pts[22, 1] -= 0.10 * brow_in
    pts[17:22, 1] -= 0.12 * brow_r
    pts[22:27, 1] -= 0.12 * brow_l
    pts[17:22, 1] += 0.06 * brow_down_r
    pts[22:27, 1] += 0.06 * brow_down_l
    _close_eye(pts, 36, blink_r)
    _close_eye(pts, 42, blink_l)
    _wide_eye(pts, 36, wide_r)
    _wide_eye(pts, 42, wide_l)
    _squint_eye(pts, 36, squint_r)
    _squint_eye(pts, 42, squint_l)
    return pts


def _close_eye(pts: np.ndarray, start: int, amount: float) -> None:
    if amount <= 1e-4:
        return
    mid_y = 0.5 * (float(pts[start + 1, 1]) + float(pts[start + 4, 1]))
    for i in (1, 2, 4, 5):
        pts[start + i, 1] += amount * (mid_y - float(pts[start + i, 1]))


def _wide_eye(pts: np.ndarray, start: int, amount: float) -> None:
    if amount <= 1e-4:
        return
    pts[start + 1, 1] -= 0.06 * amount
    pts[start + 2, 1] -= 0.06 * amount
    pts[start + 4, 1] += 0.04 * amount
    pts[start + 5, 1] += 0.04 * amount


def _squint_eye(pts: np.ndarray, start: int, amount: float) -> None:
    if amount <= 1e-4:
        return
    mid_y = 0.5 * (float(pts[start + 1, 1]) + float(pts[start + 4, 1]))
    for i in (1, 2, 4, 5):
        pts[start + i, 1] += 0.45 * amount * (mid_y - float(pts[start + i, 1]))


def mouth_2d(pts: np.ndarray) -> np.ndarray:
    extra = np.zeros((2, pts.shape[1]), dtype=pts.dtype)
    extra[0, :2] = 0.5 * (pts[37, :2] + pts[38, :2])
    extra[0, 2] = 1.0
    extra[1, :2] = 0.5 * (pts[43, :2] + pts[44, :2])
    extra[1, 2] = 1.0
    return np.vstack((pts, extra))


def blink_of(packet: IfmPacket) -> dict[str, float]:
    return {
        "l": round(_clip(packet.get("eyeBlink_L")), 3),
        "r": round(_clip(packet.get("eyeBlink_R")), 3),
    }


def brow_of(packet: IfmPacket) -> dict[str, float]:
    down_l = _clip(packet.get("browDown_L"))
    down_r = _clip(packet.get("browDown_R"))
    return {
        "inner": round(_clip(packet.get("browInnerUp")), 3),
        "l": round(_clip(packet.get("browOuterUp_L")) - down_l, 3),
        "r": round(_clip(packet.get("browOuterUp_R")) - down_r, 3),
        "down_l": round(down_l, 3),
        "down_r": round(down_r, 3),
    }


def _smoothstep(lo: float, hi: float, value: float) -> float:
    if hi <= lo:
        return 0.0 if value < lo else 1.0
    t = (float(value) - lo) / (hi - lo)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)


def _dead(value: float, dead: float = 0.12) -> float:
    mag = float(value)
    if mag <= dead:
        return 0.0
    return (mag - dead) / (1.0 - dead)


def weights_from_arkit(packet: IfmPacket) -> dict[str, float]:
    """Map iPhone blendshapes onto the shared smile/sad/A I U E set."""
    g = packet.get
    jaw = _clip(g("jawOpen") * (1.0 - 0.65 * _clip(g("mouthClose"))))
    smile = _dead(_clip(0.5 * (_clip(g("mouthSmile_L")) + _clip(g("mouthSmile_R")))))
    sad = _dead(_clip(0.5 * (_clip(g("mouthFrown_L")) + _clip(g("mouthFrown_R")))))
    stretch = _dead(_clip(0.5 * (_clip(g("mouthStretch_L")) + _clip(g("mouthStretch_R")))))
    pucker = _dead(_clip(g("mouthPucker")))
    funnel = _dead(_clip(g("mouthFunnel")))
    upper = 0.5 * (_clip(g("mouthUpperUp_L")) + _clip(g("mouthUpperUp_R")))
    lower = 0.5 * (_clip(g("mouthLowerDown_L")) + _clip(g("mouthLowerDown_R")))
    # Smile-open / teeth without a jaw drop still has to drive A.
    lip_open = _dead(_clip(0.55 * upper + 0.75 * lower + 0.25 * funnel), 0.08)
    opened = max(_dead(jaw, 0.08), lip_open)
    response = feel.response()
    high = _smoothstep(0.32, 0.72, opened)
    out = empty_weights()
    out["A"] = _clip(opened * response)
    spread = stretch * response
    out["I"] = _clip(spread * (1.0 - high))
    out["E"] = _clip(spread * high)
    out["U"] = _clip((pucker + 0.35 * funnel) * (1.0 - high) * response + (funnel + 0.35 * pucker) * high * response)
    out["smile"] = _clip(smile * (1.0 - 0.55 * opened), 0.0, 0.7)
    out["sad"] = _clip(sad * (1.0 - 0.55 * opened), 0.0, 0.7)
    if out["smile"] >= out["sad"]:
        out["sad"] = 0.0
    else:
        out["smile"] = 0.0
    total = sum(out[name] for name in VOWEL_IDS)
    if total > 1.0:
        for name in VOWEL_IDS:
            out[name] /= total
    return out


_MOUTH_SLOTS = tuple(range(20, 28))
_LEFT_EYE = (11, 12, 13)
_RIGHT_EYE = (17, 18, 19)


def _mesh_span(rest: np.ndarray) -> float:
    if len(rest) > 4:
        return max(float(np.linalg.norm(rest[4, :2] - rest[0, :2])), 1.0)
    return 1.0


def _scale_mouth(mesh: np.ndarray, gain: float) -> None:
    slots = [slot for slot in _MOUTH_SLOTS if slot < len(mesh)]
    if len(slots) < 2:
        return
    gain = float(gain)
    if abs(gain - 1.0) < 1e-4:
        return
    xy = mesh[slots, :2]
    center = np.mean(xy, axis=0)
    mesh[slots, :2] = center + (xy - center) * gain


def _procedural_mouth(rest: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    out = rest.copy()
    span = _mesh_span(rest)
    a = float(weights.get("A") or 0.0)
    smile = float(weights.get("smile") or 0.0)
    sad = float(weights.get("sad") or 0.0)
    spread = float(weights.get("I") or 0.0) + float(weights.get("E") or 0.0)
    rounded = float(weights.get("U") or 0.0)
    if len(out) > 2:
        out[2, 1] += 0.10 * span * a
    if len(out) > 25:
        out[21, 1] -= 0.08 * span * a
        out[20, 1] -= 0.06 * span * a
        out[22, 1] -= 0.06 * span * a
        out[24, 1] += 0.14 * span * a
        out[25, 1] += 0.16 * span * a
        out[27, 1] += 0.14 * span * a
        out[23, 0] -= span * (0.04 * smile + 0.03 * spread - 0.025 * rounded)
        out[23, 1] += span * (0.03 * sad - 0.03 * smile)
        out[26, 0] += span * (0.04 * smile + 0.03 * spread - 0.025 * rounded)
        out[26, 1] += span * (0.03 * sad - 0.03 * smile)
        out[25, 1] += 0.02 * span * rounded
    return out


def _apply_brows(mesh: np.ndarray, rest: np.ndarray, brow: dict[str, float]) -> None:
    span = 0.06 * _mesh_span(rest)
    inner = float(brow.get("inner") or 0.0)
    left = float(brow.get("l") or 0.0)
    right = float(brow.get("r") or 0.0)
    down_l = float(brow.get("down_l") or 0.0)
    down_r = float(brow.get("down_r") or 0.0)
    lifts = (
        (5, left, down_l),
        (6, 0.7 * left + 0.3 * inner, down_l),
        (7, inner, down_l),
        (8, inner, down_r),
        (9, 0.7 * right + 0.3 * inner, down_r),
        (10, right, down_r),
    )
    for slot, up, down in lifts:
        if slot >= len(mesh) or slot >= len(rest):
            continue
        mesh[slot, 1] = float(rest[slot, 1]) - span * up + 0.5 * span * down


def _close_lids(mesh: np.ndarray, rest: np.ndarray, blink: dict[str, float]) -> None:
    for slots, key in ((_LEFT_EYE, "l"), (_RIGHT_EYE, "r")):
        amount = _clip(float(blink.get(key, 0.0)))
        if amount < 0.03:
            continue
        lid = slots[1]
        corners = [i for i in (slots[0], slots[2]) if i < len(mesh)]
        if not corners and lid >= len(rest):
            continue
        chord_y = (
            float(np.mean(mesh[corners, 1])) if corners else float(rest[lid, 1])
        )
        if lid < len(mesh) and float(mesh[lid, 1]) < chord_y:
            mesh[lid, 1] = float(mesh[lid, 1]) * (1.0 - amount) + chord_y * amount
        for i in corners:
            mesh[i, 1] = float(mesh[i, 1]) * (1.0 - amount) + chord_y * amount


def drive_ifm(
    rest: np.ndarray | None,
    weights: dict[str, float] | None,
    blink: dict[str, float] | None = None,
    brow: dict[str, float] | None = None,
    mixed: np.ndarray | None = None,
) -> np.ndarray | None:
    """Character mesh from ARKit AUs: authored or procedural mouth, lids, brows."""
    if rest is None:
        return mixed
    out = mixed.copy() if mixed is not None else rest.copy()
    if mixed is None and weights:
        out = _procedural_mouth(rest, weights)
    apply_open_offset(out, rest, open_amount(weights))
    _apply_brows(out, rest, brow or {})
    _close_lids(out, rest, blink or {})
    _scale_mouth(out, feel.mouth_gain())
    return out


_EYE_SPAN = 25.0
_LOOK_REST = 0.2


def _look_from_shapes(packet: IfmPacket) -> dict[str, float]:
    look_y = _clip(packet.get("eyeLookDown_L")) + _clip(packet.get("eyeLookDown_R"))
    look_y -= _clip(packet.get("eyeLookUp_L")) + _clip(packet.get("eyeLookUp_R"))
    look_x = _clip(packet.get("eyeLookOut_R")) + _clip(packet.get("eyeLookIn_L"))
    look_x -= _clip(packet.get("eyeLookOut_L")) + _clip(packet.get("eyeLookIn_R"))
    return {"x": round(float(0.5 * look_x), 3), "y": round(float(0.5 * look_y), 3)}


def _look_from_eyes(packet: IfmPacket) -> dict[str, float]:
    pitches: list[float] = []
    yaws: list[float] = []
    if packet.has_right_eye:
        pitches.append(float(packet.right_eye[0]))
        yaws.append(float(packet.right_eye[1]))
    if packet.has_left_eye:
        pitches.append(float(packet.left_eye[0]))
        yaws.append(float(packet.left_eye[1]))
    if not pitches:
        return {"x": 0.0, "y": 0.0}
    pitch = sum(pitches) / float(len(pitches))
    yaw = sum(yaws) / float(len(yaws))
    # ARKit / iFacialMocap: +yaw = look left, +pitch = look up.
    # Blendshape look.x is person's right; look.y is down.
    look_x = _clip(-yaw / _EYE_SPAN, -1.0, 1.0)
    look_y = _clip(-pitch / _EYE_SPAN, -1.0, 1.0)
    return {"x": round(float(look_x), 3), "y": round(float(look_y), 3)}


def _stronger_look(a: float, b: float) -> float:
    return a if abs(a) >= abs(b) else b


def look_quiet(look: dict[str, float] | None) -> bool:
    """True when gaze is close enough to treat as rest."""
    if not isinstance(look, dict):
        return False
    try:
        return abs(float(look.get("x") or 0.0)) < _LOOK_REST and abs(
            float(look.get("y") or 0.0)
        ) < _LOOK_REST
    except (TypeError, ValueError):
        return False


def look_of(packet: IfmPacket) -> dict[str, float]:
    """Gaze from eyeLook* and leftEye/rightEye.

    Blendshapes are the real pupil move. Eye bones sit near rest, or leak a
    sideways yaw while you look up — mixing bone X with shape Y shoves both
    irises to one side. Use the blendshape vector when it is clearly on;
    otherwise fall back to the bones.
    """
    shapes = _look_from_shapes(packet)
    if not (packet.has_right_eye or packet.has_left_eye):
        return shapes
    eyes = _look_from_eyes(packet)
    sx = float(shapes["x"])
    sy = float(shapes["y"])
    if abs(sx) >= _LOOK_REST or abs(sy) >= _LOOK_REST:
        return {"x": round(sx, 3), "y": round(sy, 3)}
    return {
        "x": round(_stronger_look(float(eyes["x"]), sx), 3),
        "y": round(_stronger_look(float(eyes["y"]), sy), 3),
    }


def iris_of(pts: np.ndarray, look: dict[str, float]) -> np.ndarray:
    """Face-local pupils (right, left) inside the OSF eye boxes."""
    out = np.zeros((2, 3), dtype=np.float32)
    look_x = float(look.get("x") or 0.0)
    look_y = float(look.get("y") or 0.0)
    arr = np.asarray(pts, dtype=np.float32)
    for i, start in enumerate((36, 42)):
        if len(arr) < start + 6:
            continue
        cx = 0.5 * (float(arr[start, 0]) + float(arr[start + 3, 0]))
        cy = 0.5 * (float(arr[start + 1, 1]) + float(arr[start + 4, 1]))
        width = abs(float(arr[start + 3, 0]) - float(arr[start, 0]))
        opening = max(abs(float(arr[start + 4, 1]) - float(arr[start + 1, 1])), width * 0.35)
        out[i, 0] = cx + 0.28 * width * look_x
        out[i, 1] = cy + 0.28 * opening * look_y
        out[i, 2] = 1.0
    return out


def head_of(packet: IfmPacket) -> dict[str, float]:
    """Head in the rig's angles (the ones osf_cam._pnp_head reads a solve in).

    iFacialMocap composes ARKit's angles yaw outermost (Ry Rx Rz, the Unity
    order); the rig composes roll outermost. Read as they came, every turn
    leaked into pitch and roll: on a real phone a 50 deg turn also looked
    ~5 deg up and rolled ~17 deg.

    Signs are Unity's: +pitch looks down, as in the rig. Read as look-up, a
    real nod drew the other way. Pitch and roll only flip as a pair (a
    mirror of the head frame): flipping pitch alone made a 50 deg turn on
    the same recording also look ~19 deg down. Yaw carries over as sent.
    """
    rot = head_matrix_yaw_outer(
        float(packet.head.get("yaw", 0.0)),
        float(packet.head.get("pitch", 0.0)),
        -float(packet.head.get("roll", 0.0)),
    )
    yaw, pitch, roll = head_angles(rot)
    return {"pitch": round(pitch, 3), "yaw": round(yaw, 3), "roll": round(roll, 3)}


def pose_of(packet: IfmPacket, sway: float = 0.0) -> dict[str, float]:
    # Face-local input: no box to measure, so scale is one face width and
    # the solved distance stays off (a look is not a zoom). Nothing sees the
    # head or body move either: a pose with ``sway`` tells FaceRig so, and
    # asks it to swing the head round the neck by the drawn turn (0 keeps it
    # in place); the torso stays put. cx stays 0: viseme rest reads it.
    # Tilt is the head's roll so FaceRig does not shadow head.roll with 0.
    return {
        "cx": 0.0,
        "cy": 0.0,
        "bx": 0.0,
        "by": 0.0,
        "scale": 1.0,
        "tz": 0.0,
        "tilt": float(head_of(packet)["roll"]),
        "sway": float(sway),
        "ok": 1.0,
    }


_REST = rest_landmarks()


def rest_face() -> IfmFace:
    """Closed doodle face for the iPhone pip."""
    zero = IfmPacket()
    return IfmFace(_REST.copy(), zero.head, blink_of(zero))


def frame_from_packet(packet: IfmPacket) -> tuple[IfmFace, dict[str, float], dict[str, float]]:
    """Pip doodle plus ARKit mouth weights. The mixer ignores the doodle."""
    pts = apply_shapes(_REST, packet)
    blink = blink_of(packet)
    face = IfmFace(pts, packet.head, blink)
    return face, weights_from_arkit(packet), blink
