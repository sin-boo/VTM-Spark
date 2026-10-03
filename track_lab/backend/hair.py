"""Hair parts from models/trackers/animeseg_hair3.pt, then follow the face."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .feel import feel
from .paths import TRACKERS
from .rig import FaceRig, mesh_center, _mesh_scale

HAIR3_WEIGHTS = TRACKERS / "animeseg_hair3.pt"
HAIR_CLASSES = ("hair_middle", "hair_left", "hair_right")
_MAX_HAIR_TURN = 45.0
_HAIR_WIDTH_MIN = 0.55
_HAIR_WIDTH_MAX = 1.35
_HAIR_WIDTH_GAIN = 0.5
_HAIR_MIDDLE_SPAN = 0.2
_HAIR_WELD = 0.05
_SIDE_OF = {"hair_middle": "mid", "hair_left": "l", "hair_right": "r"}
FT_ID_TO_CLASS = {1: "hair_middle", 2: "hair_left", 3: "hair_right"}
MIN_AREA = 60.0
APPROX = 0.0015
HAIR3_SIZE = 768
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_hair3 = None


def _torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_hair3():
    global _hair3
    if _hair3 is not None:
        return _hair3
    if not HAIR3_WEIGHTS.is_file():
        raise FileNotFoundError(f"Hair model missing: {HAIR3_WEIGHTS}")
    import torch
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    raw = torch.load(str(HAIR3_WEIGHTS), map_location="cpu", weights_only=False)
    num_classes = int(raw.get("num_classes") or 4)
    base_model = raw.get("base_model") or "facebook/mask2former-swin-base-ade-semantic"
    imgsz = int(raw.get("image_size") or HAIR3_SIZE)
    cfg = Mask2FormerConfig.from_pretrained(base_model)
    cfg.num_labels = num_classes
    cfg.id2label = {0: "background", **FT_ID_TO_CLASS}
    cfg.label2id = {v: int(k) for k, v in cfg.id2label.items()}
    model = Mask2FormerForUniversalSegmentation(cfg)
    state: dict[str, Any] = {}
    for key, val in (raw.get("state_dict") or raw).items():
        name = key[6:] if str(key).startswith("model.") else str(key)
        if name.startswith("criterion."):
            continue
        state[name] = val
    model.load_state_dict(state, strict=False)
    device = _torch_device()
    _hair3 = {
        "model": model.to(device).eval(),
        "device": device,
        "size": max(32, imgsz),
        "amp": device.startswith("cuda"),
    }
    return _hair3


def _mask_to_polygons(pred: np.ndarray, img_h: int, img_w: int) -> list[dict[str, Any]]:
    if pred.shape[:2] != (img_h, img_w):
        pred = cv2.resize(pred, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out: list[dict[str, Any]] = []
    for cid, name in FT_ID_TO_CLASS.items():
        mask = np.where(pred == cid, 255, 0).astype(np.uint8)
        if int(mask.max()) == 0:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        out.extend(_polys_from_mask(mask, name))
    return out


def _drop_skin(
    image_bgr: np.ndarray,
    layer: np.ndarray,
    face_pts: np.ndarray,
) -> np.ndarray:
    """Cheeks match the nose, not the dyed hair."""
    face = np.asarray(face_pts, dtype=np.float32)
    if len(face) < 16:
        return layer
    nx, ny = int(round(float(face[15, 0]))), int(round(float(face[15, 1])))
    h, w = image_bgr.shape[:2]
    if not (0 <= nx < w and 0 <= ny < h):
        return layer
    x0, x1 = max(0, nx - 4), min(w, nx + 5)
    y0, y1 = max(0, ny - 4), min(h, ny + 5)
    patch = image_bgr[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 3:
        return layer
    mean = patch.reshape(-1, 3).mean(axis=0)
    diff = np.linalg.norm(image_bgr.astype(np.float32) - mean, axis=2)
    layer = layer.copy()
    layer[diff < 48.0] = 0
    return layer


def _chroma_fg(image_bgr: np.ndarray) -> np.ndarray:
    """Character pixels. Green screen is not hair."""
    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)
    green = (g > 40.0) & ((g - r) > 18.0) & ((g - b) > 18.0)
    fg = np.where(green, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(fg, kernel, iterations=1)


def _face_keepout(pts: np.ndarray, h: int, w: int) -> np.ndarray | None:
    """Inner face only (eyes / nose / mouth / chin). Not a jaw-width box.

    Bangs hang over the forehead; side locks sit outside the outer jaw. A hull
    that includes brows or jaw 0/4 — or a brow-to-floor rectangle — paints hard
    shell walls through that hair. Open a narrow neck slit so a punched hole
    does not refill when polygons are filled again.
    """
    face = np.asarray(pts, dtype=np.float32)
    if face.ndim != 2 or len(face) < 20:
        return None
    eye_y = None
    for index in (11, 12, 13, 17, 18, 19):
        if index >= len(face):
            continue
        if face.shape[1] >= 3 and float(face[index, 2]) < 0.05:
            continue
        y = float(face[index, 1])
        eye_y = y if eye_y is None else min(eye_y, y)
    # Skip brows (5–10) and outer jaw (0, 4): those span the hair.
    ids = (
        1,
        2,
        3,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
    )
    ring: list[list[float]] = []
    for index in ids:
        if index >= len(face):
            continue
        if face.shape[1] >= 3 and float(face[index, 2]) < 0.05:
            continue
        x, y = float(face[index, 0]), float(face[index, 1])
        if eye_y is not None and y < eye_y - 4.0:
            continue
        ring.append([x, y])
    if len(ring) < 6:
        return None
    hull = cv2.convexHull(np.asarray(ring, dtype=np.float32))
    pts2 = hull.reshape(-1, 2)
    center = pts2.mean(axis=0)
    shrunk = center + (pts2 - center) * 0.88
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(shrunk).astype(np.int32), 255)
    chin_y = float(face[2, 1]) if len(face) > 2 else float(np.max(shrunk[:, 1]))
    y_open = int(np.clip(max(chin_y + 4.0, float(np.max(shrunk[:, 1]))), 0, h))
    span = max(float(np.max(shrunk[:, 0]) - np.min(shrunk[:, 0])), 8.0)
    half = 0.18 * span
    cx = float(np.mean(shrunk[:, 0]))
    x0 = int(max(0, cx - half))
    x1 = int(min(w, cx + half + 1.0))
    if x1 > x0 and y_open < h:
        mask[y_open:h, x0:x1] = 255
    return mask


def _polys_from_mask(layer: np.ndarray, cls: str) -> list[dict[str, Any]]:
    if int(layer.max()) == 0:
        return []
    h, w = layer.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    layer = cv2.morphologyEx(layer, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: list[dict[str, Any]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(0.4, APPROX * peri), True)
        if len(approx) < 3:
            continue
        poly = [
            [round(float(np.clip(p[0][0], 0, w - 1)), 1), round(float(np.clip(p[0][1], 0, h - 1)), 1)]
            for p in approx
        ]
        out.append({"class": cls, "polygon": poly, "area": round(area, 1), "score": 1.0})
    return out


def refine_hair(
    image_bgr: np.ndarray | None,
    segments: list[dict[str, Any]],
    face_pts: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Keep the hair-model polygons. Only strip green screen.

    Track runs face / skeleton / eyes / hair as separate models. Do not punch a
    face hull through the hair mask — that box was clipping bangs and side locks.
    ``face_pts`` is ignored here; follow still parents hair to the face rig.
    """
    del face_pts
    rows = [seg for seg in (segments or []) if isinstance(seg, dict)]
    if not rows:
        return []
    h = w = 0
    if image_bgr is not None and getattr(image_bgr, "size", 0):
        h, w = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    if h < 8 or w < 8:
        xs: list[float] = []
        ys: list[float] = []
        for seg in rows:
            for vertex in seg.get("polygon") or []:
                if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                    xs.append(float(vertex[0]))
                    ys.append(float(vertex[1]))
        w = int(max(xs) + 2) if xs else 1
        h = int(max(ys) + 2) if ys else 1
    has_img = image_bgr is not None and getattr(image_bgr, "size", 0)
    fg = _chroma_fg(image_bgr) if has_img else None
    out: list[dict[str, Any]] = []
    for seg in rows:
        cls = str(seg.get("class") or "")
        pts = np.asarray(seg.get("polygon") or [], dtype=np.float32)
        if cls not in HAIR_CLASSES or pts.ndim != 2 or len(pts) < 3:
            continue
        layer = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(layer, [np.round(pts[:, :2]).astype(np.int32)], 255)
        if fg is not None:
            layer = cv2.bitwise_and(layer, fg)
        out.extend(_polys_from_mask(layer, cls))
    return out


def _detect_hair3(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    pack = _load_hair3()
    h, w = image_bgr.shape[:2]
    size = int(pack["size"])
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    pixel = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    pixel = pixel.to(pack["device"], non_blocking=True)
    with torch.inference_mode():
        if pack["amp"]:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = pack["model"](pixel_values=pixel)
        else:
            out = pack["model"](pixel_values=pixel)
        mask_logits = F.interpolate(
            out.masks_queries_logits,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        scores = out.class_queries_logits.softmax(-1)[..., :-1]
        masks = mask_logits.sigmoid()
        sem = torch.einsum("bqc,bqhw->bchw", scores, masks)
        pred = sem.argmax(dim=1)[0].detach().to(dtype=torch.uint8).cpu().numpy()
    return _mask_to_polygons(pred, h, w)


def _hair_poly(cls: str, xs: np.ndarray, ys: np.ndarray) -> dict[str, Any]:
    return {
        "class": cls,
        "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in zip(xs, ys)],
    }


@dataclass
class HairPart:
    cls: str
    local: np.ndarray
    anchor: np.ndarray
    offsets: np.ndarray


@dataclass
class HairWeld:
    mid_part: int
    mid_vert: int
    src_part: int
    src_vert: int


@dataclass
class HairRig:
    parts: list[HairPart]
    rest_cx: float
    rest_cy: float
    rest_ms: float
    welds: list[HairWeld]


def _clip_turn(yaw: float, pitch: float, max_turn: float | None) -> tuple[float, float]:
    if max_turn is None:
        return float(yaw), float(pitch)
    cap = math.radians(float(max_turn))
    yaw = float(yaw)
    pitch = float(pitch)
    if yaw < -cap:
        yaw = -cap
    elif yaw > cap:
        yaw = cap
    if pitch < -cap:
        pitch = -cap
    elif pitch > cap:
        pitch = cap
    return yaw, pitch


def _part_side(cls: str, anchor: np.ndarray, rest_ms: float) -> str:
    named = _SIDE_OF.get(cls)
    if named:
        return named
    dist = float(np.hypot(float(anchor[0]), float(anchor[1]))) / max(float(rest_ms), 1.0)
    if dist < _HAIR_MIDDLE_SPAN:
        return "mid"
    return "r" if float(anchor[0]) >= 0.0 else "l"


def _part_width(part: HairPart, yaw_r: float, rest_ms: float, gain: float) -> float:
    side = _part_side(part.cls, part.anchor, rest_ms)
    if side == "mid":
        return 1.0
    sign = 1.0 if side == "r" else -1.0
    # +yaw brings screen-left (negative rest X) toward the camera.
    facing = -sign * math.sin(float(yaw_r))
    width = 1.0 + _HAIR_WIDTH_GAIN * float(gain) * facing
    return float(np.clip(width, _HAIR_WIDTH_MIN, _HAIR_WIDTH_MAX))


def _hairline_local(rest: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Brow midpoint, rest-centered. Bangs glue here — not at the hanging tips."""
    face = np.asarray(rest, dtype=np.float64)
    brows: list[np.ndarray] = []
    for index in (5, 6, 7, 8, 9, 10):
        if index >= len(face):
            continue
        if face.shape[1] >= 3 and float(face[index, 2]) < 0.05:
            continue
        brows.append(face[index, :2])
    if brows:
        mid = np.mean(np.stack(brows, axis=0), axis=0)
    else:
        mid = np.asarray(origin, dtype=np.float64).copy()
        mid[1] = mid[1] - 0.35 * max(float(_mesh_scale(face)), 1.0)
    return np.asarray(mid, dtype=np.float64) - np.asarray(origin, dtype=np.float64)


def _build_welds(parts: list[HairPart], rest_ms: float) -> list[HairWeld]:
    """Pin middle vertices that already sit on a left/right seam."""
    thresh = max(4.0, _HAIR_WELD * max(float(rest_ms), 1.0))
    mids = [(i, part) for i, part in enumerate(parts) if part.cls == "hair_middle"]
    sides = [
        (i, part) for i, part in enumerate(parts) if part.cls in ("hair_left", "hair_right")
    ]
    if not mids or not sides:
        return []
    welds: list[HairWeld] = []
    for mi, middle in mids:
        for vi, vertex in enumerate(middle.local):
            best_i = -1
            best_j = -1
            best_d = thresh
            for si, side in sides:
                delta = side.local - vertex
                dist = np.sqrt(np.sum(delta * delta, axis=1))
                idx = int(np.argmin(dist))
                d = float(dist[idx])
                if d < best_d:
                    best_d = d
                    best_i = si
                    best_j = idx
            if best_i >= 0:
                welds.append(
                    HairWeld(mid_part=mi, mid_vert=vi, src_part=best_i, src_vert=best_j)
                )
    return welds


def _widen(
    xs: np.ndarray,
    ys: np.ndarray,
    ax: float,
    ay: float,
    width: float,
    roll: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a polygon about its posed anchor along the rolled X axis."""
    if abs(float(width) - 1.0) < 1e-6:
        return xs, ys
    dx = np.asarray(xs, dtype=np.float64) - ax
    dy = np.asarray(ys, dtype=np.float64) - ay
    c = math.cos(roll)
    s = math.sin(roll)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    lx = lx * float(width)
    return ax + lx * c - ly * s, ay + lx * s + ly * c


def _part_pose(
    part: HairPart,
    face_rig: FaceRig,
    rest_ms: float,
    max_turn: float | None,
    gain: float,
) -> tuple[float, float, float, float, float]:
    """Posed anchor, left/right width, head zoom, roll for one part."""
    ax = np.asarray([part.anchor[0]], dtype=np.float64)
    ay = np.asarray([part.anchor[1]], dtype=np.float64)
    # A pinned part keeps the wide lens: its silhouette holds on a big turn.
    posed_x, posed_y = face_rig.map_hair(ax, ay, max_turn=max_turn, wide=True)
    turn = face_rig.turn()
    yaw_r, _pitch = _clip_turn(turn["yaw"], turn["pitch"], max_turn)
    width = _part_width(part, yaw_r, rest_ms, gain)
    scale = float(face_rig.place()["scale"])
    return float(posed_x[0]), float(posed_y[0]), width, scale, float(turn["roll"])


def _rigid_part(
    part: HairPart,
    ax: float,
    ay: float,
    width: float,
    scale: float,
    roll: float,
) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(roll)
    s = math.sin(roll)
    ox = part.offsets[:, 0] * width
    oy = part.offsets[:, 1]
    rx = scale * (ox * c - oy * s)
    ry = scale * (ox * s + oy * c)
    return ax + rx, ay + ry


def build_hair_rig(segments: list[dict[str, Any]], rest: np.ndarray) -> HairRig | None:
    cx, cy = mesh_center(rest)
    parts: list[HairPart] = []
    origin = np.array([cx, cy], dtype=np.float64)
    hairline = _hairline_local(rest, origin)
    for seg in segments:
        cls = str(seg.get("class") or "")
        pts = np.asarray(seg.get("polygon") or [], dtype=np.float64)
        if cls not in HAIR_CLASSES or pts.ndim != 2 or len(pts) < 3:
            continue
        local = pts[:, :2] - origin
        if cls == "hair_middle":
            # Parent bangs / crown to the scalp, not the hanging tips.
            anchor = hairline.copy()
        else:
            idx = int(np.argmin(np.sum(local * local, axis=1)))
            anchor = local[idx].copy()
        parts.append(HairPart(cls=cls, local=local, anchor=anchor, offsets=local - anchor))
    if not parts:
        return None
    rest_ms = float(_mesh_scale(rest))
    return HairRig(
        parts,
        rest_cx=float(cx),
        rest_cy=float(cy),
        rest_ms=rest_ms,
        welds=_build_welds(parts, rest_ms),
    )


def follow_hair(
    rig: HairRig | None,
    live: np.ndarray | None,
    face_rig: FaceRig | None = None,
) -> list[dict[str, Any]]:
    del live
    if rig is None or not rig.parts:
        return []
    use_head = face_rig is not None and face_rig.locked
    pin = feel.hair_pin() if use_head else 0.0
    gain = feel.hair_width() if use_head else 1.0
    max_turn = _MAX_HAIR_TURN if pin > 1e-6 else None
    rest_ms = max(float(rig.rest_ms), 1.0)
    posed: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    for part in rig.parts:
        width = 1.0
        if use_head and face_rig is not None:
            ax, ay, width, scale, roll = _part_pose(part, face_rig, rest_ms, max_turn, gain)
            xs, ys = face_rig.map_hair(part.local[:, 0], part.local[:, 1], max_turn=max_turn)
            # Left / right visibility applies in both modes; pin only picks
            # card (0) vs rigid part (1).
            xs, ys = _widen(xs, ys, ax, ay, width, roll)
            if pin > 1e-6:
                rigid_x, rigid_y = _rigid_part(part, ax, ay, width, scale, roll)
                xs = xs + (rigid_x - xs) * pin
                ys = ys + (rigid_y - ys) * pin
        else:
            xs = part.local[:, 0] + rig.rest_cx
            ys = part.local[:, 1] + rig.rest_cy
        posed.append((xs, ys, width, _part_side(part.cls, part.anchor, rest_ms)))
    if getattr(rig, "welds", None):
        touched = {weld.mid_part for weld in rig.welds}
        for index in touched:
            xs, ys, width, side = posed[index]
            posed[index] = (np.array(xs, copy=True), np.array(ys, copy=True), width, side)
        for weld in rig.welds:
            xs, ys, width, side = posed[weld.mid_part]
            src_x, src_y, _, _ = posed[weld.src_part]
            xs[weld.mid_vert] = float(src_x[weld.src_vert])
            ys[weld.mid_vert] = float(src_y[weld.src_vert])
    out: list[dict[str, Any]] = []
    for part, (xs, ys, width, side) in zip(rig.parts, posed):
        rec = _hair_poly(part.cls, xs, ys)
        rec["side"] = side
        rec["width"] = round(float(width), 3)
        rec["pin"] = round(float(pin), 3)
        out.append(rec)
    return out


def rig_rest_hair(rig: HairRig | None) -> list[dict[str, Any]]:
    """Rest polygons exactly as the rig was built. No welds, no follow."""
    if rig is None:
        return []
    return [
        _hair_poly(part.cls, part.local[:, 0] + rig.rest_cx, part.local[:, 1] + rig.rest_cy)
        for part in rig.parts
    ]


def detect_hair(
    image_bgr: np.ndarray,
    face_pts: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Hair-part polygons from animeseg_hair3.pt. Face mesh is not a keepout."""
    del face_pts
    if image_bgr is None or image_bgr.size == 0:
        return []
    try:
        segs = _detect_hair3(image_bgr)
        return refine_hair(image_bgr, segs)
    except Exception as exc:
        print(f"animeseg_hair3 failed ({exc})")
        return []
