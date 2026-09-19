"""Hair parts from models/trackers/animeseg_hair3.pt, then follow the face."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .feel import feel
from .rig import FaceRig, mesh_center, _mesh_scale

ROOT = Path(__file__).resolve().parents[1]
TRACKERS = ROOT.parent / "models" / "trackers"
HAIR3_WEIGHTS = TRACKERS / "animeseg_hair3.pt"
HAIR_FALLBACK = TRACKERS / "hair_seg.pt"
HAIR_CLASSES = ("hair_middle", "hair_left", "hair_right")
_HAIR_PIN_INNER = 0.55
_HAIR_PIN_OUTER = 1.25
_MAX_HAIR_TURN = 45.0
FT_ID_TO_CLASS = {1: "hair_middle", 2: "hair_left", 3: "hair_right"}
MIN_AREA = 60.0
APPROX = 0.0015
HAIR3_SIZE = 768
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_hair3 = None
_yolo = None


def _torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _yolo_device():
    return 0 if _torch_device() == "cuda" else "cpu"


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


def _model_yolo():
    global _yolo
    if _yolo is not None:
        return _yolo
    if not HAIR_FALLBACK.is_file():
        raise FileNotFoundError(f"Hair fallback missing: {HAIR_FALLBACK}")
    from ultralytics import YOLO

    _yolo = YOLO(str(HAIR_FALLBACK))
    return _yolo


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
class HairRig:
    parts: list[tuple[str, np.ndarray]]
    rest_cx: float
    rest_cy: float
    rest_ms: float


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    span = max(float(edge1) - float(edge0), 1e-8)
    t = np.clip((np.asarray(x, dtype=np.float64) - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_hair_rig(segments: list[dict[str, Any]], rest: np.ndarray) -> HairRig | None:
    cx, cy = mesh_center(rest)
    parts: list[tuple[str, np.ndarray]] = []
    origin = np.array([cx, cy], dtype=np.float64)
    for seg in segments:
        cls = str(seg.get("class") or "")
        pts = np.asarray(seg.get("polygon") or [], dtype=np.float64)
        if cls not in HAIR_CLASSES or pts.ndim != 2 or len(pts) < 3:
            continue
        parts.append((cls, pts[:, :2] - origin))
    if not parts:
        return None
    return HairRig(
        parts,
        rest_cx=float(cx),
        rest_cy=float(cy),
        rest_ms=float(_mesh_scale(rest)),
    )


def follow_hair(
    rig: HairRig | None,
    live: np.ndarray | None,
    face_rig: FaceRig | None = None,
) -> list[dict[str, Any]]:
    if rig is None or not rig.parts:
        return []
    use_head = face_rig is not None and face_rig.locked
    pin = feel.hair_pin() if use_head else 0.0
    max_turn = _MAX_HAIR_TURN if pin > 1e-6 else None
    rest_ms = max(float(rig.rest_ms), 1.0)
    out: list[dict[str, Any]] = []
    for cls, local in rig.parts:
        if use_head and face_rig is not None:
            xs_loc = local[:, 0]
            ys_loc = local[:, 1]
            xs, ys = face_rig.map_local(xs_loc, ys_loc, max_turn=max_turn)
            if pin > 1e-6:
                flat_x, flat_y = face_rig.map_flat(xs_loc, ys_loc)
                weight = _smoothstep(_HAIR_PIN_INNER, _HAIR_PIN_OUTER, np.hypot(xs_loc, ys_loc) / rest_ms)
                blend = weight * pin
                xs = xs + (flat_x - xs) * blend
                ys = ys + (flat_y - ys) * blend
        else:
            xs = local[:, 0] + rig.rest_cx
            ys = local[:, 1] + rig.rest_cy
        out.append(_hair_poly(cls, xs, ys))
    return out


def detect_hair(
    image_bgr: np.ndarray,
    face_pts: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Hair-part polygons from the hair model. Face mesh is not a keepout."""
    del face_pts
    if image_bgr is None or image_bgr.size == 0:
        return []
    try:
        segs = _detect_hair3(image_bgr)
        return refine_hair(image_bgr, segs)
    except Exception as exc:
        print(f"animeseg_hair3 failed ({exc}) — falling back to hair_seg.pt")
    h, w = image_bgr.shape[:2]
    model = _model_yolo()
    model.predictor = None
    results = model.predict(
        source=image_bgr,
        conf=0.12,
        iou=0.7,
        verbose=False,
        device=_yolo_device(),
        retina_masks=True,
        imgsz=640,
    )
    if not results or results[0].masks is None or results[0].boxes is None:
        return []
    r0 = results[0]
    names = r0.names
    clss = r0.boxes.cls.cpu().numpy().astype(int)
    confs = r0.boxes.conf.cpu().numpy()
    orig_h, orig_w = (r0.orig_shape[:2] if getattr(r0, "orig_shape", None) else (h, w))
    best: dict[str, dict[str, Any]] = {}
    for i, cls_id in enumerate(clss):
        cid = int(cls_id)
        if isinstance(names, dict):
            cls_name = names.get(cid)
        elif isinstance(names, (list, tuple)) and 0 <= cid < len(names):
            cls_name = names[cid]
        else:
            cls_name = None
        if cls_name not in HAIR_CLASSES:
            if 0 <= cid < len(HAIR_CLASSES):
                cls_name = HAIR_CLASSES[cid]
            else:
                continue
        try:
            xyn = r0.masks.xyn[i]
        except Exception:
            xyn = None
        if xyn is not None and len(xyn) >= 3:
            xy = np.asarray(xyn, dtype=np.float32)
            xy = np.stack([xy[:, 0] * orig_w, xy[:, 1] * orig_h], axis=1)
        else:
            raw = r0.masks.xy[i]
            if raw is None or len(raw) < 3:
                continue
            xy = np.asarray(raw, dtype=np.float32)
        pts = xy.reshape(-1, 1, 2)
        peri = cv2.arcLength(pts, True)
        approx = cv2.approxPolyDP(pts, max(0.8, APPROX * peri), True)
        if len(approx) < 3:
            continue
        poly = [
            [round(float(np.clip(p[0][0], 0, orig_w - 1)), 1), round(float(np.clip(p[0][1], 0, orig_h - 1)), 1)]
            for p in approx
        ]
        area = float(cv2.contourArea(approx))
        if area < MIN_AREA:
            continue
        rec = {"class": cls_name, "polygon": poly, "area": round(area, 1), "score": round(float(confs[i]), 3)}
        prev = best.get(cls_name)
        if prev is None or area > float(prev.get("area") or 0):
            best[cls_name] = rec
    return refine_hair(image_bgr, [best[c] for c in HAIR_CLASSES if c in best])
