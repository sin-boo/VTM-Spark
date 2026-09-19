"""Live hair-part tracker for Live Poser.

Prefers the full-stack labeling model ``animeseg_hair3.pt`` (AnimeSeg
Mask2Former, 4-class semantic: bg + hair_middle / hair_left / hair_right).
Falls back to YOLO-seg ``hair_seg.pt`` if those weights cannot be loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
_PACKAGE_TRACKERS = ROOT.parent.parent.parent / "models" / "trackers"
# vendor/tools/live-poser → ai_vtuber/tools/model-tracking-train/...
_POSE_TRACKER_HAIR = (
    ROOT.parents[3]
    / "tools"
    / "model-tracking-train"
    / "anime-pose-tracker"
    / "weights"
    / "hair"
)
DEFAULT_HAIR3_CANDIDATES = [
    _PACKAGE_TRACKERS / "animeseg_hair3.pt",
    ROOT / "models" / "animeseg_hair3.pt",
    _POSE_TRACKER_HAIR / "animeseg_hair3.pt",
]
DEFAULT_YOLO_HAIR_CANDIDATES = [
    _PACKAGE_TRACKERS / "hair_seg.pt",
    ROOT / "models" / "hair_seg.pt",
    _POSE_TRACKER_HAIR / "hair_seg.pt",
]
DEFAULT_HAIR_CANDIDATES = DEFAULT_HAIR3_CANDIDATES + DEFAULT_YOLO_HAIR_CANDIDATES

HAIR_CLASSES = ("hair_middle", "hair_left", "hair_right")
HAIR_SWAP_LR = {"hair_left": "hair_right", "hair_right": "hair_left"}
# BGR overlay colors match the pose-tracker labeler.
HAIR_COLORS = {
    "hair_middle": (0, 200, 255),
    "hair_left": (255, 180, 0),
    "hair_right": (160, 80, 255),
}
HAIR_DETECT_CONF = 0.12
HAIR_DETECT_IOU = 0.7
HAIR_IMGSZ = 640
HAIR3_IMGSZ = 768
MIN_POLY_AREA = 60.0
APPROX_EPS_RATIO = 0.006
HAIR3_APPROX_EPS_RATIO = 0.0015
FT_ID_TO_CLASS = {1: "hair_middle", 2: "hair_left", 3: "hair_right"}
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_hair_weights(explicit: Path | str | None = None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    for c in DEFAULT_HAIR_CANDIDATES:
        if c.is_file():
            return c
    return None


def resolve_yolo_hair_weights(explicit: Path | str | None = None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    for c in DEFAULT_YOLO_HAIR_CANDIDATES:
        if c.is_file():
            return c
    return None


def _is_hair3_weights(path: Path | None) -> bool:
    if path is None:
        return False
    name = path.name.lower()
    return "animeseg" in name or name.endswith("hair3.pt")


def _as_xy(poly: Sequence) -> np.ndarray:
    if poly is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2]


def clone_hair_segments(segments: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments or []:
        cls = str(seg.get("class") or "")
        pts = _as_xy(seg.get("polygon") or [])
        if cls not in HAIR_CLASSES or pts.shape[0] < 3:
            continue
        rec: dict[str, Any] = {"class": cls, "polygon": pts.astype(np.float32).tolist()}
        if "score" in seg:
            rec["score"] = float(seg["score"])
        if "area" in seg:
            rec["area"] = float(seg["area"])
        out.append(rec)
    return out


def flip_hair_pixels(
    segments: Sequence[Mapping[str, Any]] | None,
    width: int | float,
) -> list[dict[str, Any]]:
    """Mirror source-pixel polygons and swap hair_left ↔ hair_right."""
    w = float(width)
    out: list[dict[str, Any]] = []
    for seg in segments or []:
        cls = str(seg.get("class") or "")
        pts = _as_xy(seg.get("polygon") or [])
        if cls not in HAIR_CLASSES or pts.shape[0] < 3:
            continue
        pts = pts.copy()
        pts[:, 0] = (w - 1.0) - pts[:, 0]
        rec: dict[str, Any] = {
            "class": HAIR_SWAP_LR.get(cls, cls),
            "polygon": pts.tolist(),
        }
        if "score" in seg:
            rec["score"] = float(seg["score"])
        if "area" in seg:
            rec["area"] = float(seg["area"])
        out.append(rec)
    return out


def draw_hair_segments(
    frame: np.ndarray,
    segments: Sequence[Mapping[str, Any]] | None,
    *,
    lost: bool = False,
) -> None:
    """Overlay hair-part polygons on a BGR frame."""
    if frame is None or not segments:
        return
    alpha = 0.35 if not lost else 0.18
    overlay = frame.copy()
    for seg in segments:
        cls = str(seg.get("class") or "")
        pts = _as_xy(seg.get("polygon") or [])
        if cls not in HAIR_COLORS or pts.shape[0] < 3:
            continue
        color = HAIR_COLORS[cls]
        poly = np.round(pts).astype(np.int32)
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        label = "M" if cls == "hair_middle" else ("L" if cls == "hair_left" else "R")
        cv2.putText(
            frame,
            label,
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


class HairHold:
    """Keep last good hair polygons when the detector drops a frame."""

    def __init__(self, max_age_s: float = 1.5) -> None:
        self.max_age_s = float(max_age_s)
        self.last: list[dict[str, Any]] | None = None
        self.last_t: float = 0.0
        self.method: str = "none"

    def reset(self) -> None:
        self.last = None
        self.last_t = 0.0
        self.method = "none"

    def update(
        self,
        segments: Sequence[Mapping[str, Any]] | None,
        method: str,
        *,
        now: float,
    ) -> tuple[list[dict[str, Any]] | None, bool, str]:
        segs = clone_hair_segments(segments)
        if segs:
            self.last = segs
            self.last_t = float(now)
            self.method = str(method or "hair_seg")
            return clone_hair_segments(segs), False, self.method
        if self.last and (float(now) - self.last_t) <= self.max_age_s:
            held_method = f"{self.method}_held" if self.method else "held"
            return clone_hair_segments(self.last), True, held_method
        return None, True, "none"


class HairSegTracker:
    """Thin wrapper around Ultralytics YOLO-seg ``hair_seg.pt``."""

    def __init__(self, weights: Path | str | None = None, device: str | None = "cpu"):
        path = resolve_yolo_hair_weights(weights)
        if path is None:
            raise FileNotFoundError(
                "hair_seg.pt not found. Place it at models/trackers/hair_seg.pt"
            )
        from ultralytics import YOLO  # lazy

        self.weights = Path(path)
        self.model = YOLO(str(self.weights))
        self.device = device if device is not None else "cpu"
        self.method = "hair_seg"

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        conf: float = HAIR_DETECT_CONF,
        iou: float = HAIR_DETECT_IOU,
        imgsz: int = HAIR_IMGSZ,
    ) -> list[dict[str, Any]]:
        """Return one polygon per class in source-pixel coordinates."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        kwargs = dict(
            source=image_bgr,
            conf=float(conf),
            iou=float(iou),
            verbose=False,
            device=self.device,
            retina_masks=True,
            imgsz=int(imgsz),
        )
        # Ultralytics keeps the first predictor; later imgsz/conf kwargs are ignored.
        self.model.predictor = None
        results = self.model.predict(**kwargs)
        if not results or results[0].masks is None or results[0].boxes is None:
            return []
        r0 = results[0]
        names = r0.names
        best: dict[str, dict[str, Any]] = {}
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy()
        orig_h, orig_w = r0.orig_shape[:2] if getattr(r0, "orig_shape", None) else (h, w)
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
            xy = None
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
            if xy is None or len(xy) < 3:
                continue
            pts = xy.reshape(-1, 1, 2)
            peri = cv2.arcLength(pts, True)
            approx = cv2.approxPolyDP(pts, max(0.8, APPROX_EPS_RATIO * peri), True)
            if len(approx) < 3:
                continue
            poly = [
                [
                    round(float(np.clip(p[0][0], 0, orig_w - 1)), 1),
                    round(float(np.clip(p[0][1], 0, orig_h - 1)), 1),
                ]
                for p in approx
            ]
            area = float(cv2.contourArea(approx))
            if area < MIN_POLY_AREA:
                continue
            rec = {
                "class": cls_name,
                "polygon": poly,
                "area": round(area, 1),
                "score": round(float(confs[i]), 3),
            }
            prev = best.get(cls_name)
            if prev is None or area > float(prev.get("area") or 0):
                best[cls_name] = rec
        return [best[c] for c in HAIR_CLASSES if c in best]


def _mask_to_hair_polygons(
    pred: np.ndarray,
    img_h: int,
    img_w: int,
    *,
    approx_eps: float = HAIR3_APPROX_EPS_RATIO,
) -> list[dict[str, Any]]:
    """Turn a class-id map (H,W) into one polygon per hair class in image pixels."""
    if pred.shape[:2] != (img_h, img_w):
        pred = cv2.resize(pred, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out: list[dict[str, Any]] = []
    for cid, name in FT_ID_TO_CLASS.items():
        mask = np.where(pred == cid, 255, 0).astype(np.uint8)
        if int(mask.max()) == 0:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < MIN_POLY_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(0.4, float(approx_eps) * peri), True)
        if len(approx) < 3:
            continue
        poly = [
            [
                round(float(np.clip(p[0][0], 0, img_w - 1)), 1),
                round(float(np.clip(p[0][1], 0, img_h - 1)), 1),
            ]
            for p in approx
        ]
        out.append({"class": name, "polygon": poly, "area": round(area, 1), "score": 1.0})
    return out


class AnimeSegHair3Tracker:
    """Full-stack AnimeSeg Mask2Former fine-tune (``animeseg_hair3.pt``)."""

    def __init__(self, weights: Path | str | None = None, device: str | None = None):
        import torch
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

        path = resolve_hair_weights(weights)
        if path is None or not _is_hair3_weights(path):
            path = None
            for c in DEFAULT_HAIR3_CANDIDATES:
                if c.is_file():
                    path = c
                    break
        if path is None:
            raise FileNotFoundError(
                "animeseg_hair3.pt not found. Place it at models/trackers/animeseg_hair3.pt"
            )
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        num_classes = int(raw.get("num_classes") or 4)
        base_model = raw.get("base_model") or "facebook/mask2former-swin-base-ade-semantic"
        imgsz = int(raw.get("image_size") or HAIR3_IMGSZ)
        cfg = Mask2FormerConfig.from_pretrained(base_model)
        cfg.num_labels = num_classes
        cfg.id2label = {0: "background", **FT_ID_TO_CLASS}
        cfg.label2id = {v: int(k) for k, v in cfg.id2label.items()}
        model = Mask2FormerForUniversalSegmentation(cfg)
        sd: dict[str, Any] = {}
        for key, val in (raw.get("state_dict") or raw).items():
            nk = key[6:] if str(key).startswith("model.") else str(key)
            if nk.startswith("criterion."):
                continue
            sd[nk] = val
        model.load_state_dict(sd, strict=False)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self.model = model.to(self.device).eval()
        self.weights = Path(path)
        self.image_size = max(32, int(imgsz))
        self.method = "animeseg_hair3"
        self._use_amp = self.device.startswith("cuda")

    def detect(self, image_bgr: np.ndarray, **_kwargs) -> list[dict[str, Any]]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        import torch
        import torch.nn.functional as F
        from PIL import Image

        h, w = image_bgr.shape[:2]
        size = self.image_size
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        pixel = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        pixel = pixel.to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self._use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = self.model(pixel_values=pixel)
            else:
                out = self.model(pixel_values=pixel)
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
        return _mask_to_hair_polygons(pred, h, w)


def create_hair_tracker(
    weights: Path | str | None = None,
    device: str | None = None,
):
    """Load animeseg_hair3 when present; otherwise YOLO hair_seg.pt."""
    path = resolve_hair_weights(weights)
    if path is not None and _is_hair3_weights(path):
        try:
            return AnimeSegHair3Tracker(path, device=device)
        except Exception as exc:
            print(f"animeseg_hair3 load failed ({exc}) — trying hair_seg.pt")
            path = resolve_yolo_hair_weights(None)
            if path is None:
                raise
            return HairSegTracker(path, device=device or "cpu")
    if path is None:
        raise FileNotFoundError(
            "Hair weights not found. Place animeseg_hair3.pt at models/trackers/"
        )
    return HairSegTracker(path, device=device or "cpu")
