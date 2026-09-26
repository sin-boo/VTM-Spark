"""Iris / pupil from models/trackers/iris_pose.pt, mapped onto the character."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from harness.protocol import LEFT_EYE_SLOTS, LEFT_IRIS, RIGHT_EYE_SLOTS, RIGHT_IRIS

from .paths import REPO, TRACKERS
from .travel_box import soft_barrier

IRIS_CANDIDATES = (
    TRACKERS / "iris_pose.pt",
    REPO / "vendor" / "tools" / "live-poser" / "models" / "iris_pose.pt",
    REPO / "vendor" / "tools" / "pose-traker" / "models" / "iris_pose.pt",
    REPO / "vendor" / "tools" / "pose-traker" / "iris-model" / "models" / "iris_pose.pt",
)

# OSF / dlib: person's right eye, person's left eye.
OSF_RIGHT = tuple(range(36, 42))
OSF_LEFT = tuple(range(42, 48))
OSF_GAZE_RIGHT = 66
OSF_GAZE_LEFT = 67
DETECT_CONF = 0.25
PUPIL_VIS = 0.3
GAZE_CONF = 0.15
CROP_LONG = 160
CROP_PAD = 0.40
CROP_PAD_Y = 0.70
_IN_EYE_X = 0.18
_IN_EYE_Y = 0.45
BLINK_HIDE = 0.85
_LOOK_SPAN = 0.38
_IRIS_HALF_W = 0.40

_yolo = None
_yolo_failed = False


@dataclass
class IrisHit:
    x: float = 0.0
    y: float = 0.0
    score: float = 0.0
    visible: bool = False
    side: str = ""
    method: str = ""
    box: tuple[float, float, float, float] | None = None


def _torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_weights() -> Path | None:
    for path in IRIS_CANDIDATES:
        if path.is_file():
            return path
    return None


def _load_yolo():
    global _yolo, _yolo_failed
    if _yolo is not None or _yolo_failed:
        return _yolo
    path = resolve_weights()
    if path is None:
        _yolo_failed = True
        return None
    try:
        from ultralytics import YOLO

        _yolo = YOLO(str(path))
        device = _torch_device()
        _yolo._iris_device = device
        print(f"[track-lab] iris_pose.pt {path.name} on {device}", flush=True)
    except Exception as exc:
        print(f"[track-lab] iris_pose.pt failed to load: {exc}", flush=True)
        _yolo_failed = True
        _yolo = None
    return _yolo


def _hits_from_result(result) -> list[dict[str, object]]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    kpts = None
    if result.keypoints is not None and result.keypoints.data is not None:
        kpts = result.keypoints.data.cpu().numpy()
    eyes: list[dict[str, object]] = []
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [float(v) for v in box]
        pupil = None
        visible = False
        if kpts is not None and i < len(kpts):
            kp = kpts[i][0]
            px, py, pv = float(kp[0]), float(kp[1]), float(kp[2])
            if pv >= PUPIL_VIS:
                pupil = (px, py)
                visible = True
        eyes.append(
            {
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
                "bbox": (x0, y0, x1, y1),
                "pupil": pupil,
                "score": float(confs[i]),
                "visible": visible,
            }
        )
    return eyes


def detect(image_bgr: np.ndarray) -> list[dict[str, object]]:
    model = _load_yolo()
    if model is None or image_bgr is None or image_bgr.size == 0:
        return []
    device = getattr(model, "_iris_device", None)
    try:
        results = model.predict(source=image_bgr, conf=DETECT_CONF, verbose=False, device=device)
    except Exception:
        return []
    if not results:
        return []
    return _hits_from_result(results[0])


def eye_crop_box(
    lms: np.ndarray,
    slots: tuple[int, ...],
    image_wh: tuple[int, int],
    pad: float = CROP_PAD,
    pad_y: float | None = None,
) -> tuple[int, int, int, int] | None:
    """Padded pixel box around an OSF eye, clamped to the frame.

    Vertical pad is larger so a look-up pupil still sits inside the crop.
    """
    pts = []
    for i in slots:
        if i >= len(lms) or float(lms[i, 2]) < 0.12:
            continue
        pts.append(lms[i, :2].astype(np.float64))
    if not pts:
        return None
    arr = np.stack(pts, axis=0)
    x0 = float(arr[:, 0].min())
    y0 = float(arr[:, 1].min())
    x1 = float(arr[:, 0].max())
    y1 = float(arr[:, 1].max())
    span = max(x1 - x0, y1 - y0, 8.0)
    extra_x = span * float(pad)
    extra_y = span * float(CROP_PAD_Y if pad_y is None else pad_y)
    w, h = int(image_wh[0]), int(image_wh[1])
    ix0 = int(max(0, np.floor(x0 - extra_x)))
    iy0 = int(max(0, np.floor(y0 - extra_y)))
    ix1 = int(min(w, np.ceil(x1 + extra_x)))
    iy1 = int(min(h, np.ceil(y1 + extra_y)))
    if ix1 - ix0 < 4 or iy1 - iy0 < 4:
        return None
    return ix0, iy0, ix1, iy1


def map_crop_to_frame(
    xy: tuple[float, float],
    scale: float,
    origin: tuple[float, float],
) -> tuple[float, float]:
    """Map a point in an upscaled crop back to full-frame pixels."""
    factor = float(scale) if float(scale) > 1e-6 else 1.0
    return float(origin[0]) + float(xy[0]) / factor, float(origin[1]) + float(xy[1]) / factor


def upscale_crop(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    long_side: int = CROP_LONG,
) -> tuple[np.ndarray, float, tuple[float, float]] | None:
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = image_bgr[y0:y1, x0:x1]
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    long = max(h, w, 1)
    scale = float(long_side) / float(long)
    if scale > 1.01:
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
    else:
        scale = 1.0
    return crop, scale, (float(x0), float(y0))


def detect_crops(image_bgr: np.ndarray, lms: np.ndarray | None) -> list[dict[str, object]]:
    """YOLO on padded, upscaled OSF eye crops. Hits are full-frame pixels."""
    model = _load_yolo()
    if model is None or image_bgr is None or image_bgr.size == 0 or lms is None:
        return []
    h, w = image_bgr.shape[:2]
    crops: list[np.ndarray] = []
    metas: list[tuple[float, tuple[float, float]]] = []
    for slots in (OSF_RIGHT, OSF_LEFT):
        box = eye_crop_box(lms, slots, (w, h))
        if box is None:
            continue
        packed = upscale_crop(image_bgr, box)
        if packed is None:
            continue
        crop, scale, origin = packed
        crops.append(crop)
        metas.append((scale, origin))
    if not crops:
        return []
    device = getattr(model, "_iris_device", None)
    try:
        results = model.predict(source=crops, conf=DETECT_CONF, verbose=False, device=device)
    except Exception:
        return []
    if not results:
        return []
    dets: list[dict[str, object]] = []
    for result, (scale, origin) in zip(results, metas):
        for det in _hits_from_result(result):
            cx, cy = map_crop_to_frame((float(det["cx"]), float(det["cy"])), scale, origin)
            pupil = det.get("pupil")
            if isinstance(pupil, tuple) and len(pupil) >= 2:
                pupil = map_crop_to_frame((float(pupil[0]), float(pupil[1])), scale, origin)
            bbox = det.get("bbox")
            mapped_box = None
            if isinstance(bbox, tuple) and len(bbox) >= 4:
                p0 = map_crop_to_frame((float(bbox[0]), float(bbox[1])), scale, origin)
                p1 = map_crop_to_frame((float(bbox[2]), float(bbox[3])), scale, origin)
                mapped_box = (p0[0], p0[1], p1[0], p1[1])
            dets.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "bbox": mapped_box,
                    "pupil": pupil,
                    "score": det["score"],
                    "visible": det["visible"],
                }
            )
    return dets


def _eye_anchor(lms: np.ndarray, slots: tuple[int, ...]) -> tuple[np.ndarray, float] | None:
    pts = []
    for i in slots:
        if i >= len(lms) or float(lms[i, 2]) < 0.12:
            continue
        pts.append(lms[i, :2].astype(np.float64))
    if not pts:
        return None
    arr = np.stack(pts, axis=0)
    return arr.mean(axis=0), float(max(arr[:, 0].max() - arr[:, 0].min(), 8.0))


def _eye_limits(lms: np.ndarray, slots: tuple[int, ...]) -> tuple[float, float, float, float, float, float] | None:
    pts = []
    for i in slots:
        if i >= len(lms) or float(lms[i, 2]) < 0.12:
            continue
        pts.append(lms[i, :2].astype(np.float64))
    if not pts:
        return None
    arr = np.stack(pts, axis=0)
    x0 = float(arr[:, 0].min())
    y0 = float(arr[:, 1].min())
    x1 = float(arr[:, 0].max())
    y1 = float(arr[:, 1].max())
    width = max(x1 - x0, 8.0)
    height = max(y1 - y0, width * 0.38)
    return x0, y0, x1, y1, width, height


def pupil_in_eye(
    xy: tuple[float, float],
    lms: np.ndarray,
    slots: tuple[int, ...],
    *,
    pad_x: float = _IN_EYE_X,
    pad_y: float = _IN_EYE_Y,
) -> bool:
    """True when the pupil is still inside the eye opening, not the crop gutter."""
    limits = _eye_limits(lms, slots)
    if limits is None:
        return False
    x0, y0, x1, y1, width, height = limits
    px, py = float(xy[0]), float(xy[1])
    return (x0 - width * pad_x) <= px <= (x1 + width * pad_x) and (
        y0 - height * pad_y
    ) <= py <= (y1 + height * pad_y)


def match_to_eyes(
    pts: np.ndarray | None,
    dets: list[dict[str, object]],
    right_slots: tuple[int, ...],
    left_slots: tuple[int, ...],
) -> tuple[IrisHit, IrisHit]:
    """Person-right / person-left pupils nearest the given eye slots."""
    empty_r, empty_l = IrisHit(side="r", method="iris_pose"), IrisHit(side="l", method="iris_pose")
    need = max(right_slots + left_slots)
    if pts is None or len(pts) <= need or not dets:
        return empty_r, empty_l
    used: set[int] = set()

    def _for(slots: tuple[int, ...], other: np.ndarray | None, side: str) -> IrisHit:
        packed = _eye_anchor(pts, slots)
        if packed is None:
            return IrisHit(side=side, method="iris_pose")
        anchor, eye_w = packed
        max_dist = max(40.0, eye_w * 1.5)
        best_i = None
        best_d = 1e9
        for i, det in enumerate(dets):
            if i in used:
                continue
            d = float(np.hypot(float(det["cx"]) - anchor[0], float(det["cy"]) - anchor[1]))
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is None or best_d > max_dist:
            return IrisHit(side=side, method="iris_pose")
        det = dets[best_i]
        raw_box = det.get("bbox")
        box = None
        if isinstance(raw_box, (tuple, list)) and len(raw_box) >= 4:
            box = tuple(float(v) for v in raw_box[:4])
        if other is not None:
            other_d = float(np.hypot(float(det["cx"]) - other[0], float(det["cy"]) - other[1]))
            if other_d + max(2.0, eye_w * 0.10) < best_d:
                return IrisHit(side=side, method="iris_pose")
        used.add(best_i)
        pupil = det.get("pupil")
        if det.get("visible") and isinstance(pupil, tuple) and len(pupil) >= 2:
            if not pupil_in_eye((float(pupil[0]), float(pupil[1])), pts, slots):
                return IrisHit(
                    score=float(det["score"]),
                    visible=False,
                    side=side,
                    method="iris_pose",
                    box=box,
                )
            return IrisHit(
                x=float(pupil[0]),
                y=float(pupil[1]),
                score=float(det["score"]),
                visible=True,
                side=side,
                method="iris_pose",
                box=box,
            )
        return IrisHit(
            score=float(det["score"]),
            visible=False,
            side=side,
            method="iris_pose",
            box=box,
        )

    right_pack = _eye_anchor(pts, right_slots)
    left_pack = _eye_anchor(pts, left_slots)
    right_anchor = None if right_pack is None else right_pack[0]
    left_anchor = None if left_pack is None else left_pack[0]
    right = _for(right_slots, left_anchor, "r")
    left = _for(left_slots, right_anchor, "l")
    return right, left


def match_to_osf(lms: np.ndarray | None, dets: list[dict[str, object]]) -> tuple[IrisHit, IrisHit]:
    """Pupils in camera pixels. Side "r" = OSF 36-41 = image-left eye
    (dlib names eyes from the person's view on an unflipped frame)."""
    return match_to_eyes(lms, dets, OSF_RIGHT, OSF_LEFT)


def track_camera(image_bgr: np.ndarray, lms: np.ndarray | None) -> tuple[IrisHit, IrisHit]:
    return match_to_osf(lms, detect_crops(image_bgr, lms))


def _gaze_ok(x: float, y: float, conf: float, conf_thr: float = GAZE_CONF) -> bool:
    return conf >= conf_thr and x > 1.0 and y > 1.0


def osf_gaze_hits(
    lms: np.ndarray | None = None,
    eye_state: object = None,
    *,
    conf_thr: float = GAZE_CONF,
) -> tuple[IrisHit, IrisHit]:
    """Pupils from swapped OSF lms[66]/[67] or raw eye_state [open, y, x, conf]."""
    right = IrisHit(side="r", method="osf_gaze")
    left = IrisHit(side="l", method="osf_gaze")

    def _hit(x: float, y: float, conf: float, side: str) -> IrisHit:
        ok = _gaze_ok(x, y, conf, conf_thr)
        return IrisHit(
            x=float(x),
            y=float(y),
            score=float(conf),
            visible=bool(ok),
            side=side,
            method="osf_gaze",
        )

    if lms is not None and len(lms) >= 68:
        right = _hit(float(lms[OSF_GAZE_RIGHT, 0]), float(lms[OSF_GAZE_RIGHT, 1]), float(lms[OSF_GAZE_RIGHT, 2]), "r")
        left = _hit(float(lms[OSF_GAZE_LEFT, 0]), float(lms[OSF_GAZE_LEFT, 1]), float(lms[OSF_GAZE_LEFT, 2]), "l")
        return right, left
    if eye_state is None:
        return right, left
    try:
        rows = list(eye_state)
    except TypeError:
        return right, left
    if len(rows) < 2:
        return right, left
    # eye_state[i] = [open, y, x, conf] in image x,y.
    er, el = rows[0], rows[1]
    try:
        right = _hit(float(er[2]), float(er[1]), float(er[3]), "r")
        left = _hit(float(el[2]), float(el[1]), float(el[3]), "l")
    except (TypeError, ValueError, IndexError):
        return right, left
    return right, left


def merge_hits(
    custom: tuple[IrisHit, IrisHit],
    osf: tuple[IrisHit, IrisHit],
) -> tuple[IrisHit, IrisHit, str]:
    """Prefer YOLO, else OSF gaze. Same rule as live-poser custom_then_osf."""

    def _pick(c: IrisHit, o: IrisHit) -> IrisHit:
        if c.visible:
            return c
        if o.visible:
            return o
        return c if c.method else o

    right = _pick(custom[0], osf[0])
    left = _pick(custom[1], osf[1])
    methods = set()
    if right.visible and right.method:
        methods.add(right.method)
    if left.visible and left.method:
        methods.add(left.method)
    if not methods:
        method = "none"
    elif len(methods) == 1:
        method = next(iter(methods))
    else:
        method = "mixed"
    return right, left, method


def hits_payload(right: IrisHit, left: IrisHit) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for hit in (right, left):
        out.append(
            {
                "side": hit.side,
                "x": round(float(hit.x), 3),
                "y": round(float(hit.y), 3),
                "score": round(float(hit.score), 3),
                "visible": bool(hit.visible),
                "method": str(hit.method or ""),
            }
        )
    return out


def raw_debug(iris_cam: object = None, look: object = None) -> dict[str, object]:
    """Camera-space iris hits + IFM look. No retarget, no overlay points."""
    hits: list[dict[str, object]] = []
    if isinstance(iris_cam, list):
        for row in iris_cam:
            if not isinstance(row, dict):
                continue
            try:
                hits.append(
                    {
                        "side": str(row.get("side") or ""),
                        "x": round(float(row.get("x") or 0.0), 3),
                        "y": round(float(row.get("y") or 0.0), 3),
                        "score": round(float(row.get("score") or 0.0), 3),
                        "visible": bool(row.get("visible")),
                        "method": str(row.get("method") or ""),
                    }
                )
            except (TypeError, ValueError):
                continue
    packed_look = None
    if isinstance(look, dict):
        try:
            packed_look = {
                "x": round(float(look.get("x") or 0.0), 3),
                "y": round(float(look.get("y") or 0.0), 3),
            }
        except (TypeError, ValueError):
            packed_look = None
    return {"iris_cam": hits, "look": packed_look}


def payload_to_hits(raw: object) -> tuple[IrisHit, IrisHit]:
    right, left = IrisHit(side="r"), IrisHit(side="l")
    if not isinstance(raw, list):
        return right, left
    for row in raw:
        if not isinstance(row, dict):
            continue
        method = str(row.get("method") or "")
        visible = bool(row.get("visible"))
        if visible and not method:
            method = "iris_pose"
        hit = IrisHit(
            x=float(row.get("x") or 0.0),
            y=float(row.get("y") or 0.0),
            score=float(row.get("score") or 0.0),
            visible=visible,
            side=str(row.get("side") or ""),
            method=method,
        )
        if hit.side == "r":
            right = hit
        elif hit.side == "l":
            left = hit
    return right, left


def map_into_eye(
    pupil: tuple[float, float],
    src_eye: np.ndarray,
    dest_eye: np.ndarray,
) -> tuple[float, float]:
    """Move a camera pupil into the matching character eye box."""
    src = np.asarray(src_eye, dtype=np.float64)
    dest = np.asarray(dest_eye, dtype=np.float64)
    src_c = src.mean(axis=0)
    dest_c = dest.mean(axis=0)
    src_w = max(float(src[:, 0].max() - src[:, 0].min()), 8.0)
    src_h = max(float(src[:, 1].max() - src[:, 1].min()), 6.0)
    dest_w = max(float(dest[:, 0].max() - dest[:, 0].min()), 8.0)
    dest_h = max(float(dest[:, 1].max() - dest[:, 1].min()), 6.0)
    nx = float(np.clip((pupil[0] - src_c[0]) / src_w, -0.72, 0.72))
    ny = float(np.clip((pupil[1] - src_c[1]) / src_h, -0.72, 0.72))
    return float(dest_c[0] + nx * dest_w), float(dest_c[1] + ny * dest_h)


def pupil_frac(xy: tuple[float, float], eye: np.ndarray) -> tuple[float, float] | None:
    """Pupil as a fraction of the OSF eye box, origin at the box center."""
    src = np.asarray(eye, dtype=np.float64)
    if src.ndim != 2 or len(src) < 2:
        return None
    c = src.mean(axis=0)
    w = max(float(src[:, 0].max() - src[:, 0].min()), 8.0)
    h = max(float(src[:, 1].max() - src[:, 1].min()), 6.0)
    return (float(xy[0]) - float(c[0])) / w, (float(xy[1]) - float(c[1])) / h


def rest_look_from_cam(
    lms: np.ndarray | None,
    right: IrisHit,
    left: IrisHit,
) -> dict[str, object]:
    """Snapshot each visible pupil as nx/ny of its OSF eye box."""
    out: dict[str, object] = {}
    if lms is None:
        return out
    for hit, slots, side in ((right, OSF_RIGHT, "r"), (left, OSF_LEFT, "l")):
        if not hit.visible:
            continue
        eye = _osf_eye(lms, slots)
        if eye is None:
            continue
        frac = pupil_frac((hit.x, hit.y), eye)
        if frac is None:
            continue
        out[side] = {"nx": round(float(frac[0]), 4), "ny": round(float(frac[1]), 4)}
    return out


def _side_frac(rest_look: object, side: str) -> tuple[float, float] | None:
    if not isinstance(rest_look, dict):
        return None
    raw = rest_look.get(side)
    if not isinstance(raw, dict):
        return None
    try:
        return float(raw["nx"]), float(raw["ny"])
    except (KeyError, TypeError, ValueError):
        return None


def _look_xy(rest_look: object) -> tuple[float, float]:
    if not isinstance(rest_look, dict):
        return 0.0, 0.0
    try:
        return float(rest_look.get("x") or 0.0), float(rest_look.get("y") or 0.0)
    except (TypeError, ValueError):
        return 0.0, 0.0


def _blink_mix(shut: float) -> float | None:
    """1 = full live pupil, 0 = rest. None hides the iris."""
    closed = float(shut)
    if closed >= BLINK_HIDE:
        return None
    if closed <= 0.0:
        return 1.0
    return 1.0 - (closed / BLINK_HIDE)


def _look_pairs(
    blink: dict[str, float],
) -> tuple[tuple[int, tuple[int, ...], float], tuple[int, tuple[int, ...], float]]:
    """Iris 28 shares blink.l / slots 11-13; iris 29 shares blink.r / 17-19.

    ``blink`` is keyed by character screen side here.
    """
    return (
        (RIGHT_IRIS, LEFT_EYE_SLOTS, float(blink.get("l") or 0.0)),
        (LEFT_IRIS, RIGHT_EYE_SLOTS, float(blink.get("r") or 0.0)),
    )


def _cam_pairs(
    right: IrisHit,
    left: IrisHit,
    blink: dict[str, float],
    selfie: bool = False,
) -> tuple[
    tuple[IrisHit, tuple[int, ...], tuple[int, ...], int, float, str],
    tuple[IrisHit, tuple[int, ...], tuple[int, ...], int, float, str],
]:
    """Canonical: the image-left pupil (OSF 36-41, hit side "r") fills the
    screen-left iris 28 in eye slots 11-13. Selfie swaps the camera side.
    ``blink`` is keyed by character screen side.
    """
    shut_l = float(blink.get("l") or 0.0)
    shut_r = float(blink.get("r") or 0.0)
    if selfie:
        return (
            (left, OSF_LEFT, LEFT_EYE_SLOTS, RIGHT_IRIS, shut_l, "l"),
            (right, OSF_RIGHT, RIGHT_EYE_SLOTS, LEFT_IRIS, shut_r, "r"),
        )
    return (
        (right, OSF_RIGHT, LEFT_EYE_SLOTS, RIGHT_IRIS, shut_l, "r"),
        (left, OSF_LEFT, RIGHT_EYE_SLOTS, LEFT_IRIS, shut_r, "l"),
    )


def _lid_y(pts: np.ndarray, slots: tuple[int, ...]) -> float | None:
    split = _split_eye(pts, slots)
    if split is None:
        return None
    return float(split[1][1])


def _mix_toward_lid(
    live_x: float,
    live_y: float,
    cx: float,
    lid_y: float,
    mix: float | None,
) -> tuple[float, float] | None:
    if mix is None:
        return None
    x = float(cx + mix * (live_x - cx))
    y = float(lid_y + mix * (live_y - lid_y))
    return x, y


def _char_eye(pts: np.ndarray, slots: tuple[int, ...]) -> np.ndarray | None:
    rows = []
    for i in slots:
        if i >= len(pts) or float(pts[i, 2]) < 0.05:
            continue
        rows.append(pts[i, :2])
    if len(rows) < 2:
        return None
    return np.stack(rows, axis=0)


def _split_eye(
    pts: np.ndarray, slots: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Corners + upper lid. Label28 is (outer, lid, inner) per eye."""
    if pts is None or len(slots) < 2:
        return None
    lid_xy = None
    if len(slots) >= 3:
        lid = int(slots[1])
        if lid < len(pts) and float(pts[lid, 2]) >= 0.05:
            lid_xy = pts[lid, :2].astype(np.float64)
    corners: list[np.ndarray] = []
    corner_ids = (slots[0], slots[2]) if len(slots) >= 3 else slots
    for i in corner_ids:
        if int(i) >= len(pts) or float(pts[int(i), 2]) < 0.05:
            continue
        corners.append(pts[int(i), :2].astype(np.float64))
    if len(corners) < 1:
        return None
    arr = np.stack(corners, axis=0)
    if lid_xy is None:
        lid_xy = arr.mean(axis=0)
        lid_xy[1] = float(arr[:, 1].min())
    return arr, lid_xy


def iris_anchor(
    pts: np.ndarray, slots: tuple[int, ...]
) -> tuple[float, float, float, float] | None:
    """Fallback pupil: midpoint of the eye corners, not the lid mid."""
    split = _split_eye(pts, slots)
    if split is None:
        return None
    corners, lid = split
    cx = float(corners[:, 0].mean())
    cy_c = float(corners[:, 1].mean())
    width = max(float(corners[:, 0].max() - corners[:, 0].min()), 8.0)
    opening = max(float(cy_c - float(lid[1])), width * 0.22)
    return cx, cy_c, width, opening


def iris_well(
    pts: np.ndarray,
    slots: tuple[int, ...],
    rest_xy: tuple[float, float] | None = None,
) -> np.ndarray | None:
    """Box around the iris disc, used as the live map target.

    When rest_xy is the still-detected pupil, live gaze moves around that
    original point instead of the geometric eye midpoint.
    """
    anc = iris_anchor(pts, slots)
    if anc is None:
        return _char_eye(pts, slots)
    cx, cy, width, opening = anc
    if rest_xy is not None:
        cx, cy = float(rest_xy[0]), float(rest_xy[1])
    hw = width * _IRIS_HALF_W
    hh = max(opening * 0.55, width * 0.22)
    return np.array(
        [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ],
        dtype=np.float64,
    )


def _eye_home(
    pts: np.ndarray,
    slots: tuple[int, ...],
    rest_xy: tuple[float, float] | None = None,
    rest_pts: np.ndarray | None = None,
) -> tuple[float, float, float, float] | None:
    """Live iris home. Rest pupils ride the current eye, not rest pixels.

    Track captures the still pupil in rest-face space. After a head turn those
    pixels stay on the camera. Carry the same offset into the posed eye so a
    look-with-the-head keeps the pupils facing the turn.
    """
    anc = iris_anchor(pts, slots)
    if anc is None:
        return None
    cx, cy, width, opening = anc
    if rest_xy is None:
        return cx, cy, width, opening
    rest_anc = iris_anchor(rest_pts, slots) if rest_pts is not None else None
    if rest_anc is None:
        return float(rest_xy[0]), float(rest_xy[1]), width, opening
    rcx, rcy, rwidth, ropening = rest_anc
    nx = (float(rest_xy[0]) - rcx) / max(rwidth, 1e-6)
    ny = (float(rest_xy[1]) - rcy) / max(ropening, 1e-6)
    return cx + nx * width, cy + ny * opening, width, opening


def _place_in_iris(
    raw: tuple[float, float] | None,
    pts: np.ndarray,
    slots: tuple[int, ...],
    rest_xy: tuple[float, float] | None = None,
    max_look_x: float = 1.0,
    max_look_y: float = 1.0,
    rest_pts: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Keep iris_pose.pt pupils. Nudge lid hits just inside the opening."""
    home = _eye_home(pts, slots, rest_xy, rest_pts)
    if home is None:
        return rest_xy if raw is None else raw
    cx, cy, width, opening = home
    if raw is None:
        return cx, cy
    mx = float(np.clip(max_look_x, 0.0, 1.0))
    my = float(np.clip(max_look_y, 0.0, 1.0))
    px, py = float(raw[0]), float(raw[1])
    x = soft_barrier(px, cx - width * 0.55 * mx, cx + width * 0.55 * mx, cx)
    lid_band = cy - opening * 0.55 * my
    y = soft_barrier(py, lid_band, cy + opening * 0.95 * my, cy)
    return x, y


def _osf_eye(lms: np.ndarray, slots: tuple[int, ...]) -> np.ndarray | None:
    rows = []
    for i in slots:
        if i >= len(lms) or float(lms[i, 2]) < 0.12:
            continue
        rows.append(lms[i, :2])
    if len(rows) < 2:
        return None
    return np.stack(rows, axis=0)


def _row(
    slot: int,
    x: float,
    y: float,
    score: float,
    visible: bool,
    box: tuple[float, float, float, float] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": slot,
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "score": round(float(score), 3),
        "visible": bool(visible),
    }
    if box is not None:
        row["box"] = [round(float(v), 1) for v in box]
    return row


def _rest_xy(rest_iris: object, slot: int) -> tuple[float, float] | None:
    if not isinstance(rest_iris, list):
        return None
    for row in rest_iris:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("id", -1))
        except (TypeError, ValueError):
            continue
        if idx != slot:
            continue
        try:
            return float(row.get("x") or 0.0), float(row.get("y") or 0.0)
        except (TypeError, ValueError):
            return None
    return None


def from_eye_mid(
    char_pts: np.ndarray,
    rest_iris: object = None,
    rest_pts: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Fallback pupils: still iris, else eye-corner midpoint."""
    out: list[dict[str, object]] = []
    for slot, eyes in (
        (RIGHT_IRIS, LEFT_EYE_SLOTS),
        (LEFT_IRIS, RIGHT_EYE_SLOTS),
    ):
        placed = _place_in_iris(
            None, char_pts, eyes, _rest_xy(rest_iris, slot), rest_pts=rest_pts
        )
        if placed is None:
            continue
        out.append(_row(slot, placed[0], placed[1], 0.7, True))
    return out


def rows_from_hits(right: IrisHit, left: IrisHit) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if right.visible:
        out.append(_row(RIGHT_IRIS, right.x, right.y, right.score, True))
    if left.visible:
        out.append(_row(LEFT_IRIS, left.x, left.y, left.score, True))
    return out


_SHUT_GAP = 0.13


def _eye_opening(pts: np.ndarray, slots: tuple[int, ...]) -> tuple[float, float, float, float, float] | None:
    """Corner center, lid gap, width, and gap/width. None when the eye is missing."""
    split = _split_eye(pts, slots)
    if split is None:
        return None
    corners, lid = split
    cx = float(corners[:, 0].mean())
    cy = float(corners[:, 1].mean())
    width = max(float(corners[:, 0].max() - corners[:, 0].min()), 8.0)
    gap = float(cy - float(lid[1]))
    return cx, cy, width, gap, gap / width


def catchlight_pupil(
    image_bgr: np.ndarray | None,
    pts: np.ndarray,
    slots: tuple[int, ...],
) -> tuple[float, float] | None:
    """White highlight inside an open mesh eye. Anime pupils carry that dot.

    The iris model misses a stylized pupil (it draws the eye and hides the
    keypoint). The highlight is low-saturation and bright, and it sits in
    the iris rather than on the lid line.
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return None
    opened = _eye_opening(pts, slots)
    if opened is None:
        return None
    cx, _cy, width, gap, ratio = opened
    if ratio < _SHUT_GAP:
        return None
    split = _split_eye(pts, slots)
    if split is None:
        return None
    _corners, lid = split
    rx = width * 0.42
    ry = max(gap * 1.35, 6.0)
    oy = float(lid[1]) + gap * 0.90
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    ys, xs = np.ogrid[:h, :w]
    mask = ((xs - cx) / rx) ** 2 + ((ys - oy) / ry) ** 2 <= 1.0
    spec = (mask & (value >= 210) & (sat <= 30)).astype(np.uint8)
    count, _labels, stats, cents = cv2.connectedComponentsWithStats(spec, 8)
    best: tuple[float, float, float] | None = None
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 3 or area > 400:
            continue
        x, y = float(cents[i, 0]), float(cents[i, 1])
        dist = ((x - cx) / rx) ** 2 + ((y - oy) / ry) ** 2
        if best is None or dist < best[0]:
            best = (dist, x, y)
    if best is None:
        return None
    return best[1], best[2]


def track_still(
    image_bgr: np.ndarray | None,
    pts28: np.ndarray | None,
) -> tuple[list[dict[str, object]], str]:
    """Iris on the character still (Track). Same pixel space as the mesh.

    The mesh eye says where to look and whether the lid is open. An open eye
    takes the ``iris_pose`` pupil when the model marks one. A white catchlight
    is only used when that pupil is hidden. A shut lid drops the point. A
    midpoint is only used when nothing else found the eye and the model
    returned no detections at all.
    """
    if pts28 is None or len(pts28) < 20:
        return [], "none"
    dets: list[dict[str, object]] = []
    if image_bgr is not None and getattr(image_bgr, "size", 0):
        dets = detect(image_bgr)
    right, left = match_to_eyes(pts28, dets, LEFT_EYE_SLOTS, RIGHT_EYE_SLOTS)
    used_pose = False
    rows: list[dict[str, object]] = []
    for hit, eyes, slot in (
        (right, LEFT_EYE_SLOTS, RIGHT_IRIS),
        (left, RIGHT_EYE_SLOTS, LEFT_IRIS),
    ):
        opened = _eye_opening(pts28, eyes)
        if opened is not None and opened[4] < _SHUT_GAP:
            if hit.box is not None:
                x0, y0, x1, y1 = hit.box
                rows.append(_row(slot, (x0 + x1) / 2.0, (y0 + y1) / 2.0, hit.score, False, hit.box))
                used_pose = True
            continue
        if (hit.box is not None or hit.score > 0.0) and not hit.visible:
            if hit.box is not None:
                x0, y0, x1, y1 = hit.box
                rows.append(_row(slot, (x0 + x1) / 2.0, (y0 + y1) / 2.0, hit.score, False, hit.box))
                used_pose = True
            continue
        if hit.visible:
            placed = _place_in_iris((hit.x, hit.y), pts28, eyes)
            if placed is None:
                continue
            used_pose = True
            rows.append(_row(slot, placed[0], placed[1], hit.score, True, hit.box))
            continue
        spot = catchlight_pupil(image_bgr, pts28, eyes)
        if spot is not None:
            used_pose = True
            rows.append(_row(slot, spot[0], spot[1], 0.9, True))
            continue
        if dets:
            continue
        placed = _place_in_iris(None, pts28, eyes)
        if placed is None:
            continue
        rows.append(_row(slot, placed[0], placed[1], 0.7, True))
    if not rows:
        return [], "none"
    return rows, "iris_pose" if used_pose else "eye_mid"


def from_look(
    char_pts: np.ndarray,
    look_x: float,
    look_y: float,
    blink: dict[str, float] | None = None,
    rest_iris: object = None,
    gaze_gain: float = 1.0,
    rest_look: object = None,
    selfie: bool = False,
    max_look_x: float = 1.0,
    max_look_y: float = 1.0,
    rest_pts: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Look in the canonical camera frame (+x = image-right): slide pupils
    from the still iris, or the eye box. Selfie negates X."""
    blink = blink or {}
    rx, ry = _look_xy(rest_look)
    gain = float(gaze_gain)
    mx = float(np.clip(max_look_x, 0.0, 1.0))
    my = float(np.clip(max_look_y, 0.0, 1.0))
    lx = float(np.clip(look_x - rx, -1.0, 1.0)) * _LOOK_SPAN * gain * mx
    ly = float(np.clip(look_y - ry, -1.0, 1.0)) * _LOOK_SPAN * gain * my
    if selfie:
        lx = -lx
    out: list[dict[str, object]] = []
    for slot, eyes, shut in _look_pairs(blink):
        mix = _blink_mix(shut)
        if mix is None:
            continue
        rest = _rest_xy(rest_iris, slot)
        home = _eye_home(char_pts, eyes, rest, rest_pts)
        if home is None:
            continue
        cx, cy, width, opening = home
        live_x = float(cx + lx * width)
        live_y = float(cy + ly * opening)
        lid_y = _lid_y(char_pts, eyes)
        mixed = _mix_toward_lid(live_x, live_y, cx, lid_y if lid_y is not None else cy, mix)
        if mixed is None:
            continue
        placed = _place_in_iris(
            mixed,
            char_pts,
            eyes,
            rest,
            max_look_x=mx,
            max_look_y=my,
            rest_pts=rest_pts,
        )
        if placed is None:
            continue
        out.append(_row(slot, placed[0], placed[1], 0.85, True))
    return out


def retarget(
    char_pts: np.ndarray | None,
    *,
    cam_lms: np.ndarray | None = None,
    cam_iris: object = None,
    look: object = None,
    blink: dict[str, float] | None = None,
    rest_iris: object = None,
    rest_look: object = None,
    rest_pts: np.ndarray | None = None,
    gaze_gain: float = 1.0,
    selfie: bool = False,
    max_look_x: float = 1.0,
    max_look_y: float = 1.0,
) -> tuple[list[dict[str, object]], str]:
    """Character-space iris rows for slots 28 (screen-left) and 29.

    ``cam_iris`` / ``look`` are canonical camera-frame inputs; ``blink`` is
    keyed by character screen side. ``selfie`` applies the one mirror rule.
    """
    if char_pts is None or len(char_pts) < 20:
        return [], "none"
    blink = blink or {}
    gain = float(gaze_gain)
    mx = float(np.clip(max_look_x, 0.0, 1.0))
    my = float(np.clip(max_look_y, 0.0, 1.0))
    right, left = payload_to_hits(cam_iris)
    rows: list[dict[str, object]] = []
    methods: set[str] = set()
    cam_hits = 0
    cam_hidden = 0
    if cam_lms is not None and (right.visible or left.visible):
        for hit, osf_slots, char_slots, slot, shut, side in _cam_pairs(
            right, left, blink, selfie
        ):
            if not hit.visible:
                continue
            cam_hits += 1
            mix = _blink_mix(shut)
            if mix is None:
                cam_hidden += 1
                continue
            rest = _rest_xy(rest_iris, slot)
            src = _osf_eye(cam_lms, osf_slots)
            home = _eye_home(char_pts, char_slots, rest, rest_pts)
            if src is None or home is None:
                continue
            live_frac = pupil_frac((hit.x, hit.y), src)
            if live_frac is None:
                continue
            rest_frac = _side_frac(rest_look, side)
            cx, cy, width, opening = home
            if rest_frac is None:
                dx = dy = 0.0
            else:
                dx = (live_frac[0] - rest_frac[0]) * gain
                dy = (live_frac[1] - rest_frac[1]) * gain
            if selfie:
                dx = -dx
            live_x = cx + dx * width * mx
            live_y = cy + dy * opening * my
            lid_y = _lid_y(char_pts, char_slots)
            mixed = _mix_toward_lid(
                live_x, live_y, cx, lid_y if lid_y is not None else cy, mix
            )
            if mixed is None:
                continue
            placed = _place_in_iris(
                mixed,
                char_pts,
                char_slots,
                rest,
                max_look_x=mx,
                max_look_y=my,
                rest_pts=rest_pts,
            )
            if placed is None:
                continue
            rows.append(_row(slot, placed[0], placed[1], hit.score, True))
            methods.add(hit.method or "iris_pose")
        if rows:
            method = next(iter(methods)) if len(methods) == 1 else "mixed"
            return rows, method
        if cam_hits and cam_hidden == cam_hits:
            return [], "none"
    if isinstance(look, dict):
        try:
            lx = float(look.get("x") or 0.0)
            ly = float(look.get("y") or 0.0)
        except (TypeError, ValueError):
            lx = ly = 0.0
        rows = from_look(
            char_pts,
            lx,
            ly,
            blink,
            rest_iris=rest_iris,
            gaze_gain=gain,
            rest_look=rest_look,
            selfie=selfie,
            max_look_x=mx,
            max_look_y=my,
            rest_pts=rest_pts,
        )
        if rows:
            return rows, "look"
        shut_l = float(blink.get("l") or 0.0)
        shut_r = float(blink.get("r") or 0.0)
        if _blink_mix(shut_l) is None and _blink_mix(shut_r) is None:
            return [], "none"
    rows = from_eye_mid(char_pts, rest_iris=rest_iris, rest_pts=rest_pts)
    if rows:
        return rows, "eye_mid"
    return [], "none"
