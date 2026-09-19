"""Anime face mesh: YOLOv8 box + HRNet 28 landmarks → label28 rest pose."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .hrnet import HRNetV2

ROOT = Path(__file__).resolve().parents[1]
ANIME_DIR = ROOT / "models" / "anime"
YOLO_NAME = "face_yolov8n.pt"
HRNET_NAME = "mmpose_anime-face_hrnetv2.pth"
INPUT_SIZE = 256
BOX_SCALE = 1.25
FACE_SCORE = 0.15
# Hair swirls can outscore the real face. Ignore tiny boxes when a
# character-sized hit exists (fraction of the short image side).
MIN_FACE_FRAC = 0.12
# HRNet crop / rest jaw must cover the drawn face, not a bangs swirl.
FIT_FACE_FRAC = 0.30
# Bust stills: YOLO often returns head+shoulders. Face lives in the upper band.
HEAD_BOX_FRAC = 0.40
HEAD_BOX_BOTTOM = 0.58
FACE_BAND = 0.70
FACE_BAND_CY = 0.40
FACE_SIDE_KEEP = 0.78
# Extra image below the face band so HRNet can see the round anime chin.
CHIN_EXTRA = 0.14
# Mouth→chin as a fraction of nose→mouth. Anime jaws sit well below the slit.
CHIN_MOUTH_FRAC = 1.55
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OUTLINE = (0, 1, 2, 3, 4)
LEFT_BROW = (5, 6, 7)
RIGHT_BROW = (8, 9, 10)
NOSE = (14, 15, 16)
MOUTH_UPPER = (23, 20, 21, 22, 26)
MOUTH_LOWER = (23, 24, 25, 27, 26)
CONNECTIONS = (
    (OUTLINE, (80, 200, 255)),
    (LEFT_BROW, (80, 255, 160)),
    (RIGHT_BROW, (80, 255, 160)),
    (NOSE, (255, 180, 80)),
    (MOUTH_UPPER, (180, 80, 255)),
    (MOUTH_LOWER, (180, 80, 255)),
)
POINT_COLOR = {
    **{i: (80, 200, 255) for i in range(0, 5)},
    **{i: (80, 255, 160) for i in range(5, 11)},
    **{i: (255, 120, 80) for i in list(range(11, 14)) + list(range(17, 20))},
    **{i: (255, 180, 80) for i in range(14, 17)},
    **{i: (180, 80, 255) for i in range(20, 28)},
}

_NOSE_HALF_W = 0.14
_NOSE_SIDE_LIFT = 0.04

_detector: "AnimeFaceMesh | None" = None


class AnimeMeshError(RuntimeError):
    pass


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def hrnet_native_to_label28(pts: np.ndarray) -> np.ndarray:
    """Native HRNet 28 → project label28 (eyes stay, nose/mouth expanded)."""
    src = np.asarray(pts, dtype=np.float32)
    if src.ndim != 2 or src.shape[0] < 28 or src.shape[1] < 2:
        return src
    cols = int(src.shape[1])
    out = np.zeros((28, cols), dtype=np.float32)
    for i in list(range(0, 14)) + list(range(17, 20)):
        out[i] = src[i]

    def _mid(idxs: tuple[int, ...]) -> tuple[float, float] | None:
        xs, ys = [], []
        for i in idxs:
            if src.shape[1] > 2 and float(src[i, 2]) < 0.05:
                continue
            xs.append(float(src[i, 0]))
            ys.append(float(src[i, 1]))
        if not xs:
            return None
        return float(np.mean(xs)), float(np.mean(ys))

    l_eye = _mid((11, 12, 13))
    r_eye = _mid((17, 18, 19))
    eye_dist = 1.0
    if l_eye is not None and r_eye is not None:
        eye_dist = max(abs(float(r_eye[0]) - float(l_eye[0])), 1e-6)

    if cols < 3 or float(src[23, 2]) >= 0.05:
        tip_x, tip_y = float(src[23, 0]), float(src[23, 1])
        tip_sc = float(src[23, 2]) if cols > 2 else 1.0
        half_w = _NOSE_HALF_W * eye_dist
        lift = _NOSE_SIDE_LIFT * eye_dist
        for i, (x, y) in (
            (14, (tip_x - half_w, tip_y - lift)),
            (15, (tip_x, tip_y)),
            (16, (tip_x + half_w, tip_y - lift)),
        ):
            out[i, 0] = x
            out[i, 1] = y
            if cols > 2:
                out[i, 2] = tip_sc

    if all(cols < 3 or float(src[i, 2]) >= 0.05 for i in (24, 25, 26, 27)):
        c_l, u_m, c_r, l_m = (src[i, :2].astype(np.float64) for i in (24, 25, 26, 27))
        sc = min(float(src[i, 2]) if cols > 2 else 1.0 for i in (24, 25, 26, 27))
        layout = {
            20: _lerp(c_l, u_m, 0.5),
            21: u_m,
            22: _lerp(u_m, c_r, 0.5),
            23: c_l,
            24: _lerp(c_l, l_m, 0.5),
            25: l_m,
            26: c_r,
            27: _lerp(c_r, l_m, 0.5),
        }
        for i, xy in layout.items():
            out[i, 0] = float(xy[0])
            out[i, 1] = float(xy[1])
            if cols > 2:
                out[i, 2] = sc
    return out


def _heatmaps_to_xy(heat: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Decode (K,H,W) heatmaps into image pixels. box is [x0,y0,x1,y1]."""
    k, hh, ww = heat.shape
    xs = heat.reshape(k, -1).argmax(axis=1)
    ys, xss = np.divmod(xs, ww)
    pts = np.zeros((k, 3), dtype=np.float32)
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    for i in range(k):
        x, y = int(xss[i]), int(ys[i])
        val = float(heat[i, y, x])
        if 0 < x < ww - 1:
            dx = float(heat[i, y, x + 1] - heat[i, y, x - 1])
            x = x + (0.25 if dx > 0 else -0.25)
        if 0 < y < hh - 1:
            dy = float(heat[i, y + 1, x if isinstance(x, int) else int(round(x))] - heat[i, y - 1, int(round(x)) if not isinstance(x, int) else x])
            y = y + (0.25 if dy > 0 else -0.25)
        pts[i, 0] = x0 + (x + 0.5) / ww * bw
        pts[i, 1] = y0 + (y + 0.5) / hh * bh
        pts[i, 2] = val
    return pts


def _box_side(box: np.ndarray) -> float:
    return min(max(float(box[2]) - float(box[0]), 1.0), max(float(box[3]) - float(box[1]), 1.0))


def _box_area(box: np.ndarray) -> float:
    return max(float(box[2]) - float(box[0]), 1.0) * max(float(box[3]) - float(box[1]), 1.0)


def pick_face_box(boxes: list[np.ndarray], w: int, h: int) -> np.ndarray:
    """Pick a character face, not the highest-confidence hair swirl.

    YOLO often gives a small bangs/hair box a higher score than the real
    face on green-screen busts. Prefer a box that is actually face-sized.
    """
    if not boxes:
        raise AnimeMeshError("No anime face found")
    min_side = MIN_FACE_FRAC * float(min(max(w, 1), max(h, 1)))
    sized = [box for box in boxes if _box_side(box) >= min_side]
    pool = sized or list(boxes)
    return max(pool, key=lambda box: _box_area(box) * max(float(box[4]) if len(box) > 4 else 1.0, 1e-6))


def tighten_head_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    """If YOLO returned head-and-shoulders, keep the upper face band.

    HRNet is trained on face crops. A bust-sized box makes it paint a
    stretched mesh from bangs down onto the chest.
    """
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    score = float(box[4]) if len(box) > 4 else 1.0
    img_h = float(max(h, 1))
    img_w = float(max(w, 1))
    if bh < HEAD_BOX_FRAC * img_h or y1 < HEAD_BOX_BOTTOM * img_h:
        return np.array([x0, y0, x1, y1, score], dtype=np.float32)
    cx = 0.5 * (x0 + x1)
    cy = y0 + FACE_BAND_CY * bh
    face_h = FACE_BAND * bh
    face_w = FACE_SIDE_KEEP * bw
    nx0 = min(max(cx - 0.5 * face_w, 0.0), img_w - 2.0)
    ny0 = min(max(cy - 0.5 * face_h, 0.0), img_h - 2.0)
    nx1 = min(max(cx + 0.5 * face_w, nx0 + 1.0), img_w - 1.0)
    ny1 = min(max(cy + 0.5 * face_h, ny0 + 1.0), img_h - 1.0)
    ny1 = min(img_h - 1.0, ny1 + CHIN_EXTRA * bh)
    return np.array([nx0, ny0, nx1, ny1, score], dtype=np.float32)


def expand_tiny_head_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    """Grow a bangs/hair-swirl hit down onto the drawn face.

    YOLO often returns a tight box on the highlight in the hair. HRNet then
    paints Label28 there. Keep the hairline, expand to a face-sized crop.
    """
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    score = float(box[4]) if len(box) > 4 else 1.0
    img_w = float(max(w, 1))
    img_h = float(max(h, 1))
    need = FIT_FACE_FRAC * min(img_w, img_h)
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    if min(bw, bh) >= need:
        return np.array([x0, y0, x1, y1, score], dtype=np.float32)
    cx = 0.5 * (x0 + x1)
    nx0 = min(max(cx - 0.5 * need, 0.0), img_w - 2.0)
    nx1 = min(max(cx + 0.5 * need, nx0 + 1.0), img_w - 1.0)
    ny0 = min(max(y0, 0.0), img_h - 2.0)
    ny1 = min(img_h - 1.0, ny0 + need * (1.0 + CHIN_EXTRA))
    if ny1 - ny0 < need:
        ny0 = max(0.0, ny1 - need * (1.0 + CHIN_EXTRA))
    return np.array([nx0, ny0, nx1, ny1, score], dtype=np.float32)


def rest_too_small(pts: np.ndarray | None, w: int, h: int) -> bool:
    """True when the rest jaw is a hair-swirl, not a character-sized face."""
    if pts is None:
        return True
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or len(arr) < 5:
        return True
    span = float(np.linalg.norm(arr[4, :2] - arr[0, :2]))
    if span <= 1.0:
        xs = arr[:5, 0]
        span = float(xs.max() - xs.min())
    return span < FIT_FACE_FRAC * float(min(max(w, 1), max(h, 1)))


def rest_shifted(old: np.ndarray | None, new: np.ndarray | None, w: int, h: int) -> bool:
    """True when the new rest face is not the same placement as the old one."""
    if old is None or new is None or len(old) < 28 or len(new) < 28:
        return old is None or new is None
    prev = np.asarray(old, dtype=np.float32)[:28, :2]
    nxt = np.asarray(new, dtype=np.float32)[:28, :2]
    span = max(float(np.hypot(float(w), float(h))), 1.0)
    dist = float(np.linalg.norm(nxt.mean(axis=0) - prev.mean(axis=0)))
    return dist > 0.08 * span


def drop_anime_chin(pts: np.ndarray) -> np.ndarray:
    """Slide a cropped-off chin down onto the round jaw. Eyes/mouth stay put.

    HRNet's contour-2 often sits on the lower lip when the crop clips the
    jaw. Anime chins sit about one nose→mouth below the slit.
    """
    src = np.asarray(pts, dtype=np.float32)
    if src.ndim != 2 or src.shape[0] < 28 or src.shape[1] < 2:
        return src
    out = src.copy()
    nose = out[15, :2].astype(np.float64)
    mouth_i = 25 if (src.shape[1] < 3 or float(src[25, 2]) >= 0.05) else 21
    mouth = out[mouth_i, :2].astype(np.float64)
    chin = out[2, :2].astype(np.float64)
    delta = mouth - nose
    span = float(np.hypot(delta[0], delta[1]))
    if span < 1e-3:
        return out
    down = delta / span
    if down[1] < 0.0:
        down = -down
    have = float(np.dot(chin - mouth, down))
    want = CHIN_MOUTH_FRAC * span
    if have >= want * 0.92:
        return out
    extra = want - have
    out[2, 0] = float(chin[0] + down[0] * extra)
    out[2, 1] = float(chin[1] + down[1] * extra)
    if out.shape[1] > 2:
        out[2, 2] = max(float(out[2, 2]), 0.5)
    for i, frac in ((1, 0.40), (3, 0.40)):
        out[i, 0] = float(out[i, 0] + down[0] * extra * frac)
        out[i, 1] = float(out[i, 1] + down[1] * extra * frac)
    return out


def _scale_box(box: np.ndarray, w: int, h: int, scale: float = BOX_SCALE) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    bw, bh = (x1 - x0) * scale, (y1 - y0) * scale
    side = max(bw, bh)
    nx0 = max(0.0, cx - 0.5 * side)
    ny0 = max(0.0, cy - 0.5 * side)
    nx1 = min(float(w - 1), cx + 0.5 * side)
    ny1 = min(float(h - 1), cy + 0.5 * side)
    return np.array([nx0, ny0, nx1, ny1, float(box[4]) if len(box) > 4 else 1.0], dtype=np.float32)


def _crop_face(bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(bgr.shape[1], max(x0 + 1, x1))
    y1 = min(bgr.shape[0], max(y0 + 1, y1))
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        raise AnimeMeshError("Empty face crop")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    ten = rgb.astype(np.float32) / 255.0
    ten = (ten - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(ten, (2, 0, 1))


def draw_label28(frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    if pts is None or len(pts) < 28:
        return vis
    h, w = vis.shape[:2]
    radius = max(1, int(round(min(h, w) * 0.0022)))
    font = max(0.45, min(h, w) / 1100.0)
    for idxs, color in CONNECTIONS:
        poly = []
        for i in idxs:
            if float(pts[i, 2]) < 0.05:
                continue
            poly.append([int(round(pts[i, 0])), int(round(pts[i, 1]))])
        if len(poly) >= 2:
            cv2.polylines(
                vis,
                [np.array(poly, np.int32)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
    for i in range(28):
        if float(pts[i, 2]) < 0.05:
            continue
        x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
        color = POINT_COLOR.get(i, (220, 220, 220))
        cv2.circle(vis, (x, y), radius + (1 if i == 15 else 0), color, -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            str(i),
            (x + radius + 1, y - radius),
            cv2.FONT_HERSHEY_PLAIN,
            font,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


class AnimeFaceMesh:
    def __init__(self) -> None:
        yolo_path = ANIME_DIR / YOLO_NAME
        hrnet_path = ANIME_DIR / HRNET_NAME
        if not yolo_path.is_file() or not hrnet_path.is_file():
            raise AnimeMeshError(
                f"Anime weights missing in {ANIME_DIR}. Need {YOLO_NAME} and {HRNET_NAME}."
            )
        self.device = _device()
        from ultralytics import YOLO

        self.yolo = YOLO(str(yolo_path))
        self.hrnet = HRNetV2(num_keypoints=28)
        raw = torch.load(hrnet_path, map_location="cpu", weights_only=False)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        mapped = {}
        for key, value in sd.items():
            if key.startswith("backbone."):
                mapped[key[len("backbone.") :]] = value
            elif key.startswith("keypoint_head."):
                mapped[key[len("keypoint_head.") :]] = value
        missing, unexpected = self.hrnet.load_state_dict(mapped, strict=False)
        if missing:
            raise AnimeMeshError(f"HRNet missing weights: {missing[:8]}")
        self.hrnet.to(self.device)
        self.hrnet.eval()

    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = bgr.shape[:2]
        device = 0 if self.device.startswith("cuda") else "cpu"
        results = self.yolo.predict(source=bgr, verbose=False, device=device, conf=FACE_SCORE)
        boxes: list[np.ndarray] = []
        if results and results[0].boxes is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            conf = results[0].boxes.conf.cpu().numpy()
            for box, score in zip(xyxy, conf):
                if float(score) >= FACE_SCORE:
                    boxes.append(np.append(box, float(score)))
        if not boxes:
            raise AnimeMeshError("No anime face found")
        raw = pick_face_box(boxes, w, h)
        best = tighten_head_box(raw, w, h)
        best = expand_tiny_head_box(best, w, h)
        scale = 1.08 if _box_area(best) < 0.92 * _box_area(raw) else BOX_SCALE
        crop_box = _scale_box(best, w, h, scale=scale)
        tensor = torch.from_numpy(_crop_face(bgr, crop_box)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            heat = self.hrnet(tensor)[0].float().cpu().numpy()
        native = _heatmaps_to_xy(heat, crop_box)
        return drop_anime_chin(hrnet_native_to_label28(native)), crop_box


def get_anime_mesh() -> AnimeFaceMesh:
    global _detector
    if _detector is None:
        _detector = AnimeFaceMesh()
    return _detector


def reset_anime_mesh() -> None:
    global _detector
    _detector = None


def fit_mesh(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return get_anime_mesh().detect(bgr)
