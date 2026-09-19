"""Fit KEYPOINT_SCHEMA (37,4) onto a reference image via the trained models.

Runs the same stack as pose-traker labeling, directly on the still:
  - anime face YOLO+HRNet → native 28 → ``hrnet_native_to_label28`` → slots 0..27
  - iris_pose.pt           → slots 28..29
  - dwpose_v2.pt           → slots 30..36 (chest derived from shoulders)

Native HRNet uses 6-pt eyes + 1 nose + 4-pt mouth; the adapter expands that
into the project schema (3-pt eyes, 3-pt nose, 8-pt mouth). Collapse repair
remains as a safety net after mapping. Iris/body stay model-only.
Coords are then remapped through the same crop/pad path as
``encode_reference`` so they land in model crop space [-1,1].
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import (
    anime_face_detector_src,
    ensure_import_paths,
    live_poser_dir,
    pose_traker_dir,
    torch_train_dir,
    trackers_dir,
)

ensure_import_paths()
ANIME_DET_SRC = anime_face_detector_src()
TORCH_TRAIN = torch_train_dir()
POSE_TRAKER = pose_traker_dir()
TRACKERS = trackers_dir()
from face_landmark_repair import (  # noqa: E402
    repair_collapsed_face_landmarks as _shared_repair_collapsed_face_landmarks,
)
from hrnet_schema_adapter import hrnet_native_to_label28  # noqa: E402


def repair_collapsed_face_landmarks(
    kps: np.ndarray,
    *,
    repair_nose: bool = True,
    repair_mouth: bool = True,
) -> np.ndarray:
    """Fix nose/mouth slots that collapsed onto eyes (pixel or crop [-1,1]).

    Safe to call on sidecar ``*_keypoints.npy`` and on freshly fitted refs.
    Pass ``repair_mouth=False`` to leave lips as the detector produced them.
    """
    return _shared_repair_collapsed_face_landmarks(
        kps,
        log_prefix="[ref-fit]",
        repair_nose=repair_nose,
        repair_mouth=repair_mouth,
    )


IRIS_MODEL_CANDIDATES = (
    TRACKERS / "iris_pose.pt",
    POSE_TRAKER / "models" / "iris_pose.pt",
    POSE_TRAKER / "iris-model" / "models" / "iris_pose.pt",
    live_poser_dir() / "models" / "iris_pose.pt",
)
SKELETON_MODEL_CANDIDATES = (
    TRACKERS / "dwpose_v2.pt",
    TRACKERS / "dwpose_v2.onnx",
    POSE_TRAKER / "iris-model" / "models" / "dwpose_v2.pt",
    live_poser_dir() / "models" / "dwpose_v2.pt",
    live_poser_dir() / "models" / "dwpose_v2.onnx",
    POSE_TRAKER / "iris-model" / "models" / "dwpose_v2.onnx",
)

NUM_KEYPOINTS = 37
KEYPOINT_DIM = 4
IMAGE_SIZE = 768

# Match pose-traker/run_detect.py thresholds.
FACE_SCORE_THRESHOLD = 0.5
IRIS_DETECT_CONF = 0.25
IRIS_PUPIL_VIS_THR = 0.3
SKELETON_DETECT_CONF = 0.05
YOLO_KPT_SCORE_THRESHOLD = 0.01
EDGE_SLACK_PX = 24.0

# Eye lids for iris matching — same as pose-traker/run_detect.py:
# image-left eye (viewer's left / person's right) → right_iris, and vice versa.
RIGHT_EYE_LANDMARK_IDS = (11, 12, 13)  # viewer's left / person's right
LEFT_EYE_LANDMARK_IDS = (17, 18, 19)  # viewer's right / person's left

BODY_KPT_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "left_shoulder",
    "left_elbow",
)

_detector = None
_detector_lock = threading.Lock()
_detector_device: str | None = None

_iris_model = None
_iris_lock = threading.Lock()
_iris_device: str | None = None
_iris_path: Path | None = None

_skel_model = None
_skel_lock = threading.Lock()
_skel_device: str | None = None
_skel_path: Path | None = None


class RefPoseFitError(RuntimeError):
    """Could not detect a usable face on the reference image."""


def _ensure_paths() -> None:
    if ANIME_DET_SRC.is_dir() and str(ANIME_DET_SRC) not in sys.path:
        sys.path.insert(0, str(ANIME_DET_SRC))
    if TORCH_TRAIN.is_dir() and str(TORCH_TRAIN) not in sys.path:
        sys.path.insert(0, str(TORCH_TRAIN))


def _pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _yolo_device(dev: str):
    """Ultralytics accepts 'cpu' or an int GPU index."""
    if dev.startswith("cuda"):
        parts = dev.split(":")
        return int(parts[1]) if len(parts) > 1 else 0
    return "cpu"


def _first_existing(candidates: tuple[Path, ...]) -> Path | None:
    for p in candidates:
        if p.is_file() and p.stat().st_size > 1000:
            return p
    return None


def get_anime_face_detector(device: str | None = None):
    """Lazy singleton LandmarkDetector (YOLOv8 face + HRNet 28 pts)."""
    global _detector, _detector_device
    dev = _pick_device(device)
    with _detector_lock:
        if _detector is not None and _detector_device == dev:
            return _detector
        _ensure_paths()
        from anime_face_detector import create_detector

        print(f"[ref-fit] Loading anime face detector on {dev} …")
        _detector = create_detector(device=dev)
        _detector_device = dev
        print("[ref-fit] Anime face detector ready.")
        return _detector


def get_iris_model(device: str | None = None):
    """Lazy singleton iris_pose.pt (YOLO-pose)."""
    global _iris_model, _iris_device, _iris_path
    dev = _pick_device(device)
    with _iris_lock:
        if _iris_model is not None and _iris_device == dev:
            return _iris_model
        path = _first_existing(IRIS_MODEL_CANDIDATES)
        if path is None:
            print("[ref-fit] iris_pose.pt not found — iris slots stay empty")
            _iris_model = None
            _iris_device = dev
            _iris_path = None
            return None
        from ultralytics import YOLO

        print(f"[ref-fit] Loading iris model: {path}")
        _iris_model = YOLO(str(path))
        _iris_device = dev
        _iris_path = path
        return _iris_model


def get_skeleton_model(device: str | None = None):
    """Lazy singleton dwpose_v2 (fine-tuned YOLO-pose)."""
    global _skel_model, _skel_device, _skel_path
    dev = _pick_device(device)
    with _skel_lock:
        if _skel_model is not None and _skel_device == dev:
            return _skel_model
        path = _first_existing(SKELETON_MODEL_CANDIDATES)
        if path is None:
            print("[ref-fit] dwpose_v2 not found — body slots stay empty")
            _skel_model = None
            _skel_device = dev
            _skel_path = None
            return None
        if path.suffix.lower() == ".onnx":
            # Prefer .pt; onnx via ultralytics works but pt is the trained export.
            pt = path.with_suffix(".pt")
            if pt.is_file():
                path = pt
        from ultralytics import YOLO

        print(f"[ref-fit] Loading skeleton model: {path}")
        _skel_model = YOLO(str(path))
        _skel_device = dev
        _skel_path = path
        return _skel_model


def _empty37() -> np.ndarray:
    return np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)


def _set(out: np.ndarray, i: int, x: float, y: float, score: float = 1.0) -> None:
    out[i, 0] = float(x)
    out[i, 1] = float(y)
    out[i, 2] = float(score)
    out[i, 3] = 1.0


def _mean_xy(pts: np.ndarray, idxs: tuple[int, ...]) -> tuple[float, float] | None:
    xs, ys = [], []
    for i in idxs:
        if i < 0 or i >= len(pts):
            continue
        if pts.shape[1] > 2 and float(pts[i, 2]) < 0.05:
            continue
        if pts.shape[1] > 3 and float(pts[i, 3]) < 0.5:
            continue
        xs.append(float(pts[i, 0]))
        ys.append(float(pts[i, 1]))
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _clamp_joint_xy(
    x: float, y: float, w: int, h: int, *, slack: float = EDGE_SLACK_PX
) -> tuple[float, float, bool]:
    near = (-slack <= x <= w - 1 + slack) and (-slack <= y <= h - 1 + slack)
    if not near:
        return x, y, False
    return (
        min(max(float(x), 1.0), float(w - 1)),
        min(max(float(y), 1.0), float(h - 1)),
        True,
    )


def _fill_face_from_detector(
    out: np.ndarray,
    image_bgr: np.ndarray,
    *,
    device: str | None = None,
    repair: bool = True,
) -> dict:
    detector = get_anime_face_detector(device=device)
    results = detector(image_bgr)
    if not results:
        raise RefPoseFitError("No anime face detected on reference image")

    def _score(r: dict) -> float:
        box = r.get("bbox")
        if box is None or len(box) < 5:
            return 0.0
        return float(box[4])

    best = max(results, key=_score)
    box_score = _score(best)
    if box_score < FACE_SCORE_THRESHOLD:
        raise RefPoseFitError(
            f"Best face score {box_score:.3f} < {FACE_SCORE_THRESHOLD}"
        )
    pts = np.asarray(best["keypoints"], dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 28:
        raise RefPoseFitError(f"Bad landmark shape: {pts.shape}")

    # HRNet native layout ≠ KEYPOINT_SCHEMA — map before fill / repair / iris.
    pts = hrnet_native_to_label28(pts)

    for i in range(28):
        x, y = float(pts[i, 0]), float(pts[i, 1])
        sc = float(pts[i, 2]) if pts.shape[1] > 2 else 1.0
        if sc < 0.05:
            continue
        _set(out, i, x, y, max(sc, 0.5))

    # Safety net: rebuild nose/mouth only if still collapsed after mapping.
    # TEST_MODE / raw runs skip this so you can see mapped detector output.
    if repair:
        out[:] = repair_collapsed_face_landmarks(out)
    return best


def _fill_iris_from_model(
    out: np.ndarray,
    image_bgr: np.ndarray,
    *,
    device: str | None = None,
) -> int:
    """Write slots 28/29 from iris_pose.pt matched to face eye anchors."""
    model = get_iris_model(device=device)
    if model is None:
        return 0
    dev = _yolo_device(_pick_device(device))
    results = model.predict(
        source=image_bgr, conf=IRIS_DETECT_CONF, verbose=False, device=dev
    )
    if not results:
        return 0
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return 0

    boxes = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    kpts = None
    if r0.keypoints is not None and r0.keypoints.data is not None:
        kpts = r0.keypoints.data.cpu().numpy()

    eyes: list[dict] = []
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = (float(v) for v in box)
        pupil = None
        visible = False
        score = float(confs[i])
        if kpts is not None and i < len(kpts):
            kp = kpts[i][0]
            px, py, pv = float(kp[0]), float(kp[1]), float(kp[2])
            if pv >= IRIS_PUPIL_VIS_THR:
                pupil = (px, py)
                visible = True
                score = max(score, float(pv))
        eyes.append(
            {
                "cx": 0.5 * (x0 + x1),
                "cy": 0.5 * (y0 + y1),
                "pupil": pupil,
                "visible": visible,
                "score": score,
            }
        )

    filled = 0
    used: set[int] = set()
    for slot, eye_idxs in ((28, RIGHT_EYE_LANDMARK_IDS), (29, LEFT_EYE_LANDMARK_IDS)):
        anchor = _mean_xy(out, eye_idxs)
        if anchor is None:
            continue
        ax, ay = anchor
        # Match radius from eye width when available.
        xs = [float(out[i, 0]) for i in eye_idxs if float(out[i, 3]) >= 0.5]
        max_dist = max(40.0, (max(xs) - min(xs)) * 1.5) if len(xs) >= 2 else 60.0
        best_i = None
        best_d = 1e9
        for i, det in enumerate(eyes):
            if i in used:
                continue
            d = float(np.hypot(det["cx"] - ax, det["cy"] - ay))
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is None or best_d > max_dist:
            continue
        used.add(best_i)
        det = eyes[best_i]
        if not det["visible"] or det["pupil"] is None:
            continue
        _set(out, slot, det["pupil"][0], det["pupil"][1], det["score"])
        filled += 1
    return filled


def _fill_body_from_dwpose_v2(
    out: np.ndarray,
    image_bgr: np.ndarray,
    *,
    device: str | None = None,
) -> int:
    """Write slots 30..36 from fine-tuned dwpose_v2.pt (chest derived)."""
    model = get_skeleton_model(device=device)
    if model is None:
        return 0
    h, w = image_bgr.shape[:2]
    dev = _yolo_device(_pick_device(device))
    results = model.predict(
        source=image_bgr, conf=SKELETON_DETECT_CONF, verbose=False, device=dev
    )
    if not results:
        return 0
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return 0
    if r0.keypoints is None or r0.keypoints.data is None:
        return 0

    confs = r0.boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    kpts = r0.keypoints.data.cpu().numpy()
    if best_i >= len(kpts):
        return 0
    kp = kpts[best_i]

    joints: dict[str, dict] = {}
    filled = 0
    for ki, name in enumerate(BODY_KPT_NAMES):
        if ki >= len(kp):
            break
        px, py, pv = float(kp[ki][0]), float(kp[ki][1]), float(kp[ki][2])
        if pv <= 1.0:
            score = float(pv)
            score_ok = score >= YOLO_KPT_SCORE_THRESHOLD
        else:
            score = float(min(1.0, pv / 2.0))
            score_ok = pv >= 1.0
        px, py, in_frame = _clamp_joint_xy(px, py, w, h)
        visible = bool(score_ok and in_frame)
        joints[name] = {"x": px, "y": py, "score": score, "visible": visible}
        slot = 30 + ki
        if visible:
            _set(out, slot, px, py, score)
            filled += 1

    # Chest (slot 36): midpoint of shoulders + downward offset — match pose-traker.
    ls = joints.get("left_shoulder")
    rs = joints.get("right_shoulder")
    if ls and rs and ls["visible"] and rs["visible"]:
        sw = abs(ls["x"] - rs["x"])
        cx = 0.5 * (ls["x"] + rs["x"])
        cy = 0.5 * (ls["y"] + rs["y"]) + sw * 0.28
        if h > 0 and cy > float(h - 1) and 1.0 <= cx <= float(w - 1):
            cy = float(h - 1)
        cx, cy, in_frame = _clamp_joint_xy(cx, cy, w, h)
        if in_frame:
            score = min(float(ls["score"]), float(rs["score"]))
            _set(out, 36, cx, cy, score)
            filled += 1
    return filled


def _flip_keypoints37_pixels(kps: np.ndarray, width: int) -> np.ndarray:
    """Mirror (37,4) pixel keypoints about the image vertical axis + swap L/R."""
    out = np.asarray(kps, dtype=np.float32).copy()
    w = max(int(width), 1)
    out[:, 0] = float(w - 1) - out[:, 0]
    pairs = (
        (0, 4),
        (1, 3),
        (5, 10),
        (6, 9),
        (7, 8),
        (11, 19),
        (12, 18),
        (13, 17),
        (14, 16),
        (20, 22),
        (23, 26),
        (24, 27),
        (28, 29),
        (32, 34),
        (33, 35),
    )
    for a, b in pairs:
        out[[a, b]] = out[[b, a]]
    return out


def _merge_keypoints37(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Average visible points from two detections; prefer higher score ties."""
    out = _empty37()
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    for i in range(NUM_KEYPOINTS):
        a_ok = float(aa[i, 3]) >= 0.5 and float(aa[i, 2]) >= 0.05
        b_ok = float(bb[i, 3]) >= 0.5 and float(bb[i, 2]) >= 0.05
        if a_ok and b_ok:
            # Weight by score so a confident pass dominates a weak one.
            wa = max(float(aa[i, 2]), 0.05)
            wb = max(float(bb[i, 2]), 0.05)
            s = wa + wb
            out[i, 0] = (wa * float(aa[i, 0]) + wb * float(bb[i, 0])) / s
            out[i, 1] = (wa * float(aa[i, 1]) + wb * float(bb[i, 1])) / s
            out[i, 2] = max(wa, wb)
            out[i, 3] = 1.0
        elif a_ok:
            out[i] = aa[i]
        elif b_ok:
            out[i] = bb[i]
    return out


def _refine_eye_slots(kps: np.ndarray) -> np.ndarray:
    """Snap eye mid (12/18) onto upper-lid mid and level inner/outer corners.

    Anime HRNet often parks slot 12/18 too high above the crease, or drops
    the inner corner (13/17) below the outer. Overlay treats 12 as the upper
    apex of the aperture, so a wild mid point reads as a broken eye diamond.
    """
    out = np.asarray(kps, dtype=np.float32).copy()
    # (outer, mid, inner) — person left then person right.
    for outer_i, mid_i, inner_i in ((11, 12, 13), (19, 18, 17)):
        if not (
            float(out[outer_i, 3]) >= 0.5
            and float(out[mid_i, 3]) >= 0.5
            and float(out[inner_i, 3]) >= 0.5
        ):
            continue
        ox, oy = float(out[outer_i, 0]), float(out[outer_i, 1])
        ix, iy = float(out[inner_i, 0]), float(out[inner_i, 1])
        mx, my = float(out[mid_i, 0]), float(out[mid_i, 1])
        chord = float(np.hypot(ix - ox, iy - oy))
        if chord < 1e-6:
            continue
        # Level corners toward their mean y (keep most of detector x).
        mid_corner_y = 0.5 * (oy + iy)
        out[outer_i, 1] = 0.65 * oy + 0.35 * mid_corner_y
        out[inner_i, 1] = 0.65 * iy + 0.35 * mid_corner_y
        oy, iy = float(out[outer_i, 1]), float(out[inner_i, 1])
        # Target upper-lid mid: chord midpoint, slightly above (image -y = up).
        cx = 0.5 * (ox + ix)
        cy = 0.5 * (oy + iy)
        # Perpendicular toward image-up from the chord.
        nx = -(iy - oy) / chord
        ny = (ix - ox) / chord
        if ny > 0:  # want upward (negative y)
            nx, ny = -nx, -ny
        lift = 0.18 * chord
        tx = cx + nx * lift
        ty = cy + ny * lift
        # Blend detector mid with geometric upper-lid target.
        out[mid_i, 0] = 0.45 * mx + 0.55 * tx
        out[mid_i, 1] = 0.45 * my + 0.55 * ty
    return out


def detect_face37_pixels(
    image_rgb: np.ndarray,
    *,
    device: str | None = None,
    flip_tta: bool = False,
    test_mode: bool = False,
) -> np.ndarray:
    """Run face + iris + dwpose_v2 on RGB still → (37,4) absolute pixels.

    When ``flip_tta`` is True: detect → horizontal flip → detect again →
    flip landmarks back → merge. Helps stabilize asymmetric eye/mouth
    failures on anime stills (manual Calibrate button).

    When ``test_mode`` is True: schema-mapped detector output only — no
    mouth/nose geometric repair, no eye refine, no flip-TTA. Native HRNet
    indices are always converted
    to KEYPOINT_SCHEMA before return.
    """
    import cv2

    arr = np.asarray(image_rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise RefPoseFitError(f"Expected HxWx3 RGB, got {arr.shape}")

    if test_mode:
        flip_tta = False

    def _once(rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        out = _empty37()
        _fill_face_from_detector(
            out, bgr, device=device, repair=not test_mode
        )
        n_iris = _fill_iris_from_model(out, bgr, device=device)
        n_body = _fill_body_from_dwpose_v2(out, bgr, device=device)
        if not test_mode:
            out = _refine_eye_slots(out)
            # Keep body nose coincident with face nose tip for consistent overlays.
            out = _align_body_nose_to_face(out)
        vis_face = int(np.sum(out[:28, 3] >= 0.5))
        if vis_face < 10:
            raise RefPoseFitError(f"Too few visible face landmarks ({vis_face}/28)")
        tag = "mapped test_mode" if test_mode else "models"
        print(
            f"[ref-fit] {tag} → face={vis_face}/28  iris={n_iris}/2  "
            f"body={n_body}/7"
            + (
                " (schema-mapped; no repair / no eye refine)"
                if test_mode
                else " (schema-mapped; nose/mouth repaired if collapsed)"
            )
        )
        return out

    primary = _once(arr)
    if not flip_tta:
        return primary

    print("[ref-fit] flip-calibrate: detecting on mirrored image …")
    mirrored = np.ascontiguousarray(arr[:, ::-1, :])
    flipped_det = _once(mirrored)
    flipped_back = _flip_keypoints37_pixels(flipped_det, arr.shape[1])
    merged = _merge_keypoints37(primary, flipped_back)
    merged = _refine_eye_slots(merged)
    # Re-run collapse repair after merge (averaging can re-break topology).
    merged = repair_collapsed_face_landmarks(merged)
    merged = _align_body_nose_to_face(merged)
    vis = int(np.sum(merged[:28, 3] >= 0.5))
    print(f"[ref-fit] flip-calibrate merge complete ({vis}/28 face visible)")
    return merged


def _crop_params_for_ref(
    image_rgb: np.ndarray,
    *,
    skip_crop: bool,
    image_size: int = IMAGE_SIZE,
) -> tuple[float, float, float, float]:
    """Match encode_reference crop window (x0,y0,w,h) in source pixels."""
    _ensure_paths()
    from inference_keypoint import _chroma_crop_to_square, _pad_rgb_to_square

    arr = np.asarray(image_rgb)
    if skip_crop:
        _canvas, pad_x, pad_y, side = _pad_rgb_to_square(arr)
        return float(-pad_x), float(-pad_y), float(side), float(side)
    _crop, x0, y0, w, h = _chroma_crop_to_square(arr, image_size=image_size)
    return float(x0), float(y0), float(w), float(h)


def reference_display_rgb(
    image_rgb: np.ndarray,
    *,
    skip_crop: bool,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Return the same HxWx3 RGB canvas ``encode_reference`` feeds the VAE.

    Crop-normalized keypoints in [-1, 1] overlay correctly on this image.
    """
    _ensure_paths()
    from inference_keypoint import _chroma_crop_to_square, _pad_rgb_to_square

    arr = np.asarray(image_rgb)
    if arr.dtype != np.uint8:
        if float(np.max(arr)) <= 1.5:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if skip_crop:
        canvas, _pad_x, _pad_y, _side = _pad_rgb_to_square(arr)
        img = Image.fromarray(canvas, mode="RGB").resize(
            (int(image_size), int(image_size)), resample=Image.Resampling.BILINEAR
        )
        return np.asarray(img, dtype=np.uint8)

    crop, _x0, _y0, _w, _h = _chroma_crop_to_square(arr, image_size=int(image_size))
    return np.asarray(crop, dtype=np.uint8)


def _align_body_nose_to_face(kps: np.ndarray) -> np.ndarray:
    """Snap body nose (30) onto face nose tip (15) when both are visible."""
    out = np.asarray(kps, dtype=np.float32).copy()
    if out.shape[0] < 31:
        return out
    if float(out[15, 3]) >= 0.5 and float(out[30, 3]) >= 0.5:
        out[30, 0] = float(out[15, 0])
        out[30, 1] = float(out[15, 1])
        out[30, 2] = max(float(out[30, 2]), float(out[15, 2]))
        out[30, 3] = 1.0
    return out


def fit_keypoints_to_reference(
    image_path: Path | str | None = None,
    image_rgb: np.ndarray | None = None,
    *,
    skip_crop: bool | None = None,
    image_size: int = IMAGE_SIZE,
    device: str | None = None,
    flip_tta: bool = False,
) -> np.ndarray:
    """Detect on ref with trained models → (37,4) in crop space [-1, 1].

    ``skip_crop`` should match ``StreamEngine.set_reference`` (greenscreen /
    train_crop → chroma side crop; otherwise pad-only).

    ``flip_tta``: flip image → detect → flip landmarks back → merge with the
    upright pass (manual Calibrate button). Does not touch the camera path.
    """
    _ensure_paths()
    from utils.keypoints import transform_keypoints_crop

    if image_rgb is None:
        if image_path is None:
            raise ValueError("Provide image_path or image_rgb")
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    else:
        image_rgb = np.asarray(image_rgb)
        if image_rgb.dtype != np.uint8:
            if float(np.max(image_rgb)) <= 1.5:
                image_rgb = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
            else:
                image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    if skip_crop is None:
        path_s = str(image_path or "").replace("\\", "/")
        in_train = "train_crop" in path_s
        arr = image_rgb
        step = max(1, min(arr.shape[0], arr.shape[1]) // 64)
        patch = arr[::step, ::step, :3].astype(np.float32)
        g, r, b = patch[..., 1], patch[..., 0], patch[..., 2]
        greenish = (g > 40.0) & (g > r + 15.0) & (g > b + 15.0)
        is_green = float(np.mean(greenish)) >= 0.25
        skip_crop = not (in_train or is_green)

    print(
        f"[ref-fit] Running face+iris+dwpose_v2 on reference "
        f"({image_rgb.shape[1]}x{image_rgb.shape[0]}, skip_crop={skip_crop}, "
        f"flip_tta={flip_tta}) …"
    )
    kps_px = detect_face37_pixels(image_rgb, device=device, flip_tta=flip_tta)
    x0, y0, cw, ch = _crop_params_for_ref(
        image_rgb, skip_crop=bool(skip_crop), image_size=image_size
    )
    kps = transform_keypoints_crop(
        kps_px,
        crop_x0=x0,
        crop_y0=y0,
        crop_w=cw,
        crop_h=ch,
        out_w=float(image_size),
        out_h=float(image_size),
        normalize=True,
    )
    kps = _align_body_nose_to_face(kps)
    vis = int(np.sum(kps[:, 3] >= 0.5))
    print(
        f"[ref-fit] Fitted {vis}/37 keypoints into crop space "
        f"(crop=[{x0:.0f},{y0:.0f},{cw:.0f}x{ch:.0f}] → {image_size})"
    )
    return kps.astype(np.float32)
