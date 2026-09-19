"""Turn Track Lab live state into harness packets."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .points import row_meta
from .protocol import (
    BODY_END,
    BODY_START,
    COMMANDS,
    FACE_COUNT,
    FEEL_KEYS,
    LEFT_EYE_SLOTS,
    NUM_KEYPOINTS,
    PROTOCOL,
    RIGHT_EYE_SLOTS,
    RIGHT_IRIS,
    LEFT_IRIS,
    SCHEMA,
)

_STATUS_KEEP = (
    "ok",
    "ready",
    "loaded",
    "live",
    "has_source",
    "source_path",
    "width",
    "height",
    "tracker",
    "faces",
    "ms",
    "generation",
    "error",
    "message",
    "source",
    "camera_index",
    "cameras",
    "feel",
    "calib",
    "ifm",
    "mirror",
    "presets",
    "active",
    "mouth_slots",
    "mouth_points",
    "eye_points",
    "points",
    "shapes",
    "weights",
    "head",
    "blink",
    "hair",
    "skeleton",
    "iris",
    "iris_method",
    "iris_cam",
    "look",
    "hair_method",
    "point_offsets",
    "gen",
    "gen_ms",
)


def _num(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rows_from_points(points: object) -> np.ndarray:
    out = np.zeros((FACE_COUNT, 3), dtype=np.float32)
    if not isinstance(points, (list, tuple, np.ndarray)) or len(points) < 1:
        return out
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return out
    n = min(FACE_COUNT, int(arr.shape[0]))
    out[:n, 0] = arr[:n, 0]
    out[:n, 1] = arr[:n, 1]
    if arr.shape[1] > 2:
        out[:n, 2] = arr[:n, 2]
    else:
        out[:n, 2] = 1.0
    return out


def _eye_mid(k: np.ndarray, slots: tuple[int, ...]) -> np.ndarray | None:
    pts = []
    for slot in slots:
        if k[slot, 3] < 0.5:
            continue
        pts.append(k[slot, :2])
    if not pts:
        return None
    mid = np.mean(np.stack(pts, axis=0), axis=0)
    score = float(np.mean(k[list(slots), 2]))
    return np.array([float(mid[0]), float(mid[1]), score, 1.0], dtype=np.float32)


def pack_keypoints(points: object, skeleton: object, iris: object = None) -> np.ndarray:
    """Label28 + followed skeleton + iris → (37, 4) KEYPOINT_SCHEMA."""
    k = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)
    face = _rows_from_points(points)
    k[:FACE_COUNT, 0:2] = face[:, 0:2]
    k[:FACE_COUNT, 2] = face[:, 2]
    k[:FACE_COUNT, 3] = (face[:, 2] >= 0.05).astype(np.float32)
    filled = {RIGHT_IRIS: False, LEFT_IRIS: False}
    if isinstance(iris, list):
        for row in iris:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("id", -1))
            except (TypeError, ValueError):
                continue
            if idx not in filled:
                continue
            if not row.get("visible", True):
                continue
            k[idx, 0] = _num(row.get("x"))
            k[idx, 1] = _num(row.get("y"))
            k[idx, 2] = _num(row.get("score"), 1.0)
            k[idx, 3] = 1.0 if k[idx, 2] >= 0.05 else 0.0
            filled[idx] = k[idx, 3] >= 0.5
    if not filled[RIGHT_IRIS]:
        right = _eye_mid(k, LEFT_EYE_SLOTS)
        if right is not None:
            k[RIGHT_IRIS] = right
    if not filled[LEFT_IRIS]:
        left = _eye_mid(k, RIGHT_EYE_SLOTS)
        if left is not None:
            k[LEFT_IRIS] = left
    if isinstance(skeleton, list):
        for joint in skeleton:
            if not isinstance(joint, dict):
                continue
            try:
                idx = int(joint.get("id", -1))
            except (TypeError, ValueError):
                continue
            if idx < BODY_START or idx > BODY_END:
                continue
            k[idx, 0] = _num(joint.get("x"))
            k[idx, 1] = _num(joint.get("y"))
            k[idx, 2] = _num(joint.get("score"), 1.0)
            k[idx, 3] = 1.0 if k[idx, 2] >= 0.05 else 0.0
    return k


def _keypoint_rows(k: np.ndarray, *, ndigits: int = 2) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(NUM_KEYPOINTS):
        meta = row_meta(i)
        rows.append(
            {
                "i": i,
                "ref": meta["ref"],
                "name": meta["name"],
                "legacy": meta["legacy"],
                "x": round(float(k[i, 0]), ndigits),
                "y": round(float(k[i, 1]), ndigits),
                "score": round(float(k[i, 2]), 3),
                "visible": bool(k[i, 3] >= 0.5),
            }
        )
    return rows


_FEEL_FALLBACK = {
    "response": 0.65,
    "smoothing": 0.48,
    "mouth": 0.50,
    "use_visemes": 1.0,
    "show_face": 1.0,
    "show_skeleton": 1.0,
    "show_hair": 1.0,
    "show_ids": 0.0,
    "hair_pin": 0.7,
    "max_yaw": 1.0,
    "max_roll": 1.0,
    "max_pitch_up": 1.0,
    "max_pitch_down": 1.0,
    "max_look_x": 1.0,
    "max_look_y": 1.0,
    "gaze_gain": 1.0,
    "gaze_smooth": 0.28,
}


def _feel(live: dict[str, Any]) -> dict[str, float]:
    raw = live.get("feel") if isinstance(live.get("feel"), dict) else {}
    out: dict[str, float] = {}
    for key in FEEL_KEYS:
        out[key] = round(_num(raw.get(key), _FEEL_FALLBACK[key]), 3)
    return out


def pack_frame(
    live: dict[str, Any],
    *,
    image_wh: tuple[int, int] = (0, 0),
    generation: int = 0,
    t: float | None = None,
    clients: int = 0,
) -> dict[str, Any]:
    """One live overlay / drive packet in character pixel space."""
    points = live.get("points") or live.get("live_points") or []
    skeleton = live.get("skeleton") or []
    hair = live.get("hair") or []
    iris = live.get("iris") or []
    k = pack_keypoints(points, skeleton, iris)
    width, height = int(image_wh[0]), int(image_wh[1])
    tracker = str(live.get("tracker") or "")
    live_on = bool(live.get("live"))
    body_on = any(float(k[i, 3]) >= 0.5 for i in range(BODY_START, BODY_END + 1))
    hair_on = isinstance(hair, list) and len(hair) > 0
    iris_on = float(k[RIGHT_IRIS, 3]) >= 0.5 or float(k[LEFT_IRIS, 3]) >= 0.5
    iris_method = str(live.get("iris_method") or "")
    if not iris_method:
        iris_method = "iris_pose" if iris else ("eye_mid" if iris_on else "none")
    hair_method = str(live.get("hair_method") or "")
    if not hair_method:
        hair_method = "lab_follow" if hair_on else "none"
    return {
        "type": "frame",
        "protocol": PROTOCOL,
        "schema": SCHEMA,
        "t": float(time.time() if t is None else t),
        "generation": int(generation),
        "live": live_on,
        "tracker": tracker,
        "faces": int(live.get("faces") or 0),
        "ms": round(_num(live.get("ms")), 2),
        "error": str(live.get("error") or ""),
        "source": str(live.get("source") or "camera"),
        "image_wh": [width, height],
        "coord_space": "character_px",
        "shape": [NUM_KEYPOINTS, 4],
        "keypoints": _keypoint_rows(k),
        "points": points if isinstance(points, list) else [],
        "skeleton": skeleton if isinstance(skeleton, list) else [],
        "hair": hair if isinstance(hair, list) else [],
        "iris": iris if isinstance(iris, list) else [],
        "iris_cam": live.get("iris_cam") if isinstance(live.get("iris_cam"), list) else [],
        "look": live.get("look") if isinstance(live.get("look"), dict) else None,
        "mouth_box": live.get("mouth_box") or [],
        "mouth_cage": live.get("mouth_cage") or [],
        "weights": live.get("weights") or {},
        "head": live.get("head") or {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "blink": live.get("blink") or {"l": 0.0, "r": 0.0},
        "calib": live.get("calib") or {},
        "feel": _feel(live),
        "iris_method": iris_method,
        "point_offsets": live.get("point_offsets") if isinstance(live.get("point_offsets"), list) else [],
        "skeleton_method": "lab_follow" if body_on else "none",
        "hair_method": hair_method,
        "clients": int(clients),
        "loaded": bool(live.get("loaded", True)),
        "meta": {
            "body_lost": not body_on,
            "hair_lost": not hair_on,
            "coord_space": "character_px",
        },
    }


def pack_status(
    payload: dict[str, Any],
    *,
    clients: int = 0,
    rest: object = None,
) -> dict[str, Any]:
    """Control-surface snapshot: settings, cameras, calibration, advertised ops."""
    out: dict[str, Any] = {
        "type": "status",
        "protocol": PROTOCOL,
        "commands": sorted(COMMANDS),
        "clients": int(clients),
        "feel": _feel(payload),
        "loaded": bool(payload["loaded"]) if "loaded" in payload else True,
        "ok": True,
        "live": False,
    }
    for key in _STATUS_KEEP:
        if key == "feel":
            continue
        if key in payload:
            out[key] = payload[key]
    if rest is not None:
        out["rest"] = rest
    elif isinstance(payload.get("rest"), list):
        out["rest"] = payload["rest"]
    return out


def warming_status(*, clients: int = 0) -> dict[str, Any]:
    """Host is up; torch worker has not attached yet. Desk can handshake."""
    return pack_status(
        {
            "ok": True,
            "ready": False,
            "loaded": False,
            "live": False,
            "has_source": False,
            "error": "",
            "message": "Loading tracker…",
            "source": "camera",
            "camera_index": 0,
            "cameras": [],
        },
        clients=clients,
    )


def warming_frame(*, clients: int = 0) -> dict[str, Any]:
    """Dummy KEYPOINT_SCHEMA so overlay/meters have a rest-shaped packet."""
    return pack_frame(
        {
            "live": False,
            "loaded": False,
            "error": "",
            "source": "camera",
            "tracker": "",
            "faces": 0,
            "message": "Loading tracker…",
        },
        clients=clients,
    )


def _image_wh(bench: Any) -> tuple[int, int]:
    src = getattr(bench, "source_bgr", None)
    shape = getattr(src, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    return 0, 0


def _rest_points(bench: Any) -> list[list[float]]:
    pts = getattr(bench, "rest_pts", None)
    if pts is None:
        return []
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < FACE_COUNT or arr.shape[1] < 2:
        return []
    score = arr[:, 2] if arr.shape[1] > 2 else np.ones((arr.shape[0],), dtype=np.float32)
    return [
        [round(float(arr[i, 0]), 3), round(float(arr[i, 1]), 3), round(float(score[i]), 4)]
        for i in range(FACE_COUNT)
    ]


def frame_from_bench(bench: Any, *, clients: int = 0) -> dict[str, Any]:
    live = bench.live_status() if hasattr(bench, "live_status") else {}
    if not isinstance(live, dict):
        live = {}
    return pack_frame(
        live,
        image_wh=_image_wh(bench),
        generation=int(getattr(bench, "generation", 0) or 0),
        clients=clients,
    )


def status_from_bench(bench: Any, payload: dict[str, Any] | None = None, *, clients: int = 0) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else bench.status()
    if not isinstance(data, dict):
        data = {}
    else:
        data = dict(data)
    data.setdefault("loaded", True)
    return pack_status(data, clients=clients, rest=_rest_points(bench))
