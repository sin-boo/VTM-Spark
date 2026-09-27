"""Live2D-style pose keys: freeze a tracking pose, edit points, store a keyform.

A key is the correction from the auto-retarget pose to the operator's edited
overlay, remembered at the expression that was active when they hit Insert.
Later frames with a similar mouth / brow / blink / gaze blend that correction
back in. Head travel is left to live retarget so a mouth key still works after
a turn.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .pose_controller import NUM_KEYPOINTS

PARAM_DIM = 12
KEY_WEIGHT_SIGMA = 0.38


def params_from_controls(ctrl: Any | None) -> np.ndarray:
    """Compact expression vector used to match keys (not head xyz)."""
    out = np.zeros(PARAM_DIM, dtype=np.float32)
    if ctrl is None:
        return out
    gaze_l = getattr(ctrl, "gaze_l", (0.0, 0.0)) or (0.0, 0.0)
    gaze_r = getattr(ctrl, "gaze_r", (0.0, 0.0)) or (0.0, 0.0)
    out[0] = float(getattr(ctrl, "mouth_open", 0.0) or 0.0)
    out[1] = float(getattr(ctrl, "mouth_form", 0.0) or 0.0)
    out[2] = float(getattr(ctrl, "mouth_smile", 0.0) or 0.0)
    out[3] = float(getattr(ctrl, "mouth_asym", 0.0) or 0.0)
    out[4] = float(getattr(ctrl, "brow_l", 0.0) or 0.0)
    out[5] = float(getattr(ctrl, "brow_r", 0.0) or 0.0)
    out[6] = float(getattr(ctrl, "blink_l", 1.0) if getattr(ctrl, "blink_l", None) is not None else 1.0)
    out[7] = float(getattr(ctrl, "blink_r", 1.0) if getattr(ctrl, "blink_r", None) is not None else 1.0)
    out[8] = float(gaze_l[0])
    out[9] = float(gaze_l[1])
    out[10] = float(gaze_r[0])
    out[11] = float(gaze_r[1])
    return out


def params_from_keypoints(kps: np.ndarray | None) -> np.ndarray:
    """Rough expression vector from a 37-point pose (mouth / brow / blink / gaze)."""
    out = np.zeros(PARAM_DIM, dtype=np.float32)
    arr = np.asarray(kps, dtype=np.float32) if kps is not None else None
    if arr is None or arr.ndim != 2 or arr.shape[0] < NUM_KEYPOINTS:
        return out

    def _span(ids: tuple[int, ...], axis: int) -> float:
        pts = []
        for i in ids:
            if float(arr[i, 3]) < 0.5:
                continue
            pts.append(float(arr[i, axis]))
        if len(pts) < 2:
            return 0.0
        return float(max(pts) - min(pts))

    out[0] = float(np.clip(_span((20, 21, 22, 23, 24, 25, 26, 27), 1) * 2.2, 0.0, 1.5))
    out[1] = float(np.clip(_span((20, 22, 23, 26), 0) * 1.4 - 0.35, -1.0, 1.0))
    brow_l = float(arr[6, 1]) if float(arr[6, 3]) >= 0.5 else 0.0
    brow_r = float(arr[9, 1]) if float(arr[9, 3]) >= 0.5 else 0.0
    eye_l = float(arr[12, 1]) if float(arr[12, 3]) >= 0.5 else brow_l
    eye_r = float(arr[18, 1]) if float(arr[18, 3]) >= 0.5 else brow_r
    out[4] = float(np.clip((eye_l - brow_l) * 4.0 - 0.3, -1.0, 1.0))
    out[5] = float(np.clip((eye_r - brow_r) * 4.0 - 0.3, -1.0, 1.0))
    out[6] = 1.0
    out[7] = 1.0
    if float(arr[28, 3]) >= 0.5 and float(arr[12, 3]) >= 0.5:
        out[8] = float(np.clip((arr[28, 0] - arr[12, 0]) * 6.0, -1.0, 1.0))
        out[9] = float(np.clip((arr[28, 1] - arr[12, 1]) * 6.0, -1.0, 1.0))
    if float(arr[29, 3]) >= 0.5 and float(arr[18, 3]) >= 0.5:
        out[10] = float(np.clip((arr[29, 0] - arr[18, 0]) * 6.0, -1.0, 1.0))
        out[11] = float(np.clip((arr[29, 1] - arr[18, 1]) * 6.0, -1.0, 1.0))
    return out


def _as37xy(kps: np.ndarray) -> np.ndarray:
    arr = np.asarray(kps, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < NUM_KEYPOINTS:
        raise ValueError(f"Expected keypoints (>= {NUM_KEYPOINTS}, _), got {arr.shape}")
    return arr[:NUM_KEYPOINTS, :2].copy()


def _rot(xy: np.ndarray, deg: float) -> np.ndarray:
    rad = np.radians(float(deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    x = xy[:, 0]
    y = xy[:, 1]
    out = np.empty_like(xy)
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def local_delta(auto_kps: np.ndarray, edited_kps: np.ndarray, roll_deg: float) -> np.ndarray:
    """Face-local xy correction so a mouth edit follows the head later."""
    delta = _as37xy(edited_kps) - _as37xy(auto_kps)
    return _rot(delta, -float(roll_deg))


def apply_local_delta(
    pose: np.ndarray,
    delta_local: np.ndarray,
    *,
    roll_deg: float,
    weight: float,
) -> np.ndarray:
    if abs(float(weight)) < 1e-6:
        return pose
    out = np.asarray(pose, dtype=np.float32).copy()
    d = _rot(np.asarray(delta_local, dtype=np.float32), float(roll_deg)) * float(weight)
    n = min(NUM_KEYPOINTS, out.shape[0], d.shape[0])
    out[:n, 0] = out[:n, 0] + d[:n, 0]
    out[:n, 1] = out[:n, 1] + d[:n, 1]
    return out


def key_weight(live: np.ndarray, stored: np.ndarray, *, sigma: float = KEY_WEIGHT_SIGMA) -> float:
    live_a = np.asarray(live, dtype=np.float64).reshape(-1)
    stored_a = np.asarray(stored, dtype=np.float64).reshape(-1)
    n = min(live_a.size, stored_a.size)
    dist = float(np.linalg.norm(live_a[:n] - stored_a[:n]))
    s = max(float(sigma), 1e-4)
    return float(np.exp(-((dist / s) ** 2)))


def make_key(
    *,
    auto_kps: np.ndarray,
    edited_kps: np.ndarray,
    params: np.ndarray,
    roll_deg: float,
    name: str = "",
) -> dict[str, Any]:
    delta = local_delta(auto_kps, edited_kps, roll_deg)
    slots = [i for i in range(NUM_KEYPOINTS) if float(np.linalg.norm(delta[i])) > 1e-4]
    return {
        "id": uuid.uuid4().hex[:12],
        "name": str(name or "").strip(),
        "params": np.asarray(params, dtype=np.float32).reshape(-1)[:PARAM_DIM].tolist(),
        "delta": delta.astype(np.float32).tolist(),
        "slots": slots,
        "roll": float(roll_deg),
    }


def apply_keys(
    pose: np.ndarray,
    live_params: np.ndarray,
    keys: list[dict[str, Any]],
    *,
    live_roll_deg: float = 0.0,
) -> np.ndarray:
    if not keys:
        return pose
    weights = [key_weight(live_params, np.asarray(k.get("params") or [], dtype=np.float32)) for k in keys]
    total = float(sum(weights))
    if total > 1.0:
        weights = [w / total for w in weights]
    out = np.asarray(pose, dtype=np.float32).copy()
    for key, w in zip(keys, weights):
        if w < 0.02:
            continue
        delta = np.asarray(key.get("delta") or [], dtype=np.float32)
        if delta.ndim != 2 or delta.shape[1] < 2:
            continue
        out = apply_local_delta(out, delta[:, :2], roll_deg=live_roll_deg, weight=w)
    return out


def keys_path_for(ref: Path | None) -> Path | None:
    """Legacy sidecar path. Character packs keep their keys inside the ``.vtm``."""
    if ref is None:
        return None
    path = Path(ref)
    if path.suffix.lower() == ".vtm":
        return path.with_name(f"{path.stem}.keys.json")
    return path.with_name(f"{path.stem}_pose_keys.json")


def _is_pack(ref: Path | None) -> bool:
    return ref is not None and Path(ref).suffix.lower() == ".vtm"


def pack_key_sidecars(pack: Path | str) -> list[Path]:
    """Every place older installs kept a pack's keys (beside it or in its folder)."""
    path = Path(pack)
    stem = path.stem
    out: list[Path] = []
    for folder in (path.parent, path.parent / stem):
        for name in (f"{stem}.keys.json", f"{stem}_pose_keys.json"):
            cand = folder / name
            if cand not in out:
                out.append(cand)
    return out


def _valid_keys(items: Any) -> list[dict[str, Any]]:
    if isinstance(items, dict):
        items = items.get("keys")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and item.get("delta") is not None]


def _read_keys_file(dest: Path | None) -> list[dict[str, Any]]:
    if dest is None or not dest.is_file():
        return []
    try:
        raw = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return _valid_keys(raw)


def legacy_pack_keys(pack: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    found = [p for p in pack_key_sidecars(pack) if p.is_file()]
    keys: list[dict[str, Any]] = []
    for sidecar in found:
        keys = _read_keys_file(sidecar)
        if keys:
            break
    return keys, found


def _fold_pack_sidecars(pack: Path) -> None:
    """Move a legacy ``<stem>.keys.json`` into the pack once, then delete it.

    Keys already inside the pack win; the sidecar only fills an empty pack.
    """
    from .character_pack import CharacterPackError, upgrade_character_pack

    keys, found = legacy_pack_keys(pack)
    if not found:
        return
    try:
        upgrade_character_pack(pack, pose_keys=keys)
    except (CharacterPackError, OSError):
        return  # keep the sidecar; the pack could not take it
    for sidecar in found:
        try:
            sidecar.unlink()
        except OSError:
            pass


def load_keys(ref: Path | None) -> list[dict[str, Any]]:
    if _is_pack(ref):
        from .character_pack import CharacterPackError, read_pack_pose_keys

        pack = Path(ref)  # type: ignore[arg-type]
        if not pack.is_file():
            return []
        _fold_pack_sidecars(pack)
        try:
            return _valid_keys(read_pack_pose_keys(pack))
        except (CharacterPackError, OSError):
            return legacy_pack_keys(pack)[0]
    return _read_keys_file(keys_path_for(ref))


def save_keys(ref: Path | None, keys: list[dict[str, Any]]) -> Path | None:
    if _is_pack(ref):
        from .character_pack import write_pack_pose_keys

        pack = Path(ref)  # type: ignore[arg-type]
        if not pack.is_file():
            return None
        write_pack_pose_keys(pack, list(keys))
        for sidecar in pack_key_sidecars(pack):
            if sidecar.is_file():
                try:
                    sidecar.unlink()
                except OSError:
                    pass
        return pack
    dest = keys_path_for(ref)
    if dest is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"keys": list(keys)}
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest
