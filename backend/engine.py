"""Realtime keypoint DiT engine (KEYPOINT_SCHEMA 37x4 + ref image)."""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from .paths import (
    configure_torch_compile_cache,
    data_dir,
    default_ref_path,
    ensure_import_paths,
    models_dir,
    package_root,
    torch_train_dir,
)

configure_torch_compile_cache()
ensure_import_paths()
import torch

TORCH_TRAIN_DIR = torch_train_dir()
# Resolved dynamically via models_dir() so downloads into models/dit are picked up.
STREAM_CKPT_NAME = "VTM-1.5.1.pt"
# Default still — models/refs only (user upload / shipped default).
DEFAULT_REF = default_ref_path()
DEFAULT_REF_FALLBACK = default_ref_path()

STREAM_DEFAULT_STEPS = 1
# Designed 1–2 step joint pass: pose and the still in one forward (CFG 1.0).
STREAM_DEFAULT_POSE_CFG = 1.0
STREAM_DEFAULT_ID_CFG = 1.0
STREAM_MIN_CFG = 0.0
STREAM_MAX_CFG = 6.0
# Fast mode: real speed levers (TF32 alone is a no-op — DiT runs bf16).
STREAM_FAST_MAX_STEPS = 8
# This checkpoint is a 1–2 step model. 3-way pose/identity CFG splits the
# still off the pose and melts the face, so Fast keeps the joint forward.
STREAM_FAST_DISABLE_CFG = True
# Compile DiT on Fast warmup. Off by default — Start splash only loads
# weights + last character. Toggle Compile in the app, or set
# VTM_COMPILE_MODEL=1 before launch.
STREAM_COMPILE_MODEL = os.environ.get("VTM_COMPILE_MODEL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
STREAM_COMPILE_MODE = os.environ.get("VTM_COMPILE_MODE", "default").strip() or "default"
STREAM_COMPILE_MODE_LADDER = (STREAM_COMPILE_MODE, "default")
STREAM_COMPILE_WARMUP_RUNS = 2
# Fast decode: Hybrid TinyVAE (same SD latents). SD-VAE stays for ref encode.
STREAM_FAST_TINY_VAE = True
STREAM_FIXED_SEED = 42
# Stream can denoise 1 or 2 poses per DiT call (better GPU occupancy).
STREAM_BATCH_MAX = 2
# How strongly each new frame blends over the previous display (1 = no smooth).
# Lower = less Batch×2 flicker / sample pop, more leftover hair after a turn.
STREAM_TEMPORAL_EMA = 0.58
# Extra pictures drawn between two DiT frames (0 = keys only).
STREAM_INBETWEENS = 1
STREAM_MAX_INBETWEENS = 3
# Master switch for print / inbetween. Slider still picks the count.
STREAM_INTERPOLATE = True
STREAM_MIN_BLEND = 0.05
STREAM_MAX_BLEND = 1.0
# Start the next DiT sample from the last generated latent (img2img hold).
STREAM_HOLD_LAST = True
# Mix weight toward the last latent (0 = fresh noise, ~1 = barely move).
# Stay low — 0.6 xeroxed the still into chrome after a few seconds.
STREAM_HOLD_LAST_T = 0.28
# Each hold also blends this much of the still latent so identity cannot drift.
STREAM_HOLD_REF_PULL = 0.22
IMAGE_SIZE = 768
INFERENCE_TIMESTEP_SHIFT = 0.3
NUM_KEYPOINTS = 37
KEYPOINT_DIM = 4
# Fallback before a checkpoint is loaded. Runtime/overlay stay KEYPOINT_SCHEMA;
# only the DiT boundary remaps, and only when the checkpoint says so.
DEFAULT_KEYPOINT_LAYOUT = "hrnet_native"


def _clip_cfg(value: float) -> float:
    raw = float(value)
    if raw < STREAM_MIN_CFG:
        return STREAM_MIN_CFG
    if raw > STREAM_MAX_CFG:
        return STREAM_MAX_CFG
    return raw


def _clip_blend(value: float) -> float:
    raw = float(value)
    if raw < STREAM_MIN_BLEND:
        return STREAM_MIN_BLEND
    if raw > STREAM_MAX_BLEND:
        return STREAM_MAX_BLEND
    return raw


def _clip_inbetweens(value: object) -> int:
    try:
        raw = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return STREAM_INBETWEENS
    if raw < 0:
        return 0
    if raw > STREAM_MAX_INBETWEENS:
        return STREAM_MAX_INBETWEENS
    return raw


def interpolate_on(value: object) -> bool:
    if value is None:
        return STREAM_INTERPOLATE
    return bool(value)


def effective_inbetweens(enabled: object, count: object) -> int:
    """How many mids to print. Off toggle always wins over the slider."""
    if not interpolate_on(enabled):
        return 0
    return _clip_inbetweens(count)


# Face travel in norm_crop. Below tight = full hold. Above loose = drop hold
# so a turn does not keep the last (or rest) face stuck on the new pose.
# Drift from rest uses the same curve — a slow lean is many tiny steps, and
# holding through that melts the still into the smear the holdout saw.
_HOLD_MOVE_TIGHT = 0.018
_HOLD_MOVE_LOOSE = 0.090


def face_pose_delta(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean visible face-keypoint travel between two poses (norm_crop)."""
    if a is None or b is None:
        return 0.0
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2:
        return 0.0
    if left.shape[0] < 28 or right.shape[0] < 28:
        return 0.0
    face_l = left[:28]
    face_r = right[:28]
    vis = np.minimum(face_l[:, 3], face_r[:, 3]) > 0.35
    if not np.any(vis):
        vis = np.ones((face_l.shape[0],), dtype=bool)
    delta = face_l[vis, :2] - face_r[vis, :2]
    return float(np.mean(np.linalg.norm(delta, axis=1)))


def hold_ease(delta: float, *, tight: float = _HOLD_MOVE_TIGHT, loose: float = _HOLD_MOVE_LOOSE) -> float:
    """1 = keep hold, 0 = drop it. Linear between ``tight`` and ``loose``."""
    span = max(float(loose) - float(tight), 1e-6)
    amount = (float(delta) - float(tight)) / span
    return float(min(max(1.0 - amount, 0.0), 1.0))


def hold_plan(move: float, drift: float) -> tuple[float, float]:
    """``(start_t, pull)`` for the next hold.

    Ease off on the larger of frame-to-frame travel and rest-drift so a slow
    walk to one side cannot keep recycling the last latent.
    """
    ease = min(hold_ease(move), hold_ease(drift))
    if ease <= 1e-4:
        return 0.0, 0.0
    return STREAM_HOLD_LAST_T * ease, STREAM_HOLD_REF_PULL * ease


def anchor_hold_latent(
    last: torch.Tensor,
    ref: torch.Tensor | None,
    pull: float = STREAM_HOLD_REF_PULL,
) -> torch.Tensor:
    """Blend the held latent toward the still so recursive hold cannot melt."""
    amount = float(pull)
    if ref is None or amount <= 0.0:
        return last
    amount = min(max(amount, 0.0), 1.0)
    still = ref
    if still.shape[-3:] != last.shape[-3:]:
        return last
    if still.shape[0] != last.shape[0]:
        still = still[-1:]
    still = still.to(device=last.device, dtype=last.dtype)
    return (1.0 - amount) * last + amount * still


def keypoints_for_model(keypoints: np.ndarray, layout: str) -> np.ndarray:
    """Map KEYPOINT_SCHEMA rows to the layout a checkpoint expects."""
    from .model_layout import keypoints_for_model as _convert

    return _convert(_as_keypoints37(keypoints), layout)


def _keypoints_for_model(
    keypoints: np.ndarray,
    layout: str = DEFAULT_KEYPOINT_LAYOUT,
) -> np.ndarray:
    """Module wrapper for tests / call sites that pass an explicit layout."""
    return keypoints_for_model(keypoints, layout)


def checkpoint_label(path: Path) -> str:
    """Display name for a checkpoint file (model name only, no step suffix)."""
    name = path.name
    if ".pt-" in name:
        return name.split(".pt-", 1)[0]
    if name.endswith(".pt"):
        return name[: -len(".pt")]
    return name


_MIN_STREAM_CKPT_BYTES = 1_000_000
_STREAM_CKPT_SUFFIXES = {".pt", ".pth", ".ckpt"}
_EXTRA_CKPT_LOCK = threading.Lock()


def _extra_checkpoint_state_path() -> Path:
    return data_dir() / "extra_checkpoint_dirs.json"


def _load_extra_checkpoint_state() -> dict:
    path = _extra_checkpoint_state_path()
    if not path.is_file():
        return {"dirs": [], "last_dir": ""}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"dirs": [], "last_dir": ""}
    if not isinstance(raw, dict):
        return {"dirs": [], "last_dir": ""}
    dirs = raw.get("dirs") or []
    last_dir = str(raw.get("last_dir") or "")
    return {
        "dirs": [str(d) for d in dirs if str(d).strip()],
        "last_dir": last_dir,
    }


def _save_extra_checkpoint_state(state: dict) -> None:
    path = _extra_checkpoint_state_path()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def checkpoint_browse_start_dir() -> Path:
    """Folder the native picker should open in (last custom dir, else models/dit)."""
    with _EXTRA_CKPT_LOCK:
        last = str(_load_extra_checkpoint_state().get("last_dir") or "").strip()
    if last:
        folder = Path(last)
        if folder.is_dir():
            return folder
    return models_dir()


def remember_checkpoint_location(path: Path | str) -> None:
    """Keep the parent folder of a picked checkpoint so sibling .pt files are listed."""
    folder = Path(path).resolve().parent
    if not folder.is_dir():
        return
    with _EXTRA_CKPT_LOCK:
        state = _load_extra_checkpoint_state()
        dirs: list[str] = []
        seen: set[str] = set()
        pending = list(state.get("dirs") or [])
        pending.append(str(folder))
        for raw in pending:
            try:
                resolved = Path(raw).resolve()
            except OSError:
                continue
            key = str(resolved)
            if key in seen or not resolved.is_dir():
                continue
            if resolved == models_dir().resolve():
                continue
            seen.add(key)
            dirs.append(key)
        state["dirs"] = dirs
        state["last_dir"] = str(folder)
        try:
            _save_extra_checkpoint_state(state)
        except OSError:
            pass


def is_stream_checkpoint_file(path: Path | str) -> bool:
    p = Path(path)
    if not p.is_file() or p.name.startswith("."):
        return False
    suffix = p.suffix.lower()
    name = p.name.lower()
    if suffix not in _STREAM_CKPT_SUFFIXES and ".pt" not in name:
        return False
    try:
        return p.stat().st_size >= _MIN_STREAM_CKPT_BYTES
    except OSError:
        return False


def _iter_checkpoint_files(folder: Path) -> list[Path]:
    """DiT weights in ``folder`` and one nested folder (hub unpack layouts)."""
    if not folder.is_dir():
        return []
    files: list[Path] = []
    try:
        entries = list(folder.iterdir())
    except OSError:
        return []
    nested: list[Path] = []
    for path in entries:
        if path.is_dir() and not path.name.startswith("."):
            nested.append(path)
            continue
        if is_stream_checkpoint_file(path):
            files.append(path)
    for child in nested:
        try:
            kids = list(child.iterdir())
        except OSError:
            continue
        for path in kids:
            if is_stream_checkpoint_file(path):
                files.append(path)
    return files


def list_stream_checkpoints() -> list[tuple[str, Path]]:
    """List DiT checkpoints under models/dit plus any user-picked folders."""
    folders = [models_dir()]
    with _EXTRA_CKPT_LOCK:
        extra = _load_extra_checkpoint_state().get("dirs") or []
    for raw in extra:
        try:
            folders.append(Path(raw).resolve())
        except OSError:
            continue

    items: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for folder in folders:
        for path in _iter_checkpoint_files(folder):
            try:
                key = str(path.resolve())
            except OSError:
                key = str(path)
            if key in seen:
                continue
            seen.add(key)
            items.append((checkpoint_label(path), path))
    items.sort(key=lambda x: x[0].lower())
    return items


def default_stream_checkpoint() -> Path:
    from .ui_session import load_ui_session

    last = str(load_ui_session().get("checkpoint") or "").strip()
    if last:
        try:
            cand = Path(last)
            if not cand.is_absolute():
                cand = (package_root() / last).resolve()
            else:
                cand = cand.resolve()
            if is_stream_checkpoint_file(cand):
                return cand
        except OSError:
            pass

    items = list_stream_checkpoints()
    preferred = models_dir() / STREAM_CKPT_NAME
    if not items:
        return preferred
    # Prefer explicit default if present; else highest step suffix / last sorted name.
    for _label, path in items:
        if path.name == STREAM_CKPT_NAME or path.resolve() == preferred.resolve():
            return path
    return items[-1][1]


def default_reference_path() -> Path:
    if DEFAULT_REF.is_file():
        return DEFAULT_REF
    if DEFAULT_REF_FALLBACK.is_file():
        return DEFAULT_REF_FALLBACK
    return DEFAULT_REF


def sidecar_keypoints_npy_path(image_path: Path) -> Path:
    """Canonical ``<stem>_keypoints.npy`` next to a reference image."""
    path = Path(image_path)
    return path.with_name(f"{path.stem}_keypoints.npy")


def discard_sidecar_keypoints(image_path: Path) -> None:
    """Drop overlay sidecars next to a still so Create cannot reuse an old mesh."""
    path = Path(image_path)
    for cand in (
        sidecar_keypoints_npy_path(path),
        path.with_suffix(".npy"),
        path.with_name(f"{path.stem}_keypoints.json"),
        path.with_suffix(".json"),
    ):
        try:
            if cand.is_file() and cand.suffix.lower() in {".npy", ".json"}:
                cand.unlink()
        except OSError:
            pass


def save_sidecar_keypoints(image_path: Path, keypoints: np.ndarray) -> Path:
    """Write overlay/rest keypoints beside the reference so the next load keeps them."""
    dest = sidecar_keypoints_npy_path(image_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, _as_keypoints37(keypoints))
    return dest


def apply_overlay_drag_to_rest(
    rest: np.ndarray,
    drag_base: np.ndarray,
    edited: np.ndarray,
    slots: Iterable[int],
) -> np.ndarray:
    """Bake a live overlay drag into the character rest pose.

    Overlay at press is ``rest + live expression``. Subtracting the press
    snapshot from the released overlay isolates the pointer motion so tracking
    can resume on the new rest instead of freezing or snapping back.
    """
    out = _as_keypoints37(rest).copy()
    base = _as_keypoints37(drag_base)
    ed = _as_keypoints37(edited)
    for raw in slots:
        i = int(raw)
        if i < 0 or i >= NUM_KEYPOINTS:
            continue
        out[i, 0] = float(out[i, 0] + (ed[i, 0] - base[i, 0]))
        out[i, 1] = float(out[i, 1] + (ed[i, 1] - base[i, 1]))
        out[i, 2] = max(float(out[i, 2]), float(ed[i, 2]), 0.85)
        out[i, 3] = 1.0
    return out


def find_sidecar_keypoints(image_path: Path) -> Path | None:
    """Locate a keypoints sidecar next to a reference image."""
    path = Path(image_path)
    candidates = [
        path.with_name(f"{path.stem}_keypoints.npy"),
        path.with_suffix(".npy"),
        path.with_name(f"{path.stem}_keypoints.json"),
        path.with_suffix(".json"),
    ]
    # train_crop layout: images/<char>/<id>.png → labels/<char>/<id>_keypoints.npy
    try:
        parts = path.parts
        if "images" in parts:
            i = parts.index("images")
            rel = Path(*parts[i + 1 :])
            labels_root = Path(*parts[:i]) / "labels"
            cand = labels_root / rel.parent / f"{rel.stem}_keypoints.npy"
            candidates.insert(0, cand)
    except Exception:
        pass
    for cand in candidates:
        if cand.is_file() and cand.suffix.lower() in {".npy", ".json"}:
            # Skip tiny character metadata JSONs without keypoints.
            if cand.suffix.lower() == ".json":
                try:
                    payload = json.loads(cand.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not _json_has_keypoints(payload):
                    continue
            return cand
    return None


def _json_has_keypoints(payload: object) -> bool:
    if isinstance(payload, list) and len(payload) == NUM_KEYPOINTS:
        return True
    if not isinstance(payload, dict):
        return False
    if "keypoints" in payload:
        return True
    if payload.get("schema") == "KEYPOINT_SCHEMA":
        return True
    return False


def load_keypoints_file(path: Path | str) -> np.ndarray:
    """Load (37, 4) keypoints from .npy or bridge/label JSON."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        return _as_keypoints37(arr)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return keypoints_from_json(payload)


def keypoints_from_json(payload: object) -> np.ndarray:
    """Load KEYPOINT_SCHEMA from bridge / label JSON.

    Preference order:
    1. Explicit ``keypoints_norm`` with ``coord_space`` / ``coord_space_norm`` = norm_crop
    2. Top-level ``coord_space=norm_crop`` + ``keypoints`` already in model space
    3. Legacy pixel ``keypoints`` + ``image_wh`` / ``source_wh`` → pad-square transform
    4. Ambiguous normalized packets without a declared space are rejected
    """
    if isinstance(payload, list):
        return _as_keypoints37(np.asarray(payload, dtype=np.float32))
    if not isinstance(payload, dict):
        raise ValueError("Keypoints JSON must be a list or object")

    def _rows_to_array(rows: object) -> np.ndarray:
        if rows and isinstance(rows, list) and isinstance(rows[0], dict):
            out = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
            for row in rows:
                i = int(row.get("i", -1))
                if 0 <= i < NUM_KEYPOINTS:
                    out[i, 0] = float(row.get("x", 0.0))
                    out[i, 1] = float(row.get("y", 0.0))
                    out[i, 2] = float(row.get("score", 0.0))
                    vis = row.get("visible", out[i, 2] > 0)
                    out[i, 3] = 1.0 if bool(vis) else 0.0
            return out
        return _as_keypoints37(np.asarray(rows, dtype=np.float32))

    space = str(
        payload.get("coord_space_norm")
        or payload.get("coord_space")
        or (payload.get("meta") or {}).get("coord_space")
        or ""
    ).strip().lower()

    if "keypoints_norm" in payload:
        arr = _rows_to_array(payload["keypoints_norm"])
        if space and space not in ("norm_crop", "normalized_crop", "crop"):
            raise ValueError(
                f"keypoints_norm present but coord_space is '{space}' "
                "(expected norm_crop)"
            )
        return arr

    if "keypoints" not in payload:
        raise ValueError("No keypoints array in JSON")

    rows = payload["keypoints"]
    arr = _rows_to_array(rows)

    if space in ("norm_crop", "normalized_crop", "crop"):
        return arr

    # Heuristic: already normalized without a declared space.
    vis = arr[:, 3] >= 0.5
    max_abs = float(np.abs(arr[vis, :2]).max()) if np.any(vis) else 0.0
    if max_abs <= 1.5:
        if not space:
            raise ValueError(
                "Normalized keypoints JSON missing coord_space; "
                "refuse to guess between norm_full and norm_crop"
            )
        if space in ("norm_full", "full", "full_frame"):
            raise ValueError(
                "norm_full keypoints cannot be loaded as model targets; "
                "re-emit with pad-square → norm_crop"
            )
        raise ValueError(f"Unsupported coord_space '{space}' for keypoints JSON")

    # Legacy pixel packet — transform via pad-square using source dimensions.
    wh = payload.get("source_wh") or payload.get("image_wh")
    if not (isinstance(wh, (list, tuple)) and len(wh) >= 2):
        raise ValueError(
            "Legacy pixel keypoints need image_wh/source_wh to convert to norm_crop"
        )
    src_w, src_h = int(wh[0]), int(wh[1])
    try:
        from utils.coordinate_frames import DEFAULT_IMAGE_SIZE as _DIS
        from utils.coordinate_frames import webcam_pixels_to_norm_crop

        image_size = int(payload.get("image_size") or _DIS)
        arr, _crop = webcam_pixels_to_norm_crop(arr, src_w, src_h, image_size=image_size)
        return arr
    except Exception:
        # Fallback without package import (tests / minimal env).
        from utils.keypoints import transform_keypoints_crop

        image_size = int(payload.get("image_size") or 768)
        side = max(src_w, src_h)
        pad_x = (side - src_w) // 2
        pad_y = (side - src_h) // 2
        return transform_keypoints_crop(
            arr,
            crop_x0=float(-pad_x),
            crop_y0=float(-pad_y),
            crop_w=float(side),
            crop_h=float(side),
            out_w=float(image_size),
            out_h=float(image_size),
            normalize=True,
        )


def looks_like_greenscreen(rgb: np.ndarray) -> bool:
    """True when the frame is mostly chroma-key green (training-style refs)."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    # Sample a coarse grid for speed.
    step = max(1, min(arr.shape[0], arr.shape[1]) // 64)
    patch = arr[::step, ::step, :3].astype(np.float32)
    g = patch[..., 1]
    r = patch[..., 0]
    b = patch[..., 2]
    greenish = (g > 40.0) & (g > r + 15.0) & (g > b + 15.0)
    return float(np.mean(greenish)) >= 0.25


def _reference_rgb(path: Path) -> np.ndarray:
    """RGB pixels for a still or a character pack preview. Never opens a .vtm as an image."""
    if path.suffix.lower() == ".vtm":
        from .character_pack import read_character_pack

        return np.asarray(read_character_pack(path).preview_rgb)
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


class MissingRefKeypointsError(FileNotFoundError):
    """Reference image has no matching KEYPOINT_SCHEMA sidecar."""


def looks_like_legacy_hrnet_layout(keypoints: np.ndarray) -> bool:
    """Detect the historical native-HRNet-in-schema-slots label signature.

    In that layout both the alleged nose (14..16) and upper lip (20..22) sit
    in the eye band, while 24..27 form the actual lower mouth.  Keep this
    deliberately conservative: a false negative only emits the normal sidecar,
    while a false positive would trigger a slower reference refit.
    """
    k = _as_keypoints37(keypoints)
    required = tuple(range(11, 20)) + tuple(range(20, 28))
    if not all(float(k[i, 3]) >= 0.5 for i in required):
        return False
    eye_y = float(np.mean(k[list((11, 12, 13, 17, 18, 19)), 1]))
    lower_y = float(np.mean(k[list((24, 25, 26, 27)), 1]))
    misplaced_y = float(np.mean(k[list((14, 15, 16, 20, 21, 22)), 1]))
    eye_to_lower = lower_y - eye_y
    if eye_to_lower <= 0.06:
        return False
    return misplaced_y <= eye_y + 0.32 * eye_to_lower


def _replace_legacy_sidecar(
    image_path: Path,
    loaded: np.ndarray,
    sidecar: Path,
    *,
    skip_crop: bool | None,
    fit_if_missing: bool,
    fit_device: str | None = "cpu",
) -> tuple[np.ndarray, Path | None, str]:
    if not looks_like_legacy_hrnet_layout(loaded):
        return loaded, sidecar, "sidecar"
    print(
        f"[pose-diag] WARNING: {sidecar.name} matches legacy native-HRNet "
        "slot layout; refitting reference pose."
    )
    if not fit_if_missing:
        return loaded, sidecar, "legacy-sidecar"
    try:
        from .ref_pose_fit import fit_keypoints_to_reference

        fitted = fit_keypoints_to_reference(
            image_path,
            skip_crop=skip_crop,
            image_size=IMAGE_SIZE,
            device=fit_device,
        )
        return (
            sanitize_normalized_keypoints(fitted),
            None,
            "fitted-legacy-replacement",
        )
    except Exception as exc:
        print(f"Legacy sidecar replacement fit failed for {image_path.name}: {exc}")
        return loaded, sidecar, "legacy-sidecar"


def resolve_ref_keypoints(
    image_path: Path,
    keypoints: np.ndarray | Path | str | None = None,
    *,
    skip_crop: bool | None = None,
    fit_if_missing: bool = True,
    fit_device: str | None = "cpu",
) -> tuple[np.ndarray, Path | None, str]:
    """Load ref (37,4) keypoints.

    Returns ``(keypoints, sidecar_path_or_none, source)`` where source is
    ``sidecar`` | ``explicit`` | ``fitted`` | ``neutral`` plus legacy warning
    variants.

    Prefer sidecar labels. If missing and ``fit_if_missing``, detect the face
    on the reference image and build a character-aligned skeleton. Neutral
    template is last resort only.

    Face-fit defaults to CPU so it never races DiT/VAE on CUDA (that hang
    looked like a stuck \"Encoding reference…\" in the UI).
    """
    image_path = Path(image_path)
    if isinstance(keypoints, (str, Path)):
        kps_path = Path(keypoints)
        if not kps_path.is_file():
            raise MissingRefKeypointsError(f"Keypoints file not found: {kps_path}")
        loaded = sanitize_normalized_keypoints(load_keypoints_file(kps_path))
        return _replace_legacy_sidecar(
            image_path,
            loaded,
            kps_path,
            skip_crop=skip_crop,
            fit_if_missing=fit_if_missing,
            fit_device=fit_device,
        )
    if keypoints is not None:
        return sanitize_normalized_keypoints(np.asarray(keypoints, dtype=np.float32)), None, "explicit"
    sidecar = find_sidecar_keypoints(image_path)
    if sidecar is not None:
        loaded = sanitize_normalized_keypoints(load_keypoints_file(sidecar))
        return _replace_legacy_sidecar(
            image_path,
            loaded,
            sidecar,
            skip_crop=skip_crop,
            fit_if_missing=fit_if_missing,
            fit_device=fit_device,
        )

    if fit_if_missing and image_path.is_file():
        try:
            from .ref_pose_fit import fit_keypoints_to_reference

            print(
                f"[ref-fit] fitting keypoints on CPU/GPU={fit_device!r} for {image_path.name}…"
            )
            fitted = fit_keypoints_to_reference(
                image_path,
                skip_crop=skip_crop,
                image_size=IMAGE_SIZE,
                device=fit_device,
            )
            return sanitize_normalized_keypoints(fitted), None, "fitted"
        except Exception as exc:
            print(f"Ref face-fit failed for {image_path.name}: {exc}")

    print(f"No sidecar for {image_path.name} - using neutral pose template")
    return sanitize_normalized_keypoints(neutral_keypoints()), None, "neutral"


def _as_keypoints37(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2 or out.shape != (NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(f"Expected keypoints ({NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {out.shape}")
    return out


def sanitize_normalized_keypoints(
    kps: np.ndarray,
    *,
    repair_mouth: bool = True,
    repair_nose: bool = True,
) -> np.ndarray:
    """Zero xy on invisible points; optionally repair nose/mouth on eyes."""
    out = _as_keypoints37(kps).copy()
    invisible = out[:, 3] < 0.5
    out[invisible, 0:2] = 0.0
    # Sidecar / HRNet labels often park upper lip on an eye — but this rewrite
    # is a limiter. Callers that already repaired (or want raw lips) can disable.
    try:
        from .ref_pose_fit import repair_collapsed_face_landmarks

        out = repair_collapsed_face_landmarks(
            out, repair_mouth=repair_mouth, repair_nose=repair_nose
        )
    except Exception as exc:
        print(f"[pose-diag] face landmark repair skipped: {exc}")
    return out


def _looks_normalized(kps: np.ndarray) -> bool:
    """True when visible xy mostly live in crop-normalized space."""
    k = _as_keypoints37(kps)
    vis = k[:, 3] >= 0.5
    if not np.any(vis):
        return True
    return float(np.abs(k[vis, :2]).max()) <= 1.5


def apply_live_deltas_to_ref(
    ref_keypoints: np.ndarray,
    live_keypoints: np.ndarray,
    live_origin: np.ndarray,
    *,
    gain: float = 1.0,
    body_method: str = "unknown",
    body_lost: bool = False,
    prev_body: np.ndarray | None = None,
    live_coord_space: str | None = "norm_crop",
    origin_coord_space: str | None = "norm_crop",
    head_yaw_deg: float | None = None,
    head_pitch_deg: float | None = None,
    head_roll_deg: float | None = None,
    head_tx_norm: float | None = None,
    head_ty_norm: float | None = None,
    limit_face: bool = True,
    limit_brows: bool = True,
    limit_eyes: bool = True,
    limit_nose: bool = True,
    limit_mouth: bool = True,
    reference_rig: object | None = None,
    prev_controls: object | None = None,
    controls_out: list | None = None,
    motion: dict | None = None,
) -> np.ndarray:
    """Map webcam motion onto the character's ref keypoints (training space).

    Live and origin must already be in model ``norm_crop`` space (same contract
    as reference sidecars / ``encode_reference``). Mixing full-frame normalized
    coords with crop-space refs is rejected.

    Modifiers / ``gain`` (the motion multiplier)
    --------------------------------------------
    Calibrate stores a live **origin** (the webcam pose at Center). Every later
    frame is measured as a delta from that origin, then multiplied by ``gain``
    and applied on top of the character's rest pose::

        character = rest_pose + gain * (live_pose - origin)

    ``gain = 1.0`` copies the webcam delta at full strength.
    ``gain = 0.5`` is half as much motion (smaller nods, smaller smiles).
    ``gain = 0.0`` would freeze at rest. Retracting the skeleton overlay does
    not change ``gain`` or these deltas — it only hides the drawn bones.

    Uses constrained Live2D-style retarget: reference bone lengths / face layout
    stay locked; only semantic controls (head, blink, gaze, brows, mouth, body
    joint angles) drive the character. Final pass runs character-relative sanitize
    plus proportion invariant guards.

    ``limit_face`` / ``limit_brows`` / ``limit_eyes`` / ``limit_nose`` /
    ``limit_mouth`` gate expression travel/semantic caps for those regions.
    """
    from .live_poser_client import body_method_kind, is_body_tracked
    from .live_retarget import apply_constrained_retarget
    from .pose_controller import sanitize_pose

    try:
        from utils.coordinate_frames import COORD_NORM_CROP, assert_norm_crop
    except Exception:
        COORD_NORM_CROP = "norm_crop"

        def assert_norm_crop(space: str | None, *, label: str = "keypoints") -> None:
            if str(space or "").strip().lower() != COORD_NORM_CROP:
                raise ValueError(
                    f"{label} must be in '{COORD_NORM_CROP}' space; got '{space}'"
                )

    assert_norm_crop(live_coord_space, label="live keypoints")
    assert_norm_crop(origin_coord_space, label="live origin keypoints")

    ref = _as_keypoints37(ref_keypoints)
    live = _as_keypoints37(live_keypoints)
    origin = _as_keypoints37(live_origin)

    out = apply_constrained_retarget(
        ref,
        live,
        origin,
        gain=float(gain),
        body_method=body_method,
        body_lost=body_lost,
        prev_body=prev_body,
        head_yaw_deg=head_yaw_deg,
        head_pitch_deg=head_pitch_deg,
        head_roll_deg=head_roll_deg,
        head_tx_norm=head_tx_norm,
        head_ty_norm=head_ty_norm,
        limit_face=limit_face,
        limit_brows=limit_brows,
        limit_eyes=limit_eyes,
        limit_nose=limit_nose,
        limit_mouth=limit_mouth,
        reference_rig=reference_rig,
        prev_controls=prev_controls,
        controls_out=controls_out,
        motion=motion,
    )

    # Log occasionally (every ~30 calls) so live drive doesn't flood the console.
    log = False
    apply_live_deltas_to_ref._n = getattr(apply_live_deltas_to_ref, "_n", 0) + 1  # type: ignore[attr-defined]
    if apply_live_deltas_to_ref._n == 1 or apply_live_deltas_to_ref._n % 30 == 0:  # type: ignore[attr-defined]
        log = True
        from .pose_controller import format_body_diagnostics

        print(
            f"[pose-diag] constrained_retarget n={apply_live_deltas_to_ref._n} "  # type: ignore[attr-defined]
            f"gain={gain:.3f} body={body_method} lost={body_lost} "
            f"yaw={head_yaw_deg} pitch={head_pitch_deg} roll={head_roll_deg} "
            f"tx={head_tx_norm} ty={head_ty_norm} "
            f"space={live_coord_space}"
        )
        print(format_body_diagnostics(live, label="live/WEBCAM"))
        print(format_body_diagnostics(out, label="live/PRE_SANITIZE"))
    body_tracked = is_body_tracked(body_method) and not body_lost
    # Preserve held/lost body exactly — do not let sanitize clamp it back to ref.
    sanitize_body = not (body_lost or body_method_kind(body_method) == "held")
    # Travel/topology stay off on live; per-feature flags still gate any future
    # sanitize clamps and face proportion locks.
    return sanitize_pose(
        out,
        ref,
        recenter=False,
        clamp_travel=False,
        topology=False,
        log=log,
        log_label="live",
        body_tracked=body_tracked,
        sanitize_body=sanitize_body,
        lock_proportions=True,
        lock_face_proportions=False,
        limit_face=limit_face,
        limit_brows=limit_brows,
        limit_eyes=limit_eyes,
        limit_nose=limit_nose,
        limit_mouth=limit_mouth,
    )


_KP_SHORT = (
    *(f"face_{i}" for i in range(28)),
    "right_iris",
    "left_iris",
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "left_shoulder",
    "left_elbow",
    "chest",
)


def format_keypoints_table(kps: np.ndarray, *, max_rows: int = 12) -> str:
    """Training-style snippet of the (37,4) tensor sent to the model."""
    k = _as_keypoints37(kps)
    lines = [
        "# index  name                     x          y       score  visible",
    ]
    for i in range(min(int(max_rows), NUM_KEYPOINTS)):
        name = _KP_SHORT[i] if i < len(_KP_SHORT) else f"kp_{i}"
        x, y, s, v = (float(k[i, 0]), float(k[i, 1]), float(k[i, 2]), float(k[i, 3]))
        lines.append(f"{i:2d}  {name:<22} {x:+.4f}   {y:+.4f}   {s:.2f}    {int(v >= 0.5)}")
    if NUM_KEYPOINTS > max_rows:
        lines.append(f"... ({NUM_KEYPOINTS} rows total: face / brows / eyes / iris / body)")
    return "\n".join(lines)


def neutral_keypoints() -> np.ndarray:
    """Fallback centered face+body template in normalized [-1, 1] space.

    Vertical layout matches train_crop stats: eyes ≈ -0.30, nose ≈ -0.25,
    upper mouth ≈ -0.26, lower mouth ≈ -0.12, chin ≈ +0.06.
    """
    out = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
    # Rough upright face — proportions from historical train_crop medians.
    face_xy = [
        # outline 0-4
        (-0.42, -0.05), (-0.28, 0.15), (0.00, 0.22), (0.28, 0.15), (0.42, -0.05),
        # brows 5-10
        (-0.32, -0.42), (-0.20, -0.46), (-0.08, -0.42),
        (0.08, -0.42), (0.20, -0.46), (0.32, -0.42),
        # left eye 11-13
        (-0.30, -0.30), (-0.20, -0.32), (-0.10, -0.30),
        # nose 14-16
        (-0.08, -0.22), (0.00, -0.18), (0.08, -0.22),
        # right eye 17-19
        (0.10, -0.30), (0.20, -0.32), (0.30, -0.30),
        # mouth 20-27: UL UM UR CL LL LM CR LR (closed slit, train-like near nose)
        (-0.12, -0.26), (0.00, -0.26), (0.12, -0.26),
        (-0.18, -0.24), (-0.10, -0.22), (0.00, -0.21), (0.18, -0.24), (0.10, -0.22),
    ]
    for i, (x, y) in enumerate(face_xy):
        out[i] = (x, y, 1.0, 1.0)
    out[28] = (-0.20, -0.30, 1.0, 1.0)  # right iris
    out[29] = (0.20, -0.30, 1.0, 1.0)   # left iris
    body = [
        (0.00, -0.18),  # nose
        (0.00, 0.28),   # neck (must sit below chin for topology)
        (-0.35, 0.42),  # r shoulder
        (-0.55, 0.75),  # r elbow
        (0.35, 0.42),   # l shoulder
        (0.55, 0.75),   # l elbow
        (0.00, 0.55),   # chest
    ]
    for i, (x, y) in enumerate(body):
        out[30 + i] = (x, y, 1.0, 1.0)
    return out


from inference_keypoint import (  # noqa: E402
    build_keypoint_model,
    decode_sd_vae,
    decode_tiny_vae,
    denoise_keypoint,
    encode_reference,
    load_sd_vae,
    load_tiny_vae,
    preferred_sd_vae_dtype,
)


def _enable_tf32() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    _hide_compiler_consoles()


def _hide_compiler_consoles() -> None:
    """torch.compile launches a console per kernel on Windows. Keep those hidden."""
    if os.name != "nt":
        return
    import subprocess

    if getattr(subprocess.Popen, "_vtm_no_window", False):
        return
    orig = subprocess.Popen
    no_window = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))

    class _QuietPopen(orig):
        _vtm_no_window = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            flags = int(kwargs.get("creationflags") or 0)
            kwargs["creationflags"] = flags | no_window
            super().__init__(*args, **kwargs)

    subprocess.Popen = _QuietPopen


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def _is_compile_runtime_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "triton",
        "torch._inductor",
        "inductor",
        "dynamo",
        "cuda graph",
        "cudagraph",
        "during capture",
        "capture",
        "cuda error",
        "operation failed",
        "graph",
    )
    return any(n in msg for n in needles) or exc.__class__.__name__ in {
        "BackendCompilerFailed",
        "TorchRuntimeError",
    }


def _clear_cuda_errors() -> None:
    """Recover after a failed CUDA-graph capture so later eager runs can proceed."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _mod_to(mod: Any, device: torch.device) -> Any:
    if mod is None:
        return None
    try:
        return mod.to(device)
    except Exception:
        return mod


def _ten_to(tensor: "torch.Tensor | None", device: torch.device) -> "torch.Tensor | None":
    if tensor is None:
        return None
    try:
        if tensor.device == device:
            return tensor
        return tensor.to(device)
    except Exception:
        return tensor


class StreamEngine:
    """Keypoint-conditioned DiT: live (37,4) + cached reference latent/keypoints."""

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        device: str = "cuda",
        num_steps: int = STREAM_DEFAULT_STEPS,
        pose_cfg_scale: float = STREAM_DEFAULT_POSE_CFG,
        id_cfg_scale: float = STREAM_DEFAULT_ID_CFG,
        fast_mode: bool = False,
        compile_model: bool | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else default_stream_checkpoint()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.num_steps = int(num_steps)
        self.pose_cfg_scale = float(pose_cfg_scale)
        self.id_cfg_scale = float(id_cfg_scale)
        # Back-compat alias used by older UI bits.
        self.cfg_scale = float(pose_cfg_scale)
        self.fast_mode = bool(fast_mode)
        self.compile_model = (
            STREAM_COMPILE_MODEL if compile_model is None else bool(compile_model)
        )
        self.stream_batch_size = 1

        self.model = None
        self.vae = None
        self.vae_tiny = None
        self._tiny_vae_id: str | None = None
        self._tiny_vae_failed = False
        self._cfg: dict = {}
        self.keypoint_layout: str = DEFAULT_KEYPOINT_LAYOUT
        self._ref_path: Path | None = None
        self._ref_kps_path: Path | None = None
        self._ref_neutral_fallback: bool = False
        self._ref_pose_source: str = "none"  # sidecar | fitted | neutral | explicit
        self._ref_skip_crop: bool = False
        self._ref_latent: torch.Tensor | None = None
        self._ref_face_latent: torch.Tensor | None = None
        self._ref_keypoints: np.ndarray | None = None
        self._ref_keypoints_model: np.ndarray | None = None
        # Rest pose at apply/calibrate time — mesh reset restores this, not the
        # last drag (which is already baked into ``_ref_keypoints``).
        self._ref_keypoints_session_base: np.ndarray | None = None
        self._ref_rig = None
        self._ready = False
        self._gpu_resident = False
        self._model_compiled = False
        self._compile_failed = False
        self._compile_mode_active: str | None = None
        self._compile_verified = False
        self._eager_model = None
        self.last_timings: dict[str, float] = {}
        self.last_decode_backend: str = "sd"
        self.last_target_keypoints: np.ndarray | None = None
        self.last_target_keypoints_batch: np.ndarray | None = None
        self.hold_last = STREAM_HOLD_LAST
        self._last_gen_latent: torch.Tensor | None = None
        self._last_hold_kps: np.ndarray | None = None
        self._body_skel_method: str = "unknown"
        self._body_lost: bool = False
        self._last_driven_body: np.ndarray | None = None
        self._cuda_lock = threading.RLock()
        # Decode pipeline: dedicated lock + CUDA stream so frame N's decode can
        # overlap frame N+1's denoise (which holds _cuda_lock).
        self._decode_lock = threading.Lock()
        self._decode_stream: "torch.cuda.Stream | None" = None
        self.image_size = IMAGE_SIZE

    def set_stream_batch_size(self, batch_size: int) -> None:
        """1 = one pose/denoise; 2 = two poses in one DiT forward."""
        self.stream_batch_size = max(1, min(int(batch_size), STREAM_BATCH_MAX))

    def set_hold_last(self, enabled: bool) -> None:
        self.hold_last = bool(enabled)

    def clear_last_gen_latent(self) -> None:
        self._last_gen_latent = None
        self._last_hold_kps = None

    @property
    def fast_status(self) -> str:
        if not self.fast_mode:
            return "off"
        parts = ["tf32"]
        if STREAM_FAST_DISABLE_CFG:
            parts.append("no-cfg")
        if self.device.type == "cuda":
            parts.append("tf32")
        else:
            parts.append("cpu")
        parts.append(f"compile:{self.compile_status}")
        if self.vae_tiny is not None:
            label = (self._tiny_vae_id or "tiny").split("/")[-1]
            parts.append(f"tinyvae:{label}")
        elif self._tiny_vae_failed:
            parts.append("tinyvae-fail")
        elif STREAM_FAST_TINY_VAE:
            parts.append("tinyvae-pending")
        return ", ".join(parts)

    @property
    def compile_status(self) -> str:
        """Compact torch.compile state for UI: on | pending | fail | skip | off."""
        if not self.fast_mode or not self.compile_model:
            return "off"
        if self.device.type != "cuda":
            return "skip"
        if self._compile_failed:
            return "fail"
        if not hasattr(torch, "compile"):
            return "fail"
        if not _triton_available():
            return "fail"
        if self._model_compiled and self._compile_verified:
            return "on"
        return "pending"

    @property
    def compile_detail(self) -> str:
        """Human reason for the compile light (UI tooltip / status line)."""
        st = self.compile_status
        if st == "on":
            mode = self._compile_mode_active or STREAM_COMPILE_MODE
            return f"torch.compile verified ({mode})"
        if st == "pending":
            return "torch.compile pending — apply a reference to compile + test"
        if st == "skip":
            return "torch.compile skipped (CUDA required)"
        if st == "off":
            if not self.compile_model:
                return "torch.compile off"
            if not self.fast_mode:
                return "Fast off — compile idle"
            return "torch.compile off"
        if not hasattr(torch, "compile"):
            return "torch.compile unavailable in this PyTorch build"
        if not _triton_available():
            return (
                "triton missing — install triton-windows "
                "(pip install \"triton-windows>=3.6,<3.7\")"
            )
        if self._compile_failed:
            return "torch.compile failed — using eager Fast"
        return "torch.compile unavailable"

    def set_fast_mode(self, enabled: bool) -> None:
        self.fast_mode = bool(enabled)
        if self.fast_mode:
            _enable_tf32()
        else:
            # Drop compiled wrapper when leaving Fast so the next enable re-tests.
            try:
                self._restore_eager_model()
            except Exception:
                pass
            self._compile_failed = False
            self._compile_verified = False

    def set_compile_model(self, enabled: bool) -> None:
        """Turn torch.compile on or off without restarting the app."""
        want = bool(enabled)
        if want == bool(self.compile_model):
            return
        self.compile_model = want
        try:
            self._restore_eager_model()
        except Exception:
            pass
        self._compile_failed = False
        self._compile_verified = False

    def verify_compile(self) -> bool:
        """One-shot runtime test that torch.compile is actually usable.

        Runs a short denoise under the compiled graph (or confirms skip/fail).
        Sets ``_compile_verified`` only when the compiled path succeeds.
        """
        self._compile_verified = False
        if not self.fast_mode or not self.compile_model:
            return False
        if self.device.type != "cuda" or self.model is None:
            return False
        if self._compile_failed:
            return False
        if not self._model_compiled:
            if not self._maybe_compile_model(STREAM_COMPILE_MODE):
                return False
        if not self._model_compiled:
            return False
        if self._ref_latent is None or self._ref_keypoints_model is None:
            # Compiled wrapper is up; full graph test waits for a reference.
            self._compile_verified = True
            return True

        try:
            steps, pose_cfg, id_cfg = self._resolve_generate_settings(None)
            warm_steps = max(1, min(int(steps), 2))
            bsz = max(1, min(int(self.stream_batch_size), STREAM_BATCH_MAX))
            kps_target = self._ref_keypoints_model
            if bsz > 1:
                kps_target = np.stack([kps_target] * bsz, axis=0)
            with self._cuda_lock:
                with torch.inference_mode():
                    _ = denoise_keypoint(
                        self.model,
                        keypoints_target=kps_target,
                        ref_latent=self._ref_latent,
                        ref_keypoints=self._ref_keypoints_model,
                        ref_face_latent=self._ref_face_latent,
                        num_steps=warm_steps,
                        pose_cfg_scale=pose_cfg,
                        id_cfg_scale=id_cfg,
                        inference_timestep_shift=INFERENCE_TIMESTEP_SHIFT,
                        seed=STREAM_FIXED_SEED,
                        shared_noise=True,
                    )
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
            self._compile_verified = True
            print(
                f"[compile] verify OK (mode={self._compile_mode_active or STREAM_COMPILE_MODE}, "
                f"batch={bsz})"
            )
            return True
        except Exception as exc:
            print(f"[compile] verify failed: {exc}")
            if _is_compile_runtime_error(exc):
                self._fallback_eager_model(exc)
            else:
                self._compile_failed = True
                self._restore_eager_model()
                _clear_cuda_errors()
            self._compile_verified = False
            return False

    def _ensure_tiny_vae(self) -> bool:
        """Lazy-load Hybrid TinyVAE for Fast decode. SD-VAE stays for encode."""
        if not STREAM_FAST_TINY_VAE:
            return False
        if self.vae_tiny is not None:
            return True
        if self._tiny_vae_failed:
            return False
        try:
            print("Loading Hybrid TinyVAE (fast decode) ...")
            vae_dtype = preferred_sd_vae_dtype(self.device)
            self.vae_tiny, self._tiny_vae_id = load_tiny_vae(
                self.device, dtype=vae_dtype
            )
            print(
                f"TinyVAE ready ({self._tiny_vae_id}, "
                f"dtype={str(vae_dtype).replace('torch.', '')})"
            )
            return True
        except Exception as exc:
            self._tiny_vae_failed = True
            self.vae_tiny = None
            self._tiny_vae_id = None
            print(f"TinyVAE skipped; using SD-VAE decode: {exc}")
            return False

    def _decode_latents(self, latents: torch.Tensor) -> tuple[np.ndarray, str]:
        """Decode latents; Fast path prefers TinyVAE when available."""
        if self.fast_mode and self._ensure_tiny_vae():
            return decode_tiny_vae(self.vae_tiny, latents), "tiny"
        return decode_sd_vae(self.vae, latents), "sd"

    def _restore_eager_model(self) -> None:
        eager = self._eager_model
        if eager is None:
            eager = getattr(self.model, "_orig_mod", None)
        if eager is not None:
            self.model = eager
        self._model_compiled = False
        self._compile_mode_active = None
        self._compile_verified = False

    def _maybe_compile_model(self, mode: str | None = None) -> bool:
        """Compile DiT once for Fast path. Returns True if model is compiled."""
        if not self.compile_model:
            return False
        if self.device.type != "cuda" or self.model is None:
            return False
        if self._model_compiled:
            return True
        if self._compile_failed:
            return False
        if not hasattr(torch, "compile"):
            self._compile_failed = True
            print("torch.compile unavailable; continuing eager.")
            return False
        if not _triton_available():
            self._compile_failed = True
            print(
                "torch.compile skipped: triton not installed in this venv "
                "(Fast still uses TinyVAE)."
            )
            return False

        compile_mode = mode or STREAM_COMPILE_MODE
        try:
            self._restore_eager_model()
            if self._eager_model is None:
                self._eager_model = self.model
            print(f"torch.compile DiT (mode={compile_mode}) ...")
            self.model = torch.compile(
                self._eager_model,
                mode=compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            self._model_compiled = True
            self._compile_mode_active = compile_mode
            print("torch.compile applied (warmup will finish capture).")
            return True
        except Exception as exc:
            self._compile_failed = True
            self._restore_eager_model()
            _clear_cuda_errors()
            print(f"torch.compile skipped ({compile_mode}): {exc}")
            return False

    def _fallback_eager_model(self, reason: BaseException | str) -> None:
        """Drop a broken compiled wrapper and keep Fast path alive."""
        self._restore_eager_model()
        self._compile_failed = True
        _clear_cuda_errors()
        print(f"torch.compile runtime failed; falling back to eager: {reason}")

    def set_guidance(self, pose_cfg: float | None = None, id_cfg: float | None = None) -> None:
        if pose_cfg is not None:
            self.pose_cfg_scale = _clip_cfg(pose_cfg)
            self.cfg_scale = self.pose_cfg_scale
        if id_cfg is not None:
            self.id_cfg_scale = _clip_cfg(id_cfg)

    def _resolve_generate_settings(
        self, num_steps: int | None
    ) -> tuple[int, float, float]:
        """Apply Fast TF32. Fast stays on the 1–2 step joint (no 3-way CFG)."""
        steps = int(num_steps if num_steps is not None else self.num_steps)
        pose_cfg = float(self.pose_cfg_scale)
        id_cfg = float(self.id_cfg_scale)
        if self.fast_mode:
            _enable_tf32()
            if STREAM_FAST_DISABLE_CFG:
                pose_cfg = 1.0
                id_cfg = 1.0
        return steps, pose_cfg, id_cfg

    def load(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        _enable_tf32()

        print(f"Loading keypoint DiT from {self.checkpoint} ...")
        self.model, cfg = build_keypoint_model(
            self.checkpoint, self.device, dtype=torch.float32
        )
        self._eager_model = self.model
        self._model_compiled = False
        self._compile_failed = False
        self._compile_mode_active = None
        self._cfg = dict(cfg)
        if not bool(cfg.get("use_keypoint_conditioning", False)):
            raise RuntimeError(
                f"{self.checkpoint.name} is not keypoint-conditioned; "
                "VTM Noble requires use_keypoint_conditioning=True"
            )
        from .model_layout import keypoint_layout_from_config

        self.keypoint_layout = keypoint_layout_from_config(cfg)
        print(
            f"[pose-diag] checkpoint keypoint layout: {self.keypoint_layout} "
            f"({self.checkpoint.name})"
        )
        self.image_size = int(cfg.get("image_resolution", IMAGE_SIZE))

        if self.vae is None:
            print("Loading SD-VAE ...")
            vae_dtype = preferred_sd_vae_dtype(self.device)
            self.vae = load_sd_vae(self.device, dtype=vae_dtype)
            print(f"SD-VAE ready (dtype={str(vae_dtype).replace('torch.', '')})")

        self._ready = True
        self._gpu_resident = True
        print(
            f"Stream model ready ({checkpoint_label(self.checkpoint)} @ {self.image_size}, "
            "dtype=float32)."
        )

    def _release_dit_weights(self) -> None:
        """Drop compiled DiT + CUDA graphs so a different checkpoint can load."""
        try:
            self._restore_eager_model()
        except Exception:
            pass
        self.model = None
        self._eager_model = None
        self._model_compiled = False
        self.clear_last_gen_latent()
        self._compile_mode_active = None
        self._compile_verified = False
        self._ready = False
        self._gpu_resident = False
        try:
            import torch._dynamo as dynamo

            dynamo.reset()
        except Exception:
            pass
        gc.collect()
        _clear_cuda_errors()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def offload_to_cpu(self) -> bool:
        """Move DiT / VAE / latents to RAM. Stop stream uses this to free VRAM.

        Weights stay loaded (``_ready``) so the next Start does not re-read disk.
        Compiled CUDA graphs cannot follow, so the next GPU pass re-wraps
        torch.compile (inductor cache keeps that recapture short).
        """
        if self.device.type != "cuda":
            return False
        with self._cuda_lock:
            if not self._gpu_resident:
                return True
            try:
                self._restore_eager_model()
            except Exception:
                pass
            cpu = torch.device("cpu")
            self.model = _mod_to(self.model, cpu)
            self._eager_model = self.model
            self.vae = _mod_to(self.vae, cpu)
            self.vae_tiny = _mod_to(self.vae_tiny, cpu)
            self._ref_latent = _ten_to(self._ref_latent, cpu)
            self._ref_face_latent = _ten_to(self._ref_face_latent, cpu)
            self._last_gen_latent = _ten_to(self._last_gen_latent, cpu)
            self._decode_stream = None
            self._gpu_resident = False
            self._model_compiled = False
            self._compile_verified = False
            gc.collect()
            _clear_cuda_errors()
            print("[engine] models offloaded to CPU")
            return True

    def ensure_gpu(self) -> None:
        """Put weights back on ``self.device`` after ``offload_to_cpu``."""
        if self.device.type != "cuda":
            self._gpu_resident = bool(self.model is not None or self.vae is not None)
            return
        if self._gpu_resident:
            return
        if self.model is None and self.vae is None:
            return
        with self._cuda_lock:
            if self._gpu_resident:
                return
            print("[engine] moving models back to GPU …")
            self.model = _mod_to(self.model, self.device)
            self._eager_model = self.model
            self.vae = _mod_to(self.vae, self.device)
            self.vae_tiny = _mod_to(self.vae_tiny, self.device)
            self._ref_latent = _ten_to(self._ref_latent, self.device)
            self._ref_face_latent = _ten_to(self._ref_face_latent, self.device)
            self._last_gen_latent = _ten_to(self._last_gen_latent, self.device)
            self._gpu_resident = True
            self._model_compiled = False
            self._compile_verified = False

    def set_checkpoint(self, path: Path | str) -> Path:
        """Swap DiT weights. Keeps VAE; re-encodes reference if present."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with self._cuda_lock:
            if self.checkpoint == path and self._ready and self.model is not None:
                return path
            previous = self.checkpoint
            print(f"Switching model -> {path.name} ...")
            self._release_dit_weights()
            self._compile_failed = False
            self.checkpoint = path
            try:
                self.load()
            except Exception as exc:
                if previous != path and previous.is_file():
                    print(f"Reload previous checkpoint after failed swap: {previous.name}")
                    self.checkpoint = previous
                    try:
                        self.load()
                    except Exception:
                        raise exc
                    raise RuntimeError(
                        f"Could not load {path.name} ({exc}). Restored {previous.name}."
                    ) from exc
                raise
            # A .vtm is a character pack, not an image. The desk reloads it
            # after this returns so Pillow never opens the zip.
            ref = self._ref_path
            if (
                ref is not None
                and self._ref_keypoints is not None
                and Path(ref).suffix.lower() != ".vtm"
            ):
                self._set_reference_locked(ref, self._ref_keypoints)
            return path

    def checkpoint_name(self) -> str:
        return checkpoint_label(self.checkpoint)

    def set_reference(
        self,
        path: Path | str,
        keypoints: np.ndarray | Path | str | None = None,
        *,
        skip_crop: bool | None = None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> Path:
        """Encode and cache a reference image + pose labels.

        Uses sidecar keypoints when present; otherwise a neutral template so
        plain greenscreen refs still work with live deltas / Test Skeleton.

        Face-fit runs **outside** ``_cuda_lock`` (CPU) so it cannot deadlock
        against DiT generate / Fast warmup. Only the VAE encode is locked.

        ``on_progress(fraction, label)`` is optional UI feedback (0..1).
        """
        def _tick(frac: float, label: str) -> None:
            if on_progress is not None:
                try:
                    on_progress(float(frac), str(label))
                except Exception:
                    pass

        if not self._ready:
            with self._cuda_lock:
                if not self._ready:
                    self.load()

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Reference image not found: {path}")

        _tick(0.08, "Reading reference…")
        arr = _reference_rgb(path)
        if skip_crop is None:
            in_train_crop = "train_crop" in str(path).replace("\\", "/")
            skip_crop = not (in_train_crop or looks_like_greenscreen(arr))

        # Keypoint resolve / face-fit off the CUDA lock (defaults to CPU).
        print(f"Resolving reference keypoints for {path.name} …")
        _tick(0.22, "Fitting pose…")
        kps, kps_path, pose_source = resolve_ref_keypoints(
            path, keypoints, skip_crop=bool(skip_crop), fit_device="cpu"
        )

        print(f"Waiting for GPU lock to encode reference {path.name} …")
        _tick(0.55, "Encoding reference…")
        with self._cuda_lock:
            out = self._set_reference_locked(
                path,
                kps,
                kps_path=kps_path,
                pose_source=pose_source,
                skip_crop=bool(skip_crop),
                image_arr=arr,
            )
        _tick(0.92, "Building preview…")
        return out

    def _set_reference_locked(
        self,
        path: Path | str,
        keypoints: np.ndarray | Path | str | None = None,
        *,
        skip_crop: bool | None = None,
        kps_path: Path | None = None,
        pose_source: str | None = None,
        image_arr: np.ndarray | None = None,
    ) -> Path:
        if not self._ready:
            self.load()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Reference image not found: {path}")

        if image_arr is None:
            arr = _reference_rgb(path)
        else:
            arr = np.asarray(image_arr)
        if skip_crop is None:
            in_train_crop = "train_crop" in str(path).replace("\\", "/")
            skip_crop = not (in_train_crop or looks_like_greenscreen(arr))

        if isinstance(keypoints, np.ndarray):
            kps = sanitize_normalized_keypoints(np.asarray(keypoints, dtype=np.float32))
            if pose_source is None:
                pose_source = "explicit"
        else:
            kps, kps_path, pose_source = resolve_ref_keypoints(
                path, keypoints, skip_crop=bool(skip_crop), fit_device="cpu"
            )
        used_neutral = pose_source == "neutral"

        if (
            self._ref_path == path
            and self._ref_latent is not None
            and self._ref_keypoints is not None
            and self._ref_keypoints_model is not None
            and np.allclose(self._ref_keypoints, kps, atol=1e-5)
        ):
            if self._ref_rig is None:
                from .live_retarget import build_reference_rig

                self._ref_rig = build_reference_rig(kps)
            if self._ref_keypoints_session_base is None:
                self._ref_keypoints_session_base = kps.copy()
            self._ref_neutral_fallback = used_neutral
            self._ref_pose_source = pose_source
            self._ref_skip_crop = bool(skip_crop)
            return path

        print(f"Encoding reference {path} ...")
        if kps_path is not None:
            print(f"Ref keypoints from {kps_path}")
        elif pose_source == "fitted":
            print("Ref keypoints: FITTED from face+iris+dwpose_v2 on reference image")
        elif used_neutral:
            print("Ref keypoints: NEUTRAL FALLBACK (no sidecar / fit failed)")
        print("Ref keypoints (KEYPOINT_SCHEMA):\n" + format_keypoints_table(kps, max_rows=8))
        from .pose_controller import format_body_diagnostics

        print(format_body_diagnostics(kps, label="set_reference"))
        if used_neutral:
            print(
                "[pose-diag] WARNING: no sidecar — body/face template is generic; "
                "mesh will not match this character's labeled proportions."
            )
        use_face = bool(self._cfg.get("use_ref_face_tokens", True))
        face_size = int(self._cfg.get("ref_face_size", 32))
        kps_model = keypoints_for_model(kps, self.keypoint_layout)
        print(
            f"[pose-diag] model keypoint layout: {self.keypoint_layout} "
            "(runtime mesh remains KEYPOINT_SCHEMA)"
        )

        _clear_cuda_errors()
        print("VAE encode starting…")
        with torch.inference_mode():
            ref_latent, ref_face, _ref_kps_t = encode_reference(
                self.vae,
                arr,
                kps_model,
                image_size=self.image_size,
                ref_face_size=face_size,
                use_ref_face_tokens=use_face,
                skip_crop=bool(skip_crop),
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        print("VAE encode done.")
        if self.model is not None:
            dtype = next(self.model.parameters()).dtype
            ref_latent = ref_latent.to(device=self.device, dtype=dtype)
            if ref_face is not None:
                ref_face = ref_face.to(device=self.device, dtype=dtype)

        self._ref_latent = ref_latent
        self._ref_face_latent = ref_face
        self.clear_last_gen_latent()
        # Keep the training-format labels we loaded (already normalized). Do not
        # replace them with encode_reference's tensor after a bad pixel remap.
        self._ref_keypoints = kps
        self._ref_keypoints_model = kps_model
        self._ref_keypoints_session_base = kps.copy()
        from .live_retarget import build_reference_rig

        self._ref_rig = build_reference_rig(kps)
        self._ref_path = path
        self._ref_kps_path = kps_path
        self._ref_neutral_fallback = used_neutral
        self._ref_pose_source = pose_source
        self._ref_skip_crop = bool(skip_crop)
        self._last_driven_body = None
        print(
            f"Reference ready (skip_crop={bool(skip_crop)}, face_tokens={ref_face is not None}, "
            f"pose_source={pose_source})"
        )
        return path

    def export_encoded_reference(self) -> dict[str, Any]:
        """CPU copies of the current encoded reference (for ``.vtm`` write)."""
        if self._ref_latent is None or self._ref_keypoints is None:
            raise RuntimeError("No encoded reference to export")
        face = None
        if self._ref_face_latent is not None:
            face = (
                self._ref_face_latent.detach().float().cpu().numpy().astype(np.float16)
            )
        return {
            "ref_latent": self._ref_latent.detach().float().cpu().numpy().astype(np.float16),
            "ref_face_latent": face,
            "keypoints": np.asarray(self._ref_keypoints, dtype=np.float32).copy(),
            "image_size": int(self.image_size),
            "skip_crop": bool(self._ref_skip_crop),
        }

    def load_encoded_reference(
        self,
        *,
        keypoints: np.ndarray,
        ref_latent: np.ndarray,
        ref_face_latent: np.ndarray | None,
        skip_crop: bool,
        path: Path | str,
        pose_source: str = "character_pack",
    ) -> Path:
        """Install a pre-encoded reference without running the VAE."""
        if not self._ready:
            with self._cuda_lock:
                if not self._ready:
                    self.load()
        dest = Path(path)
        kps = sanitize_normalized_keypoints(np.asarray(keypoints, dtype=np.float32))
        kps_model = keypoints_for_model(kps, self.keypoint_layout)
        latent = torch.from_numpy(np.asarray(ref_latent, dtype=np.float32))
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        face = None
        if ref_face_latent is not None:
            face = torch.from_numpy(np.asarray(ref_face_latent, dtype=np.float32))
            if face.ndim == 3:
                face = face.unsqueeze(0)
        if self.model is not None:
            dtype = next(self.model.parameters()).dtype
            latent = latent.to(device=self.device, dtype=dtype)
            if face is not None:
                face = face.to(device=self.device, dtype=dtype)
        else:
            latent = latent.to(device=self.device)
            if face is not None:
                face = face.to(device=self.device)
        self._ref_latent = latent
        self._ref_face_latent = face
        self.clear_last_gen_latent()
        self._ref_keypoints = kps
        self._ref_keypoints_model = kps_model
        self._ref_keypoints_session_base = kps.copy()
        from .live_retarget import build_reference_rig

        self._ref_rig = build_reference_rig(kps)
        self._ref_path = dest
        self._ref_kps_path = None
        self._ref_neutral_fallback = False
        self._ref_pose_source = str(pose_source or "character_pack")
        self._ref_skip_crop = bool(skip_crop)
        self._last_driven_body = None
        print(
            f"Reference loaded from pack (skip_crop={bool(skip_crop)}, "
            f"face_tokens={face is not None}, pose_source={self._ref_pose_source})"
        )
        return dest

    def adopt_ref_keypoints(
        self,
        keypoints: np.ndarray,
        *,
        persist: bool = True,
        pose_source: str = "manual",
    ) -> Path | None:
        """Replace the character rest pose without re-encoding the identity VAE.

        Used after a mesh drag so the next live retarget starts from the edited
        layout. Writes ``<ref_stem>_keypoints.npy`` when ``persist`` is set.
        """
        kps = _as_keypoints37(keypoints).copy()
        self._ref_keypoints = kps
        self._ref_keypoints_model = keypoints_for_model(kps, self.keypoint_layout)
        from .live_retarget import build_reference_rig

        self._ref_rig = build_reference_rig(kps)
        saved: Path | None = None
        if persist and self._ref_path is not None and self._ref_path.suffix.lower() != ".vtm":
            saved = save_sidecar_keypoints(self._ref_path, kps)
            self._ref_kps_path = saved
            self._ref_pose_source = str(pose_source or "manual")
            print(f"Saved mesh edits → {saved}")
        return saved

    def restore_session_ref_keypoints(self) -> np.ndarray:
        """Undo manual mesh edits back to the rest pose from apply/calibrate."""
        base = self._ref_keypoints_session_base
        if base is None:
            if self._ref_keypoints is None:
                raise RuntimeError("No reference keypoints to restore")
            return np.asarray(self._ref_keypoints, dtype=np.float32).copy()
        self.adopt_ref_keypoints(base, persist=True, pose_source="reset")
        assert self._ref_keypoints is not None
        return np.asarray(self._ref_keypoints, dtype=np.float32).copy()

    def calibrate_reference(
        self,
        path: Path | str | None = None,
        *,
        flip_tta: bool = True,
    ) -> np.ndarray:
        """Re-fit face/body landmarks on the reference still (not the camera).

        Runs detect → horizontal flip → detect → flip landmarks back → merge.
        Ignores any sidecar so the mesh matches the current character image.

        Detector fit runs **outside** ``_cuda_lock`` (and on CPU) so LivePoser /
        Fast warmup cannot deadlock the GPU; only VAE encode is locked.
        """
        from .ref_pose_fit import fit_keypoints_to_reference

        if not self._ready:
            with self._cuda_lock:
                if not self._ready:
                    self.load()
        ref_path = Path(path) if path is not None else self._ref_path
        if ref_path is None or not Path(ref_path).is_file():
            raise FileNotFoundError("No reference image to calibrate — pick one first.")
        ref_path = Path(ref_path)

        arr = _reference_rgb(ref_path)
        in_train_crop = "train_crop" in str(ref_path).replace("\\", "/")
        skip_crop = not (in_train_crop or looks_like_greenscreen(arr))

        print(
            f"[calibrate] Refitting pose on {ref_path.name} "
            f"(flip_tta={flip_tta}, skip_crop={skip_crop}, device=cpu) …"
        )
        # CPU avoids racing LivePoser iris YOLO / DiT on the same CUDA context.
        fitted = fit_keypoints_to_reference(
            None if ref_path.suffix.lower() == ".vtm" else ref_path,
            image_rgb=arr,
            skip_crop=bool(skip_crop),
            image_size=self.image_size,
            flip_tta=bool(flip_tta),
            device="cpu",
        )
        # Fit already ran collapse repair once — do not rewrite mouth again.
        kps = sanitize_normalized_keypoints(
            fitted, repair_mouth=False, repair_nose=False
        )
        with self._cuda_lock:
            # Force encode even if path matches (keypoints changed).
            self._ref_path = None
            self._set_reference_locked(
                ref_path, kps, skip_crop=bool(skip_crop), image_arr=arr
            )
            self._ref_pose_source = "calibrated"
            saved = save_sidecar_keypoints(ref_path, kps)
            self._ref_kps_path = saved
        gap = None
        if kps[21, 3] >= 0.5 and kps[25, 3] >= 0.5:
            gap = float(kps[25, 1] - kps[21, 1])
        if gap is not None:
            print(f"[calibrate] Done — mouth gap={gap:.4f}")
        else:
            print("[calibrate] Done")
        return kps

    def plan_warmup_stages(
        self,
        *,
        batch_size: int | None = None,
        include_compile: bool | None = None,
    ) -> list[str]:
        """Named stages for a single ``warmup`` call (for UI progress)."""
        warm_batch = int(
            batch_size if batch_size is not None else self.stream_batch_size
        )
        warm_batch = max(1, min(warm_batch, STREAM_BATCH_MAX))
        stages: list[str] = []
        if self.fast_mode and STREAM_FAST_TINY_VAE and self.vae_tiny is None:
            stages.append("TinyVAE")
        do_compile = (
            self.fast_mode
            if include_compile is None
            else bool(include_compile)
        )
        if (
            do_compile
            and self.compile_model
            and self.device.type == "cuda"
            and not self._model_compiled
            and not self._compile_failed
        ):
            stages.append("Compiling…")
        # Denoise runs: compile path uses STREAM_COMPILE_WARMUP_RUNS; else 1.
        will_compile = (
            do_compile
            and self.compile_model
            and self.device.type == "cuda"
            and not self._compile_failed
        )
        runs = (
            max(1, int(STREAM_COMPILE_WARMUP_RUNS))
            if (self._model_compiled or will_compile)
            else 1
        )
        for i in range(runs):
            stages.append(f"Warmup {i + 1}/{runs}")
        stages.append("Decode warmup…")
        return stages

    def warmup(
        self,
        num_steps: int | None = None,
        *,
        batch_size: int | None = None,
        on_progress=None,
    ) -> None:
        """Compile (optional) + prime kernels for Fast path.

        ``batch_size`` primes a specific DiT batch (needed when Batch×2 is on
        and torch.compile was captured at batch=1).

        ``on_progress(done, total, label)`` reports completed stage count.
        """
        warm_batch = int(
            batch_size if batch_size is not None else self.stream_batch_size
        )
        warm_batch = max(1, min(warm_batch, STREAM_BATCH_MAX))
        stages = self.plan_warmup_stages(batch_size=warm_batch)
        total = max(1, len(stages))
        done = 0

        def _tick(label: str) -> None:
            nonlocal done
            done = min(done + 1, total)
            if on_progress is not None:
                on_progress(done, total, label)

        with self._cuda_lock:
            if not self._ready or self._ref_latent is None or self.model is None:
                return
            if self._ref_keypoints is None:
                return
            if self._ref_keypoints_model is None:
                self._ref_keypoints_model = keypoints_for_model(
                    self._ref_keypoints, self.keypoint_layout
                )
            if self.fast_mode and "TinyVAE" in stages:
                self._ensure_tiny_vae()
                _tick("TinyVAE")
            elif self.fast_mode:
                self._ensure_tiny_vae()

        # Compile outside the lock — inductor can take a long time and was
        # blocking Apply ref ("Encoding reference…") the whole time.
        if self.fast_mode:
            if "Compiling…" in stages:
                if on_progress is not None:
                    on_progress(float(done), total, "Compiling… please wait")
                stop_hb = threading.Event()

                def _compile_heartbeat() -> None:
                    t0 = time.perf_counter()
                    while not stop_hb.wait(0.4):
                        if on_progress is None:
                            continue
                        # Asymptote toward the end of this stage so the bar
                        # keeps moving during the long inductor compile.
                        elapsed = time.perf_counter() - t0
                        soft = float(done) + min(0.92, 1.0 - math.exp(-elapsed / 22.0))
                        on_progress(soft, total, "Compiling… please wait")

                hb = threading.Thread(
                    target=_compile_heartbeat, name="rs-compile-hb", daemon=True
                )
                hb.start()
                try:
                    self._maybe_compile_model(STREAM_COMPILE_MODE)
                finally:
                    stop_hb.set()
                    hb.join(timeout=1.5)
                _tick("Compiling…")
            else:
                self._maybe_compile_model(STREAM_COMPILE_MODE)

        with self._cuda_lock:
            if not self._ready or self._ref_latent is None or self.model is None:
                return
            if self._ref_keypoints is None:
                return
            steps, pose_cfg, id_cfg = self._resolve_generate_settings(num_steps)
            warm_steps = max(1, min(steps, 4))

            def _run_denoise_warmups(
                runs: int, *, batch_size: int = 1
            ) -> torch.Tensor | None:
                latents = None
                bsz = max(1, min(int(batch_size), STREAM_BATCH_MAX))
                kps_target = self._ref_keypoints_model
                if bsz > 1:
                    kps_target = np.stack([kps_target] * bsz, axis=0)
                with torch.inference_mode():
                    for i in range(runs):
                        label = f"Warmup {i + 1}/{runs}"
                        if on_progress is not None:
                            on_progress(done, total, label)
                        latents = denoise_keypoint(
                            self.model,
                            keypoints_target=kps_target,
                            ref_latent=self._ref_latent,
                            ref_keypoints=self._ref_keypoints_model,
                            ref_face_latent=self._ref_face_latent,
                            num_steps=warm_steps,
                            pose_cfg_scale=pose_cfg,
                            id_cfg_scale=id_cfg,
                            inference_timestep_shift=INFERENCE_TIMESTEP_SHIFT,
                            seed=STREAM_FIXED_SEED,
                            shared_noise=True,
                        )
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        if self._model_compiled:
                            print(f"compile warmup {i + 1}/{runs} (batch={bsz})")
                        _tick(label)
                return latents

            def _try_compiled_warmups(bsz: int) -> torch.Tensor | None:
                runs = (
                    max(1, int(STREAM_COMPILE_WARMUP_RUNS))
                    if self._model_compiled
                    else 1
                )
                return _run_denoise_warmups(runs, batch_size=bsz)

            latents: torch.Tensor | None = None
            try:
                latents = _try_compiled_warmups(warm_batch)
            except Exception as exc:
                if not (self._model_compiled and _is_compile_runtime_error(exc)):
                    raise
                failed_mode = self._compile_mode_active or STREAM_COMPILE_MODE
                print(f"compile mode '{failed_mode}' failed at warmup: {exc}")
                self._restore_eager_model()
                _clear_cuda_errors()

                ladder = [
                    m
                    for m in STREAM_COMPILE_MODE_LADDER
                    if m != failed_mode
                ]
                recovered = False
                for mode in ladder:
                    self._compile_failed = False
                    if on_progress is not None:
                        on_progress(done, total, f"Compiling ({mode})…")
                    if not self._maybe_compile_model(mode):
                        continue
                    try:
                        latents = _try_compiled_warmups(warm_batch)
                        recovered = True
                        break
                    except Exception as exc2:
                        if not _is_compile_runtime_error(exc2):
                            raise
                        print(f"compile mode '{mode}' failed at warmup: {exc2}")
                        self._restore_eager_model()
                        _clear_cuda_errors()

                if not recovered:
                    self._fallback_eager_model(exc)
                    latents = _run_denoise_warmups(1, batch_size=warm_batch)

            if latents is not None:
                with torch.inference_mode():
                    if on_progress is not None:
                        on_progress(done, total, "Decode warmup…")
                    _images, decode_kind = self._decode_latents(latents)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    print(f"decode warmup ({decode_kind}, batch={warm_batch})")
                    _tick("Decode warmup…")

            # One-shot compile verification so the UI green light is truthful.
            if self.fast_mode and self.compile_model and self.device.type == "cuda":
                if on_progress is not None:
                    on_progress(done, total, "Compile test…")
                ok = self.verify_compile()
                if ok:
                    print("[compile] Fast torch.compile verified — green light")
                else:
                    print(
                        f"[compile] Fast torch.compile not active "
                        f"(status={self.compile_status})"
                    )
                if on_progress is not None:
                    on_progress(total, total, "Ready" if ok else "Compile off")
            elif on_progress is not None and done < total:
                on_progress(total, total, "Ready")

    def generate_from_keypoints(
        self,
        keypoints: np.ndarray | torch.Tensor,
        num_steps: int | None = None,
        *,
        sanitize: bool | str = True,
        hair_maps: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[Image.Image, float]:
        with self._cuda_lock:
            return self._generate_from_keypoints_locked(
                keypoints, num_steps, sanitize=sanitize, hair_maps=hair_maps
            )

    def denoise_to_latents(
        self,
        keypoints: np.ndarray | torch.Tensor,
        num_steps: int | None = None,
        *,
        sanitize: bool | str = True,
        hair_maps: np.ndarray | torch.Tensor | None = None,
    ) -> dict:
        """Pipeline stage 1: pose prep + DiT denoise only (holds the CUDA lock).

        Returns a dict with ``latents`` plus timing/meta so the caller can
        decode later (possibly overlapped with the next frame's denoise) via
        :meth:`decode_latents_to_image`.
        """
        with self._cuda_lock:
            return self._denoise_to_latents_locked(
                keypoints, num_steps, sanitize=sanitize, hair_maps=hair_maps
            )

    def _ensure_decode_stream(self) -> "torch.cuda.Stream | None":
        if self.device.type != "cuda":
            return None
        if self._decode_stream is None:
            self._decode_stream = torch.cuda.Stream()
        return self._decode_stream

    def decode_latents_to_images(
        self, latents: torch.Tensor
    ) -> tuple[list[Image.Image], float, str]:
        """Pipeline stage 2: VAE decode on a side CUDA stream (batch-aware).

        Does NOT take ``_cuda_lock`` so it can run while the next frame's
        denoise holds it. ``denoise_to_latents`` synchronizes before returning,
        so ``latents`` are always complete by the time we get here.
        """
        t0 = time.perf_counter()
        with self._decode_lock:
            stream = self._ensure_decode_stream()
            with torch.inference_mode():
                if stream is not None:
                    latents.record_stream(stream)
                    with torch.cuda.stream(stream):
                        images, decode_kind = self._decode_latents(latents)
                    stream.synchronize()
                else:
                    images, decode_kind = self._decode_latents(latents)
        decode_s = time.perf_counter() - t0
        self.last_decode_backend = decode_kind
        arr = np.asarray(images)
        if arr.ndim == 3:
            arr = arr[None, ...]
        return [Image.fromarray(arr[i]) for i in range(arr.shape[0])], decode_s, decode_kind

    def decode_latents_to_image(
        self, latents: torch.Tensor
    ) -> tuple[Image.Image, float, str]:
        """Decode and return the first image (single-frame convenience)."""
        images, decode_s, decode_kind = self.decode_latents_to_images(latents)
        return images[0], decode_s, decode_kind

    def _generate_from_keypoints_locked(
        self,
        keypoints: np.ndarray | torch.Tensor,
        num_steps: int | None = None,
        *,
        sanitize: bool | str = True,
        hair_maps: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[Image.Image, float]:
        images, elapsed = self._generate_batch_from_keypoints_locked(
            keypoints, num_steps, sanitize=sanitize, hair_maps=hair_maps
        )
        return images[0], elapsed

    def generate_batch_from_keypoints(
        self,
        keypoints: np.ndarray | torch.Tensor,
        num_steps: int | None = None,
        *,
        sanitize: bool | str = True,
        hair_maps: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[list[Image.Image], float]:
        """Denoise + decode one or more poses; returns every image in the batch."""
        with self._cuda_lock:
            return self._generate_batch_from_keypoints_locked(
                keypoints, num_steps, sanitize=sanitize, hair_maps=hair_maps
            )

    def _generate_batch_from_keypoints_locked(
        self,
        keypoints: np.ndarray | torch.Tensor,
        num_steps: int | None = None,
        *,
        sanitize: bool | str = True,
        hair_maps: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[list[Image.Image], float]:
        result = self._denoise_to_latents_locked(
            keypoints, num_steps, sanitize=sanitize, hair_maps=hair_maps
        )

        t1 = time.perf_counter()
        with torch.inference_mode():
            images_arr, decode_kind = self._decode_latents(result["latents"])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        decode_s = time.perf_counter() - t1

        elapsed = time.perf_counter() - result["start"]
        self.last_decode_backend = decode_kind
        arr = np.asarray(images_arr)
        if arr.ndim == 3:
            arr = arr[None, ...]
        images = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        self.last_timings = {
            "total": elapsed,
            "denoise": result["denoise_s"],
            "decode": decode_s,
            "pipeline": "keypoint_fast" if self.fast_mode else "keypoint",
            "fast": 1.0 if self.fast_mode else 0.0,
            "steps": float(result["steps"]),
            "pose_cfg": float(result["pose_cfg"]),
            "id_cfg": float(result["id_cfg"]),
            "batch": float(len(images)),
            "fps": (len(images) / elapsed) if elapsed > 0 else 0.0,
            "hold_last": 1.0 if result.get("hold_last") else 0.0,
        }
        return images, elapsed

    def _prepare_target_keypoints_batch(
        self,
        keypoints: np.ndarray | torch.Tensor,
        *,
        sanitize: bool | str = True,
    ) -> np.ndarray:
        """Sanitize one or more poses → ``(B, 37, 4)`` in KEYPOINT_SCHEMA."""
        if isinstance(keypoints, torch.Tensor):
            kps = keypoints.detach().float().cpu().numpy()
        else:
            kps = np.asarray(keypoints, dtype=np.float32)
        if kps.ndim == 2:
            kps_batch = kps[None, ...]
        elif kps.ndim == 3:
            kps_batch = kps
        else:
            raise ValueError(f"Expected keypoints (37,4) or (B,37,4), got {kps.shape}")
        if kps_batch.shape[1:] != (NUM_KEYPOINTS, KEYPOINT_DIM):
            raise ValueError(
                f"Expected keypoints (B,{NUM_KEYPOINTS},{KEYPOINT_DIM}), got {kps_batch.shape}"
            )
        if kps_batch.shape[0] < 1 or kps_batch.shape[0] > STREAM_BATCH_MAX:
            raise ValueError(
                f"Batch size must be 1..{STREAM_BATCH_MAX}, got {kps_batch.shape[0]}"
            )

        from .pose_controller import block_model_slots, format_body_diagnostics, sanitize_pose

        sanitize_mode = (
            str(sanitize).strip().lower()
            if isinstance(sanitize, str)
            else ("full" if sanitize else "none")
        )
        if sanitize_mode not in {"full", "constrained", "none"}:
            raise ValueError(f"Unknown target sanitize mode: {sanitize!r}")

        body_tracked = bool(
            getattr(self, "_body_skel_method", None)
            and not getattr(self, "_body_lost", False)
        )
        try:
            from .live_poser_client import is_body_tracked as _is_body_tracked

            body_tracked = _is_body_tracked(
                getattr(self, "_body_skel_method", "unknown")
            ) and not bool(getattr(self, "_body_lost", False))
        except Exception:
            body_tracked = False

        bsz = int(kps_batch.shape[0])
        # Verbose pose dumps only for single-frame; batch path stays quiet.
        verbose = bsz == 1
        if verbose:
            print(
                f"[pose-diag] generate: sanitize={sanitize_mode} "
                f"ref_neutral={self._ref_neutral_fallback} "
                f"ref={self._ref_path.name if self._ref_path else None} "
                f"sidecar={self._ref_kps_path.name if self._ref_kps_path else None}"
            )
        else:
            print(f"[pose-diag] generate: batch={bsz} sanitize={sanitize_mode}")

        out = np.empty_like(kps_batch)
        for i in range(bsz):
            # Driven expression geometry is intentional (especially a closed
            # mouth). Collapse repair is for imported reference labels, not targets.
            one = sanitize_normalized_keypoints(
                kps_batch[i], repair_mouth=False, repair_nose=False
            )
            if verbose:
                print(format_body_diagnostics(one, label="generate/IN"))
            if sanitize_mode != "none":
                if sanitize_mode == "constrained":
                    # Retarget already applied the reference-local face envelope.
                    # Validate visibility and reference body lengths without a
                    # second travel/topology pass changing the intended expression.
                    one = sanitize_pose(
                        one,
                        self._ref_keypoints,
                        recenter=False,
                        clamp_travel=False,
                        topology=False,
                        log=verbose,
                        log_label="generate/constrained",
                        body_tracked=body_tracked,
                        lock_proportions=True,
                        lock_face_proportions=False,
                    )
                else:
                    one = sanitize_pose(
                        one,
                        self._ref_keypoints,
                        log=verbose,
                        log_label="generate",
                        body_tracked=body_tracked,
                    )
            elif verbose:
                print("[pose-diag] generate: SKIP sanitize")
                print(format_body_diagnostics(one, label="generate/RAW"))
            out[i] = block_model_slots(one)

        if verbose:
            print(
                "Generating with KEYPOINT_SCHEMA target (no text prompt):\n"
                + format_keypoints_table(out[0], max_rows=12)
            )
            print(format_body_diagnostics(out[0], label="generate/SENT_TO_MODEL"))
        return out

    def _denoise_to_latents_locked(
        self,
        keypoints: np.ndarray | torch.Tensor,
        num_steps: int | None = None,
        *,
        sanitize: bool | str = True,
        hair_maps: np.ndarray | torch.Tensor | None = None,
    ) -> dict:
        if not self._ready:
            self.load()
        if self._ref_latent is None or self._ref_keypoints is None:
            raise RuntimeError("Set a reference image (+ keypoints) before generating.")

        kps_batch = self._prepare_target_keypoints_batch(keypoints, sanitize=sanitize)
        self.last_target_keypoints = kps_batch[-1].copy()
        self.last_target_keypoints_batch = kps_batch.copy()
        kps_model = np.stack(
            [
                keypoints_for_model(kps_batch[i], self.keypoint_layout)
                for i in range(kps_batch.shape[0])
            ],
            axis=0,
        )
        if self._ref_keypoints_model is None:
            self._ref_keypoints_model = keypoints_for_model(
                self._ref_keypoints, self.keypoint_layout
            )

        steps, pose_cfg, id_cfg = self._resolve_generate_settings(num_steps)

        if self.fast_mode:
            print(
                f"[fast] batch={kps_batch.shape[0]} steps={steps} "
                f"pose_cfg={pose_cfg:.2f} id_cfg={id_cfg:.2f} "
                f"({self.fast_status})"
            )

        start = time.perf_counter()
        hold_last = bool(self.hold_last)
        now_kps = kps_batch[-1]
        prev = self._last_gen_latent if hold_last else None
        start_t = 0.0
        if prev is not None:
            move = face_pose_delta(self._last_hold_kps, now_kps)
            drift = face_pose_delta(self._ref_keypoints, now_kps)
            start_t, pull = hold_plan(move, drift)
            if start_t <= 1e-4:
                prev = None
                start_t = 0.0
            else:
                prev = anchor_hold_latent(prev, self._ref_latent, pull=pull)
        with torch.inference_mode():
            t0 = time.perf_counter()
            latents = denoise_keypoint(
                self.model,
                keypoints_target=kps_model,
                ref_latent=self._ref_latent,
                ref_keypoints=self._ref_keypoints_model,
                ref_face_latent=self._ref_face_latent,
                num_steps=steps,
                pose_cfg_scale=pose_cfg,
                id_cfg_scale=id_cfg,
                inference_timestep_shift=INFERENCE_TIMESTEP_SHIFT,
                seed=STREAM_FIXED_SEED,
                # Same noise for every batch item — otherwise Batch×2 A/B frames
                # look like two different samples and flicker hard on display.
                shared_noise=True,
                hair_maps=hair_maps,
                last_latent=prev,
                start_t=start_t,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            denoise_s = time.perf_counter() - t0
        # Clone so decode (side CUDA stream) does not race the next mix.
        self._last_gen_latent = latents[-1:].detach().clone()
        self._last_hold_kps = now_kps.copy()

        return {
            "latents": latents,
            "denoise_s": denoise_s,
            "start": start,
            "steps": steps,
            "pose_cfg": pose_cfg,
            "id_cfg": id_cfg,
            "keypoints_used": kps_batch,
            "batch": int(kps_batch.shape[0]),
            "hold_last": bool(prev is not None),
        }
