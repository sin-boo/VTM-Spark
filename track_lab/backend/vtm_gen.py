"""Run VTM Spark generate from the current Track Lab overlay.

The harness host stays torch-free. This module is imported by the worker.
Parent ``backend`` is loaded as ``vtm_backend`` so it does not collide with
Track Lab's own ``backend`` package.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np

from harness.pack import _image_wh, frame_from_bench, pack_frame

NUM_KEYPOINTS = 37
_REPO = Path(__file__).resolve().parents[2]
_PKG = "vtm_backend"

_engine: Any = None
_ref_sig: tuple[str, int] | None = None
_jpeg: bytes | None = None


def last_jpeg() -> bytes | None:
    return _jpeg


def overlay_norm(frame: dict[str, Any] | None) -> np.ndarray | None:
    """Harness character_px keypoints → (37, 4) norm_crop for DiT."""
    if not isinstance(frame, dict):
        return None
    rows = frame.get("keypoints")
    if not isinstance(rows, list) or not rows:
        return None
    k = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)
    visible = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("i", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= NUM_KEYPOINTS:
            continue
        try:
            k[idx, 0] = float(row.get("x", 0.0))
            k[idx, 1] = float(row.get("y", 0.0))
            k[idx, 2] = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        on = bool(row.get("visible")) or float(k[idx, 2]) >= 0.5
        k[idx, 3] = 1.0 if on else 0.0
        if on:
            visible += 1
    if visible < 1:
        return None
    w, h = _frame_wh(frame)
    out = k.copy()
    out[:, 0] = k[:, 0] / float(w) * 2.0 - 1.0
    out[:, 1] = k[:, 1] / float(h) * 2.0 - 1.0
    return out


def hair_norm(frame: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Harness character_px hair polygons → norm_crop segments."""
    if not isinstance(frame, dict) or "hair" not in frame:
        return []
    raw = frame.get("hair")
    if not isinstance(raw, list):
        return []
    w, h = _frame_wh(frame)
    sx = 2.0 / float(w)
    sy = 2.0 / float(h)
    out: list[dict[str, Any]] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        cls = str(seg.get("class") or "")
        poly = seg.get("polygon") or []
        if not cls or not isinstance(poly, list) or len(poly) < 3:
            continue
        pts: list[list[float]] = []
        for vertex in poly:
            if not isinstance(vertex, (list, tuple)) or len(vertex) < 2:
                continue
            try:
                pts.append([float(vertex[0]) * sx - 1.0, float(vertex[1]) * sy - 1.0])
            except (TypeError, ValueError):
                continue
        if len(pts) < 3:
            continue
        out.append({"class": cls, "polygon": pts})
    return out


def generate(bench: Any, overlay: dict[str, Any] | None = None) -> dict[str, Any]:
    """Encode the still + current overlay and run one DiT sample."""
    global _jpeg

    src = _source()
    if not src.is_file():
        raise RuntimeError("Load a still first")
    frame = _frame_for_generate(bench, overlay)
    kps = overlay_norm(frame)
    if kps is None:
        raise RuntimeError("No overlay points — press Track, then Gen")
    engine = _engine_for(src)
    hair_maps = _hair_maps(hair_norm(frame))
    image, elapsed = engine.generate_from_keypoints(
        kps, sanitize="none", hair_maps=hair_maps
    )
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    _jpeg = buf.getvalue()
    payload = bench.status() if hasattr(bench, "status") else {}
    if not isinstance(payload, dict):
        payload = {}
    else:
        payload = dict(payload)
    payload["error"] = ""
    payload["gen"] = True
    payload["gen_ms"] = round(float(elapsed) * 1000.0, 1)
    payload["message"] = f"Generated in {elapsed:.1f}s"
    return payload


def _source() -> Path:
    from .face import source_path

    return source_path()


def _frame_for_generate(bench: Any, overlay: dict[str, Any] | None) -> dict[str, Any]:
    """Pack the live overlay, then replace it with on-screen geometry when given."""
    if not isinstance(overlay, dict) or not overlay:
        return frame_from_bench(bench)
    live = bench.live_status() if hasattr(bench, "live_status") else {}
    if not isinstance(live, dict):
        live = {}
    else:
        live = dict(live)
    for key in ("points", "hair", "skeleton", "iris"):
        if overlay.get(key) is not None:
            live[key] = overlay[key]
    return pack_frame(
        live,
        image_wh=_image_wh(bench),
        generation=int(getattr(bench, "generation", 0) or 0),
    )


def _frame_wh(frame: dict[str, Any]) -> tuple[int, int]:
    wh = frame.get("image_wh") if isinstance(frame.get("image_wh"), (list, tuple)) else ()
    w = int(wh[0]) if len(wh) > 0 else 0
    h = int(wh[1]) if len(wh) > 1 else 0
    return max(w, 1), max(h, 1)


def _ensure_vtm() -> None:
    if _PKG in sys.modules:
        return
    pkg_dir = _REPO / "backend"
    init = pkg_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        _PKG,
        init,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load VTM Spark backend")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PKG] = mod
    spec.loader.exec_module(mod)


def _engine_for(src: Path) -> Any:
    global _engine, _ref_sig
    _ensure_vtm()
    from vtm_backend.engine import StreamEngine, default_stream_checkpoint

    if _engine is None:
        ckpt = default_stream_checkpoint()
        if not ckpt.is_file():
            raise RuntimeError(f"Missing DiT weights at {ckpt}")
        _engine = StreamEngine(checkpoint=ckpt, compile_model=False)
    sig = (str(src.resolve()), int(src.stat().st_mtime_ns))
    if _ref_sig != sig:
        _engine.set_reference(src)
        _ref_sig = sig
    return _engine


def _hair_maps(segments: list[dict[str, Any]]) -> np.ndarray | None:
    if not segments:
        return None
    _ensure_vtm()
    from vtm_backend.paths import ensure_import_paths

    ensure_import_paths()
    from utils.hair import rasterize_hair_maps

    return rasterize_hair_maps(segments)
