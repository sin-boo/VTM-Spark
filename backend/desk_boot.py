"""Desk launch stages: model, last character, Track Lab handshake."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BOOT_STAGE_KEYS = ("model", "character", "lab")
STAGE_WEIGHTS = {"model": 0.46, "character": 0.34, "lab": 0.2}

_STAGE_LABELS = {
    "model": "Loading model",
    "character": "Loading character",
    "lab": "Connecting Track Lab",
}


def new_boot_state() -> dict[str, Any]:
    return {
        "ready": False,
        "running": False,
        "error": "",
        "awaiting": "",
        "suggested": "",
        "stages": {
            key: {"state": "idle", "progress": 0.0, "label": _STAGE_LABELS[key]}
            for key in BOOT_STAGE_KEYS
        },
    }


def _overall_from_stages(stages: dict[str, Any], *, ready: bool) -> tuple[float, str]:
    total = 0.0
    for key in BOOT_STAGE_KEYS:
        row = stages.get(key) if isinstance(stages.get(key), dict) else {}
        total += max(0.0, min(1.0, float(row.get("progress") or 0.0))) * STAGE_WEIGHTS[key]
    running = next(
        (key for key in ("model", "character", "lab") if str((stages.get(key) or {}).get("state")) == "run"),
        None,
    )
    if running is None:
        running = next(
            (
                key
                for key in BOOT_STAGE_KEYS
                if str((stages.get(key) or {}).get("state") or "idle") == "idle"
            ),
            None,
        )
    label = "Ready"
    if running is not None:
        row = stages.get(running) if isinstance(stages.get(running), dict) else {}
        label = str(row.get("label") or _STAGE_LABELS.get(running) or "Starting…")
    elif not ready:
        label = "Starting…"
    return max(0.0, min(1.0, total)), label


def snapshot_boot(state: dict[str, Any]) -> dict[str, Any]:
    stages = {}
    raw_stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    for key in BOOT_STAGE_KEYS:
        row = raw_stages.get(key) if isinstance(raw_stages.get(key), dict) else {}
        stages[key] = {
            "state": str(row.get("state") or "idle"),
            "progress": float(row.get("progress") or 0.0),
            "label": str(row.get("label") or _STAGE_LABELS[key]),
        }
    ready = bool(state.get("ready"))
    frac, label = _overall_from_stages(stages, ready=ready)
    return {
        "ready": ready,
        "running": bool(state.get("running")),
        "error": str(state.get("error") or ""),
        "awaiting": str(state.get("awaiting") or ""),
        "suggested": str(state.get("suggested") or ""),
        "stages": stages,
        "progress": frac,
        "progress_label": label,
    }


def patch_boot_stage(
    state: dict[str, Any],
    key: str,
    *,
    stage_state: str,
    progress: float | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    if key not in BOOT_STAGE_KEYS:
        return snapshot_boot(state)
    row = state.setdefault("stages", {}).setdefault(
        key, {"state": "idle", "progress": 0.0, "label": _STAGE_LABELS[key]}
    )
    row["state"] = str(stage_state)
    if progress is not None:
        row["progress"] = max(0.0, min(1.0, float(progress)))
    if label is not None:
        row["label"] = str(label)
    return snapshot_boot(state)


def overall_progress(state: dict[str, Any]) -> tuple[float, str]:
    """0..1 weighted bar + the stage that is currently running."""
    snap = snapshot_boot(state)
    return float(snap.get("progress") or 0.0), str(snap.get("progress_label") or "Starting…")


def previous_load_target(session: dict[str, Any] | None) -> tuple[str, str]:
    """Kind + ident for the last still/pack. Kind is character, still, or none."""
    raw = str(
        (session or {}).get("character_path")
        or (session or {}).get("reference_path")
        or ""
    ).strip()
    if not raw:
        return ("none", "")
    path = Path(raw)
    if path.suffix.lower() == ".vtm":
        return ("character", path.stem)
    return ("still", raw)
