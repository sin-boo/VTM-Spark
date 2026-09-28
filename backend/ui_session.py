"""Persist last-selected checkpoint / reference and Hub catalog names."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .paths import models_root

_LOCK = threading.Lock()


def session_path() -> Path:
    return models_root() / "session.json"


def load_ui_session() -> dict[str, Any]:
    path = session_path()
    empty: dict[str, Any] = {
        "checkpoint": "",
        "reference_path": "",
        "character_path": "",
        "hub_files": [],
        "mouth_osf": {},
        "steps": None,
        "pose_cfg": None,
        "id_cfg": None,
        "frame_blend": None,
        "inbetweens": None,
        "interpolate": None,
        "max_fps": None,
        "batch": None,
        "hold_last": None,
        "compile_model": None,
        "travel_box": {},
    }
    if not path.is_file():
        return dict(empty)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(empty)
    if not isinstance(raw, dict):
        return dict(empty)
    hub = raw.get("hub_files") or []
    mouth_osf = raw.get("mouth_osf") if isinstance(raw.get("mouth_osf"), dict) else {}
    travel_box = raw.get("travel_box") if isinstance(raw.get("travel_box"), dict) else {}
    return {
        "checkpoint": str(raw.get("checkpoint") or "").strip(),
        "reference_path": str(raw.get("reference_path") or "").strip(),
        "character_path": str(raw.get("character_path") or "").strip(),
        "hub_files": [str(n) for n in hub if str(n).strip()],
        "mouth_osf": dict(mouth_osf),
        "travel_box": dict(travel_box),
        "steps": raw.get("steps"),
        "pose_cfg": raw.get("pose_cfg"),
        "id_cfg": raw.get("id_cfg"),
        "frame_blend": raw.get("frame_blend"),
        "inbetweens": raw.get("inbetweens"),
        "interpolate": raw.get("interpolate"),
        "max_fps": raw.get("max_fps"),
        "batch": raw.get("batch"),
        "hold_last": raw.get("hold_last"),
        "compile_model": raw.get("compile_model"),
    }


def save_ui_session(**kwargs: Any) -> None:
    with _LOCK:
        state = load_ui_session()
        if "checkpoint" in kwargs and kwargs["checkpoint"] is not None:
            state["checkpoint"] = str(kwargs["checkpoint"]).strip()
        if "reference_path" in kwargs and kwargs["reference_path"] is not None:
            state["reference_path"] = str(kwargs["reference_path"]).strip()
        if "character_path" in kwargs and kwargs["character_path"] is not None:
            state["character_path"] = str(kwargs["character_path"]).strip()
        if "hub_files" in kwargs and kwargs["hub_files"] is not None:
            names = [str(n).strip() for n in kwargs["hub_files"] if str(n).strip()]
            state["hub_files"] = names
        if "travel_box" in kwargs and kwargs["travel_box"] is not None:
            state["travel_box"] = (
                dict(kwargs["travel_box"]) if isinstance(kwargs["travel_box"], dict) else {}
            )
        for key in ("steps", "pose_cfg", "id_cfg", "frame_blend", "inbetweens", "max_fps", "batch"):
            if key in kwargs and kwargs[key] is not None:
                state[key] = kwargs[key]
        if "interpolate" in kwargs and kwargs["interpolate"] is not None:
            state["interpolate"] = bool(kwargs["interpolate"])
        if "hold_last" in kwargs and kwargs["hold_last"] is not None:
            state["hold_last"] = bool(kwargs["hold_last"])
        if "compile_model" in kwargs and kwargs["compile_model"] is not None:
            state["compile_model"] = bool(kwargs["compile_model"])
        path = session_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except OSError:
            pass
