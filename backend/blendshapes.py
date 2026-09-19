"""Track Lab plan / blend shapes stored next to character packs.

Location: ``models/blendshapes/``

- ``current.json`` — latest plan copied from Track Lab (read-only for the lab)
- ``<character-id>.json`` — snapshot taken when that character was created
  or last repaired

VTM Noble reads lab shapes and writes these files. It does not write shapes
back into Track Lab.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .paths import blendshapes_dir, display_path

FORMAT_ID = "vtm-blendshapes"
FORMAT_VERSION = 1
CURRENT_ID = "current"
SHAPE_IDS = ("rest", "smile", "sad", "A", "I", "U", "E", "O")
INCOMPATIBLE_MESSAGE = (
    "Incompatible. This character's blend shapes do not match the current plan. "
    "Would you like us to repair this character?"
)


def current_plan_path() -> Path:
    return blendshapes_dir() / "current.json"


def character_plan_path(ident: str) -> Path:
    stem = Path(str(ident or "").strip()).name
    if not stem or stem in {".", ".."}:
        raise ValueError("Invalid character id")
    return blendshapes_dir() / f"{stem}.json"


def normalize_shapes(raw: object) -> dict[str, list[list[float]]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[list[float]]] = {}
    for name in SHAPE_IDS:
        pts = raw.get(name)
        if not isinstance(pts, list) or len(pts) < 28:
            continue
        rows: list[list[float]] = []
        for row in pts[:28]:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                rows.append([0.0, 0.0, 0.0])
                continue
            score = float(row[2]) if len(row) > 2 else 1.0
            rows.append([round(float(row[0]), 3), round(float(row[1]), 3), round(score, 4)])
        out[name] = rows
    return out


def has_plan(shapes: object) -> bool:
    if not isinstance(shapes, dict):
        return False
    return any(name in shapes for name in SHAPE_IDS)


def fingerprint(shapes: object) -> str:
    norm = normalize_shapes(shapes)
    if not has_plan(norm):
        return ""
    payload = json.dumps(norm, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def empty_plan(ident: str = CURRENT_ID) -> dict[str, Any]:
    return {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "id": ident,
        "fingerprint": "",
        "shapes": {},
    }


def write_plan(path: Path | str, shapes: object, *, ident: str) -> dict[str, Any]:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    norm = normalize_shapes(shapes)
    payload = {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "id": str(ident or dest.stem),
        "fingerprint": fingerprint(norm),
        "shapes": norm,
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def read_plan(path: Path | str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return empty_plan(file_path.stem)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_plan(file_path.stem)
    if not isinstance(raw, dict):
        return empty_plan(file_path.stem)
    shapes = normalize_shapes(raw.get("shapes"))
    ident = str(raw.get("id") or file_path.stem)
    return {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "id": ident,
        "fingerprint": str(raw.get("fingerprint") or fingerprint(shapes)),
        "shapes": shapes,
    }


def load_current() -> dict[str, Any]:
    return read_plan(current_plan_path())


def load_character_plan(ident: str) -> dict[str, Any]:
    return read_plan(character_plan_path(ident))


def save_current(shapes: object) -> dict[str, Any]:
    return write_plan(current_plan_path(), shapes, ident=CURRENT_ID)


def save_character(ident: str, shapes: object) -> dict[str, Any]:
    return write_plan(character_plan_path(ident), shapes, ident=ident)


def delete_character_plan(ident: str) -> None:
    path = character_plan_path(ident)
    if path.is_file():
        path.unlink()


def shapes_from_lab(packet: dict[str, Any] | None) -> dict[str, list[list[float]]]:
    if not isinstance(packet, dict):
        return {}
    return normalize_shapes(packet.get("shapes"))


def refresh_current_from_lab(packet: dict[str, Any] | None) -> dict[str, Any]:
    """Copy authored lab shapes into ``current.json``. Never writes to the lab."""
    shapes = shapes_from_lab(packet)
    if has_plan(shapes):
        return save_current(shapes)
    return load_current()


def apply_current_to_character(ident: str) -> dict[str, Any]:
    current = load_current()
    if not has_plan(current.get("shapes")):
        raise ValueError("No current blend shapes to repair this character with")
    return save_character(ident, current["shapes"])


def compatibility(ident: str, current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare a character snapshot to the current plan."""
    plan = current if isinstance(current, dict) else load_current()
    cur_shapes = plan.get("shapes") if isinstance(plan.get("shapes"), dict) else {}
    char = load_character_plan(ident)
    char_shapes = char.get("shapes") if isinstance(char.get("shapes"), dict) else {}
    current_fp = fingerprint(cur_shapes)
    character_fp = fingerprint(char_shapes)
    has_current = bool(current_fp)
    has_character = bool(character_fp)
    compatible = (not has_current) or (has_character and character_fp == current_fp)
    return {
        "compatible": compatible,
        "has_current_plan": has_current,
        "has_character_plan": has_character,
        "current_fingerprint": current_fp,
        "character_fingerprint": character_fp,
        "path": display_path(character_plan_path(ident)),
        "current_path": display_path(current_plan_path()),
    }


def plan_card_fields(ident: str, current: dict[str, Any] | None = None) -> dict[str, Any]:
    info = compatibility(ident, current)
    return {
        "shapes_compatible": bool(info["compatible"]),
        "has_shapes": bool(info["has_character_plan"]),
        "shapes_path": str(info["path"]),
    }
