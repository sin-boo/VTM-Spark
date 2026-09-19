"""Scan required / optional runtime weights and report a setup checklist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import (
    live_poser_dir,
    models_dir,
    openseeface_dir,
    package_root,
    pose_traker_dir,
    trackers_dir,
)

_MIN_BYTES = 1_000_000


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    label: str
    purpose: str
    required: bool
    # Relative paths from package root (posix). First existing match wins.
    candidates: tuple[str, ...]
    min_bytes: int = 1
    # If True, absence is OK when an alternate auto-download path exists.
    auto_download: bool = False


# Canonical local weights the user should upload / ship.
# Prefer models/dit + models/trackers; vendor mirrors remain as fallbacks.
CHECKLIST: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        id="dit",
        label="VTM-ELF.pt",
        purpose="DiT generation checkpoint",
        required=True,
        candidates=("models/dit/VTM-ELF.pt",),
        min_bytes=_MIN_BYTES,
        auto_download=True,
    ),
    ChecklistItem(
        id="iris",
        label="iris_pose.pt",
        purpose="Iris / pupil tracker (fine-tune)",
        required=True,
        candidates=(
            "models/trackers/iris_pose.pt",
            "vendor/tools/live-poser/models/iris_pose.pt",
            "vendor/tools/pose-traker/models/iris_pose.pt",
            "vendor/tools/pose-traker/iris-model/models/iris_pose.pt",
        ),
        min_bytes=_MIN_BYTES,
    ),
    ChecklistItem(
        id="body",
        label="dwpose_v2.pt",
        purpose="Body pose for reference fit (fine-tune)",
        required=True,
        candidates=(
            "models/trackers/dwpose_v2.pt",
            "models/trackers/dwpose_v2.onnx",
            "vendor/tools/live-poser/models/dwpose_v2.pt",
            "vendor/tools/live-poser/models/dwpose_v2.onnx",
            "vendor/tools/pose-traker/iris-model/models/dwpose_v2.pt",
            "vendor/tools/pose-traker/iris-model/models/dwpose_v2.onnx",
        ),
        min_bytes=_MIN_BYTES,
    ),
    ChecklistItem(
        id="mediapipe_body",
        label="pose_landmarker_lite.task",
        purpose="Live body landmarks (MediaPipe)",
        required=True,
        candidates=(
            "models/trackers/pose_landmarker_lite.task",
            "vendor/tools/live-poser/models/pose_landmarker_lite.task",
        ),
        min_bytes=100_000,
        auto_download=True,
    ),
    ChecklistItem(
        id="hair",
        label="animeseg_hair3.pt",
        purpose="Hair-part tracker (full-stack Mask2Former fine-tune)",
        required=False,
        candidates=(
            "models/trackers/animeseg_hair3.pt",
            "models/trackers/hair_seg.pt",
            "vendor/tools/live-poser/models/hair_seg.pt",
        ),
        min_bytes=_MIN_BYTES,
    ),
    ChecklistItem(
        id="osf_landmarks",
        label="lm_model3_opt.onnx",
        purpose="OpenSeeFace face landmarks (default)",
        required=True,
        candidates=("vendor/tools/openseeface/models/lm_model3_opt.onnx",),
        min_bytes=_MIN_BYTES,
    ),
    ChecklistItem(
        id="osf_retinaface",
        label="retinaface_640x640_opt.onnx",
        purpose="OpenSeeFace face detector",
        required=True,
        candidates=("vendor/tools/openseeface/models/retinaface_640x640_opt.onnx",),
        min_bytes=_MIN_BYTES,
    ),
    ChecklistItem(
        id="osf_detection",
        label="mnv3_detection_opt.onnx",
        purpose="OpenSeeFace mobile detection",
        required=True,
        candidates=("vendor/tools/openseeface/models/mnv3_detection_opt.onnx",),
        min_bytes=100_000,
    ),
    ChecklistItem(
        id="osf_gaze",
        label="mnv3_gaze32_split_opt.onnx",
        purpose="OpenSeeFace gaze",
        required=True,
        candidates=("vendor/tools/openseeface/models/mnv3_gaze32_split_opt.onnx",),
        min_bytes=100_000,
    ),
    ChecklistItem(
        id="osf_priorbox",
        label="priorbox_640x640.json",
        purpose="OpenSeeFace RetinaFace priors",
        required=True,
        candidates=("vendor/tools/openseeface/models/priorbox_640x640.json",),
        min_bytes=100,
    ),
)


def _resolve_candidate(rel: str) -> Path:
    return (package_root() / rel).resolve()


def _find_item(item: ChecklistItem) -> tuple[Path | None, int]:
    for rel in item.candidates:
        path = _resolve_candidate(rel)
        if path.is_file() and path.stat().st_size >= item.min_bytes:
            return path, path.stat().st_size
    if item.id == "dit":
        dit = models_dir()
        if dit.is_dir():
            for path in sorted(dit.iterdir()):
                if (
                    path.is_file()
                    and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
                    and path.stat().st_size >= item.min_bytes
                ):
                    return path, path.stat().st_size
    return None, 0


def scan_models() -> dict[str, Any]:
    """Return a full checklist: each item present/missing + overall pass/fail."""
    items: list[dict[str, Any]] = []
    required_total = 0
    required_ok = 0
    optional_ok = 0
    optional_total = 0

    for spec in CHECKLIST:
        found, size = _find_item(spec)
        ok = found is not None
        entry = {
            "id": spec.id,
            "label": spec.label,
            "purpose": spec.purpose,
            "required": spec.required,
            "auto_download": spec.auto_download,
            "ok": ok,
            "path": found.as_posix().replace("\\", "/") if found else "",
            "rel_preferred": spec.candidates[0],
            "bytes": size,
            "candidates": list(spec.candidates),
        }
        items.append(entry)
        if spec.required:
            required_total += 1
            if ok:
                required_ok += 1
        else:
            optional_total += 1
            if ok:
                optional_ok += 1

    # Extra DiT files beyond the preferred name still count for "dit" if preferred missing.
    dit_dir = models_dir()
    extra_dit = [
        {
            "name": p.name,
            "path": p.as_posix().replace("\\", "/"),
            "bytes": p.stat().st_size,
        }
        for p in sorted(dit_dir.iterdir())
        if p.is_file() and not p.name.startswith(".") and p.stat().st_size >= _MIN_BYTES
    ]
    dit_item = next((x for x in items if x["id"] == "dit"), None)
    if dit_item is not None and not dit_item["ok"] and extra_dit:
        dit_item["ok"] = True
        dit_item["path"] = extra_dit[0]["path"]
        dit_item["bytes"] = extra_dit[0]["bytes"]
        dit_item["label"] = extra_dit[0]["name"]
        required_ok = sum(1 for x in items if x["required"] and x["ok"])

    success = required_ok == required_total and required_total > 0
    missing = [x for x in items if x["required"] and not x["ok"]]
    return {
        "success": success,
        "message": (
            "All required models found."
            if success
            else f"Missing {len(missing)} required model(s)."
        ),
        "required_ok": required_ok,
        "required_total": required_total,
        "optional_ok": optional_ok,
        "optional_total": optional_total,
        "items": items,
        "dit_files": extra_dit,
        "preferred_dirs": {
            "dit": "models/dit",
            "trackers": "models/trackers",
            "openseeface": "vendor/tools/openseeface/models",
        },
        "hub_auto": [
            {
                "id": "sd_vae",
                "label": "stabilityai/sd-vae-ft-mse",
                "purpose": "SD VAE encode/decode (Hugging Face, first load)",
            },
            {
                "id": "tiny_vae",
                "label": "cqyan/hybrid-sd-tinyvae",
                "purpose": "Fast TinyVAE decode (Hugging Face, first load)",
            },
            {
                "id": "anime_face",
                "label": "face_yolov8n.pt + mmpose_anime-face_hrnetv2.pth",
                "purpose": "Anime face detector for ref-fit (auto-download)",
            },
        ],
    }


def checklist_success() -> bool:
    return bool(scan_models().get("success"))


def format_checklist_text(report: dict[str, Any] | None = None) -> str:
    report = report or scan_models()
    lines = [
        f"Model checklist: {report['required_ok']}/{report['required_total']} required OK",
        f"  {report['message']}",
        "",
    ]
    for item in report["items"]:
        mark = "[x]" if item["ok"] else "[ ]"
        req = "required" if item["required"] else "optional"
        where = item["path"] or item["rel_preferred"]
        lines.append(f"  {mark} {item['label']} ({req}) - {item['purpose']}")
        lines.append(f"      {where}")
    lines.append("")
    lines.append("Upload fine-tunes to preferred paths:")
    for key, rel in report["preferred_dirs"].items():
        lines.append(f"  - {key}: {rel}/")
    return "\n".join(lines)


def print_checklist() -> int:
    report = scan_models()
    print(format_checklist_text(report))
    return 0 if report["success"] else 1


# Keep import-side helpers available for resolvers / docs.
def tracker_search_roots() -> list[Path]:
    return [
        trackers_dir(),
        live_poser_dir() / "models",
        pose_traker_dir() / "models",
        pose_traker_dir() / "iris-model" / "models",
        openseeface_dir() / "models",
    ]


if __name__ == "__main__":
    raise SystemExit(print_checklist())
