"""Desk UI preferences that must outlive the WebView.

pywebview runs WebView2 in private mode, so ``localStorage`` is wiped on every
launch. The language pick lives in ``models/ui_prefs.json`` instead. The native
splash reads it too, so this module stays stdlib-only (no torch / FastAPI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

LANGUAGES = ("en", "ja")
PREFS_NAME = "ui_prefs.json"

# Windows primary language id for Japanese (LANG_JAPANESE).
_LANG_JAPANESE = 0x11


def prefs_path() -> Path:
    from .paths import data_dir

    return data_dir() / PREFS_NAME


def _clean_language(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in LANGUAGES else ""


def system_language() -> str:
    """``ja`` when the Windows display language is Japanese, else ``en``."""
    if sys.platform != "win32":
        return "en"
    try:
        import ctypes

        lang_id = int(ctypes.windll.kernel32.GetUserDefaultUILanguage())
    except Exception:
        return "en"
    return "ja" if lang_id & 0x3FF == _LANG_JAPANESE else "en"


def load_ui_prefs(path: Path | None = None) -> dict[str, Any]:
    """Saved prefs. ``language`` is blank until the user picks one."""
    target = path or prefs_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return {"language": _clean_language(raw.get("language"))}


def save_ui_prefs(patch: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    target = path or prefs_path()
    prefs = load_ui_prefs(target)
    if "language" in patch:
        language = _clean_language(patch.get("language"))
        if not language:
            raise ValueError(f"Unknown language: {patch.get('language')!r}")
        prefs["language"] = language
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    return prefs


def ui_language(path: Path | None = None) -> str:
    """The language the desk should paint in: the saved pick, else the OS language."""
    return load_ui_prefs(path)["language"] or system_language()


# Boot lines the native splash shows before React loads. The desk has its own
# copy of these (ui/src/i18n.tsx) for the React splash and progress bars.
SPLASH_JA: dict[str, str] = {
    "Starting…": "起動中…",
    "Loading resources…": "リソースを読み込み中…",
    "Ready": "準備完了",
    "Loading model": "モデルを読み込み中",
    "Loading image decoder": "画像デコーダーを読み込み中",
    "Moving model to GPU": "モデルを GPU に転送中",
    "Downloading model": "モデルをダウンロード中",
    "Model ready": "モデルの準備完了",
    "Model failed": "モデルの読み込みに失敗しました",
    "Loading character": "キャラクターを読み込み中",
    "No character": "キャラクターなし",
    "Character failed": "キャラクターの読み込みに失敗しました",
    "Connecting Track Lab": "Track Lab に接続中",
    "Track Lab connected": "Track Lab に接続しました",
    "Track Lab still starting": "Track Lab を起動中",
    "Skipped": "スキップ",
    # Prefix: "Character: <name>" keeps the name as-is.
    "Character: ": "キャラクター: ",
    "Desk API failed to start": "デスク API を起動できませんでした",
}


def splash_strings(language: str | None = None) -> dict[str, str]:
    """English → display map for the native splash (empty for English)."""
    return dict(SPLASH_JA) if (language or ui_language()) == "ja" else {}
