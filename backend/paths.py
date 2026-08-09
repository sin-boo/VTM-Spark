"""Resolve project / packaged data paths (self-contained install root only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def package_root() -> Path:
    """Directory that contains `backend/`, `ui/`, `vendor/`, and `models/`.

    - Dev / GitHub tree: VTM-Noble /
    - Packaged thin-launcher layout: dist/VTMNoble/
    - Legacy PyInstaller onedir: exe folder or `_internal`
    """
    env_root = (
        os.environ.get("VTM_NOBLE_ROOT", "").strip()
        or os.environ.get("REAL_STREAM_ROOT", "").strip()
    )
    if env_root:
        p = Path(env_root)
        if p.is_dir():
            return p.resolve()

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
        for candidate in (exe_dir, exe_dir / "_internal", meipass):
            if (
                (candidate / "data").is_dir()
                or (candidate / "ui" / "dist").is_dir()
                or (candidate / "vendor" / "torch_train").is_dir()
            ):
                return candidate
        return meipass

    # backend/paths.py -> backend/ -> install root
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Install / GitHub root (same as package_root — no parent monorepo)."""
    return package_root()


def _path_roots() -> list[Path]:
    """Candidate roots for relative display / resolve (package only)."""
    return [package_root().resolve()]


def display_path(path: Path | str) -> str:
    """Package-relative posix path for UI / API responses."""
    p = Path(path)
    try:
        resolved = p.resolve() if p.is_absolute() else (package_root() / p).resolve()
    except OSError:
        return Path(path).as_posix().replace("\\", "/")
    root = package_root().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix().replace("\\", "/")


def resolve_user_path(path: Path | str) -> Path:
    """Resolve an absolute or package-relative path from the UI/API."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (package_root() / p).resolve()


def ensure_under_models(path: Path | str) -> Path:
    """Resolve *path* and require it stay under the package ``models/`` tree."""
    resolved = resolve_user_path(path).resolve()
    root = models_root().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Checkpoint must be inside models/ (got {display_path(resolved)})"
        ) from exc
    return resolved


def data_dir() -> Path:
    d = package_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_root() -> Path:
    """User-facing models tree (DiT download + bundled trackers)."""
    root = package_root() / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def models_dir() -> Path:
    """DiT checkpoint directory (relative models/dit; downloaded on setup)."""
    preferred = models_root() / "dit"
    preferred.mkdir(parents=True, exist_ok=True)

    # Migrate typo folder models/iamge → models/dit (one-time, best-effort).
    typo = models_root() / "iamge"
    if typo.is_dir():
        try:
            for src in typo.iterdir():
                if not src.is_file() or src.name.startswith("."):
                    continue
                dest = preferred / src.name
                if not dest.exists():
                    src.replace(dest)
            if not any(typo.iterdir()):
                typo.rmdir()
        except OSError:
            pass

    return preferred


def refs_dir() -> Path:
    d = data_dir() / "refs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def trackers_dir() -> Path:
    """Small tracker weights shipped with the app / downloaded on setup."""
    preferred = models_root() / "trackers"
    preferred.mkdir(parents=True, exist_ok=True)
    if any(preferred.iterdir()):
        return preferred
    legacy = data_dir() / "trackers"
    if legacy.is_dir() and any(legacy.iterdir()):
        return legacy
    return preferred


def outputs_dir() -> Path:
    """Legacy folder — do not write app frames or logs here."""
    return package_root() / "outputs"


def ui_dist_dir() -> Path:
    """Return packaged Vite output."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
        for candidate in (exe_dir, exe_dir / "_internal", meipass, package_root()):
            dist = candidate / "ui" / "dist"
            if (dist / "index.html").is_file():
                return dist
    return package_root() / "ui" / "dist"


def _first_existing(*candidates: Path, marker: str | None = None) -> Path:
    for path in candidates:
        if marker is None:
            if path.is_dir():
                return path
        elif (path / marker).exists():
            return path
    return candidates[0]


def torch_train_dir() -> Path:
    """DiT inference helpers (`inference_keypoint`, `models`, `vae`, …)."""
    return _first_existing(
        package_root() / "vendor" / "torch_train",
        marker="inference_keypoint.py",
    )


def tools_dir() -> Path:
    """Vendored tools (live-poser + OpenSeeFace + pose-traker)."""
    return _first_existing(
        package_root() / "vendor" / "tools",
        marker="live-poser",
    )


def live_poser_dir() -> Path:
    return tools_dir() / "live-poser"


def pose_traker_dir() -> Path:
    return tools_dir() / "pose-traker"


def anime_face_detector_src() -> Path:
    return pose_traker_dir() / "anime-face-detector" / "src"


def openseeface_dir() -> Path:
    return _first_existing(
        tools_dir() / "openseeface",
        marker="tracker.py",
    )


def default_ref_candidates() -> list[Path]:
    """Reference stills under data/refs only (user uploads / shipped defaults)."""
    return [
        refs_dir() / "default.png",
        refs_dir() / "train_char_1.png",
        refs_dir() / "upload.png",
    ]


def default_ref_path() -> Path:
    for path in default_ref_candidates():
        if path.is_file():
            return path
    return default_ref_candidates()[0]


def ensure_import_paths() -> None:
    """Add vendored tool roots to sys.path for tracker / DiT helpers."""
    for path in (
        torch_train_dir(),
        live_poser_dir(),
        pose_traker_dir(),
        anime_face_detector_src(),
        openseeface_dir(),
    ):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
