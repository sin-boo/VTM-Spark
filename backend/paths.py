"""Resolve project / packaged data paths (dev vs VTM Noble install)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def package_root() -> Path:
    """Directory that contains `backend/`, `ui/`, and `models/`.

    - Dev / GitHub tree: vtm-noble/ (or legacy real_stream/)
    - Packaged thin-launcher layout: dist/VTMNoble/ (runtime python, not frozen)
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
                or (candidate / "send2pod" / "torch_train").is_dir()
            ):
                return candidate
        return meipass

    # backend/paths.py -> backend/ -> install root
    return Path(__file__).resolve().parent.parent


def _path_roots() -> list[Path]:
    """Candidate roots for relative display / resolve (deduped)."""
    roots: list[Path] = []
    for root in (package_root(), repo_root(), package_root().parent):
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def display_path(path: Path | str) -> str:
    """Package- or repo-relative posix path for UI / API responses."""
    p = Path(path)
    try:
        resolved = p.resolve() if p.is_absolute() else (package_root() / p).resolve()
    except OSError:
        return Path(path).as_posix().replace("\\", "/")
    for root in _path_roots():
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return resolved.as_posix().replace("\\", "/")


def resolve_user_path(path: Path | str) -> Path:
    """Resolve an absolute, package-relative, or repo-relative path from the UI/API."""
    p = Path(path)
    if p.is_absolute():
        return p
    for root in _path_roots():
        candidate = (root / p).resolve()
        if candidate.exists():
            return candidate
    return (package_root() / p).resolve()


def repo_root() -> Path:
    """Monorepo root in development; install root when packaged."""
    root = package_root()
    if getattr(sys, "frozen", False):
        return root
    if (root / "vendor" / "torch_train").is_dir() or (root / "runtime").is_dir():
        return root
    if (root / "send2pod" / "torch_train").is_dir():
        return root
    return root.parent


def data_dir() -> Path:
    return package_root() / "data"


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
            # Remove empty typo directory (ignore leftovers).
            if not any(typo.iterdir()):
                typo.rmdir()
        except OSError:
            pass

    if any(p.is_file() and not p.name.startswith(".") for p in preferred.iterdir()):
        return preferred

    # Legacy packaged path (older installs put DiT under data/models/dit).
    legacy_packaged = data_dir() / "models" / "dit"
    if legacy_packaged.is_dir() and any(
        p.is_file() and not p.name.startswith(".") for p in legacy_packaged.iterdir()
    ):
        return legacy_packaged

    return preferred


def refs_dir() -> Path:
    d = data_dir() / "refs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def trackers_dir() -> Path:
    """Small tracker weights shipped with the app."""
    preferred = models_root() / "trackers"
    preferred.mkdir(parents=True, exist_ok=True)
    if any(preferred.iterdir()):
        return preferred
    legacy = data_dir() / "trackers"
    if legacy.is_dir() and any(legacy.iterdir()):
        return legacy
    return preferred


def outputs_dir() -> Path:
    d = package_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    root = package_root()
    return _first_existing(
        root / "vendor" / "torch_train",
        root / "send2pod" / "torch_train",
        repo_root() / "send2pod" / "torch_train",
        marker="inference_keypoint.py",
    )


def tools_dir() -> Path:
    """Vendored or monorepo tools (live-poser + OSF layout)."""
    root = package_root()
    return _first_existing(
        root / "vendor" / "tools",
        root / "tools",
        repo_root() / "tools",
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
        tools_dir() / "vedio traker" / "OpenSeeFace",
        marker="tracker.py",
    )


def default_ref_candidates() -> list[Path]:
    return [
        refs_dir() / "default.png",
        refs_dir() / "train_char_1.png",
        repo_root()
        / "send2pod"
        / "data"
        / "train_crop"
        / "images"
        / "cherecter 3"
        / "001.png",
        repo_root() / "UI" / "outputs" / "_refs" / "train_char_1.png",
        package_root()
        / "send2pod"
        / "data"
        / "train_crop"
        / "images"
        / "cherecter 3"
        / "001.png",
    ]


def default_ref_path() -> Path:
    for path in default_ref_candidates():
        if path.is_file():
            return path
    return default_ref_candidates()[0]


def ensure_import_paths() -> None:
    """Add vendored / monorepo tool roots to sys.path for tracker / DiT helpers."""
    for path in (
        torch_train_dir(),
        live_poser_dir(),
        pose_traker_dir(),
        anime_face_detector_src(),
        openseeface_dir(),
    ):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
