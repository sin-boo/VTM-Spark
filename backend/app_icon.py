"""Hat mark -> Windows ICO with real alpha (no white square)."""

from __future__ import annotations

import struct
import sys
from io import BytesIO
from pathlib import Path

ICO_SIZES = (16, 24, 32, 48, 64, 256)
APP_USER_MODEL_ID = "VTM.Studio"


def hat_png_path() -> Path | None:
    from backend.paths import package_root, ui_dist_dir

    for folder in (package_root() / "ui" / "public", ui_dist_dir()):
        path = folder / "splash-mark.png"
        if path.is_file():
            return path
    return None


def hat_ico_path() -> Path:
    from backend.paths import package_root

    return package_root() / "backend" / "packaging" / "run.ico"


def _rgba_frames(png: Path) -> list:
    from PIL import Image

    src = Image.open(png).convert("RGBA")
    frames = []
    for size in ICO_SIZES:
        im = src.copy()
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(
            im,
            ((size - im.width) // 2, (size - im.height) // 2),
            im,
        )
        frames.append(canvas)
    return frames


def write_hat_ico(png: Path, ico: Path) -> Path:
    """Write a PNG-in-ICO so Windows keeps the transparent brim."""
    frames = _rgba_frames(png)
    entries: list[tuple[int, int, int, int]] = []
    payloads: list[bytes] = []
    offset = 6 + 16 * len(frames)
    for im in frames:
        buf = BytesIO()
        im.save(buf, format="PNG")
        data = buf.getvalue()
        w, h = im.size
        entries.append((0 if w >= 256 else w, 0 if h >= 256 else h, len(data), offset))
        payloads.append(data)
        offset += len(data)

    out = bytearray()
    out += struct.pack("<HHH", 0, 1, len(frames))
    for w, h, nbytes, off in entries:
        out += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, nbytes, off)
    for data in payloads:
        out += data
    ico.parent.mkdir(parents=True, exist_ok=True)
    ico.write_bytes(bytes(out))
    return ico


def ensure_hat_ico() -> Path | None:
    png = hat_png_path()
    if png is None:
        return None
    ico = hat_ico_path()
    public_ico = png.parent / "favicon.ico"
    if (
        ico.is_file()
        and public_ico.is_file()
        and ico.stat().st_mtime >= png.stat().st_mtime
        and public_ico.stat().st_mtime >= png.stat().st_mtime
    ):
        return ico
    write_hat_ico(png, ico)
    if public_ico.resolve() != ico.resolve():
        write_hat_ico(png, public_ico)
    return ico


def claim_app_id() -> None:
    """Stop Windows grouping the desk under pythonw.exe."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)  # type: ignore[attr-defined]
    except Exception:
        pass


_ICON_HANDLES: list[int] = []


def apply_window_icon(hwnd: int, ico: Path | None = None) -> bool:
    if sys.platform != "win32" or int(hwnd) <= 0:
        return False
    path = ico if ico is not None else ensure_hat_ico()
    if path is None or not path.is_file():
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        GCLP_HICON = -14
        GCLP_HICONSM = -34

        user32.LoadImageW.restype = ctypes.c_void_p
        user32.LoadImageW.argtypes = [
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            wintypes.UINT,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        resolved = str(path.resolve())
        small = user32.LoadImageW(None, resolved, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
        big = user32.LoadImageW(None, resolved, IMAGE_ICON, 32, 32, LR_LOADFROMFILE)
        if not small and not big:
            return False
        if small:
            _ICON_HANDLES.append(int(small))
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)
        if big:
            _ICON_HANDLES.append(int(big))
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
        try:
            set_class = getattr(user32, "SetClassLongPtrW", user32.SetClassLongW)
            if small:
                set_class(hwnd, GCLP_HICONSM, small)
            if big:
                set_class(hwnd, GCLP_HICON, big)
        except Exception:
            pass
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    png = Path(args[0]) if args else hat_png_path()
    ico = Path(args[1]) if len(args) > 1 else hat_ico_path()
    if png is None or not png.is_file():
        print("missing splash-mark.png", file=sys.stderr)
        return 1
    write_hat_ico(png, ico)
    print(ico)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
