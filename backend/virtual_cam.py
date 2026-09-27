"""Push generated frames to the bundled 'VTM Studio Cam' virtual webcam."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from .vcam_device import DEVICE_NAME, ensure_installed

VCAM_FPS = 30.0
_FrameSource = Callable[[], Image.Image | np.ndarray | None]


def vcam_even_size(width: int, height: int) -> tuple[int, int]:
    """Unity Capture wants even dims; width in steps of 4."""
    w = max(16, int(width))
    h = max(16, int(height))
    w -= w % 4
    h -= h % 2
    return max(16, w), max(16, h)


def trim_solid_edges(arr: np.ndarray, *, limit: int = 16) -> np.ndarray:
    """Drop uniform near-black letterbox rows/cols. Leaves chroma / content alone."""
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
        return arr
    h, w = arr.shape[:2]
    dark = arr.max(axis=2) <= int(limit)
    rows = ~dark.all(axis=1)
    cols = ~dark.all(axis=0)
    if not rows.any() or not cols.any():
        return arr
    y0 = int(np.argmax(rows))
    y1 = int(h - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(w - np.argmax(cols[::-1]))
    if (y1 - y0) < h * 0.5 or (x1 - x0) < w * 0.5:
        return arr
    if y0 == 0 and x0 == 0 and y1 == h and x1 == w:
        return arr
    return arr[y0:y1, x0:x1]


def cover_rgb(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale uniformly and center-crop so the frame is filled (no black bars)."""
    src = np.asarray(arr)
    if src.ndim != 3:
        raise ValueError("cover_rgb expects HxWxC")
    src = src[..., :3]
    sh, sw = int(src.shape[0]), int(src.shape[1])
    if sh == height and sw == width:
        out = src
    else:
        scale = max(width / max(sw, 1), height / max(sh, 1))
        nw = max(1, int(round(sw * scale)))
        nh = max(1, int(round(sh * scale)))
        img = Image.fromarray(np.ascontiguousarray(src), mode="RGB").resize(
            (nw, nh), Image.Resampling.BILINEAR
        )
        x0 = max(0, (nw - width) // 2)
        y0 = max(0, (nh - height) // 2)
        img = img.crop((x0, y0, x0 + width, y0 + height))
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.BILINEAR)
        out = np.asarray(img, dtype=np.uint8)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


class VirtualCameraOut:
    """Thread-safe pyvirtualcam wrapper for VTM Studio Cam.

    A pump thread keeps resending the latest picture at camera FPS so OBS
    stays live on the reference still — Unity Capture goes black if we only
    send when DiT produces a new frame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cam: Any = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._device = ""
        self._backend = ""
        self._error = ""
        self._source: _FrameSource | None = None
        self._held: Image.Image | np.ndarray | None = None
        self._frame: np.ndarray | None = None
        self._pump_stop = threading.Event()
        self._pump_thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._cam is not None

    @property
    def device(self) -> str:
        with self._lock:
            return self._device or DEVICE_NAME

    @property
    def backend(self) -> str:
        with self._lock:
            return self._backend

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def size(self) -> tuple[int, int]:
        with self._lock:
            return int(self._width), int(self._height)

    @property
    def fps(self) -> float:
        with self._lock:
            return float(self._fps)

    def start(
        self,
        width: int,
        height: int,
        fps: float = VCAM_FPS,
        *,
        source: _FrameSource | None = None,
    ) -> str:
        """Open VTM Studio Cam at the generated-frame resolution and pump frames."""
        w, h = vcam_even_size(width, height)
        rate = float(max(1.0, min(60.0, fps)))
        self._stop_pump()

        with self._lock:
            self._source = source
            if self._cam is not None and self._width == w and self._height == h:
                self._fps = rate
                device = self._device
            else:
                if self._cam is not None:
                    self._close_locked()
                device = self._open_locked(w, h, rate)

        self._start_pump()
        return device

    def stop(self) -> None:
        self._stop_pump()
        with self._lock:
            self._close_locked()
            self._error = ""
            self._source = None
            self._held = None
            self._frame = None

    def send(self, image: Image.Image | np.ndarray) -> None:
        """Hold a picture; the pump thread is the only caller of cam.send."""
        with self._lock:
            self._held = image

    def _open_locked(self, width: int, height: int, fps: float) -> str:
        try:
            import pyvirtualcam
        except ImportError as exc:
            self._error = "pyvirtualcam is not installed — re-run install.bat"
            raise RuntimeError(self._error) from exc

        try:
            ensure_installed(allow_prompt=True)
        except Exception as exc:
            self._error = str(exc)
            raise

        try:
            cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=fps,
                fmt=pyvirtualcam.PixelFormat.RGB,
                backend="unitycapture",
                device=DEVICE_NAME,
                print_fps=False,
            )
        except Exception as exc:
            self._error = (
                f"Could not open {DEVICE_NAME}. "
                f"Re-run install.bat and approve UAC once. ({exc})"
            )
            raise RuntimeError(self._error) from exc

        self._cam = cam
        self._width = int(getattr(cam, "width", width) or width)
        self._height = int(getattr(cam, "height", height) or height)
        self._fps = float(getattr(cam, "fps", fps) or fps)
        self._device = str(getattr(cam, "device", "") or DEVICE_NAME)
        self._backend = str(getattr(cam, "backend", "") or "unitycapture")
        self._error = ""
        return self._device

    def _start_pump(self) -> None:
        self._pump_stop = threading.Event()
        thread = threading.Thread(
            target=self._pump_loop, name="vtm-vcam", daemon=True
        )
        self._pump_thread = thread
        thread.start()

    def _stop_pump(self) -> None:
        self._pump_stop.set()
        thread = self._pump_thread
        self._pump_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _pump_loop(self) -> None:
        while not self._pump_stop.is_set():
            with self._lock:
                cam = self._cam
                source = self._source
                held = self._held
                w, h = self._width, self._height
                fps = float(self._fps or VCAM_FPS)
            if cam is None or w <= 0 or h <= 0:
                break
            image: Image.Image | np.ndarray | None = None
            if source is not None:
                try:
                    image = source()
                except Exception:
                    image = None
            if image is None:
                image = held
            if isinstance(image, Image.Image):
                try:
                    image = image.copy()
                except Exception:
                    image = None
            if image is not None:
                try:
                    frame = self._as_rgb(image, w, h)
                    cam.send(frame)
                    with self._lock:
                        if self._cam is cam:
                            self._frame = frame
                except Exception as exc:  # noqa: BLE001
                    self._fail(f"Virtual camera send failed: {exc}")
                    break
            if self._pump_stop.is_set():
                break
            try:
                sleeper = getattr(cam, "sleep_until_next_frame", None)
                if callable(sleeper):
                    sleeper()
                else:
                    time.sleep(1.0 / max(fps, 1.0))
            except Exception:
                if self._pump_stop.wait(1.0 / max(fps, 1.0)):
                    break

    def _fail(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._close_locked()

    def _close_locked(self) -> None:
        cam = self._cam
        self._cam = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._device = ""
        self._backend = ""
        if cam is None:
            return
        try:
            cam.close()
        except Exception:
            pass

    @staticmethod
    def _as_rgb(
        image: Image.Image | np.ndarray, width: int, height: int
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = trim_solid_edges(arr)
        return cover_rgb(arr, int(width), int(height))


_shared: VirtualCameraOut | None = None
_shared_lock = threading.Lock()


def get_virtual_cam() -> VirtualCameraOut:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = VirtualCameraOut()
        return _shared
