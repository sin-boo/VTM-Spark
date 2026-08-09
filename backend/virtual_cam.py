"""Push generated frames to the bundled 'VTM Noble Cam' virtual webcam."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PIL import Image

from .vcam_device import DEVICE_NAME, ensure_installed


class VirtualCameraOut:
    """Thread-safe pyvirtualcam wrapper for VTM Noble Cam."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cam: Any = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._device = ""
        self._backend = ""
        self._error = ""

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

    def start(self, width: int, height: int, fps: float = 30.0) -> str:
        """Open VTM Noble Cam at the generated-frame resolution."""
        w = max(16, int(width))
        h = max(16, int(height))
        w -= w % 2
        h -= h % 2
        rate = float(max(1.0, min(60.0, fps)))

        with self._lock:
            if self._cam is not None:
                if self._width == w and self._height == h:
                    return self._device
                self._close_locked()

            try:
                import pyvirtualcam
            except ImportError as exc:
                self._error = (
                    "pyvirtualcam is not installed — run Smart Build [1]"
                )
                raise RuntimeError(self._error) from exc

            try:
                ensure_installed(allow_prompt=True)
            except Exception as exc:
                self._error = str(exc)
                raise

            try:
                cam = pyvirtualcam.Camera(
                    width=w,
                    height=h,
                    fps=rate,
                    fmt=pyvirtualcam.PixelFormat.RGB,
                    backend="unitycapture",
                    device=DEVICE_NAME,
                    print_fps=False,
                )
            except Exception as exc:
                self._error = (
                    f"Could not open {DEVICE_NAME}. "
                    f"Run Smart Build [1] and approve UAC once. ({exc})"
                )
                raise RuntimeError(self._error) from exc

            self._cam = cam
            self._width = int(cam.width)
            self._height = int(cam.height)
            self._fps = float(getattr(cam, "fps", rate) or rate)
            self._device = str(getattr(cam, "device", "") or DEVICE_NAME)
            self._backend = str(getattr(cam, "backend", "") or "unitycapture")
            self._error = ""
            return self._device

    def stop(self) -> None:
        with self._lock:
            self._close_locked()
            self._error = ""

    def send(self, image: Image.Image | np.ndarray) -> None:
        """Push one RGB frame at the camera's native size (1:1 when sizes match)."""
        with self._lock:
            cam = self._cam
            if cam is None:
                return
            try:
                frame = self._as_rgb(image, self._width, self._height)
                cam.send(frame)
            except Exception as exc:  # noqa: BLE001 — keep generate loop alive
                self._error = f"Virtual camera send failed: {exc}"
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
            img = image.convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[0] != height or arr.shape[1] != width:
                img = Image.fromarray(arr, mode="RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                )
                arr = np.asarray(img, dtype=np.uint8)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr


_shared: VirtualCameraOut | None = None
_shared_lock = threading.Lock()


def get_virtual_cam() -> VirtualCameraOut:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = VirtualCameraOut()
        return _shared
