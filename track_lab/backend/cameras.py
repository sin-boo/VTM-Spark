"""List capture devices for the OSF bench.

Windows uses OpenSeeFace DirectShow so virtual cams show up by name.
Falls back to probing OpenCV indices.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import cv2


def _no_mic_params() -> list[int]:
    """Keep VideoCapture from binding the Windows microphone.

    OpenCV 4.7+/5 MSMF (and some DirectShow graphs) open audio stream 0 with
    the webcam. Other apps then see a silent exclusive mic.
    """
    params: list[int] = []
    video = getattr(cv2, "CAP_PROP_VIDEO_STREAM", None)
    if video is not None:
        params.extend([int(video), 0])
    stream = getattr(cv2, "CAP_PROP_AUDIO_STREAM", None)
    if stream is not None:
        params.extend([int(stream), -1])
    return params


def open_video(index: int, backend: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture()
    params = _no_mic_params()
    if params:
        cap.open(int(index), int(backend), params)
    else:
        cap.open(int(index), int(backend))
    return cap


ROOT = Path(__file__).resolve().parents[1]
VENDOR_OSF = ROOT.parent / "vendor" / "tools" / "openseeface"
CAM_SAVE = ROOT / "output" / "camera.json"
_PREFER = ("droidcam", "obs", "webcam", "usb", "hd ", "integrated", "nizima", "camera")
_AUDIO_DEVICE = ("microphone", "stereo mix", "wave in", "what u hear", "speaks")
_com = threading.local()


def _is_camera_name(name: str) -> bool:
    n = name.lower()
    if any(key in n for key in ("camera", "webcam", "droidcam", "obs", "virtual", "capture")):
        return True
    return not any(key in n for key in _AUDIO_DEVICE)


def _ensure_com() -> None:
    if os.name != "nt":
        return
    if getattr(_com, "ready", False):
        return
    import ctypes

    ctypes.windll.ole32.CoInitializeEx(None, 0x0)
    _com.ready = True


def load_camera_index() -> int | None:
    if not CAM_SAVE.is_file():
        return None
    try:
        data = json.loads(CAM_SAVE.read_text(encoding="utf-8"))
        return int(data.get("index", 0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_camera_index(index: int) -> None:
    CAM_SAVE.parent.mkdir(parents=True, exist_ok=True)
    CAM_SAVE.write_text(json.dumps({"index": int(index)}, indent=2), encoding="utf-8")


def pick_default(cameras: list[dict[str, object]], saved: int | None = None) -> int:
    if saved is not None:
        for cam in cameras:
            if int(cam["index"]) == int(saved):
                return int(saved)
    for key in _PREFER:
        for cam in cameras:
            if key in str(cam.get("name", "")).lower():
                return int(cam["index"])
    return int(cameras[0]["index"]) if cameras else 0


def list_cameras() -> list[dict[str, object]]:
    _ensure_com()
    cams = _list_dshow()
    if cams:
        return cams
    return _list_opencv()


def _list_dshow() -> list[dict[str, object]]:
    if os.name != "nt" or not (VENDOR_OSF / "dshowcapture.py").is_file():
        return []
    path = str(VENDOR_OSF)
    if path not in sys.path:
        sys.path.append(path)
    try:
        import dshowcapture  # noqa: E402
    except Exception:
        return []
    _ensure_com()
    cap = dshowcapture.DShowCapture()
    try:
        try:
            cap.get_devices()
        except Exception:
            pass
        info = cap.get_info() or []
        cameras: list[dict[str, object]] = []
        for i, cam in enumerate(info):
            if not isinstance(cam, dict):
                continue
            idx = int(cam.get("index", cam.get("id", i)))
            name = str(cam.get("name") or f"Camera {idx}")
            if not _is_camera_name(name):
                continue
            cameras.append({"index": idx, "name": name, "backend": "dshow"})
        return cameras
    except Exception:
        return []
    finally:
        try:
            cap.destroy_capture()
        except Exception:
            pass


def _list_opencv(max_probe: int = 8) -> list[dict[str, object]]:
    cameras: list[dict[str, object]] = []
    for i in range(max_probe):
        cap = None
        try:
            if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
                cap = open_video(i, cv2.CAP_DSHOW)
            else:
                cap = open_video(i, cv2.CAP_ANY)
            if cap is None or not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, "size", 0):
                cameras.append({"index": i, "name": f"Camera {i}", "backend": "opencv"})
        except Exception:
            continue
        finally:
            if cap is not None:
                cap.release()
    return cameras


def open_capture(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    _ensure_com()
    backends: list[int] = []
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(cv2.CAP_ANY)
    for backend in backends:
        cap = open_video(int(index), backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0):
            return cap
        cap.release()
    return None
