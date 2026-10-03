"""List capture devices for the OSF bench.

Windows uses OpenSeeFace DirectShow so virtual cams show up by name.
Falls back to probing OpenCV indices.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2

from . import debug_log
from .paths import OUTPUT


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
    """Open one index. Mic-off params first; some virtual cams reject them."""
    cap = cv2.VideoCapture()
    params = _no_mic_params()
    if params:
        cap.open(int(index), int(backend), params)
        if cap.isOpened():
            return cap
        cap.release()
        cap = cv2.VideoCapture()
    cap.open(int(index), int(backend))
    return cap


ROOT = Path(__file__).resolve().parents[1]
VENDOR_OSF = ROOT.parent / "vendor" / "tools" / "openseeface"
CAM_SAVE = OUTPUT / "camera.json"
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


def load_camera_choice() -> tuple[int | None, str]:
    """Last camera the user picked: index plus the device name."""
    if not CAM_SAVE.is_file():
        return None, ""
    try:
        data = json.loads(CAM_SAVE.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, ""
    if not isinstance(data, dict):
        return None, ""
    name = str(data.get("name") or "").strip()
    try:
        return int(data.get("index", 0)), name
    except (TypeError, ValueError):
        return None, name


def load_camera_index() -> int | None:
    index, _name = load_camera_choice()
    return index


def save_camera_index(index: int, name: str | None = None) -> None:
    """Remember this camera. A blank name keeps the previous name for the same index."""
    prev_index, prev_name = load_camera_choice()
    if name is None:
        chosen = prev_name if prev_index == int(index) else ""
    else:
        chosen = str(name).strip()
    CAM_SAVE.parent.mkdir(parents=True, exist_ok=True)
    CAM_SAVE.write_text(
        json.dumps({"index": int(index), "name": chosen}, indent=2),
        encoding="utf-8",
    )


def pick_default(
    cameras: list[dict[str, object]],
    saved: int | None = None,
    *,
    saved_name: str = "",
) -> int:
    """Use the last camera the user chose, then a known webcam, then the first device."""
    wanted = str(saved_name or "").strip().lower()
    if wanted:
        for cam in cameras:
            if str(cam.get("name") or "").strip().lower() == wanted:
                return int(cam["index"])
    # A named pick that is gone: its old index now belongs to another device.
    elif saved is not None:
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
            cameras.append({
                "index": idx,
                "name": name,
                "backend": "dshow",
                "caps": list(cam.get("caps") or []),
            })
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


def _frame_ready(frame: object) -> bool:
    return frame is not None and int(getattr(frame, "size", 0) or 0) > 0


def _pull_frame(cap: object, tries: int = 12, pause: float = 0.04) -> object | None:
    """Virtual cams (DroidCam, OBS) often open before the first real frame."""
    read = getattr(cap, "read", None)
    if not callable(read):
        return None
    for i in range(max(1, tries)):
        try:
            ok, frame = read()
        except Exception:
            ok, frame = False, None
        if ok and _frame_ready(frame):
            return frame
        if i + 1 < tries and pause > 0:
            time.sleep(pause)
    return None


def _opencv_backends() -> list[int]:
    if os.name != "nt":
        return [int(cv2.CAP_ANY)]
    found: list[int] = []
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
        backend = getattr(cv2, name, None)
        if backend is None:
            continue
        value = int(backend)
        if value not in found:
            found.append(value)
    return found or [int(cv2.CAP_ANY)]


def _size_plan(width: int, height: int) -> list[tuple[int, int]]:
    """Native mode first. Forcing 640x480 makes some virtual cams refuse frames."""
    plan = [(0, 0), (int(width), int(height)), (1280, 720), (640, 480)]
    out: list[tuple[int, int]] = []
    for size in plan:
        if size not in out and size[0] >= 0 and size[1] >= 0:
            out.append(size)
    return out


def _apply_size(cap: cv2.VideoCapture, width: int, height: int) -> None:
    if width > 0 and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


class _DshowHold:
    """Same read/release surface as VideoCapture, backed by OpenSeeFace DirectShow."""

    def __init__(self, reader: object) -> None:
        self._reader = reader

    def read(self) -> tuple[bool, object]:
        return self._reader.read()  # type: ignore[attr-defined]

    def isOpened(self) -> bool:
        opened = getattr(self._reader, "is_open", None)
        if callable(opened):
            return bool(opened())
        return True

    def release(self) -> None:
        closer = getattr(self._reader, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
        self._reader = None


# DirectShow frame intervals, 100 ns units: 30 fps, and the slowest still
# counted as 30 (29.97 fps NTSC modes report 333667).
_WANT_IV = 333333
_SLOWEST_IV = 340000


def _rank_dcaps(
    caps: list[dict],
    width: int,
    height: int,
    *,
    prefer_mid: bool,
) -> list[int]:
    """One DirectShow capability id per resolution. DroidCam repeats each size.

    Of one size's modes, the one whose top rate is nearest 30 fps without
    falling short: the tracker keeps ~30 frames a second, so a 60 fps mode
    only doubles the decode and copy work in the grab thread.
    """

    def rate(min_iv: int) -> tuple[int, int]:
        return (1 if min_iv > _SLOWEST_IV else 0, abs(min_iv - _WANT_IV))

    def score(cap: dict) -> tuple:
        cx = int(cap.get("minCX") or 0)
        cy = int(cap.get("minCY") or 0)
        min_iv = int(cap.get("minInterval") or _WANT_IV)
        if prefer_mid:
            mid = abs(cx - 640) + abs(cy - 480)
            huge = 1 if (cx * cy) >= (1280 * 720) else 0
            return (huge, mid, abs(cx - width) + abs(cy - height), rate(min_iv))
        return (abs(cx - width) + abs(cy - height), abs(cx - 960) + abs(cy - 720), rate(min_iv))

    best: dict[tuple[int, int], dict] = {}
    for cap in caps:
        cx = int(cap.get("minCX") or 0)
        cy = int(cap.get("minCY") or 0)
        if cx <= 0 or cy <= 0 or cap.get("id") is None:
            continue
        key = (cx, cy)
        prev = best.get(key)
        if prev is None or score(cap) < score(prev):
            best[key] = cap
    ranked = sorted(best.values(), key=score)
    return [int(cap["id"]) for cap in ranked[:4]]


def _camera_caps(index: int) -> tuple[str, list[dict]]:
    devices = _list_dshow()
    for cam in devices:
        if int(cam["index"]) != int(index):
            continue
        raw = cam.get("caps")
        caps = list(raw) if isinstance(raw, list) else []
        return str(cam.get("name") or ""), caps
    return "", []


def _open_dshow_reader(index: int, width: int, height: int) -> tuple[object | None, bool]:
    """Open the same DirectShow index the camera list uses.

    OpenCV's index often does not match that list, so virtual cams at
    index 1 fail even when nothing else is reading them.
    """
    if os.name != "nt" or not (VENDOR_OSF / "dshowcapture.py").is_file():
        return None, False
    path = str(VENDOR_OSF)
    if path not in sys.path:
        sys.path.append(path)
    try:
        from input_reader import DShowCaptureReader  # noqa: E402
    except Exception:
        return None, False
    name, caps = _camera_caps(index)
    if not name:
        return None, False
    prefer_mid = "droidcam" in name.lower()
    attempts: list[int | None] = _rank_dcaps(caps, width, height, prefer_mid=prefer_mid)
    attempts.extend([None, -1])
    fallback: object | None = None
    for attempt in attempts:
        reader = None
        try:
            reader = DShowCaptureReader(
                int(index),
                int(width),
                int(height),
                30,
                use_dshowcapture=True,
                dcap=attempt,
            )
            if not reader.is_open():
                reader.close()
                continue
            frame = _pull_frame(reader, tries=10, pause=0.05)
            if frame is None:
                reader.close()
                time.sleep(0.12)
                continue
            if _picture(frame):
                # #region agent log
                if debug_log.ENABLED:
                    import numpy as np

                    _sample = np.asarray(frame)
                    _step = max(1, min(_sample.shape[:2]) // 16) if getattr(_sample, "ndim", 0) >= 2 else 1
                    debug_log.log(
                        "D",
                        "cameras.py:_open_dshow_reader",
                        "dshow pictured",
                        {
                            "index": int(index),
                            "name": name,
                            "dcap": attempt,
                            "shape": list(_sample.shape),
                            "std": round(float(np.std(_sample[::_step, ::_step])), 2),
                        },
                    )
                # #endregion
                if fallback is not None:
                    _release_reader(fallback)
                return reader, True
            if fallback is None:
                fallback = reader
            else:
                reader.close()
        except Exception:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
        time.sleep(0.12)
    # #region agent log
    debug_log.log(
        "D",
        "cameras.py:_open_dshow_reader",
        "dshow fallback",
        {"index": int(index), "name": name, "pictured": False, "has_reader": fallback is not None},
    )
    # #endregion
    return fallback, False


def _release_reader(reader: object) -> None:
    closer = getattr(reader, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _picture(frame: object) -> bool:
    """A solid black buffer is a DroidCam mode, not a live picture."""
    try:
        import numpy as np

        sample = np.asarray(frame)
        if sample.size == 0:
            return False
        step = max(1, min(sample.shape[:2]) // 32) if sample.ndim >= 2 else 1
        picked = sample[::step, ::step] if sample.ndim >= 2 else sample
        return float(np.std(picked)) >= 2.0
    except Exception:
        return True


def _open_opencv_capture(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    for backend in _opencv_backends():
        for w, h in _size_plan(width, height):
            cap = None
            try:
                cap = open_video(int(index), backend)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    break
                _apply_size(cap, w, h)
                if _pull_frame(cap) is not None:
                    # #region agent log
                    debug_log.log(
                        "D",
                        "cameras.py:_open_opencv_capture",
                        "opencv open",
                        {"index": int(index), "backend": int(backend), "size": [int(w), int(h)]},
                    )
                    # #endregion
                    return cap
                cap.release()
            except Exception:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
    return None


def open_capture(index: int, width: int, height: int) -> cv2.VideoCapture | _DshowHold | None:
    """Open a real or virtual camera. Listing index is a DirectShow index."""
    _ensure_com()
    opened = _open_dshow_reader(int(index), int(width), int(height))
    reader: object | None
    pictured = False
    if isinstance(opened, tuple):
        reader, pictured = opened
    else:
        reader, pictured = opened, opened is not None
    if reader is not None and pictured:
        return _DshowHold(reader)
    cap = _open_opencv_capture(int(index), int(width), int(height))
    if cap is not None:
        if reader is not None:
            _release_reader(reader)
        return cap
    if reader is not None:
        # #region agent log
        debug_log.log("D", "cameras.py:open_capture", "using dull dshow", {"index": int(index)})
        # #endregion
        return _DshowHold(reader)
    # #region agent log
    debug_log.log("D", "cameras.py:open_capture", "open failed", {"index": int(index)})
    # #endregion
    return None
