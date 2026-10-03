"""Live tracking must drop stale camera / UDP frames instead of queuing delay."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np

from backend.ifm_cam import recv_latest
from backend.osf_cam import OsfCam
from backend import osf_cam as osf_cam_mod


def test_recv_latest_keeps_newest_datagram() -> None:
    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        recv.bind(("127.0.0.1", 0))
        recv.settimeout(1.0)
        addr = ("127.0.0.1", int(recv.getsockname()[1]))
        for payload in (b"one", b"two", b"three"):
            send.sendto(payload, addr)
        got, _peer = recv_latest(recv)
        assert got == b"three"
    finally:
        recv.close()
        send.close()


def test_tracker_is_built_before_the_camera_opens(monkeypatch) -> None:
    order: list[str] = []

    class FakeTracker:
        def close(self) -> None:
            return None

    def make_tracker(w: int, h: int) -> FakeTracker:
        order.append(f"tracker:{w}x{h}")
        return FakeTracker()

    def open_camera(index: int):
        order.append(f"open:{index}")
        return None

    monkeypatch.setattr(osf_cam_mod, "_make_tracker", make_tracker)
    monkeypatch.setattr(osf_cam_mod, "_open_camera", open_camera)
    cam = OsfCam()
    try:
        cam.start(index=3)
    except RuntimeError:
        pass
    assert order == ["tracker:640x480", "open:3"]


def test_osf_processes_latest_frame_not_backlog(monkeypatch) -> None:
    produced = {"n": 0}
    processed: list[int] = []

    class FakeCap:
        def read(self):
            produced["n"] += 1
            n = produced["n"]
            time.sleep(0.004)
            return True, np.full((12, 12, 3), n % 255, dtype=np.uint8)

        def release(self) -> None:
            return None

    class FakeTracker:
        def predict(self, frame):
            processed.append(int(frame[0, 0, 0]))
            time.sleep(0.03)
            return []

        def close(self) -> None:
            return None

    monkeypatch.setattr(osf_cam_mod, "_make_tracker", lambda w, h: FakeTracker())
    monkeypatch.setattr(osf_cam_mod, "_encode_jpeg", lambda img: b"x")

    cam = OsfCam()
    cam._cap = FakeCap()
    cam._running = True
    thread = threading.Thread(target=cam._loop, args=(lambda snap: None,), daemon=True)
    cam._thread = thread
    thread.start()
    time.sleep(0.22)
    cam.stop()
    thread.join(timeout=2.0)
    assert produced["n"] >= 8
    assert len(processed) >= 2
    assert produced["n"] > len(processed) * 2


def test_ifm_processes_latest_packet_not_backlog(monkeypatch) -> None:
    from backend import ifm_cam as ifm_cam_mod
    from backend.ifm_cam import IfmCam

    monkeypatch.setattr(ifm_cam_mod, "_encode_jpeg", lambda img: b"x")

    listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    processed: list[float] = []
    try:
        listen.bind(("127.0.0.1", 0))
        listen.settimeout(0.05)
        port = int(listen.getsockname()[1])
        addr = ("127.0.0.1", port)
        cam = IfmCam()
        cam._sock = listen
        cam._running = True
        cam._started = time.perf_counter()

        def on_frame(snap) -> None:
            if not snap.faces:
                return
            processed.append(float((snap.head or {}).get("yaw") or 0.0))
            time.sleep(0.04)

        thread = threading.Thread(target=cam._loop, args=(on_frame,), daemon=True)
        cam._thread = thread
        thread.start()
        for index in range(1, 25):
            send.sendto(f"jawOpen-0|=head#0,{index},0".encode("utf-8"), addr)
            time.sleep(0.004)
        time.sleep(0.12)
        cam.stop()
        thread.join(timeout=2.0)
        assert len(processed) >= 1
        assert len(processed) < 18
        assert processed[-1] >= 18
    finally:
        send.close()
        try:
            listen.close()
        except OSError:
            pass


def _settle_s(smooth: float, fps: float, move: float = 0.5, turn_s: float = 0.0) -> float:
    """Seconds for the head ease to cover 90 % of a turn (radians) the head
    makes in ``turn_s`` seconds (0 = at once)."""
    from backend.ease import HeadEase

    ease = HeadEase()
    ease.step((0.0,), (1.0,), smooth, 0.0)
    for i in range(1, int(fps * 5)):
        now = i / fps
        target = move * min(now / turn_s, 1.0) if turn_s > 0.0 else move
        if float(ease.step((target,), (1.0,), smooth, now)[0]) >= 0.9 * move:
            return now
    return 5.0


def test_stronger_smooth_eases_a_move_longer() -> None:
    """The old blend let go on any move: Smooth only touched a still head."""
    light = _settle_s(0.25, 30.0)
    mid = _settle_s(0.5, 30.0)
    strong = _settle_s(1.0, 30.0)
    assert light < mid < strong
    assert strong > 0.3
    # Smooth 0 is the raw head.
    assert _settle_s(0.0, 30.0) <= 1.0 / 30.0


def test_head_ease_is_timed_in_seconds_not_frames() -> None:
    """One Smooth feels the same on a 12 fps webcam and a 60 fps iPhone."""
    for smooth in (0.5, 1.0):
        # A brisk turn, 0.3 s end to end; 12 fps lands only every 83 ms.
        slow = _settle_s(smooth, 12.0, turn_s=0.3)
        fast = _settle_s(smooth, 60.0, turn_s=0.3)
        assert abs(slow - fast) <= 1.0 / 12.0 + 0.2 * max(slow, fast)


def test_head_ease_holds_jitter_and_keeps_easing_a_move() -> None:
    import numpy as np

    from backend.ease import HeadEase

    rng = np.random.default_rng(0)
    ease = HeadEase()
    out = [float(ease.step((v,), (1.0,), 0.5, i / 30.0)[0]) for i, v in enumerate(rng.normal(0, 0.004, 90))]
    # Tracker jitter held still: well under half of it gets through.
    assert np.std(np.diff(out[30:])) < 0.5 * 0.004 * np.sqrt(2.0)
    # A steady 12 fps turn: every frame trails the reading (still easing,
    # not snapped to it), then settles on it once the head stops.
    ease = HeadEase()
    ease.step((0.0,), (1.0,), 0.5, 0.0)
    for i in range(1, 7):
        target = 0.08 * i
        assert float(ease.step((target,), (1.0,), 0.5, i / 12.0)[0]) < target - 0.005
    for i in range(7, 40):
        got = float(ease.step((0.48,), (1.0,), 0.5, i / 12.0)[0])
    assert abs(got - 0.48) < 0.005


def test_jitter_does_not_hold_a_still_head_ease_open() -> None:
    """Webcam jitter on a still head must not read as movement: the ease keeps
    close to its designed still time constant instead of a third of it."""
    import math

    import numpy as np

    from backend.ease import HeadEase, cutoffs

    rng = np.random.default_rng(1)
    fps = 24.0
    jitter = math.radians(0.7)
    for smooth in (0.88, 1.0):
        ease = HeadEase()
        noise = rng.normal(0.0, jitter, 240)
        out = [float(ease.step((v,), (1.0,), smooth, i / fps)[0]) for i, v in enumerate(noise)]
        still_hz, _ = cutoffs(smooth)
        designed = 1.0 / (2.0 * math.pi * still_hz)
        # A first-order ease with time constant tau passes std * sqrt(dt / (2 tau + dt)).
        tau_seen = (1.0 / fps) * (np.var(noise[48:]) / np.var(out[48:]) - 1.0) / 2.0
        assert tau_seen > 0.75 * designed


def test_rig_eases_the_turn_but_not_the_mouth(monkeypatch) -> None:
    """Smooth eases the head in the rig; the mesh it is handed (mouth,
    brows) goes through at once, so a strong Smooth does not blur lip sync."""
    import math
    from types import SimpleNamespace

    import numpy as np

    from backend import rig as rig_mod
    from backend.feel import feel
    from backend.rig import FaceRig

    clock = {"t": 0.0}
    monkeypatch.setattr(rig_mod, "time", SimpleNamespace(perf_counter=lambda: clock["t"]))
    rest = np.zeros((28, 3), dtype=np.float32)
    rest[:, 2] = 1.0
    rest[0, 0], rest[4, 0] = 0.0, 100.0
    rest[15] = [50.0, 40.0, 1.0]
    rest[21] = [50.0, 70.0, 1.0]
    rest[25] = [50.0, 72.0, 1.0]
    pose = {"cx": 200.0, "cy": 200.0, "bx": 200.0, "by": 200.0, "scale": 100.0, "tilt": 0.0, "ok": 1.0}
    still = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    turned = {"pitch": 0.0, "yaw": 20.0, "roll": 0.0}

    def yaw_after(smooth: float, frames: int) -> float:
        feel.update({"smoothing": smooth, "max_yaw": 1.0})
        rig = FaceRig()
        clock["t"] = 0.0
        rig.apply(rest, rest, still, pose)
        for i in range(1, frames + 1):
            clock["t"] = i / 30.0
            rig.apply(rest, rest, turned, pose)
        return math.degrees(rig.turn()["yaw"])

    assert abs(yaw_after(0.0, 1) - 20.0) < 0.01
    assert yaw_after(1.0, 1) < 0.25 * 20.0
    assert yaw_after(0.5, 6) > yaw_after(1.0, 6)
    assert abs(yaw_after(1.0, 90) - 20.0) < 0.5

    feel.update({"smoothing": 1.0})
    rig = FaceRig()
    clock["t"] = 0.0
    rig.apply(rest, rest, still, pose)
    opened = rest.copy()
    opened[25, 1] = 84.0
    clock["t"] = 1.0 / 30.0
    out = rig.apply(opened, rest, still, pose)
    assert out is not None
    assert abs(float(out[25, 1] - out[21, 1]) - 14.0) < 0.01


# --- CPU next to a game ------------------------------------------------------


class _PacedCap:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.n = 0
        self.t0 = time.perf_counter()

    def read(self):
        self.n += 1
        wait = self.t0 + self.n / self.fps - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        return True, np.full((12, 12, 3), self.n % 255, dtype=np.uint8)

    def release(self) -> None:
        return None


class _CountTracker:
    def __init__(self, face: bool = True) -> None:
        self.calls = 0
        self.face = face

    def predict(self, frame):
        self.calls += 1
        return [object()] if self.face else []

    def close(self) -> None:
        return None


def _run_osf(monkeypatch, cap: _PacedCap, tracker: _CountTracker, seconds: float) -> OsfCam:
    monkeypatch.setattr(osf_cam_mod, "_make_tracker", lambda w, h: tracker)
    cam = OsfCam()
    cam._cap = cap
    cam._tracker = tracker
    cam._running = True
    thread = threading.Thread(target=cam._loop, args=(lambda snap: None,), daemon=True)
    cam._thread = thread
    thread.start()
    time.sleep(seconds)
    return cam


def test_camera_preview_is_encoded_only_while_read(monkeypatch) -> None:
    encoded = {"n": 0}

    def encode(img) -> bytes:
        encoded["n"] += 1
        return b"jpeg%d" % encoded["n"]

    monkeypatch.setattr(osf_cam_mod, "_encode_jpeg", encode)
    monkeypatch.setattr(osf_cam_mod, "_preview_read_t", -1e9)
    cam = _run_osf(monkeypatch, _PacedCap(100.0), _CountTracker(face=False), 0.25)
    try:
        # Nobody reading (Track Lab shut while streaming): no JPEG work.
        assert encoded["n"] == 0
        # The first read encodes the newest grab on the spot...
        first = osf_cam_mod.want_preview()
        assert first and encoded["n"] == 1
        # ...and while reads keep coming, each grab is encoded for them.
        time.sleep(0.15)
        assert encoded["n"] >= 5
        assert osf_cam_mod.want_preview() is None
        assert cam._latest_preview().startswith(b"jpeg")
    finally:
        cam.stop()


def test_tracker_keeps_about_30_frames_a_second_off_a_60_fps_camera(monkeypatch) -> None:
    tracker = _CountTracker()
    monkeypatch.setattr(osf_cam_mod, "_encode_jpeg", lambda img: b"x")
    cam = _run_osf(monkeypatch, _PacedCap(60.0), tracker, 1.0)
    cam.stop()
    assert 22 <= tracker.calls <= 36


def test_tracker_tracks_every_frame_of_a_30_fps_camera(monkeypatch) -> None:
    tracker = _CountTracker()
    cap = _PacedCap(30.0)
    monkeypatch.setattr(osf_cam_mod, "_encode_jpeg", lambda img: b"x")
    cam = _run_osf(monkeypatch, cap, tracker, 1.0)
    cam.stop()
    assert tracker.calls >= cap.n - 3


def test_tracker_hunts_a_missing_face_less_often(monkeypatch) -> None:
    tracker = _CountTracker(face=False)
    monkeypatch.setattr(osf_cam_mod, "_encode_jpeg", lambda img: b"x")
    monkeypatch.setattr(osf_cam_mod, "_NO_FACE_HOLD_S", 0.2)
    cam = _run_osf(monkeypatch, _PacedCap(30.0), tracker, 1.2)
    cam.stop()
    # ~0.2 s at 30 a second, then ~10 a second: about 16, not 36.
    assert 8 <= tracker.calls <= 22


def test_onnx_threads_fit_the_pc() -> None:
    plan = osf_cam_mod._onnx_plan
    assert plan(1) == (1, False)
    assert plan(4) == (2, False)
    assert plan(8) == (2, False)
    assert plan(32) == (4, True)


def test_osf_sessions_are_capped_and_stop_spinning() -> None:
    import pytest

    ort = pytest.importorskip("onnxruntime")
    opened: list = []

    class Real:
        SessionOptions = ort.SessionOptions
        ExecutionMode = ort.ExecutionMode

        @staticmethod
        def InferenceSession(path, sess_options=None, **kwargs):  # noqa: N802
            opened.append(sess_options)
            return path

    quiet = osf_cam_mod._QuietOrt(Real, threads=2, spin=False)
    assert quiet.ExecutionMode is ort.ExecutionMode
    wide = ort.SessionOptions()
    wide.intra_op_num_threads = 4
    quiet.InferenceSession("retina.onnx", sess_options=wide, providers=["CPUExecutionProvider"])
    one = ort.SessionOptions()
    one.intra_op_num_threads = 1
    # OpenSeeFace hands one options object to two sessions.
    quiet.InferenceSession("gaze.onnx", sess_options=one)
    quiet.InferenceSession("detect.onnx", sess_options=one)
    assert [o.intra_op_num_threads for o in opened] == [2, 1, 1]
    assert all(o.get_session_config_entry("session.intra_op.allow_spinning") == "0" for o in opened)

    big = osf_cam_mod._QuietOrt(Real, threads=4, spin=True)
    big.InferenceSession("lm.onnx", sess_options=ort.SessionOptions())
    assert opened[-1].intra_op_num_threads == 4
    assert opened[-1].get_session_config_entry("session.force_spinning_stop") == "1"


def test_camera_mode_prefers_30_fps_over_60() -> None:
    from backend.cameras import _rank_dcaps

    caps = [
        {"id": 0, "minCX": 640, "minCY": 480, "minInterval": 166666},
        {"id": 1, "minCX": 640, "minCY": 480, "minInterval": 333333},
        {"id": 2, "minCX": 640, "minCY": 480, "minInterval": 400000},
        # A size with only 10 fps and 60 fps modes keeps 60, not 10.
        {"id": 3, "minCX": 1280, "minCY": 720, "minInterval": 1000000},
        {"id": 4, "minCX": 1280, "minCY": 720, "minInterval": 166666},
    ]
    assert _rank_dcaps(caps, 640, 480, prefer_mid=False) == [1, 4]
    assert _rank_dcaps(caps, 640, 480, prefer_mid=True) == [1, 4]
