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


def test_smoothing_holds_still_and_lets_go_on_a_move() -> None:
    """A fixed Smooth dragged every head move ~50 ms; still jitter needs it."""
    import numpy as np

    from backend.face import motion_alpha

    base = 0.4
    face = np.random.default_rng(0).normal(0, 40, (28, 2))
    # Jitter well under 1 % of the face: full smoothing.
    assert motion_alpha(base, face + 0.5, face, 200.0) == base
    # A real move (5 % of the face this frame): follow at once.
    assert motion_alpha(base, face + 10.0, face, 200.0) == 1.0
    mid = motion_alpha(base, face + 3.0, face, 200.0)
    assert base < mid < 1.0
    # One point jumping (a blink, the mouth) is not the head moving.
    blink = face.copy()
    blink[3] += 30.0
    assert motion_alpha(base, blink, face, 200.0) == base
