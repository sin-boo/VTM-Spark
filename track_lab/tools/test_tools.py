"""Recorders: a fake iPhone over loopback and a fake camera."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import numpy as np

from tools.guide import Guide


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _short_guide(voice=None) -> Guide:
    return Guide(steps=[("rest", "Hold", 0.4), ("left", "Left", 0.4)], voice=voice)


def test_guide_walks_the_steps_once_each() -> None:
    guide = _short_guide()
    assert guide.total == 0.8
    assert guide.tick(0.0) == ("rest", True, False)
    assert guide.tick(0.1) == ("rest", False, False)
    assert guide.tick(0.5) == ("left", True, False)
    assert guide.tick(0.9)[2] is True


def test_ifm_recorder_keeps_every_packet_with_its_step(tmp_path: Path, monkeypatch) -> None:
    import tools.record_ifm as rec

    port = _free_port()
    phone = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    phone.bind(("127.0.0.1", 0))
    phone.settimeout(5.0)
    monkeypatch.setattr(rec, "saved_phone", lambda: {})
    monkeypatch.setattr(rec, "handshake_targets", lambda *_a: [phone.getsockname()])
    monkeypatch.setattr(rec, "Guide", lambda voice=None: _short_guide(voice))
    stop = threading.Event()

    def stream() -> None:
        data, addr = phone.recvfrom(4096)
        assert data.decode().startswith("iFacialMocap_")
        yaw = 0.0
        while not stop.is_set():
            yaw += 1.0
            phone.sendto(f"jawOpen-0|=head#2.0,{yaw},0.5,0,0,0".encode(), ("127.0.0.1", port))
            stop.wait(0.01)

    worker = threading.Thread(target=stream, daemon=True)
    worker.start()
    out = tmp_path / "ifm.jsonl"
    try:
        assert rec.record(out, phone="", port=port, free=0.0, voice_on=False) == 0
    finally:
        stop.set()
        worker.join(timeout=2.0)
        phone.close()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "header" and rows[0]["guided"] is True
    steps = [row["step"] for row in rows if row.get("kind") == "step"]
    assert steps == ["rest", "left"]
    packets = [row for row in rows if "raw" in row]
    assert len(packets) > 20
    assert {row["step"] for row in packets} == {"rest", "left"}
    assert all(packets[i]["t"] <= packets[i + 1]["t"] for i in range(len(packets) - 1))
    lines = rec.summarize(packets)
    assert lines[0].startswith("step") and any(line.startswith("left") for line in lines)


def test_ifm_recorder_says_when_the_port_is_taken() -> None:
    import pytest

    import tools.record_ifm as rec

    holder = rec.listen(_free_port())
    try:
        with pytest.raises(SystemExit, match="Track Lab"):
            rec.listen(holder.getsockname()[1])
    finally:
        holder.close()


class _FakeCamera:
    def __init__(self) -> None:
        self.released = False

    def read(self) -> tuple[bool, np.ndarray]:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = 200
        return True, frame

    def release(self) -> None:
        self.released = True


def test_video_recorder_writes_frames_and_their_steps(tmp_path: Path, monkeypatch) -> None:
    import tools.record_video as rec

    cam = _FakeCamera()
    monkeypatch.setattr(rec, "open_capture", lambda *_a: cam)
    monkeypatch.setattr(rec, "saved_camera", lambda: {"index": 3, "name": "Fake"})
    monkeypatch.setattr(rec, "Guide", lambda voice=None: _short_guide(voice))
    out = tmp_path / "cam.mp4"
    assert rec.record(out, camera=None, free=0.0, voice_on=False) == 0
    assert cam.released
    assert out.is_file() and out.stat().st_size > 0
    rows = [json.loads(line) for line in out.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["camera"] == 3 and rows[0]["size"] == [64, 48]
    frames = rows[1:]
    assert frames and {row["step"] for row in frames} == {"rest", "left"}
    assert [row["i"] for row in frames] == list(range(len(frames)))


def test_video_recorder_reports_a_camera_that_will_not_open(tmp_path: Path, monkeypatch) -> None:
    import tools.record_video as rec

    monkeypatch.setattr(rec, "open_capture", lambda *_a: None)
    monkeypatch.setattr(rec, "saved_camera", lambda: {})
    assert rec.record(tmp_path / "cam.mp4", camera=0, free=1.0, voice_on=False) == 1
