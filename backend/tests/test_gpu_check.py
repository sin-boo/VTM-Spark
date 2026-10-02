from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import gpu_check as gc


def _card(name: str, cap: tuple[int, int] | None, uuid: str = "GPU-1", driver: str = "560.94") -> dict:
    return {"uuid": uuid, "name": name, "driver": driver, "cap": cap}


@pytest.mark.parametrize(
    "cap, want",
    [
        ((3, 5), []),  # Kepler: no current torch build
        ((5, 2), ["cu126"]),  # GTX 970
        ((6, 1), ["cu126"]),  # GTX 1080: cu128 dropped Pascal
        ((7, 0), ["cu126"]),  # Titan V
        ((7, 5), ["cu128", "cu126"]),  # RTX 2070
        ((8, 6), ["cu128", "cu126"]),  # RTX 3060
        ((8, 9), ["cu128", "cu126"]),  # RTX 4070
        ((12, 0), ["cu128"]),  # RTX 5060 Ti: cu126 has no Blackwell
        (None, ["cu128", "cu126"]),  # driver too old to report compute_cap
    ],
)
def test_builds_for_each_generation(cap, want) -> None:
    assert gc.builds_for(cap) == want
    assert gc.want_build(cap) == (want[0] if want else "cu128")


def test_parse_cards_with_and_without_compute_cap() -> None:
    rows = gc.parse_cards("GPU-a, NVIDIA GeForce RTX 3060, 552.44, 8.6\nnoise\n", with_cap=True)
    assert rows == [_card("NVIDIA GeForce RTX 3060", (8, 6), "GPU-a", "552.44")]
    rows = gc.parse_cards("GPU-b, NVIDIA GeForce GTX 1080, 472.12\n", with_cap=False)
    assert rows == [_card("NVIDIA GeForce GTX 1080", None, "GPU-b", "472.12")]


def test_query_cards_falls_back_when_the_driver_lacks_compute_cap(monkeypatch) -> None:
    monkeypatch.setattr("backend.gpu_select._nvidia_smi", lambda: "nvidia-smi")
    seen = []

    def run(cmd, **_kw):
        seen.append(cmd[1])
        if "compute_cap" in cmd[1]:
            return SimpleNamespace(returncode=2, stdout="")
        return SimpleNamespace(returncode=0, stdout="GPU-x, NVIDIA GeForce GTX 1070, 456.71\n")

    assert gc.query_cards(run=run) == [_card("NVIDIA GeForce GTX 1070", None, "GPU-x", "456.71")]
    assert len(seen) == 2
    monkeypatch.setattr("backend.gpu_select._nvidia_smi", lambda: None)
    assert gc.query_cards(run=run) == []


def test_target_card_is_the_pinned_one_else_the_newest() -> None:
    old, new = _card("GTX 1080", (6, 1), "GPU-old"), _card("RTX 4070", (8, 9), "GPU-new")
    assert gc.target_card([old, new], env={}) is new
    assert gc.target_card([old, new], env={"CUDA_VISIBLE_DEVICES": "GPU-old"}) is old
    assert gc.target_card([], env={}) is None


FAIL = {"ok": False, "error": "CUDA error: no kernel image is available for execution on the device", "build": "cu128"}


def test_diagnose_kinds() -> None:
    assert gc.diagnose(_card("RTX 3060", (8, 6)), {"ok": True, "error": "", "build": "cu128"}, []) is None
    assert gc.diagnose(None, {**FAIL, "error": "CUDA is not available"}, [])["kind"] == "no_nvidia"
    assert gc.diagnose(_card("GTX 780", (3, 5)), FAIL, [])["kind"] == "card_too_old"
    old_driver = {**FAIL, "error": "The NVIDIA driver on your system is too old (found version 11040)."}
    problem = gc.diagnose(_card("RTX 3060", (8, 6), driver="472.12"), old_driver, [])
    assert problem["kind"] == "driver_old" and problem["repair"] is False  # no build helps
    # A GTX 1080 under a cu128 build: the cu126 build fits, Repair can put it in.
    problem = gc.diagnose(_card("GTX 1080", (6, 1)), FAIL, [])
    assert problem["kind"] == "wrong_build" and problem["repair"] is True and problem["build"] == "CUDA 12.8"
    # ...unless cu126 already failed on this card.
    tried = [{"what": "build", "build": "cu126", "ok": False, "error": "x"}]
    assert gc.diagnose(_card("GTX 1080", (6, 1)), FAIL, tried)["repair"] is False
    # A CPU wheel on an RTX card.
    problem = gc.diagnose(_card("RTX 4070", (8, 9)), {**FAIL, "build": "cpu"}, [])
    assert problem["kind"] == "wrong_build" and problem["repair"] is True
    # The right build failing on an RTX 3060: try the other one.
    problem = gc.diagnose(_card("RTX 3060", (8, 6)), FAIL, [])
    assert problem["kind"] == "cuda_error" and problem["repair"] is True
    # Blackwell has only cu128: nothing left.
    assert gc.diagnose(_card("RTX 5070", (12, 0)), FAIL, [])["repair"] is False


def test_every_problem_has_a_story() -> None:
    for kind in ("no_nvidia", "card_too_old", "driver_old", "wrong_build", "cuda_error"):
        story = gc.describe({"kind": kind, "gpu": "RTX 3060", "error": "boom", "tried": [
            {"what": "build", "build": "cu128", "ok": False, "error": "boom"},
            {"what": "start_check", "ok": False, "error": "boom"},
        ]})
        assert story["title"] and story["what"] and story["next"]
        assert story["tried"][-2:] == [
            "Installed the AI engine for CUDA 12.8: the card still failed (boom)",
            "Tested the card when VTM Spark started: boom",
        ]
        assert "!!" in gc.format_story({"kind": kind})


@pytest.fixture
def report(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / gc.REPORT_NAME
    monkeypatch.setattr(gc, "report_path", lambda: path)
    return path


def _install_pc(monkeypatch, card, probes: list[dict]) -> None:
    monkeypatch.setattr(gc, "query_cards", lambda: [card] if card else [])
    monkeypatch.setattr(gc, "probe_torch", lambda: probes.pop(0))


def test_install_retries_the_other_build_then_passes(monkeypatch, report, capsys) -> None:
    gtx = _card("NVIDIA GeForce GTX 1080", (6, 1))
    _install_pc(monkeypatch, gtx, [FAIL, {"ok": True, "error": "", "build": "cu126"}])
    assert gc.main(["verify"]) == gc.EXIT_RETRY
    assert "retry:cu126" in capsys.readouterr().out
    assert gc.main(["verify"]) == gc.EXIT_OK
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["ok"] is True and saved["build"] == "cu126" and saved["problem"] is None
    assert [r["build"] for r in saved["tried"]] == ["cu128", "cu126"]
    assert gc.main(["summary"]) == gc.EXIT_OK
    # A later install that passes again starts the story over instead of growing it.
    _install_pc(monkeypatch, gtx, [{"ok": True, "error": "", "build": "cu126"}])
    gc.main(["verify"])
    assert len(json.loads(report.read_text(encoding="utf-8"))["tried"]) == 1


def test_install_reports_when_every_build_failed(monkeypatch, report, capsys) -> None:
    rtx = _card("NVIDIA GeForce RTX 3060", (8, 6))
    _install_pc(monkeypatch, rtx, [FAIL, {**FAIL, "build": "cu126"}])
    assert gc.main(["verify"]) == gc.EXIT_RETRY
    capsys.readouterr()
    assert gc.main(["verify"]) == gc.EXIT_FAILED
    out = capsys.readouterr().out
    assert "The graphics card could not run the AI engine" in out
    assert "CUDA 12.8: the card still failed" in out and "CUDA 12.6: the card still failed" in out
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["problem"]["repair"] is False  # nothing left for Repair to try
    assert gc.main(["summary"]) == gc.EXIT_FAILED
    assert "could not run the AI engine" in capsys.readouterr().out


def test_install_does_not_redownload_for_an_old_driver(monkeypatch, report, capsys) -> None:
    rtx = _card("NVIDIA GeForce RTX 4070", (8, 9), driver="516.94")
    _install_pc(monkeypatch, rtx, [{**FAIL, "error": "The NVIDIA driver on your system is too old (found version 11070)."}])
    assert gc.main(["verify"]) == gc.EXIT_FAILED
    assert "The NVIDIA driver is too old" in capsys.readouterr().out


def test_desk_problem_includes_what_install_tried(monkeypatch, report) -> None:
    rtx = _card("NVIDIA GeForce RTX 3060", (8, 6))
    gc.save_report({"uuid": rtx["uuid"], "ok": False, "tried": [{"what": "build", "build": "cu128", "ok": False, "error": "x"}]})
    _install_pc(monkeypatch, rtx, [FAIL])
    problem = gc.desk_problem()
    assert problem["kind"] == "cuda_error"
    assert [r["what"] for r in problem["tried"]] == ["build", "start_check"]
    assert problem["repair"] is True  # cu126 not tried yet
    # A swapped card does not inherit the old card's attempts.
    _install_pc(monkeypatch, {**rtx, "uuid": "GPU-other"}, [FAIL])
    assert [r["what"] for r in gc.desk_problem()["tried"]] == ["start_check"]


def test_desk_problem_none_when_the_card_works_or_cpu_is_allowed(monkeypatch, report) -> None:
    _install_pc(monkeypatch, None, [{"ok": True, "error": "", "build": "cu128"}])
    assert gc.desk_problem() is None
    monkeypatch.setenv(gc.ALLOW_CPU_ENV, "1")
    monkeypatch.setattr(gc, "probe_torch", lambda: pytest.fail("not probed"))
    assert gc.desk_problem() is None


def test_launch_repair_opens_install_bat_outside_the_desk_tree(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(gc.subprocess, "Popen", lambda args, **kw: calls.append((args, kw)))
    with pytest.raises(FileNotFoundError):
        gc.launch_repair(tmp_path)
    (tmp_path / "install.bat").write_text("@echo off\n", encoding="utf-8")
    gc.launch_repair(tmp_path)
    args, kw = calls[0]
    assert args[:3] == ["cmd", "/c", "start"] and args[-1] == str(tmp_path / "install.bat")
    assert kw["cwd"] == str(tmp_path)


def test_probe_runs_on_this_machine() -> None:
    probe = gc.probe_torch()
    assert set(probe) == {"ok", "error", "build"}
    assert probe["ok"] or probe["error"]


def test_boot_stops_on_a_gpu_problem(monkeypatch) -> None:
    from backend.desk_boot import new_boot_state, snapshot_boot
    from backend.stream import StreamRuntime

    rt = StreamRuntime.__new__(StreamRuntime)
    import threading

    rt._boot_lock = threading.Lock()
    rt._boot = new_boot_state()
    published = []
    rt._publish_boot = lambda **kw: published.append(kw)
    problem = {"kind": "wrong_build", "gpu": "GTX 1080", "repair": True, "tried": []}
    rt._boot_gpu_problem = lambda: problem
    rt._boot_model = lambda: pytest.fail("the model must not load on the CPU")
    rt._run_boot()
    snap = snapshot_boot(rt._boot)
    assert snap["gpu_problem"] == problem
    assert snap["ready"] is False and snap["running"] is False
    # The desk keeps polling start_boot: it must not start over.
    rt._boot_thread = None
    assert rt.start_boot()["gpu_problem"] == problem


def test_a_broken_checker_never_blocks_the_desk(monkeypatch) -> None:
    from backend.stream import StreamRuntime

    def boom():
        raise RuntimeError("checker bug")

    monkeypatch.setattr(gc, "desk_problem", boom)
    rt = StreamRuntime.__new__(StreamRuntime)
    assert rt._boot_gpu_problem() is None
