from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import gpu_select
from backend.gpu_select import (
    CUDA_ENV,
    OWNED_ENV,
    apply_saved_gpu,
    list_gpus,
    load_gpu_pref,
    parse_smi_csv,
    save_gpu_pref,
)

SMI = (
    "0, GPU-aaaa-1111, NVIDIA GeForce RTX 4090, 24564\n"
    "1, GPU-bbbb-2222, NVIDIA GeForce RTX 3060, 12288\n"
)
GPUS = parse_smi_csv(SMI)


def test_parse_smi_csv() -> None:
    assert GPUS == [
        {"index": 0, "uuid": "GPU-aaaa-1111", "name": "NVIDIA GeForce RTX 4090", "memory_mb": 24564},
        {"index": 1, "uuid": "GPU-bbbb-2222", "name": "NVIDIA GeForce RTX 3060", "memory_mb": 12288},
    ]
    assert parse_smi_csv("No devices were found\n") == []


def test_list_gpus_handles_missing_or_failing_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_select, "_nvidia_smi", lambda: None)
    assert list_gpus() == []
    monkeypatch.setattr(gpu_select, "_nvidia_smi", lambda: "nvidia-smi")
    assert list_gpus(run=lambda *a, **k: SimpleNamespace(returncode=9, stdout="")) == []

    def boom(*a, **k):
        raise OSError("gone")

    assert list_gpus(run=boom) == []
    assert list_gpus(run=lambda *a, **k: SimpleNamespace(returncode=0, stdout=SMI)) == GPUS


def test_pref_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "gpu.json"
    assert load_gpu_pref(path) == ""
    assert save_gpu_pref("GPU-bbbb-2222", path) == "GPU-bbbb-2222"
    assert load_gpu_pref(path) == "GPU-bbbb-2222"
    save_gpu_pref("", path)
    assert load_gpu_pref(path) == ""
    with pytest.raises(ValueError):
        save_gpu_pref("1", path)


def test_apply_pins_saved_gpu() -> None:
    env: dict[str, str] = {}
    assert apply_saved_gpu(env, saved="GPU-bbbb-2222", gpus=GPUS, log=lambda m: None) == "GPU-bbbb-2222"
    assert env[CUDA_ENV] == "GPU-bbbb-2222"
    assert env[OWNED_ENV] == "GPU-bbbb-2222"


def test_apply_missing_gpu_falls_back_to_automatic() -> None:
    env: dict[str, str] = {}
    assert apply_saved_gpu(env, saved="GPU-gone", gpus=GPUS, log=lambda m: None) == ""
    assert CUDA_ENV not in env


def test_reload_replaces_our_own_pin() -> None:
    # A reload inherits the old process env; the new pick must win.
    env = {CUDA_ENV: "GPU-aaaa-1111", OWNED_ENV: "GPU-aaaa-1111"}
    apply_saved_gpu(env, saved="GPU-bbbb-2222", gpus=GPUS, log=lambda m: None)
    assert env[CUDA_ENV] == "GPU-bbbb-2222"
    # Back to automatic clears it entirely.
    apply_saved_gpu(env, saved="", gpus=GPUS, log=lambda m: None)
    assert CUDA_ENV not in env and OWNED_ENV not in env


def test_user_set_cuda_visible_devices_wins() -> None:
    env = {CUDA_ENV: "1"}
    assert apply_saved_gpu(env, saved="GPU-aaaa-1111", gpus=GPUS, log=lambda m: None) == ""
    assert env == {CUDA_ENV: "1"}
    assert gpu_select.external_pin(env) == "1"


def test_api_exposes_gpu_routes() -> None:
    src = (Path(__file__).resolve().parents[1] / "api.py").read_text(encoding="utf-8")
    assert '@app.get("/api/gpus")' in src
    assert '@app.post("/api/gpu")' in src
