"""Can this PC's NVIDIA card run VTM Spark? If not: what went wrong, what we tried, what is left.

install.bat picks the torch build for the card, then makes the card run real work.
CUDA 12.8 builds (cu128) run Turing (RTX 20) and newer, Blackwell included, but
dropped Maxwell / Pascal / Volta (GTX 900 / 10, Titan V); CUDA 12.6 builds (cu126)
still run those, but not Blackwell. When the card fails on one build and the other
one supports it, the installer tries that one too. Each try lands in
``models/gpu_check.json``.

The desk checks again at start: a card torch cannot use stops the launch with that
story (and a Repair button when another build could help) instead of streaming on
the CPU at seconds a frame. Stdlib only at import: the installer runs this before
torch exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BUILD_NEW = "cu128"
BUILD_OLD = "cu126"
BUILD_LABELS = {BUILD_NEW: "CUDA 12.8", BUILD_OLD: "CUDA 12.6"}
REPORT_NAME = "gpu_check.json"
# Oldest card any current torch build runs: Maxwell (GTX 900).
MIN_CAP = (5, 0)
# cu128 starts at Turing; cu126 ends at Hopper (no Blackwell).
NEW_BUILD_MIN_CAP = (7, 5)
OLD_BUILD_MAX_CAP = (9, 0)
# Developer escape hatch: let the desk run on the CPU anyway.
ALLOW_CPU_ENV = "VTM_ALLOW_CPU"
# `verify` exit codes for build.ps1.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_RETRY = 3

CREATE_NO_WINDOW = 0x08000000

Cap = tuple[int, int]


def builds_for(cap: Cap | None) -> list[str]:
    """Torch builds that run a card of this compute capability, best first."""
    if cap is None:
        return [BUILD_NEW, BUILD_OLD]
    if cap < MIN_CAP:
        return []
    out = []
    if cap >= NEW_BUILD_MIN_CAP:
        out.append(BUILD_NEW)
    if cap <= OLD_BUILD_MAX_CAP:
        out.append(BUILD_OLD)
    return out


def want_build(cap: Cap | None) -> str:
    """The build to install for this card (cu128 when nothing runs it, so the app still installs)."""
    builds = builds_for(cap)
    return builds[0] if builds else BUILD_NEW


def parse_cap(text: str) -> Cap | None:
    try:
        major, minor = str(text).strip().split(".")[:2]
        return int(major), int(minor)
    except (ValueError, TypeError):
        return None


def cap_text(cap: Cap | None) -> str:
    return f"{cap[0]}.{cap[1]}" if cap else ""


# ---------------------------------------------------------------------- cards
def parse_cards(text: str, *, with_cap: bool) -> list[dict[str, Any]]:
    """Rows of ``uuid, name, driver_version[, compute_cap]`` from nvidia-smi."""
    cards: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].startswith("GPU-"):
            continue
        cards.append(
            {
                "uuid": parts[0],
                "name": parts[1],
                "driver": parts[2],
                "cap": parse_cap(parts[3]) if with_cap and len(parts) > 3 else None,
            }
        )
    return cards


def query_cards(run=subprocess.run) -> list[dict[str, Any]]:
    """NVIDIA cards the driver sees. Empty when there is none, or no driver."""
    from .gpu_select import _nvidia_smi

    exe = _nvidia_smi()
    if not exe:
        return []
    kw: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 10}
    if os.name == "nt":
        kw["creationflags"] = CREATE_NO_WINDOW
    # compute_cap needs a 2022+ driver; older ones still list the card without it.
    for fields, with_cap in (("uuid,name,driver_version,compute_cap", True), ("uuid,name,driver_version", False)):
        try:
            out = run([exe, f"--query-gpu={fields}", "--format=csv,noheader"], **kw)
        except (OSError, subprocess.SubprocessError):
            return []
        if getattr(out, "returncode", 1) == 0:
            cards = parse_cards(out.stdout or "", with_cap=with_cap)
            if cards:
                return cards
    return []


def target_card(cards: list[dict[str, Any]], env: dict[str, str] | None = None) -> dict[str, Any] | None:
    """The card torch will run on: the pinned one, else the newest (CUDA lists it first)."""
    if not cards:
        return None
    env = os.environ if env is None else env
    pin = str(env.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    for card in cards:
        if pin and card["uuid"] == pin:
            return card
    return max(cards, key=lambda c: c["cap"] or (0, 0))


# ---------------------------------------------------------------------- torch
def _first_line(exc: BaseException | str) -> str:
    text = str(exc).strip()
    return text.splitlines()[0].strip() if text else type(exc).__name__


def installed_build() -> str:
    """'cu128' / 'cu126' / 'cpu' from the torch wheel, or '' without torch."""
    try:
        import torch
    except Exception:
        return ""
    ver = str(torch.__version__)
    return ver.split("+", 1)[1] if "+" in ver else ("cpu" if not torch.version.cuda else "")


def probe_torch() -> dict[str, Any]:
    """Make the card run a matmul and a convolution; ``{"ok", "error", "build"}``.

    ``is_available()`` alone passes on a card the wheel has no kernels for (GTX 10 on
    a cu128 build): the first real kernel then fails, so run one.
    """
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": f"torch did not import: {_first_line(exc)}", "build": ""}
    build = installed_build()
    if not torch.cuda.is_available():
        error = ""
        try:
            torch.cuda.init()
        except Exception as exc:  # torch's own words, e.g. "driver ... is too old"
            error = _first_line(exc)
        return {"ok": False, "error": error or "CUDA is not available", "build": build}
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "sm_61 is not compatible": the kernel below says it
            dev = torch.device("cuda")
            x = torch.randn(64, 64, device=dev)
            y = x @ x
            img = torch.randn(1, 4, 16, 16, device=dev)
            z = torch.nn.functional.conv2d(img, torch.randn(8, 4, 3, 3, device=dev), padding=1)
            torch.cuda.synchronize()
            if not (bool(torch.isfinite(y).all()) and bool(torch.isfinite(z).all())):
                raise RuntimeError("the card returned NaN/Inf on a test calculation")
    except Exception as exc:
        return {"ok": False, "error": _first_line(exc), "build": build}
    return {"ok": True, "error": "", "build": build}


# ---------------------------------------------------------------------- report
def report_path() -> Path:
    from .paths import data_dir

    return data_dir() / REPORT_NAME


def load_report(path: Path | None = None) -> dict[str, Any]:
    try:
        raw = json.loads((path or report_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_report(report: dict[str, Any], path: Path | None = None) -> None:
    target = path or report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")


def tried_for(report: dict[str, Any], card: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Failed installer attempts on this same card (a swapped card, or one that passed, starts over)."""
    rows = report.get("tried")
    if not isinstance(rows, list) or report.get("ok"):
        return []
    if card is not None and str(report.get("uuid") or "") != card["uuid"]:
        return []
    return [dict(r) for r in rows if isinstance(r, dict)]


def builds_left(cap: Cap | None, tried: list[dict[str, Any]], current: str) -> list[str]:
    """Other builds that support the card and have not failed on it yet."""
    failed = {str(r.get("build")) for r in tried if r.get("what") == "build" and not r.get("ok")}
    return [b for b in builds_for(cap) if b != current and b not in failed]


# ---------------------------------------------------------------------- diagnosis
def _driver_too_old(error: str) -> bool:
    low = error.lower()
    return "driver" in low and any(w in low for w in ("too old", "insufficient", "older than"))


def diagnose(
    card: dict[str, Any] | None,
    probe: dict[str, Any],
    tried: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """None when the card works; else what went wrong, what was tried, and whether Repair can help."""
    if probe.get("ok"):
        return None
    error = str(probe.get("error") or "")
    current = str(probe.get("build") or "")
    cap = card["cap"] if card else None
    problem: dict[str, Any] = {
        "gpu": card["name"] if card else "",
        "driver": card["driver"] if card else "",
        "cap": cap_text(cap),
        "build": BUILD_LABELS.get(current, current),
        "error": error,
        "tried": tried,
        "repair": False,
    }
    if card is None:
        return {**problem, "kind": "no_nvidia"}
    if cap is not None and cap < MIN_CAP:
        return {**problem, "kind": "card_too_old"}
    if _driver_too_old(error):
        # Every build needs a CUDA 12 driver: another one would not help.
        return {**problem, "kind": "driver_old"}
    problem["repair"] = bool(builds_left(cap, tried, current))
    if current not in builds_for(cap):  # a GTX 1080 swapped in under cu128, or a CPU wheel
        return {**problem, "kind": "wrong_build"}
    return {**problem, "kind": "cuda_error"}


_TITLES = {
    "no_nvidia": "No NVIDIA graphics card found",
    "card_too_old": "This graphics card is too old for VTM Spark",
    "driver_old": "The NVIDIA driver is too old",
    "wrong_build": "The AI engine does not match this graphics card",
    "cuda_error": "The graphics card could not run the AI engine",
}


def describe(problem: dict[str, Any]) -> dict[str, Any]:
    """Plain-words story for the console: title, what happened, what we tried, what is left."""
    kind = str(problem.get("kind") or "cuda_error")
    gpu = problem.get("gpu") or "The graphics card"
    err = problem.get("error") or "unknown error"
    what = {
        "no_nvidia": "VTM Spark runs its AI on an NVIDIA graphics card (GeForce GTX 900 series or newer). "
        "Windows does not show one on this PC.",
        "card_too_old": f"{gpu} (CUDA {problem.get('cap') or '?'}) is older than any card the AI engine "
        "supports. VTM Spark needs a GeForce GTX 900 series or newer.",
        "driver_old": f"{gpu} is supported, but its driver ({problem.get('driver') or '?'}) is too old "
        f"for the AI engine ({err}).",
        "wrong_build": f"The AI engine installed here ({problem.get('build') or '?'}) cannot run on {gpu}. "
        "This happens after a graphics card change.",
        "cuda_error": f"{gpu} failed a test calculation: {err}",
    }[kind]
    tried = [tried_line(r) for r in problem.get("tried") or []]
    if kind == "no_nvidia":
        tried = ["Looked for NVIDIA cards with the NVIDIA driver's nvidia-smi tool: none found"] + tried
    if kind == "no_nvidia":
        nxt = "If this PC has an NVIDIA card, install its driver from https://www.nvidia.com/drivers and " \
              "start VTM Spark again. Without one, this PC cannot run VTM Spark."
    elif kind == "card_too_old":
        nxt = "This PC cannot run VTM Spark with this card."
    elif kind == "driver_old":
        nxt = "Update the NVIDIA driver (https://www.nvidia.com/drivers or the NVIDIA app), then start VTM Spark again."
    elif problem.get("repair"):
        nxt = "Repair installs the AI engine version for this card (about 2.5 GB) and tests it again."
    else:
        nxt = "Restart the PC. If it keeps happening, reinstall the NVIDIA driver from https://www.nvidia.com/drivers."
    return {"title": _TITLES[kind], "what": what, "tried": tried, "next": nxt}


def tried_line(row: dict[str, Any]) -> str:
    if row.get("what") == "start_check":
        return f"Tested the card when VTM Spark started: {row.get('error') or 'failed'}"
    label = BUILD_LABELS.get(str(row.get("build")), str(row.get("build") or "?"))
    if row.get("ok"):
        return f"Installed the AI engine for {label}: the card ran it"
    return f"Installed the AI engine for {label}: the card still failed ({row.get('error') or 'unknown error'})"


def format_story(problem: dict[str, Any]) -> str:
    story = describe(problem)
    lines = ["", "  !! " + story["title"], "", "  What happened:", "    " + story["what"]]
    if story["tried"]:
        lines += ["", "  What we tried:"] + [f"    - {t}" for t in story["tried"]]
    lines += ["", "  What is left:", "    " + story["next"], ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------- desk
def desk_problem() -> dict[str, Any] | None:
    """At desk start: None when the card works, else the problem (with what install tried)."""
    if str(os.environ.get(ALLOW_CPU_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
        return None
    probe = probe_torch()
    if probe["ok"]:
        return None
    card = target_card(query_cards())
    tried = tried_for(load_report(), card)
    tried.append({"what": "start_check", "ok": False, "error": probe["error"]})
    return diagnose(card, probe, tried)


def launch_repair(root: Path | None = None) -> None:
    """Open install.bat in its own console. It closes the desk, puts the right AI engine
    in for this card and tests it. Started through ``cmd /c start`` so the desk's process
    tree, which the installer kills, does not include it."""
    from .paths import package_root

    bat = (root or package_root()) / "install.bat"
    if not bat.is_file():
        raise FileNotFoundError(f"install.bat not found at {bat}")
    subprocess.Popen(
        ["cmd", "/c", "start", "VTM Spark repair", str(bat)],
        cwd=str(bat.parent),
        creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        close_fds=True,
    )


# ---------------------------------------------------------------------- installer CLI
def _verify() -> int:
    probe = probe_torch()
    card = target_card(query_cards())
    tried = tried_for(load_report(), card)
    tried.append({"what": "build", "build": probe["build"], "ok": bool(probe["ok"]), "error": probe["error"]})
    problem = diagnose(card, probe, tried)
    left = builds_left(card["cap"] if card else None, tried, probe["build"])
    retry = bool(problem and problem["kind"] in {"wrong_build", "cuda_error"} and left)
    if problem is not None and not retry:
        problem["repair"] = False  # the installer already tried every build that fits
    save_report(
        {
            "uuid": card["uuid"] if card else "",
            "gpu": card["name"] if card else "",
            "driver": card["driver"] if card else "",
            "cap": cap_text(card["cap"]) if card else "",
            "ok": bool(probe["ok"]),
            "build": probe["build"],
            "tried": tried,
            "problem": problem,
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if problem is None:
        name = card["name"] if card else "GPU"
        print(f"    graphics card OK: {name} runs the AI engine ({BUILD_LABELS.get(probe['build'], probe['build'])})")
        return EXIT_OK
    if retry:
        print(f"    the graphics card failed on {BUILD_LABELS.get(probe['build'], probe['build'])}: {probe['error']}")
        print(f"retry:{left[0]}")
        return EXIT_RETRY
    print(format_story(problem))
    return EXIT_FAILED


def _summary() -> int:
    """Install summary line: exit 0 when the card ran the AI engine, else print the title and what is left."""
    report = load_report()
    if report.get("ok"):
        return EXIT_OK
    problem = report.get("problem")
    if isinstance(problem, dict) and problem.get("kind") in _TITLES:
        story = describe(problem)
        print(f"{story['title']}. {story['next']}")
    else:
        print("The graphics card was not tested. Re-running install.bat tests it.")
    return EXIT_FAILED


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else ""
    if cmd == "want-build":
        card = target_card(query_cards())
        print(want_build(card["cap"] if card else None))
        return EXIT_OK
    if cmd == "verify":
        return _verify()
    if cmd == "summary":
        return _summary()
    print("usage: python -m backend.gpu_check want-build | verify | summary", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
