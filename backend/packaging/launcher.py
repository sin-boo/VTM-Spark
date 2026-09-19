"""Tiny VTM Noble launcher (stdlib only — never import torch/backend here).



PyInstaller must analyze ONLY this file. The real app runs from the side-by-side

``runtime\\Scripts\\python.exe -m backend``.



Always kills the child process *tree* on exit so closing the window cannot leave

a multi-GB Python orphan.

"""



from __future__ import annotations



import atexit

import os

import signal

import subprocess

import sys

from pathlib import Path



_child: subprocess.Popen[bytes] | None = None





def _kill_tree(pid: int) -> None:

    if pid <= 0:

        return

    if os.name == "nt":

        try:

            # /T = kill child processes too (uvicorn threads are in-proc, but

            # any spawned helpers die with the tree).

            subprocess.call(

                ["taskkill", "/F", "/T", "/PID", str(pid)],

                stdout=subprocess.DEVNULL,

                stderr=subprocess.DEVNULL,

                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0,

            )

        except Exception:

            pass

    else:

        try:

            os.kill(pid, signal.SIGKILL)

        except Exception:

            pass





def _cleanup_child() -> None:

    global _child

    proc = _child

    if proc is None:

        return

    if proc.poll() is None:

        _kill_tree(proc.pid)

        try:

            proc.wait(timeout=3)

        except Exception:

            pass

    _child = None





def _package_root() -> Path:
    """Directory that contains runtime/, backend/, etc.

    Supports two layouts:
      - Shipped folder:  <dir>/VTMNoble.exe + <dir>/runtime/...
      - Repo convenience: real_stream/VTMNoble.exe → uses real_stream/dist/VTMNoble/
    """
    if getattr(sys, "frozen", False):
        here = Path(sys.executable).resolve().parent
    else:
        here = Path(__file__).resolve().parents[2]

    def _has_runtime(base: Path) -> bool:
        return (base / "runtime" / "Scripts" / "python.exe").is_file()

    if _has_runtime(here):
        return here

    dist = here / "dist" / "VTMNoble"
    if _has_runtime(dist):
        return dist

    return here


def main() -> int:

    global _child

    root = _package_root()

    runtime_py = root / "runtime" / "Scripts" / "python.exe"

    if not runtime_py.is_file():

        print(f"Missing runtime Python:\n  {runtime_py}", flush=True)

        print(
            "Run install.bat, then double-click VTMNoble.exe "
            "next to run.exe.",
            flush=True,
        )

        try:

            input("Press Enter to close…")

        except Exception:

            pass

        return 1



    env = os.environ.copy()

    env["PYTHONUTF8"] = "1"

    env["PYTHONIOENCODING"] = "utf-8"

    env.setdefault("VTM_NOBLE_ROOT", str(root))

    # Cap BLAS/OMP oversubscription before torch import (reduces RAM/CPU thrash).
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")
    env.setdefault("TORCH_NUM_THREADS", "4")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Refuse to spawn a second backend if the API is already healthy.
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=0.35) as resp:
            body = resp.read(64)
            if resp.status == 200 and b"ok" in body:
                print(
                    "VTM Noble is already running on port 8765.\n"
                    "Close the other window, or close the other desk first.",
                    flush=True,
                )
                try:
                    input("Press Enter to close…")
                except Exception:
                    pass
                return 2
    except Exception:
        pass

    cmd = [str(runtime_py), "-m", "backend", *sys.argv[1:]]

    creationflags = 0

    if os.name == "nt":

        # Hide the runtime console; new process group so taskkill /T can target the tree.

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)



    atexit.register(_cleanup_child)

    try:

        _child = subprocess.Popen(

            cmd,

            cwd=str(root),

            env=env,

            creationflags=creationflags,

        )

        return int(_child.wait())

    except KeyboardInterrupt:

        _cleanup_child()

        return 130

    except OSError as exc:

        print(f"Failed to start runtime:\n{exc}", flush=True)

        try:

            input("Press Enter to close…")

        except Exception:

            pass

        return 1

    finally:

        _cleanup_child()





if __name__ == "__main__":

    raise SystemExit(main())

