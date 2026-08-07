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





def main() -> int:

    global _child



    if getattr(sys, "frozen", False):

        root = Path(sys.executable).resolve().parent

    else:

        root = Path(__file__).resolve().parent.parent



    runtime_py = root / "runtime" / "Scripts" / "python.exe"

    if not runtime_py.is_file():

        print(f"Missing runtime Python:\n  {runtime_py}", flush=True)

        print("Rebuild with packaging\\build.ps1", flush=True)

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
                    "Close the other window, or run start.bat → [K] Kill leftovers.",
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

        # New process group so Ctrl+C / taskkill /T can target the tree.

        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)



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

