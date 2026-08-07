"""Legacy Tk entry — use `python -m backend` or start.bat instead."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "Tk UI retired. Launch with: python -m backend\n"
        "Dev: python -m backend --ui none   +   npm run dev (in ui/)\n"
        "     python -m backend --ui dev"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
