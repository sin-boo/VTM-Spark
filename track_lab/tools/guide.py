"""Timed, spoken prompts for a recording, so each stretch of data is labelled.

Directions are the performer's own: "left" is your left, whichever way the
screen or a mirror shows it. Every hold is followed by a short "center".
"""

from __future__ import annotations

import subprocess
import sys

HOLD = 3.0
BACK = 1.5


def _move(tag: str, say: str) -> list[tuple[str, str, float]]:
    return [(tag, say, HOLD), ("center", "Center", BACK)]


STEPS: list[tuple[str, str, float]] = [
    ("rest", "Recording. Look at the screen and hold still", 4.0),
    *_move("left", "Turn your head to your left"),
    *_move("right", "Now to your right"),
    *_move("left", "Left"),
    *_move("right", "Right"),
    *_move("up", "Look up"),
    *_move("down", "Look down"),
    *_move("up", "Up"),
    *_move("down", "Down"),
    *_move("up_left", "Look up and to your left"),
    *_move("up_right", "Up and to your right"),
    *_move("down_left", "Look down and to your left"),
    *_move("down_right", "Down and to your right"),
    *_move("tilt_left", "Tilt your head toward your left shoulder"),
    *_move("tilt_right", "Now toward your right shoulder"),
    ("rest", "Done. Hold still", 2.5),
]

# Windows' built-in voice, one PowerShell child reading a line per prompt.
_SPEAKER = (
    "Add-Type -AssemblyName System.Speech;"
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$s.Volume = {volume};"
    "while (($l = [Console]::In.ReadLine()) -ne $null) {{"
    " $s.SpeakAsyncCancelAll(); $s.SpeakAsync($l) | Out-Null }};"
    "while ($s.State -eq 'Speaking') {{ Start-Sleep -Milliseconds 50 }}"
)


class Voice:
    """Speaks prompts aloud on Windows; prints only, anywhere else."""

    def __init__(self, enabled: bool = True, volume: int = 100) -> None:
        self._proc: subprocess.Popen[str] | None = None
        if not enabled or sys.platform != "win32":
            return
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", _SPEAKER.format(volume=int(volume))],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            self._proc = None

    def say(self, text: str) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(text.replace("\n", " ") + "\n")
            proc.stdin.flush()
        except OSError:
            self._proc = None

    def close(self, timeout: float = 6.0) -> None:
        """Let the last prompt finish, then stop the speaker."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()


class Guide:
    """Walks the steps against a clock and announces each one once."""

    def __init__(self, steps: list[tuple[str, str, float]] | None = None, voice: Voice | None = None) -> None:
        self.steps = list(STEPS if steps is None else steps)
        self.voice = voice
        self.total = sum(seconds for _tag, _say, seconds in self.steps)
        self._index = -1

    def step_at(self, t: float) -> int:
        """Index of the step running at ``t`` seconds (len(steps) once done)."""
        end = 0.0
        for i, (_tag, _say, seconds) in enumerate(self.steps):
            end += seconds
            if t < end:
                return i
        return len(self.steps)

    def tick(self, t: float) -> tuple[str, bool, bool]:
        """(tag, just changed, finished) at ``t``; speaks a new step."""
        i = self.step_at(t)
        if i >= len(self.steps):
            return "rest", False, True
        changed = i != self._index
        if changed:
            self._index = i
            _tag, say, _seconds = self.steps[i]
            print(f"[{t:5.1f}s] {say}", flush=True)
            if self.voice is not None:
                self.voice.say(say)
        return self.steps[i][0], changed, False
