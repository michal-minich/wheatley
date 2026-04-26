from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional


_PLAYBACK_LOCK = threading.Lock()
_CURRENT_PLAYBACK: Optional[subprocess.Popen] = None
_CURRENT_PLAYBACK_STARTED_AT: Optional[float] = None


def play_audio(path: Path, command: Optional[List[str]] = None) -> bool:
    if command:
        resolved = [part.format(path=str(path)) for part in command]
    else:
        resolved = _default_command(path)
    if not resolved:
        return False
    return run_playback_command(resolved)


def run_playback_command(command: List[str]) -> bool:
    global _CURRENT_PLAYBACK, _CURRENT_PLAYBACK_STARTED_AT
    process = subprocess.Popen(command, shell=False)
    with _PLAYBACK_LOCK:
        _CURRENT_PLAYBACK = process
        _CURRENT_PLAYBACK_STARTED_AT = time.monotonic()
    try:
        while process.poll() is None:
            time.sleep(0.03)
        return process.returncode == 0
    finally:
        with _PLAYBACK_LOCK:
            if _CURRENT_PLAYBACK is process:
                _CURRENT_PLAYBACK = None
                _CURRENT_PLAYBACK_STARTED_AT = None


def stop_audio_playback() -> None:
    with _PLAYBACK_LOCK:
        process = _CURRENT_PLAYBACK
    if not process or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=0.4)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=0.4)


def current_playback_age_seconds() -> Optional[float]:
    with _PLAYBACK_LOCK:
        process = _CURRENT_PLAYBACK
        started_at = _CURRENT_PLAYBACK_STARTED_AT
    if not process or process.poll() is not None or started_at is None:
        return None
    return time.monotonic() - started_at


def _default_command(path: Path) -> List[str]:
    if shutil.which("afplay"):
        return ["afplay", str(path)]
    if shutil.which("aplay"):
        return ["aplay", str(path)]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
    return []
