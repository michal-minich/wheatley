from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def play_audio(path: Path, command: Optional[List[str]] = None) -> None:
    if command:
        resolved = [part.format(path=str(path)) for part in command]
    else:
        resolved = _default_command(path)
    if not resolved:
        return
    subprocess.run(resolved, shell=False, check=False)


def _default_command(path: Path) -> List[str]:
    if shutil.which("afplay"):
        return ["afplay", str(path)]
    if shutil.which("aplay"):
        return ["aplay", str(path)]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
    return []

