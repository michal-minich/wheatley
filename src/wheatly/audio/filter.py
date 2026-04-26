from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from wheatly.config import FilterConfig


def apply_voice_filter(input_path: Path, output_path: Path, cfg: FilterConfig) -> Path:
    if not cfg.enabled:
        return input_path
    ffmpeg = shutil.which(cfg.ffmpeg_binary)
    if not ffmpeg:
        return input_path

    filtergraph = _filtergraph(cfg.preset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-af",
        filtergraph,
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, shell=False)
    if completed.returncode != 0:
        return input_path
    return output_path


def _filtergraph(preset: str) -> str:
    if preset == "wheatley_light":
        return (
            "highpass=f=180,"
            "lowpass=f=4300,"
            "acompressor=threshold=-18dB:ratio=2.6:attack=8:release=120,"
            "acrusher=bits=13:mix=0.045,"
            "volume=1.08"
        )
    if preset == "wheatley_bright":
        return (
            "highpass=f=210,"
            "lowpass=f=5600,"
            "equalizer=f=2800:t=q:w=1.4:g=2.8,"
            "equalizer=f=4300:t=q:w=1.0:g=1.4,"
            "acompressor=threshold=-20dB:ratio=2.1:attack=6:release=100,"
            "volume=1.08"
        )
    if preset == "radio_robot":
        return (
            "highpass=f=260,"
            "lowpass=f=3600,"
            "acompressor=threshold=-20dB:ratio=3.5,"
            "acrusher=bits=11:mix=0.10,"
            "volume=1.1"
        )
    return "highpass=f=180,lowpass=f=4300,volume=1.0"
