from __future__ import annotations

import importlib.util
import json
import platform
import shutil
from typing import Dict

from wheatly.config import Config


def collect_diagnostics(cfg: Config) -> Dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": {
            "llm_backend": cfg.llm.backend,
            "llm_model": cfg.llm.model,
            "stt_backend": cfg.stt.backend,
            "tts_backend": cfg.resolved_tts_backend(),
            "tts_enabled": cfg.tts.enabled,
        },
        "commands": {
            "say": bool(shutil.which("say")),
            "piper": bool(shutil.which(cfg.tts.piper_binary)),
            "ffmpeg": bool(shutil.which(cfg.tts.filter.ffmpeg_binary)),
            "ollama": bool(shutil.which("ollama")),
            "whisper_cpp": bool(shutil.which(cfg.stt.whisper_cpp_binary)),
        },
        "python_packages": {
            "sounddevice": _has_package("sounddevice"),
            "numpy": _has_package("numpy"),
            "faster_whisper": _has_package("faster_whisper"),
            "edge_tts": _has_package("edge_tts"),
        },
    }


def diagnostics_json(cfg: Config) -> str:
    return json.dumps(collect_diagnostics(cfg), indent=2, ensure_ascii=True)


def _has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None
