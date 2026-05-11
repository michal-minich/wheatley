from __future__ import annotations

import importlib.util
import json
import platform
import shutil
from typing import Dict

from wheatley.audio.devices import list_audio_devices
from wheatley.config import Config
from wheatley.llm.backends import model_supports_images


def collect_diagnostics(cfg: Config) -> Dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": {
            "llm_backend": cfg.llm.backend,
            "llm_model": cfg.llm.model,
            "llm_image_input_by_model_name": model_supports_images(cfg.llm.model),
            "stt_backend": cfg.stt.backend,
            "stt_model": cfg.stt.model,
            "stt_preview_model": cfg.stt.preview_model,
            "stt_preview_use_remote": cfg.stt.preview_use_remote,
            "stt_final_model": cfg.stt.final_model,
            "stt_final_use_remote": cfg.stt.final_use_remote,
            "stt_remote_base_url": cfg.stt.remote_base_url,
            "stt_remote_model": cfg.stt.remote_model,
            "stt_fallback_backend": cfg.stt.remote_fallback_backend,
            "audio_input_device_mode": cfg.audio.input_device_mode,
            "audio_input_device_preferred_names": cfg.audio.input_device_preferred_names,
            "audio_input_device_name": cfg.audio.input_device_name,
            "audio_input_device_index": cfg.audio.input_device_index,
            "tts_backend": cfg.resolved_tts_backend(),
            "tts_enabled": cfg.tts.enabled,
            "photo_short_side": cfg.tools.photo_short_side,
            "photo_command_configured": bool(cfg.tools.photo_command),
        },
        "commands": {
            "say": bool(shutil.which("say")),
            "piper": bool(shutil.which(cfg.tts.piper_binary)),
            "ffmpeg": bool(shutil.which(cfg.tts.filter.ffmpeg_binary)),
            "imagesnap": bool(shutil.which("imagesnap")),
            "fswebcam": bool(shutil.which("fswebcam")),
            "libcamera_still": bool(shutil.which("libcamera-still")),
            "rpicam_still": bool(shutil.which("rpicam-still")),
            "ollama": bool(shutil.which("ollama")),
            "whisper_cpp": bool(shutil.which(cfg.stt.whisper_cpp_binary)),
        },
        "python_packages": {
            "sounddevice": _has_package("sounddevice"),
            "numpy": _has_package("numpy"),
            "faster_whisper": _has_package("faster_whisper"),
            "edge_tts": _has_package("edge_tts"),
        },
        "audio_devices": list_audio_devices(),
    }


def diagnostics_json(cfg: Config) -> str:
    return json.dumps(collect_diagnostics(cfg), indent=2, ensure_ascii=True)


def _has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None
