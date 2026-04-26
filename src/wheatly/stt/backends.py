from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from wheatly.config import STTConfig
from wheatly.stt.base import STTBackend, Transcription


class KeyboardSTT(STTBackend):
    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        if audio_path:
            raise RuntimeError("keyboard STT cannot transcribe audio files")
        return Transcription(text=input("you> ").strip(), language=None)


class FasterWhisperSTT(STTBackend):
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self._model = None

    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        if not audio_path:
            raise RuntimeError("faster-whisper requires an audio file")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Install faster-whisper first: pip install '.[stt]'"
            ) from exc

        if self._model is None:
            self._model = WhisperModel(
                self.cfg.model,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
        segments, info = self._model.transcribe(
            str(audio_path),
            language=self.cfg.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcription(
            text=text,
            language=getattr(info, "language", None),
            duration_seconds=getattr(info, "duration", None),
        )


class WhisperCppSTT(STTBackend):
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg

    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        if not audio_path:
            raise RuntimeError("whisper.cpp requires an audio file")
        command = [
            self.cfg.whisper_cpp_binary,
            "-m",
            self.cfg.whisper_cpp_model,
            "-f",
            str(audio_path),
        ] + self.cfg.whisper_cpp_args
        completed = subprocess.run(
            command, capture_output=True, text=True, shell=False, timeout=120
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "whisper.cpp failed")
        return Transcription(text=_clean_whisper_cpp_output(completed.stdout))


def build_stt(cfg: STTConfig) -> STTBackend:
    backend = cfg.backend.lower()
    if backend == "keyboard":
        return KeyboardSTT()
    if backend in {"faster_whisper", "faster-whisper"}:
        return FasterWhisperSTT(cfg)
    if backend in {"whisper_cpp", "whisper.cpp"}:
        return WhisperCppSTT(cfg)
    raise ValueError(f"Unsupported STT backend: {cfg.backend}")


def _clean_whisper_cpp_output(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("whisper_") or line.startswith("main:"):
            continue
        lines.append(line)
    return " ".join(lines).strip()
