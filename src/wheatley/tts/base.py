from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SpeechResult:
    audio_path: Optional[Path]
    spoken: bool


@dataclass
class PreparedSpeech:
    text: str
    audio_path: Optional[Path]


class TTSBackend:
    def speak(self, text: str) -> SpeechResult:
        raise NotImplementedError

    def supports_stream_pipelining(self) -> bool:
        return False

    def prepare_for_playback(self, text: str) -> PreparedSpeech:
        raise RuntimeError("prepare_for_playback is not implemented for this backend")

    def play_prepared(self, prepared: PreparedSpeech) -> bool:
        del prepared
        return False
