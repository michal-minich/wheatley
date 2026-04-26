from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SpeechResult:
    audio_path: Optional[Path]
    spoken: bool


class TTSBackend:
    def speak(self, text: str) -> SpeechResult:
        raise NotImplementedError

