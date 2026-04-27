from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Transcription:
    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None


class STTBackend:
    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        raise NotImplementedError

