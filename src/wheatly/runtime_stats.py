from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

STATS_VERSION = 2


@dataclass
class LatencyStatsData:
    version: int = STATS_VERSION
    llm_words_per_second: float = 3.0
    tts_words_per_second: float = 3.0
    llm_observations: int = 0
    tts_observations: int = 0
    updated_at: float = 0.0


class LatencyStats:
    def __init__(self, path: Path):
        self.path = path
        self.data = LatencyStatsData()
        self._lock = threading.Lock()
        self._load()

    def recommended_initial_words(
        self,
        min_words: int,
        max_words: int,
        adaptive: bool,
    ) -> int:
        if not adaptive:
            return min_words
        with self._lock:
            llm_wps = max(0.5, self.data.llm_words_per_second)
            tts_wps = max(0.5, self.data.tts_words_per_second)
        # If TTS consumes speech faster than the LLM creates it, buffer more text
        # before the first spoken chunk so playback does not run dry immediately.
        recommended = round(min_words * (tts_wps / llm_wps))
        recommended = max(min_words, recommended)
        return min(max_words, recommended)

    def record_llm(self, words: int, duration_seconds: float) -> None:
        if words <= 0 or duration_seconds <= 0:
            return
        self._update("llm", words / duration_seconds)

    def record_tts(self, words: int, duration_seconds: float) -> None:
        if words <= 0 or duration_seconds <= 0:
            return
        self._update("tts", words / duration_seconds)

    def _update(self, kind: str, words_per_second: float) -> None:
        words_per_second = max(0.1, min(80.0, words_per_second))
        alpha = 0.28
        with self._lock:
            if kind == "llm":
                if self.data.llm_observations == 0:
                    self.data.llm_words_per_second = words_per_second
                else:
                    self.data.llm_words_per_second = (
                        alpha * words_per_second
                        + (1 - alpha) * self.data.llm_words_per_second
                    )
                self.data.llm_observations += 1
            else:
                if self.data.tts_observations == 0:
                    self.data.tts_words_per_second = words_per_second
                else:
                    self.data.tts_words_per_second = (
                        alpha * words_per_second
                        + (1 - alpha) * self.data.tts_words_per_second
                    )
                self.data.tts_observations += 1
            self.data.updated_at = time.time()
            self._save_locked()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("version") != STATS_VERSION:
                return
            self.data = LatencyStatsData(**raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.data = LatencyStatsData()

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self.data), indent=2), encoding="utf-8")
