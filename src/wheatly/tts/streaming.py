from __future__ import annotations

import queue
import re
import threading
import time
from typing import Callable, Optional

from wheatly.tts.base import TTSBackend


class StreamingSpeaker:
    """Queues short text segments so TTS can begin before the LLM finishes."""

    def __init__(
        self,
        tts: TTSBackend,
        enabled: bool,
        min_words: int = 8,
        max_words: int = 18,
        initial_min_words: Optional[int] = None,
        feedback_min_words: int = 8,
        max_initial_wait_seconds: float = 2.0,
        on_spoken: Optional[Callable[[str, float], None]] = None,
    ):
        self.tts = tts
        self.enabled = enabled
        self.min_words = min_words
        self.max_words = max_words
        self.initial_min_words = initial_min_words or min_words
        self.feedback_min_words = feedback_min_words
        self.max_initial_wait_seconds = max_initial_wait_seconds
        self.on_spoken = on_spoken
        self._buffer = ""
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._segments_queued = 0
        self._first_buffered_at: Optional[float] = None

    def __enter__(self) -> "StreamingSpeaker":
        if self.enabled:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    def feed(self, text: str) -> None:
        if not self.enabled or not text:
            return
        if self._first_buffered_at is None:
            self._first_buffered_at = time.perf_counter()
        self._buffer += text
        while True:
            segment = self._pop_segment(final=False)
            if not segment:
                break
            self._segments_queued += 1
            self._queue.put(segment)

    def finish(self) -> None:
        if not self.enabled:
            return
        remaining = self._pop_segment(final=True)
        if remaining:
            self._segments_queued += 1
            self._queue.put(remaining)
        self._queue.put(None)
        if self._worker:
            self._worker.join()
        if self._error:
            raise RuntimeError(f"streaming TTS failed: {self._error}") from self._error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                started_at = time.perf_counter()
                self.tts.speak(item)
                if self.on_spoken:
                    self.on_spoken(item, time.perf_counter() - started_at)
            except BaseException as exc:  # Preserve worker errors for caller.
                self._error = exc
                return

    def _pop_segment(self, final: bool) -> str:
        text = self._buffer.strip()
        if not text:
            self._buffer = ""
            return ""

        if final:
            self._buffer = ""
            return text

        words = text.split()
        min_words = self.initial_min_words if self._segments_queued == 0 else self.min_words
        first_segment = self._segments_queued == 0
        waited_too_long = (
            first_segment
            and self._first_buffered_at is not None
            and time.perf_counter() - self._first_buffered_at >= self.max_initial_wait_seconds
        )
        match = re.search(r"([.!?;:]\s+|\n+)", self._buffer)
        if match:
            segment = self._buffer[: match.end()].strip()
            segment_words = len(segment.split())
            if segment_words >= min_words or (
                waited_too_long and segment_words >= self.feedback_min_words
            ):
                self._buffer = self._buffer[match.end() :]
                return segment

        if len(words) < min_words:
            if not waited_too_long or len(words) < self.feedback_min_words:
                return ""

        if waited_too_long and len(words) >= self.feedback_min_words:
            boundary = _word_boundary_index(self._buffer, min(len(words), self.feedback_min_words))
            segment = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:]
            return segment

        max_words = max(self.max_words, min_words)
        if len(words) >= max_words:
            boundary = _word_boundary_index(self._buffer, max_words)
            segment = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:]
            return segment

        return ""


def _word_boundary_index(text: str, word_count: int) -> int:
    matches = list(re.finditer(r"\S+\s*", text))
    if len(matches) < word_count:
        return len(text)
    return matches[word_count - 1].end()
