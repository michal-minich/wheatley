from __future__ import annotations

import queue
import re
import threading
import time
from typing import Callable, Optional

from wheatley.tts.base import PreparedSpeech, TTSBackend

_SENTENCE_BOUNDARY_PATTERN = r"([.!?;:]\s+|\n+)"
_CLAUSE_BOUNDARY_PATTERN = r"(,\s+)"
_MIN_COMPLETE_SENTENCE_WORDS_ON_TIMEOUT = 5


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
        max_inter_chunk_wait_seconds: float = 0.55,
        playback_prebuffer_chunks: int = 2,
        playback_prebuffer_max_wait_seconds: float = 0.35,
        on_spoken: Optional[Callable[[str, float], None]] = None,
        stop_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
    ):
        self.tts = tts
        self.enabled = enabled
        self.min_words = min_words
        self.max_words = max_words
        self.initial_min_words = initial_min_words or min_words
        self.feedback_min_words = feedback_min_words
        self.max_initial_wait_seconds = max_initial_wait_seconds
        self.max_inter_chunk_wait_seconds = max(0.0, max_inter_chunk_wait_seconds)
        self.playback_prebuffer_chunks = max(1, playback_prebuffer_chunks)
        self.playback_prebuffer_max_wait_seconds = max(
            0.0, playback_prebuffer_max_wait_seconds
        )
        self.on_spoken = on_spoken
        self.stop_event = stop_event
        self.pause_event = pause_event
        self._buffer = ""
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._prepared_queue: queue.Queue[Optional[PreparedSpeech]] = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._prepare_worker: Optional[threading.Thread] = None
        self._play_worker: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._error_lock = threading.Lock()
        self._segments_queued = 0
        self._first_buffered_at: Optional[float] = None
        self._last_buffer_update_at: Optional[float] = None
        self._last_segment_queued_at: Optional[float] = None
        self._pipeline_mode = self.tts.supports_stream_pipelining()

    def __enter__(self) -> "StreamingSpeaker":
        if self.enabled:
            if self._pipeline_mode:
                self._prepare_worker = threading.Thread(target=self._run_prepare, daemon=True)
                self._play_worker = threading.Thread(target=self._run_playback, daemon=True)
                self._prepare_worker.start()
                self._play_worker.start()
            else:
                self._worker = threading.Thread(target=self._run_single_stage, daemon=True)
                self._worker.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    def feed(self, text: str) -> None:
        if not self.enabled or not text or self._stopped():
            return
        if self._first_buffered_at is None:
            self._first_buffered_at = time.perf_counter()
        self._last_buffer_update_at = time.perf_counter()
        self._buffer += text
        while True:
            segment = self._pop_segment(final=False)
            if not segment:
                break
            self._segments_queued += 1
            self._queue.put(segment)
            self._last_segment_queued_at = time.perf_counter()

    def finish(self) -> None:
        if not self.enabled:
            return
        if not self._stopped():
            remaining = self._pop_segment(final=True)
            if remaining:
                self._segments_queued += 1
                self._queue.put(remaining)
                self._last_segment_queued_at = time.perf_counter()
        self._queue.put(None)
        if self._pipeline_mode:
            if self._prepare_worker:
                self._prepare_worker.join()
            if self._play_worker:
                self._play_worker.join()
        elif self._worker:
            self._worker.join()
        if self._error:
            raise RuntimeError(f"streaming TTS failed: {self._error}") from self._error

    def _run_single_stage(self) -> None:
        while True:
            item = self._queue.get()
            if item is None or self._stopped():
                return
            try:
                self._wait_if_paused()
                if self._stopped():
                    return
                started_at = time.perf_counter()
                self.tts.speak(item)
                if self._stopped():
                    return
                if self.on_spoken:
                    self.on_spoken(item, time.perf_counter() - started_at)
            except BaseException as exc:  # Preserve worker errors for caller.
                self._set_error(exc)
                return

    def _run_prepare(self) -> None:
        while True:
            item = self._queue.get()
            if item is None or self._stopped():
                self._prepared_queue.put(None)
                return
            try:
                prepared = self.tts.prepare_for_playback(item)
                if not prepared.text:
                    prepared = PreparedSpeech(text=item, audio_path=prepared.audio_path)
                self._prepared_queue.put(prepared)
            except BaseException as exc:
                self._set_error(exc)
                self._prepared_queue.put(None)
                return

    def _run_playback(self) -> None:
        first = self._prepared_queue.get()
        if first is None or self._stopped():
            return
        buffered: list[PreparedSpeech] = [first]
        stream_done = False
        started_buffering_at = time.perf_counter()
        while (
            len(buffered) < self.playback_prebuffer_chunks
            and not self._stopped()
        ):
            remaining = (
                self.playback_prebuffer_max_wait_seconds
                - (time.perf_counter() - started_buffering_at)
            )
            if remaining <= 0:
                break
            try:
                item = self._prepared_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                stream_done = True
                break
            buffered.append(item)

        while buffered or not stream_done:
            if self._stopped():
                return
            if not buffered:
                item = self._prepared_queue.get()
                if item is None:
                    return
                buffered.append(item)

            current = buffered.pop(0)
            try:
                self._wait_if_paused()
                if self._stopped():
                    return
                started_at = time.perf_counter()
                self.tts.play_prepared(current)
                if self._stopped():
                    return
                if self.on_spoken:
                    self.on_spoken(current.text, time.perf_counter() - started_at)
            except BaseException as exc:
                self._set_error(exc)
                return

            if stream_done:
                continue
            try:
                item = self._prepared_queue.get_nowait()
            except queue.Empty:
                continue
            if item is None:
                stream_done = True
            else:
                buffered.append(item)

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
        now = time.perf_counter()
        waited_too_long = (
            first_segment
            and self._first_buffered_at is not None
            and now - self._first_buffered_at >= self.max_initial_wait_seconds
        )
        waited_too_long = waited_too_long or (
            not first_segment
            and self._last_buffer_update_at is not None
            and now - self._last_buffer_update_at >= self.max_inter_chunk_wait_seconds
        )
        max_words = max(self.max_words, min_words)

        sentence_min_words = (
            1
            if first_segment
            else (
                min_words
                if not waited_too_long
                else min(
                    self.feedback_min_words,
                    _MIN_COMPLETE_SENTENCE_WORDS_ON_TIMEOUT,
                )
            )
        )
        sentence_boundary = _preferred_boundary_index(
            self._buffer,
            _SENTENCE_BOUNDARY_PATTERN,
            min_words=sentence_min_words,
            max_words=max_words,
        )
        if sentence_boundary > 0:
            segment = self._buffer[:sentence_boundary].strip()
            self._buffer = self._buffer[sentence_boundary:]
            return segment

        if first_segment:
            # Keep the opening chunk sentence-aligned for fluency.
            # If there is never a sentence boundary, finish(final=True) flushes
            # the remaining text so short/no-punctuation replies are still spoken.
            return ""

        if len(words) < min_words:
            if not waited_too_long or len(words) < self.feedback_min_words:
                return ""

        if waited_too_long and len(words) >= self.feedback_min_words:
            clause_boundary = _preferred_boundary_index(
                self._buffer,
                _CLAUSE_BOUNDARY_PATTERN,
                min_words=self.feedback_min_words,
                max_words=max_words,
            )
            if clause_boundary > 0:
                segment = self._buffer[:clause_boundary].strip()
                self._buffer = self._buffer[clause_boundary:]
                return segment
            boundary = _word_boundary_index(self._buffer, min(len(words), self.feedback_min_words))
            segment = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:]
            return segment

        if len(words) >= max_words:
            clause_boundary = _preferred_boundary_index(
                self._buffer,
                _CLAUSE_BOUNDARY_PATTERN,
                min_words=self.feedback_min_words,
                max_words=max_words,
            )
            if clause_boundary > 0:
                segment = self._buffer[:clause_boundary].strip()
                self._buffer = self._buffer[clause_boundary:]
                return segment
            boundary = _word_boundary_index(self._buffer, max_words)
            segment = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:]
            return segment

        return ""

    def _stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def _wait_if_paused(self) -> None:
        if self.pause_event is None:
            return
        while self.pause_event.is_set() and not self._stopped():
            time.sleep(0.03)

    def _set_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc


def _word_boundary_index(text: str, word_count: int) -> int:
    matches = list(re.finditer(r"\S+\s*", text))
    if len(matches) < word_count:
        return len(text)
    return matches[word_count - 1].end()


def _preferred_boundary_index(
    text: str,
    pattern: str,
    min_words: int,
    max_words: int,
) -> int:
    chosen = 0
    for match in re.finditer(pattern, text):
        candidate = text[: match.end()].strip()
        words = len(candidate.split())
        if words < min_words or words > max_words:
            continue
        chosen = match.end()
    return chosen
