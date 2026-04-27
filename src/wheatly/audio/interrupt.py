from __future__ import annotations

import queue
import re
import threading
import time
import unicodedata
import wave
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from wheatly.audio.playback import current_playback_age_seconds, stop_audio_playback
from wheatly.config import AudioConfig
from wheatly.stt.base import Transcription


TranscribeAudio = Callable[[Path], Transcription]


class SpeechInterruptMonitor:
    def __init__(
        self,
        cfg: AudioConfig,
        transcribe: TranscribeAudio,
        interrupt_event: threading.Event,
        enabled: bool = True,
    ):
        self.cfg = cfg
        self.transcribe = transcribe
        self.interrupt_event = interrupt_event
        self.enabled = enabled and cfg.speech_interrupt_enabled
        self.pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "SpeechInterruptMonitor":
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        self.pause_event.clear()
        if self._thread:
            self._thread.join(timeout=0.5)

    def _run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            return

        audio_queue: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):  # pragma: no cover - hardware
            del frames, time_info, status
            audio_queue.put(indata.copy())

        blocksize = 1024
        pre_roll_blocks = max(
            1,
            int(self.cfg.speech_interrupt_pre_roll_seconds * self.cfg.sample_rate)
            // blocksize,
        )
        pre_roll = deque(maxlen=pre_roll_blocks)
        baseline_rms: Optional[float] = None

        try:
            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype="int16",
                blocksize=blocksize,
                callback=callback,
            ):
                while not self._stop_event.is_set() and not self.interrupt_event.is_set():
                    try:
                        block = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    rms = _rms(np, block)
                    pre_roll.append(block)
                    playback_age = current_playback_age_seconds()
                    if playback_age is None:
                        baseline_rms = None
                        pre_roll.clear()
                        continue
                    if playback_age < self.cfg.speech_interrupt_grace_seconds:
                        baseline_rms = _update_baseline(baseline_rms, rms)
                        continue

                    threshold = self._threshold(baseline_rms)
                    if baseline_rms is None:
                        baseline_rms = rms
                    elif rms < threshold:
                        baseline_rms = baseline_rms * 0.94 + rms * 0.06

                    if rms >= threshold:
                        self._verify_candidate(list(pre_roll), audio_queue, np)
                        pre_roll.clear()
                        baseline_rms = None
        except Exception:
            return

    def _threshold(self, baseline_rms: Optional[float]) -> float:
        threshold = max(
            self.cfg.speech_interrupt_min_rms,
            self.cfg.vad_threshold * self.cfg.speech_interrupt_vad_multiplier,
        )
        if baseline_rms is not None:
            threshold = max(
                threshold,
                baseline_rms * self.cfg.speech_interrupt_baseline_multiplier,
            )
        return threshold

    def _verify_candidate(self, frames, audio_queue: queue.Queue, np) -> None:
        pause_playback = self.cfg.speech_interrupt_pause_tts_while_verifying
        if pause_playback:
            self.pause_event.set()
        started_at = time.monotonic()
        target_samples = int(
            self.cfg.speech_interrupt_record_seconds * self.cfg.sample_rate
        )
        sample_count = sum(len(frame) for frame in frames)
        while (
            sample_count < target_samples
            and not self._stop_event.is_set()
            and not self.interrupt_event.is_set()
        ):
            try:
                block = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if time.monotonic() - started_at >= self.cfg.speech_interrupt_record_seconds:
                    break
                continue
            frames.append(block)
            sample_count += len(block)

        try:
            path = self._write_candidate(np, frames)
            text = self.transcribe(path).text
            if is_stop_interrupt(
                text,
                phrase=self.cfg.speech_interrupt_phrase,
                max_words=self.cfg.speech_interrupt_max_words,
            ):
                self.interrupt_event.set()
                stop_audio_playback()
        except Exception:
            pass
        finally:
            if pause_playback:
                self.pause_event.clear()

    def _write_candidate(self, np, frames) -> Path:
        output_dir = Path(self.cfg.utterance_dir) / "interrupts"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"interrupt_{time.time_ns()}.wav"
        audio = np.concatenate(frames, axis=0)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(self.cfg.channels)
            handle.setsampwidth(2)
            handle.setframerate(self.cfg.sample_rate)
            handle.writeframes(audio.tobytes())
        return path


def is_stop_interrupt(text: str, phrase: str = "stop", max_words: int = 4) -> bool:
    normalized = _normalize(text)
    target = _normalize(phrase)
    if not normalized or not target:
        return False
    words = normalized.split()
    if len(words) > max_words:
        return False
    return target in words


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def _rms(np, block) -> float:
    samples = block.astype("float32") / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


def _update_baseline(current: Optional[float], rms: float) -> float:
    if current is None:
        return rms
    return current * 0.94 + rms * 0.06
