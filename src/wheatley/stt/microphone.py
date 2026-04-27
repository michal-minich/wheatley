from __future__ import annotations

import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

from wheatley.config import AudioConfig


class MicrophoneRecorder:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg

    def record_utterance(
        self,
        output_path: Path,
        partial_transcriber: Optional[Callable[[Path], str]] = None,
        on_partial_transcript: Optional[Callable[[str], None]] = None,
    ) -> Path:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Install audio extras first: pip install '.[audio]'"
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_queue: queue.Queue = queue.Queue()
        partial_worker = _PartialTranscriptWorker(
            cfg=self.cfg,
            output_path=output_path,
            transcriber=partial_transcriber,
            callback=on_partial_transcript,
        )

        def callback(indata, frames, time_info, status):  # pragma: no cover - hardware
            del frames, time_info, status
            audio_queue.put(indata.copy())

        started = False
        speech_started_at = None
        last_voice_at = None
        last_voice_frame_count = 0
        frames = []
        wait_started_at = time.monotonic()

        try:
            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype="int16",
                blocksize=1024,
                callback=callback,
            ):
                while True:
                    block = audio_queue.get()
                    samples = block.astype("float32") / 32768.0
                    rms = float(np.sqrt(np.mean(samples * samples)))
                    now = time.monotonic()

                    has_voice = rms >= self.cfg.vad_threshold
                    if has_voice:
                        if not started:
                            started = True
                            speech_started_at = now
                        last_voice_at = now

                    if started:
                        frames.append(block)
                        if has_voice:
                            last_voice_frame_count = len(frames)
                        partial_worker.maybe_submit(frames, now)

                    if (
                        not started
                        and now - wait_started_at > self.cfg.max_wait_seconds
                    ):
                        raise TimeoutError("no speech detected before max_wait_seconds")

                    if started and speech_started_at is not None:
                        enough_speech = (
                            now - speech_started_at >= self.cfg.min_speech_seconds
                        )
                        enough_silence = (
                            last_voice_at is not None
                            and now - last_voice_at >= self.cfg.silence_seconds
                        )
                        too_long = (
                            now - speech_started_at >= self.cfg.max_utterance_seconds
                        )
                        if (enough_speech and enough_silence) or too_long:
                            break
        finally:
            partial_worker.stop()

        if not frames:
            raise RuntimeError("recording ended without audio frames")

        frames = _trim_trailing_silence(frames, last_voice_frame_count, self.cfg)
        _write_wav(output_path, frames, self.cfg)
        return output_path


class _PartialTranscriptWorker:
    def __init__(
        self,
        cfg: AudioConfig,
        output_path: Path,
        transcriber: Optional[Callable[[Path], str]],
        callback: Optional[Callable[[str], None]],
    ):
        self.cfg = cfg
        self.output_path = output_path
        self.transcriber = transcriber
        self.callback = callback
        self.last_submit_at = 0.0
        self.started_at: Optional[float] = None
        self.active = bool(
            cfg.partial_transcript_enabled
            and transcriber is not None
            and callback is not None
        )
        self.busy = False
        self.lock = threading.Lock()

    def maybe_submit(self, frames, now: float) -> None:
        if not self.active:
            return
        if self.started_at is None:
            self.started_at = now
        if now - self.started_at < self.cfg.partial_transcript_min_audio_seconds:
            return
        if now - self.last_submit_at < self.cfg.partial_transcript_interval_seconds:
            return
        with self.lock:
            if self.busy:
                return
            self.busy = True
        self.last_submit_at = now
        snapshot = list(frames)
        thread = threading.Thread(target=self._run, args=(snapshot,), daemon=True)
        thread.start()

    def stop(self) -> None:
        self.active = False

    def _run(self, frames) -> None:
        partial_path = self.output_path.with_name(
            f"{self.output_path.stem}.partial_{time.time_ns()}.wav"
        )
        try:
            _write_wav(partial_path, frames, self.cfg)
            text = (
                self.transcriber(partial_path) if self.transcriber else ""
            ).strip()
            if text and self.active and self.callback:
                self.callback(text)
        except Exception:
            pass
        finally:
            with self.lock:
                self.busy = False


def _write_wav(output_path: Path, frames, cfg: AudioConfig) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install audio extras first: pip install '.[audio]'") from exc

    audio = np.concatenate(frames, axis=0)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(cfg.channels)
        handle.setsampwidth(2)
        handle.setframerate(cfg.sample_rate)
        handle.writeframes(audio.tobytes())


def _trim_trailing_silence(frames, last_voice_frame_count: int, cfg: AudioConfig):
    if not frames or last_voice_frame_count <= 0:
        return frames
    keep_samples = max(0, int(cfg.trailing_silence_keep_seconds * cfg.sample_rate))
    trimmed = list(frames[:last_voice_frame_count])
    for block in frames[last_voice_frame_count:]:
        if keep_samples <= 0:
            break
        trimmed.append(block)
        keep_samples -= len(block)
    return trimmed or frames
