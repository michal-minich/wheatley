from __future__ import annotations

import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

from wheatley.audio.devices import input_stream_device_kwargs
from wheatley.audio.log_paths import timestamped_audio_filename
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
        speech_started_sample_count = None
        last_voice_sample_count = None
        captured_sample_count = 0
        last_voice_frame_count = 0
        frames = []
        pre_roll_frames = []
        pre_roll_sample_count = 0
        pre_roll_sample_limit = _pre_roll_sample_limit(self.cfg)
        wait_started_at = time.monotonic()

        try:
            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype="int16",
                blocksize=1024,
                callback=callback,
                **input_stream_device_kwargs(self.cfg, sd),
            ):
                while True:
                    block = audio_queue.get()
                    block_sample_count = len(block)
                    audio_position_samples = captured_sample_count + block_sample_count
                    samples = block.astype("float32") / 32768.0
                    rms = float(np.sqrt(np.mean(samples * samples)))
                    now = time.monotonic()

                    has_voice = rms >= self.cfg.vad_threshold
                    if has_voice:
                        if not started:
                            started = True
                            speech_started_sample_count = captured_sample_count
                            frames.extend(pre_roll_frames)
                        last_voice_sample_count = audio_position_samples
                    elif not started:
                        pre_roll_frames, pre_roll_sample_count = _append_pre_roll_frame(
                            pre_roll_frames,
                            pre_roll_sample_count,
                            block,
                            pre_roll_sample_limit,
                        )

                    if started:
                        frames.append(block)
                        if has_voice:
                            last_voice_frame_count = len(frames)
                        partial_worker.maybe_submit(frames, now)

                    if (
                        not started
                        and self.cfg.max_wait_seconds > 0
                        and now - wait_started_at > self.cfg.max_wait_seconds
                    ):
                        raise TimeoutError("no speech detected before max_wait_seconds")

                    if started and speech_started_sample_count is not None:
                        enough_speech, enough_silence, too_long = (
                            _endpoint_timing_reached(
                                self.cfg,
                                audio_position_samples=audio_position_samples,
                                speech_started_sample_count=(
                                    speech_started_sample_count
                                ),
                                last_voice_sample_count=last_voice_sample_count,
                                has_voice=has_voice,
                            )
                        )
                        if (enough_speech and enough_silence) or too_long:
                            break
                    captured_sample_count = audio_position_samples
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
        timestamp_ns = time.time_ns()
        partial_path = self.output_path.parent / "partials" / timestamped_audio_filename(
            "user_partial",
            ".wav",
            timestamp_ns=timestamp_ns,
            extra=self.output_path.stem,
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(cfg.channels)
        handle.setsampwidth(2)
        handle.setframerate(cfg.sample_rate)
        handle.writeframes(audio.tobytes())


def _pre_roll_sample_limit(cfg: AudioConfig) -> int:
    return max(0, int(cfg.pre_roll_seconds * cfg.sample_rate))


def _append_pre_roll_frame(frames, sample_count: int, block, sample_limit: int):
    if sample_limit <= 0:
        return [], 0
    frames = list(frames)
    frames.append(block)
    sample_count += len(block)
    while frames and sample_count - len(frames[0]) >= sample_limit:
        sample_count -= len(frames.pop(0))
    return frames, sample_count


def _trim_trailing_silence(frames, last_voice_frame_count: int, cfg: AudioConfig):
    if not frames or last_voice_frame_count <= 0:
        return frames
    keep_samples = _trailing_silence_keep_samples(cfg)
    trimmed = list(frames[:last_voice_frame_count])
    for block in frames[last_voice_frame_count:]:
        if keep_samples <= 0:
            break
        trimmed.append(block)
        keep_samples -= len(block)
    return trimmed or frames


def _trailing_silence_keep_samples(cfg: AudioConfig) -> int:
    keep_seconds = max(
        0.0,
        cfg.trailing_silence_keep_seconds,
        min(max(0.0, cfg.silence_seconds), 2.0),
    )
    return int(keep_seconds * cfg.sample_rate)


def _endpoint_timing_reached(
    cfg: AudioConfig,
    audio_position_samples: int,
    speech_started_sample_count: Optional[int],
    last_voice_sample_count: Optional[int],
    has_voice: bool,
) -> tuple[bool, bool, bool]:
    sample_rate = float(cfg.sample_rate or 1)
    speech_seconds = 0.0
    if speech_started_sample_count is not None:
        speech_seconds = max(
            0.0, (audio_position_samples - speech_started_sample_count) / sample_rate
        )
    silence_seconds = 0.0
    if last_voice_sample_count is not None:
        silence_seconds = max(
            0.0, (audio_position_samples - last_voice_sample_count) / sample_rate
        )

    enough_speech = speech_seconds >= max(0.0, cfg.min_speech_seconds)
    enough_silence = (
        last_voice_sample_count is not None
        and silence_seconds >= cfg.silence_seconds
    )
    too_long = (
        cfg.max_utterance_seconds > 0
        and speech_seconds >= cfg.max_utterance_seconds
        and enough_silence
        and not has_voice
    )
    return enough_speech, enough_silence, too_long
