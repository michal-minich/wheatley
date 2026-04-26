from __future__ import annotations

import queue
import time
import wave
from pathlib import Path

from wheatly.config import AudioConfig


class MicrophoneRecorder:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg

    def record_utterance(self, output_path: Path) -> Path:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Install audio extras first: pip install '.[audio]'"
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_queue: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):  # pragma: no cover - hardware
            del frames, time_info, status
            audio_queue.put(indata.copy())

        started = False
        speech_started_at = None
        last_voice_at = None
        frames = []
        wait_started_at = time.monotonic()

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

                if rms >= self.cfg.vad_threshold:
                    if not started:
                        started = True
                        speech_started_at = now
                    last_voice_at = now

                if started:
                    frames.append(block)

                if not started and now - wait_started_at > self.cfg.max_wait_seconds:
                    raise TimeoutError("no speech detected before max_wait_seconds")

                if started and speech_started_at is not None:
                    enough_speech = now - speech_started_at >= self.cfg.min_speech_seconds
                    enough_silence = (
                        last_voice_at is not None
                        and now - last_voice_at >= self.cfg.silence_seconds
                    )
                    too_long = now - speech_started_at >= self.cfg.max_utterance_seconds
                    if (enough_speech and enough_silence) or too_long:
                        break

        if not frames:
            raise RuntimeError("recording ended without audio frames")

        audio = np.concatenate(frames, axis=0)
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(self.cfg.channels)
            handle.setsampwidth(2)
            handle.setframerate(self.cfg.sample_rate)
            handle.writeframes(audio.tobytes())
        return output_path

