from __future__ import annotations

import math
import wave
from pathlib import Path

from wheatly.audio.playback import play_audio
from wheatly.config import AudioConfig


CHIME_SAMPLE_RATE = 44100
CHIME_VERSION = 2


def play_listening_chime(event: str, cfg: AudioConfig) -> None:
    if not cfg.listening_chimes_enabled:
        return
    try:
        path = ensure_listening_chime(event, cfg)
        play_audio(path)
    except OSError:
        return


def ensure_listening_chime(event: str, cfg: AudioConfig) -> Path:
    output_dir = Path(cfg.utterance_dir) / "chimes"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"listening_{event}_v{CHIME_VERSION}.wav"
    if path.exists():
        return path
    if event == "start":
        audio = _render_chime(392.0, 659.25, 0.36, cfg.listening_chime_volume)
    elif event == "stop":
        audio = _render_deep_gong(293.66, 146.83, 0.78, cfg.listening_chime_volume)
    else:
        raise ValueError(f"Unknown listening chime event: {event}")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(CHIME_SAMPLE_RATE)
        handle.writeframes(audio)
    return path


def _render_chime(
    start_hz: float, end_hz: float, duration_seconds: float, volume: float
) -> bytes:
    frame_count = int(CHIME_SAMPLE_RATE * duration_seconds)
    volume = max(0.0, min(1.0, volume))
    frames = bytearray()
    phase = 0.0
    phase_bell = 0.0
    for index in range(frame_count):
        t = index / CHIME_SAMPLE_RATE
        progress = index / max(1, frame_count - 1)
        frequency = start_hz + (end_hz - start_hz) * _smoothstep(progress)
        phase += math.tau * frequency / CHIME_SAMPLE_RATE
        phase_bell += math.tau * frequency * 2.41 / CHIME_SAMPLE_RATE
        attack = min(1.0, t / 0.018)
        decay = math.exp(-5.0 * progress)
        shimmer = math.exp(-8.0 * progress)
        sample = (
            math.sin(phase) * 0.72 * decay
            + math.sin(phase_bell) * 0.28 * shimmer
        )
        value = int(max(-1.0, min(1.0, sample * attack * volume)) * 32767)
        frames.extend(value.to_bytes(2, byteorder="little", signed=True))
    return bytes(frames)


def _render_deep_gong(
    start_hz: float, end_hz: float, duration_seconds: float, volume: float
) -> bytes:
    frame_count = int(CHIME_SAMPLE_RATE * duration_seconds)
    volume = max(0.0, min(1.0, volume))
    frames = bytearray()
    phase_low = 0.0
    phase_body = 0.0
    phase_metal = 0.0
    for index in range(frame_count):
        t = index / CHIME_SAMPLE_RATE
        progress = index / max(1, frame_count - 1)
        frequency = start_hz + (end_hz - start_hz) * _smoothstep(progress)
        phase_low += math.tau * frequency / CHIME_SAMPLE_RATE
        phase_body += math.tau * frequency * 1.51 / CHIME_SAMPLE_RATE
        phase_metal += math.tau * frequency * 2.03 / CHIME_SAMPLE_RATE
        attack = min(1.0, t / 0.035)
        low_decay = math.exp(-2.8 * progress)
        metal_decay = math.exp(-7.0 * progress)
        sample = (
            math.sin(phase_low) * 0.74 * low_decay
            + math.sin(phase_body) * 0.20 * low_decay
            + math.sin(phase_metal) * 0.06 * metal_decay
        )
        value = int(max(-1.0, min(1.0, sample * attack * volume)) * 32767)
        frames.extend(value.to_bytes(2, byteorder="little", signed=True))
    return bytes(frames)


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)
