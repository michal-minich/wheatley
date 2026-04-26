from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
import time
import wave
from pathlib import Path

from wheatly.audio.filter import apply_voice_filter
from wheatly.audio.playback import play_audio, run_playback_command
from wheatly.config import Config
from wheatly.tts.base import SpeechResult, TTSBackend


class NoTTS(TTSBackend):
    def speak(self, text: str) -> SpeechResult:
        del text
        return SpeechResult(audio_path=None, spoken=False)


class MacOSSayTTS(TTSBackend):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def speak(self, text: str) -> SpeechResult:
        if not self.cfg.tts.enabled:
            return SpeechResult(audio_path=None, spoken=False)
        command = ["say", "-v", self.cfg.tts.voice, text]
        return SpeechResult(audio_path=None, spoken=run_playback_command(command))


class PiperTTS(TTSBackend):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def speak(self, text: str) -> SpeechResult:
        if not self.cfg.tts.enabled:
            return SpeechResult(audio_path=None, spoken=False)
        output_dir = Path(self.cfg.tts.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base = _safe_audio_name(text)
        raw_path = output_dir / f"{base}.wav"
        command = [
            self.cfg.tts.piper_binary,
            "--model",
            self.cfg.tts.piper_model,
            "--output_file",
            str(raw_path),
            "--length-scale",
            str(self.cfg.tts.length_scale),
            "--noise-scale",
            str(self.cfg.tts.noise_scale),
            "--noise-w-scale",
            str(self.cfg.tts.noise_w_scale),
            "--sentence-silence",
            str(self.cfg.tts.sentence_silence),
            "--volume",
            str(self.cfg.tts.volume),
        ]
        if self.cfg.tts.piper_config:
            command.extend(["--config", self.cfg.tts.piper_config])
        if self.cfg.tts.piper_speaker is not None:
            command.extend(["--speaker", str(self.cfg.tts.piper_speaker)])
        completed = subprocess.run(
            command,
            input=text,
            capture_output=True,
            text=True,
            shell=False,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "piper failed")

        final_path = _postprocess_audio(raw_path, output_dir, base, self.cfg)
        spoken = False
        if self.cfg.tts.playback:
            spoken = play_audio(final_path, self.cfg.tts.playback_command)
        return SpeechResult(audio_path=final_path, spoken=spoken)


class EdgeTTSTTS(TTSBackend):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def speak(self, text: str) -> SpeechResult:
        if not self.cfg.tts.enabled:
            return SpeechResult(audio_path=None, spoken=False)
        output_dir = Path(self.cfg.tts.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base = _safe_audio_name(text)
        raw_path = output_dir / f"{base}.edge.mp3"
        _run_edge_tts(
            text=text,
            output_path=raw_path,
            voice=self.cfg.tts.edge_voice,
            rate=self.cfg.tts.edge_rate,
            pitch=self.cfg.tts.edge_pitch,
            volume=self.cfg.tts.edge_volume,
        )
        final_path = _postprocess_audio(raw_path, output_dir, base, self.cfg)
        spoken = False
        if self.cfg.tts.playback:
            spoken = play_audio(final_path, self.cfg.tts.playback_command)
        return SpeechResult(audio_path=final_path, spoken=spoken)


class ExternalCommandTTS(TTSBackend):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def speak(self, text: str) -> SpeechResult:
        if not self.cfg.tts.enabled:
            return SpeechResult(audio_path=None, spoken=False)
        if not self.cfg.tts.external_command:
            raise RuntimeError("tts.external_command is required for external TTS")
        output_dir = Path(self.cfg.tts.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base = _safe_audio_name(text)
        raw_path = output_dir / f"{base}.wav"
        command = [
            part.format(text=text, output=str(raw_path))
            for part in self.cfg.tts.external_command
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, shell=False, timeout=90
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "external TTS failed")
        if not raw_path.exists():
            return SpeechResult(audio_path=None, spoken=False)
        final_path = apply_voice_filter(
            raw_path, output_dir / f"{base}.wheatley.wav", self.cfg.tts.filter
        )
        spoken = False
        if self.cfg.tts.playback:
            spoken = play_audio(final_path, self.cfg.tts.playback_command)
        return SpeechResult(audio_path=final_path, spoken=spoken)


def build_tts(cfg: Config) -> TTSBackend:
    backend = cfg.resolved_tts_backend().lower()
    if backend in {"none", "disabled"}:
        return NoTTS()
    if backend == "macos_say":
        return MacOSSayTTS(cfg)
    if backend == "piper":
        return PiperTTS(cfg)
    if backend == "edge_tts":
        return EdgeTTSTTS(cfg)
    if backend == "external":
        return ExternalCommandTTS(cfg)
    raise ValueError(f"Unsupported TTS backend: {backend}")


def _safe_audio_name(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"reply_{int(time.time())}_{digest}"


def _postprocess_audio(raw_path: Path, output_dir: Path, base: str, cfg: Config) -> Path:
    tts_input_path = raw_path
    if cfg.tts.leading_silence_ms > 0:
        padded_path = output_dir / f"{base}.padded.wav"
        tts_input_path = _add_leading_silence_any(
            raw_path,
            padded_path,
            cfg.tts.leading_silence_ms,
            cfg.tts.filter.ffmpeg_binary,
        )
    filtered_path = output_dir / f"{base}.wheatley.wav"
    return apply_voice_filter(tts_input_path, filtered_path, cfg.tts.filter)


def _add_leading_silence(input_path: Path, output_path: Path, milliseconds: int) -> Path:
    if milliseconds <= 0:
        return input_path
    try:
        with wave.open(str(input_path), "rb") as source:
            params = source.getparams()
            frames = source.readframes(source.getnframes())
        silence_frames = int(params.framerate * milliseconds / 1000)
        silence = b"\x00" * silence_frames * params.nchannels * params.sampwidth
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as target:
            target.setparams(params)
            target.writeframes(silence + frames)
        return output_path
    except (OSError, wave.Error):
        return input_path


def _add_leading_silence_any(
    input_path: Path,
    output_path: Path,
    milliseconds: int,
    ffmpeg_binary: str,
) -> Path:
    if milliseconds <= 0:
        return input_path
    if input_path.suffix.lower() == ".wav":
        return _add_leading_silence(input_path, output_path, milliseconds)
    ffmpeg = shutil.which(ffmpeg_binary)
    if not ffmpeg:
        return input_path
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-af",
            f"adelay={milliseconds}:all=1",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        return input_path
    return output_path


def _run_edge_tts(
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    pitch: str,
    volume: str,
) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError(
            "edge_tts backend requires the edge-tts package. "
            "Install it with: python3 -m pip install 'edge-tts>=7.0.0'"
        ) from exc

    async def save() -> None:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )
        await communicate.save(str(output_path))

    asyncio.run(save())
