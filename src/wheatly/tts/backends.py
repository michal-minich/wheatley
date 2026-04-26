from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

from wheatly.audio.filter import apply_voice_filter
from wheatly.audio.playback import play_audio
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
        completed = subprocess.run(command, shell=False, check=False)
        return SpeechResult(audio_path=None, spoken=completed.returncode == 0)


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

        final_path = raw_path
        filtered_path = output_dir / f"{base}.wheatley.wav"
        final_path = apply_voice_filter(raw_path, filtered_path, self.cfg.tts.filter)
        if self.cfg.tts.playback:
            play_audio(final_path, self.cfg.tts.playback_command)
        return SpeechResult(audio_path=final_path, spoken=self.cfg.tts.playback)


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
        if self.cfg.tts.playback:
            play_audio(final_path, self.cfg.tts.playback_command)
        return SpeechResult(audio_path=final_path, spoken=self.cfg.tts.playback)


def build_tts(cfg: Config) -> TTSBackend:
    backend = cfg.resolved_tts_backend().lower()
    if backend in {"none", "disabled"}:
        return NoTTS()
    if backend == "macos_say":
        return MacOSSayTTS(cfg)
    if backend == "piper":
        return PiperTTS(cfg)
    if backend == "external":
        return ExternalCommandTTS(cfg)
    raise ValueError(f"Unsupported TTS backend: {backend}")


def _safe_audio_name(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"reply_{int(time.time())}_{digest}"
