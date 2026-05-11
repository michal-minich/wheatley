from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from wheatley.audio.filter import apply_voice_filter
from wheatley.audio.log_paths import dated_audio_dir, timestamped_audio_filename
from wheatley.audio.playback import play_audio, run_playback_command
from wheatley.config import Config
from wheatley.tts.base import PreparedSpeech, SpeechResult, TTSBackend


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
        text = _normalize_tts_text(text)
        if not text:
            return SpeechResult(audio_path=None, spoken=False)
        command = ["say", "-v", self.cfg.tts.voice, text]
        return SpeechResult(audio_path=None, spoken=run_playback_command(command))


class _PreparedAudioTTS(TTSBackend):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def speak(self, text: str) -> SpeechResult:
        if not self.cfg.tts.enabled:
            return SpeechResult(audio_path=None, spoken=False)
        prepared = self.prepare_for_playback(text)
        spoken = self.play_prepared(prepared)
        return SpeechResult(audio_path=prepared.audio_path, spoken=spoken)

    def supports_stream_pipelining(self) -> bool:
        return True

    def play_prepared(self, prepared: PreparedSpeech) -> bool:
        return _play_prepared_audio(self.cfg, prepared)

    def _prepare_text(self, text: str) -> tuple[str, PreparedSpeech | None]:
        if not self.cfg.tts.enabled:
            return "", PreparedSpeech(text=text, audio_path=None)
        text = _normalize_tts_text(text)
        if not text:
            return "", PreparedSpeech(text="", audio_path=None)
        return text, None


class PiperTTS(_PreparedAudioTTS):
    def prepare_for_playback(self, text: str) -> PreparedSpeech:
        text, empty = self._prepare_text(text)
        if empty:
            return empty
        spoken_text = _apply_piper_pronunciation_replacements(text, self.cfg)
        paths = _tts_audio_paths(
            Path(self.cfg.tts.output_dir),
            spoken_text,
            ".wav",
            self.cfg,
        )
        raw_path = paths.raw_path
        piper_binary = self.cfg.tts.piper_binary
        command = []
        if Path(piper_binary).name.lower().startswith("python"):
            command.extend([piper_binary, "-m", "piper"])
        else:
            command.append(piper_binary)
        command.extend(
            [
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
        )
        if self.cfg.tts.piper_config:
            command.extend(["--config", self.cfg.tts.piper_config])
        if self.cfg.tts.piper_speaker is not None:
            command.extend(["--speaker", str(self.cfg.tts.piper_speaker)])
        completed = subprocess.run(
            command,
            input=spoken_text,
            capture_output=True,
            text=True,
            shell=False,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "piper failed")

        final_path = _postprocess_audio(paths, self.cfg)
        return PreparedSpeech(text=text, audio_path=final_path)


class EdgeTTSTTS(_PreparedAudioTTS):
    def prepare_for_playback(self, text: str) -> PreparedSpeech:
        text, empty = self._prepare_text(text)
        if empty:
            return empty
        paths = _tts_audio_paths(
            Path(self.cfg.tts.output_dir),
            text,
            ".edge.mp3",
            self.cfg,
        )
        raw_path = paths.raw_path
        _run_edge_tts(
            text=text,
            output_path=raw_path,
            voice=self.cfg.tts.edge_voice,
            rate=self.cfg.tts.edge_rate,
            pitch=self.cfg.tts.edge_pitch,
            volume=self.cfg.tts.edge_volume,
        )
        final_path = _postprocess_audio(paths, self.cfg)
        return PreparedSpeech(text=text, audio_path=final_path)


class ExternalCommandTTS(_PreparedAudioTTS):
    def prepare_for_playback(self, text: str) -> PreparedSpeech:
        text, empty = self._prepare_text(text)
        if empty:
            return empty
        if not self.cfg.tts.external_command:
            raise RuntimeError("tts.external_command is required for external TTS")
        paths = _tts_audio_paths(Path(self.cfg.tts.output_dir), text, ".wav", self.cfg)
        raw_path = paths.raw_path
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
            return PreparedSpeech(text=text, audio_path=None)
        final_path = _postprocess_audio(paths, self.cfg)
        return PreparedSpeech(text=text, audio_path=final_path)


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


@dataclass
class _TTSAudioPaths:
    raw_path: Path
    final_path: Path
    padded_path: Path


def _tts_audio_paths(
    output_root: Path,
    text: str,
    raw_suffix: str,
    cfg: Config,
) -> _TTSAudioPaths:
    timestamp_ns = time.time_ns()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    date_dir = dated_audio_dir(output_root, timestamp_ns)
    intermediate_dir = date_dir / "intermediate"
    final_suffix = ".wheatley.wav" if cfg.tts.filter.enabled else raw_suffix
    final_name = timestamped_audio_filename(
        "agent",
        final_suffix,
        timestamp_ns=timestamp_ns,
        extra=digest,
    )
    raw_name = timestamped_audio_filename(
        "agent_raw",
        raw_suffix,
        timestamp_ns=timestamp_ns,
        extra=digest,
    )
    padded_name = timestamped_audio_filename(
        "agent_padded",
        ".wav",
        timestamp_ns=timestamp_ns,
        extra=digest,
    )
    needs_raw_intermediate = cfg.tts.filter.enabled or cfg.tts.leading_silence_ms > 0
    final_path = date_dir / final_name
    raw_path = intermediate_dir / raw_name if needs_raw_intermediate else final_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return _TTSAudioPaths(
        raw_path=raw_path,
        final_path=final_path,
        padded_path=intermediate_dir / padded_name,
    )


def _normalize_tts_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    # Trailing ellipsis can produce audible "dot dot dot" artifacts in some voices.
    text = re.sub(r"(?:\.{3,}|…+)\s*$", ".", text)
    return text


def _apply_piper_pronunciation_replacements(text: str, cfg: Config) -> str:
    for pattern, replacement in cfg.tts.piper_pronunciation_replacements.items():
        try:
            text = re.sub(pattern, replacement, text)
        except re.error as exc:
            raise ValueError(
                f"Invalid tts.piper_pronunciation_replacements pattern {pattern!r}: {exc}"
            ) from exc
    return text


def _play_prepared_audio(cfg: Config, prepared: PreparedSpeech) -> bool:
    if not cfg.tts.playback:
        return False
    if prepared.audio_path is None:
        return False
    return play_audio(prepared.audio_path, cfg.tts.playback_command)


def _postprocess_audio(paths: _TTSAudioPaths, cfg: Config) -> Path:
    tts_input_path = paths.raw_path
    if cfg.tts.leading_silence_ms > 0:
        padded_path = paths.padded_path if cfg.tts.filter.enabled else paths.final_path
        tts_input_path = _add_leading_silence_with_ffmpeg(
            paths.raw_path,
            padded_path,
            cfg.tts.leading_silence_ms,
            cfg.tts.filter.ffmpeg_binary,
        )
    if cfg.tts.filter.enabled:
        return apply_voice_filter(tts_input_path, paths.final_path, cfg.tts.filter)
    return tts_input_path


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


def _add_leading_silence_with_ffmpeg(
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
