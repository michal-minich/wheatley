from __future__ import annotations

import json
import mimetypes
import subprocess
import urllib.error
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Optional

from wheatley.config import STTConfig
from wheatley.stt.base import STTBackend, Transcription


class KeyboardSTT(STTBackend):
    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        if audio_path:
            raise RuntimeError("keyboard STT cannot transcribe audio files")
        return Transcription(text=input("you> ").strip(), language=None)


class FasterWhisperSTT(STTBackend):
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self._model = None

    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        if not audio_path:
            raise RuntimeError("faster-whisper requires an audio file")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Install faster-whisper first: pip install '.[stt]'"
            ) from exc

        if self._model is None:
            self._model = WhisperModel(
                self.cfg.model,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
        segments, info = self._model.transcribe(
            str(audio_path),
            language=self.cfg.language,
            task="transcribe",
            beam_size=1,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            max_new_tokens=160,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcription(
            text=text,
            language=getattr(info, "language", None),
            duration_seconds=getattr(info, "duration", None),
        )


class RemoteFallbackSTT(STTBackend):
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self._fallback: Optional[STTBackend] = None

    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        if not audio_path:
            raise RuntimeError("remote STT requires an audio file")
        if not remote_stt_available(self.cfg):
            return self._fallback_backend().transcribe(audio_path)
        try:
            return _transcribe_remote(audio_path, self.cfg)
        except (OSError, RuntimeError, TimeoutError, urllib.error.URLError):
            return self._fallback_backend().transcribe(audio_path)

    def _fallback_backend(self) -> STTBackend:
        if self._fallback is None:
            fallback_backend = self.cfg.remote_fallback_backend.lower().replace("-", "_")
            if fallback_backend in {"remote", "remote_fallback"}:
                raise RuntimeError("remote STT fallback backend cannot be remote")
            fallback_cfg = replace(self.cfg, backend=self.cfg.remote_fallback_backend)
            self._fallback = build_stt(fallback_cfg)
        return self._fallback


class WhisperCppSTT(STTBackend):
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg

    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        if not audio_path:
            raise RuntimeError("whisper.cpp requires an audio file")
        command = [
            self.cfg.whisper_cpp_binary,
            "-m",
            self.cfg.whisper_cpp_model,
            "-f",
            str(audio_path),
        ] + self.cfg.whisper_cpp_args
        completed = subprocess.run(
            command, capture_output=True, text=True, shell=False, timeout=120
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "whisper.cpp failed")
        return Transcription(text=_clean_whisper_cpp_output(completed.stdout))


def build_stt(cfg: STTConfig) -> STTBackend:
    backend = cfg.backend.lower()
    if backend == "keyboard":
        return KeyboardSTT()
    if backend in {"remote_fallback", "remote-fallback", "remote"}:
        return RemoteFallbackSTT(cfg)
    if backend in {"faster_whisper", "faster-whisper"}:
        return FasterWhisperSTT(cfg)
    if backend in {"whisper_cpp", "whisper.cpp"}:
        return WhisperCppSTT(cfg)
    raise ValueError(f"Unsupported STT backend: {cfg.backend}")


def _clean_whisper_cpp_output(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("whisper_") or line.startswith("main:"):
            continue
        lines.append(line)
    return " ".join(lines).strip()


def _transcribe_remote(audio_path: Path, cfg: STTConfig) -> Transcription:
    endpoint = _remote_stt_endpoint(cfg.remote_base_url)
    fields = {
        "model": cfg.remote_model or cfg.model or "default",
        "response_format": "json",
    }
    if cfg.language:
        fields["language"] = cfg.language
    body, content_type = _multipart_body(audio_path, fields)
    headers = {
        "Content-Type": content_type,
        "Accept": "application/json",
    }
    if cfg.remote_api_key and cfg.remote_api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {cfg.remote_api_key}"
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(
            request, timeout=cfg.remote_request_timeout_seconds
        ) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"remote STT failed: HTTP {exc.code}: {detail}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("remote STT returned non-JSON response") from exc
    text = str(payload.get("text", "")).strip()
    if not text:
        raise RuntimeError("remote STT returned empty text")
    duration = payload.get("duration_seconds")
    return Transcription(
        text=text,
        language=payload.get("language") or cfg.language,
        duration_seconds=duration if isinstance(duration, (int, float)) else None,
    )


def remote_stt_available(cfg: STTConfig) -> bool:
    if cfg.remote_probe_timeout_seconds <= 0:
        return True
    request = urllib.request.Request(_remote_health_endpoint(cfg.remote_base_url))
    try:
        with urllib.request.urlopen(
            request, timeout=cfg.remote_probe_timeout_seconds
        ) as response:
            return 200 <= response.status < 300
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def _remote_stt_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/audio/transcriptions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/audio/transcriptions"
    return f"{base}/v1/audio/transcriptions"


def _remote_health_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/health"
    if base.endswith("/audio/transcriptions"):
        return base.rsplit("/audio/transcriptions", 1)[0] + "/health"
    return f"{base}/health"


def _multipart_body(audio_path: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"wheatley-{uuid.uuid4().hex}"
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "ascii"
                ),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    filename = audio_path.name
    content_type = mimetypes.guess_type(filename)[0] or "audio/wav"
    lines.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            audio_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"
