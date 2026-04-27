from __future__ import annotations

import argparse
import json
import tempfile
import time
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from wheatley.config import STTConfig
from wheatley.stt.backends import build_stt
from wheatley.stt.base import STTBackend


class STTModelRouter:
    def __init__(
        self,
        backend: str,
        default_model: str,
        model_map: dict[str, str],
        device: str,
        compute_type: str,
    ):
        self.backend = backend
        self.default_model = default_model
        self.model_map = model_map
        self.device = device
        self.compute_type = compute_type
        self._backends: dict[tuple[str, str], STTBackend] = {}

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str],
        requested_model: Optional[str],
    ):
        model = self._select_model(language, requested_model)
        key = (language or "", model)
        if key not in self._backends:
            cfg = STTConfig(
                backend=self.backend,
                model=model,
                language=language,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._backends[key] = build_stt(cfg)
        return self._backends[key].transcribe(audio_path)

    def models(self) -> list[dict[str, str]]:
        rows = [{"language": "default", "model": self.default_model}]
        rows.extend(
            {"language": language, "model": model}
            for language, model in sorted(self.model_map.items())
        )
        return rows

    def _select_model(self, language: Optional[str], requested_model: Optional[str]) -> str:
        if requested_model and requested_model != "default":
            return requested_model
        if language and language in self.model_map:
            return self.model_map[language]
        return self.default_model


def serve(
    host: str,
    port: int,
    router: STTModelRouter,
) -> None:
    class Handler(_STTRequestHandler):
        model_router = router

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"wheatley STT server listening on http://{host}:{port}/v1")
    server.serve_forever()


class _STTRequestHandler(BaseHTTPRequestHandler):
    model_router: STTModelRouter

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/health"}:
            self._json({"ok": True})
            return
        if self.path in {"/v1/models", "/models"}:
            self._json({"data": self.model_router.models()})
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        if self.path not in {"/v1/audio/transcriptions", "/audio/transcriptions"}:
            self.send_error(404, "not found")
            return
        try:
            fields, file_bytes, filename = self._read_multipart()
            language = _blank_to_none(fields.get("language"))
            model = _blank_to_none(fields.get("model"))
            suffix = Path(filename or "audio.wav").suffix or ".wav"
            started = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as audio:
                audio.write(file_bytes)
                audio.flush()
                result = self.model_router.transcribe(Path(audio.name), language, model)
            self._json(
                {
                    "text": result.text,
                    "language": result.language or language,
                    "duration_seconds": result.duration_seconds,
                    "wall_seconds": round(time.perf_counter() - started, 4),
                }
            )
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:
        print("%s - %s" % (self.address_string(), format % args))

    def _read_multipart(self) -> tuple[dict[str, str], bytes, str]:
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        header = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n"
            "\r\n"
        ).encode("utf-8")
        message = BytesParser(policy=policy.default).parsebytes(header + body)
        if not message.is_multipart():
            raise RuntimeError("expected multipart/form-data")
        fields: dict[str, str] = {}
        file_bytes = b""
        filename = "audio.wav"
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            if name == "file":
                file_bytes = payload
                filename = (
                    part.get_filename()
                    or part.get_param("filename", header="content-disposition")
                    or filename
                )
            else:
                fields[name] = payload.decode("utf-8", errors="replace")
        if not file_bytes:
            raise RuntimeError("missing audio file")
        return fields, file_bytes, filename

    def _json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wheatley-stt-server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", default="faster_whisper")
    parser.add_argument("--default-model", default="small.en")
    parser.add_argument("--model", action="append", default=[], help="language=model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args(argv)

    router = STTModelRouter(
        backend=args.backend,
        default_model=args.default_model,
        model_map=_parse_model_map(args.model),
        device=args.device,
        compute_type=args.compute_type,
    )
    serve(args.host, args.port, router)
    return 0


def _parse_model_map(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--model must use language=model format: {item}")
        language, model = item.split("=", 1)
        result[language.strip()] = model.strip()
    return result


def _blank_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


if __name__ == "__main__":
    raise SystemExit(main())
