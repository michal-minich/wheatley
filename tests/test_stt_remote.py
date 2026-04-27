import json
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from wheatley.config import Config, STTConfig
from wheatley.language import apply_configured_language
from wheatley.stt.backends import RemoteFallbackSTT, build_stt
from wheatley.stt.base import STTBackend, Transcription


class FixedSTT(STTBackend):
    def transcribe(self, audio_path=None):
        return Transcription(text="local fallback", language="en")


class RemoteSTTTests(unittest.TestCase):
    def test_remote_stt_posts_audio_and_parses_text(self):
        server = _start_server()
        try:
            cfg = STTConfig(
                backend="remote_fallback",
                remote_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                remote_model="small.en",
                language="en",
            )
            stt = build_stt(cfg)
            result = stt.transcribe(_write_wav())
            self.assertEqual(result.text, "remote transcript")
            self.assertEqual(result.language, "en")
        finally:
            server.shutdown()
            server.server_close()

    def test_remote_stt_uses_local_fallback_when_server_is_unreachable(self):
        cfg = STTConfig(
            backend="remote_fallback",
            remote_base_url="http://127.0.0.1:9/v1",
            remote_request_timeout_seconds=0.1,
        )
        stt = RemoteFallbackSTT(cfg)
        with patch.object(stt, "_fallback_backend", return_value=FixedSTT()):
            result = stt.transcribe(_write_wav())
        self.assertEqual(result.text, "local fallback")

    def test_language_mode_sets_remote_stt_model_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.runtime.data_dir = tmp
            cfg.runtime.state_dir = str(Path(tmp) / "state")
            cfg.language.enabled = True
            cfg.language.default = "sk"
            cfg.ensure_dirs()

            apply_configured_language(cfg, "sk")

            self.assertEqual(
                cfg.stt.model,
                "models/whisper/whisper-large-v3-turbo-sk-ct2-int8",
            )
            self.assertEqual(cfg.stt.remote_model, "models/whisper/whisper-large-v3-sk-ct2-int8")


def _start_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if b'name="file"' not in body:
                self.send_response(400)
                self.end_headers()
                return
            data = json.dumps({"text": "remote transcript", "language": "en"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _write_wav():
    path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    return path


if __name__ == "__main__":
    unittest.main()
