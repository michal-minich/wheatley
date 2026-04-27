import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from wheatley.config import Config
from wheatley.tts.backends import PiperTTS, _add_leading_silence


class TTSBackendTests(unittest.TestCase):
    def test_add_leading_silence_prepends_audio_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            target = Path(tmp) / "target.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(1000)
                handle.writeframes(b"\x01\x02" * 10)

            result = _add_leading_silence(source, target, 100)

            self.assertEqual(result, target)
            with wave.open(str(target), "rb") as handle:
                self.assertEqual(handle.getnframes(), 110)
                self.assertEqual(handle.readframes(100), b"\x00\x00" * 100)

    def test_piper_binary_python_uses_module_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.tts.enabled = True
            cfg.tts.playback = False
            cfg.tts.output_dir = tmp
            cfg.tts.piper_binary = ".venv/bin/python"
            backend = PiperTTS(cfg)

            captured: dict[str, list[str]] = {}

            def fake_run(command, **kwargs):
                del kwargs
                captured["command"] = command

                class Completed:
                    returncode = 0
                    stderr = ""

                return Completed()

            expected_output = Path(tmp) / "out.wav"
            with mock.patch("wheatley.tts.backends.subprocess.run", side_effect=fake_run):
                with mock.patch(
                    "wheatley.tts.backends._postprocess_audio",
                    return_value=expected_output,
                ):
                    prepared = backend.prepare_for_playback("hello")

            self.assertEqual(
                captured["command"][:3],
                [".venv/bin/python", "-m", "piper"],
            )
            self.assertEqual(prepared.audio_path, expected_output)


if __name__ == "__main__":
    unittest.main()
