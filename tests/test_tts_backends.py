import tempfile
import unittest
import wave
from pathlib import Path

from wheatly.tts.backends import _add_leading_silence


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


if __name__ == "__main__":
    unittest.main()
