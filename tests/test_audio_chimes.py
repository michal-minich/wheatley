import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from wheatley.audio.chimes import ensure_listening_chime, play_listening_chime
from wheatley.config import AudioConfig


class AudioChimeTests(unittest.TestCase):
    def test_listening_chimes_are_generated_as_wav_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig(utterance_dir=tmp)

            start_path = ensure_listening_chime("start", cfg)
            stop_path = ensure_listening_chime("stop", cfg)

            self.assertEqual(start_path.parent, Path(tmp) / "chimes")
            self.assertNotEqual(start_path.read_bytes(), stop_path.read_bytes())
            with wave.open(str(start_path), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), 44100)
                self.assertGreater(handle.getnframes(), 1000)

    def test_disabled_chime_does_not_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig(utterance_dir=tmp, listening_chimes_enabled=False)
            with patch("wheatley.audio.chimes.play_audio") as play:
                play_listening_chime("start", cfg)

            play.assert_not_called()


if __name__ == "__main__":
    unittest.main()
