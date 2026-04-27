import unittest
from pathlib import Path
from unittest.mock import patch

from wheatley.audio.interrupt import SpeechInterruptMonitor, is_stop_interrupt
from wheatley.config import AudioConfig
from wheatley.stt.base import Transcription


class AudioInterruptTests(unittest.TestCase):
    def test_stop_interrupt_matches_single_stop_word(self):
        self.assertTrue(is_stop_interrupt("Stop."))
        self.assertTrue(is_stop_interrupt("stóp"))

    def test_stop_interrupt_rejects_long_transcripts(self):
        self.assertFalse(is_stop_interrupt("please stop talking now because this is long"))

    def test_stop_interrupt_rejects_other_words(self):
        self.assertFalse(is_stop_interrupt("keep going"))

    def test_false_candidate_does_not_stop_playback(self):
        cfg = AudioConfig()
        cfg.speech_interrupt_record_seconds = 0.0
        monitor = SpeechInterruptMonitor(
            cfg,
            transcribe=lambda path: Transcription(
                text="assistant speech", language="en", duration_seconds=None
            ),
            interrupt_event=__import__("threading").Event(),
        )
        frames = [_FakeBlock(1024)]
        audio_queue = __import__("queue").Queue()

        with patch("wheatley.audio.interrupt.stop_audio_playback") as stop:
            monitor._write_candidate = lambda np, frames: Path("candidate.wav")
            monitor._verify_candidate(frames, audio_queue, _FakeNumpy())

        stop.assert_not_called()


class _FakeBlock:
    def __init__(self, length: int):
        self._length = length

    def __len__(self):
        return self._length


class _FakeNumpy:
    def concatenate(self, frames, axis=0):
        del frames, axis
        return b""


if __name__ == "__main__":
    unittest.main()
