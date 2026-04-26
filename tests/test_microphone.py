import unittest

from wheatly.config import AudioConfig
from wheatly.stt.microphone import _trim_trailing_silence


class MicrophoneTests(unittest.TestCase):
    def test_trim_trailing_silence_keeps_only_configured_tail(self):
        cfg = AudioConfig(sample_rate=10, trailing_silence_keep_seconds=0.3)
        frames = [[index] for index in range(8)]

        trimmed = _trim_trailing_silence(frames, last_voice_frame_count=2, cfg=cfg)

        self.assertEqual(trimmed, [[0], [1], [2], [3], [4]])


if __name__ == "__main__":
    unittest.main()
