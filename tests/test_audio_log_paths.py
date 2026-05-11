import unittest
from datetime import datetime
from pathlib import Path

from wheatley.audio.log_paths import dated_audio_path


class AudioLogPathTests(unittest.TestCase):
    def test_dated_audio_path_uses_zero_padded_date_and_sortable_time(self):
        timestamp = datetime(2026, 4, 1, 9, 8, 7, 123456).astimezone()
        path = dated_audio_path(
            Path("audio"),
            "user",
            timestamp_ns=int(timestamp.timestamp() * 1_000_000_000),
        )

        self.assertEqual(
            path,
            Path("audio/2026/04/01/09-08-07-123456_user.wav"),
        )

    def test_dated_audio_path_can_place_runtime_subfolders(self):
        timestamp = datetime(2026, 4, 1, 9, 8, 7, 123456).astimezone()
        path = dated_audio_path(
            Path("audio"),
            "interrupt",
            timestamp_ns=int(timestamp.timestamp() * 1_000_000_000),
            subdir="interrupts",
        )

        self.assertEqual(
            path,
            Path("audio/2026/04/01/interrupts/09-08-07-123456_interrupt.wav"),
        )


if __name__ == "__main__":
    unittest.main()
