import sys
import threading
import time
import unittest

from wheatly.audio.playback import (
    current_playback_age_seconds,
    run_playback_command,
    stop_audio_playback,
)


class AudioPlaybackTests(unittest.TestCase):
    def test_stop_audio_playback_terminates_current_command(self):
        results = []
        thread = threading.Thread(
            target=lambda: results.append(
                run_playback_command(
                    [sys.executable, "-c", "import time; time.sleep(5)"]
                )
            )
        )
        thread.start()
        try:
            deadline = time.monotonic() + 1.0
            while current_playback_age_seconds() is None and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertIsNotNone(current_playback_age_seconds())
            stop_audio_playback()
            thread.join(timeout=1.0)

            self.assertFalse(thread.is_alive())
            self.assertEqual(results, [False])
        finally:
            stop_audio_playback()
            thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
