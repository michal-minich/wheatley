import unittest
import threading

from wheatly.tts.base import SpeechResult, TTSBackend
from wheatly.tts.streaming import StreamingSpeaker


class RecordingTTS(TTSBackend):
    def __init__(self):
        self.spoken = []

    def speak(self, text: str) -> SpeechResult:
        self.spoken.append(text)
        return SpeechResult(audio_path=None, spoken=True)


class StreamingSpeakerTests(unittest.TestCase):
    def test_initial_wait_does_not_emit_tiny_first_sentence(self):
        tts = RecordingTTS()
        with StreamingSpeaker(
            tts,
            enabled=True,
            min_words=24,
            max_words=60,
            initial_min_words=24,
            feedback_min_words=8,
            max_initial_wait_seconds=0.0,
        ) as speaker:
            speaker.feed("Sure. I can tell you a longer story after that.")

        self.assertEqual(tts.spoken[0], "Sure. I can tell you a longer story")

    def test_initial_wait_emits_feedback_chunk_without_sentence(self):
        tts = RecordingTTS()
        with StreamingSpeaker(
            tts,
            enabled=True,
            min_words=24,
            max_words=60,
            initial_min_words=24,
            feedback_min_words=4,
            max_initial_wait_seconds=0.0,
        ) as speaker:
            speaker.feed("one two three four five six")

        self.assertEqual(tts.spoken[0], "one two three four")

    def test_stop_event_prevents_queued_speech(self):
        tts = RecordingTTS()
        stop_event = threading.Event()
        stop_event.set()
        with StreamingSpeaker(
            tts,
            enabled=True,
            min_words=1,
            max_words=2,
            stop_event=stop_event,
        ) as speaker:
            speaker.feed("one two three four")

        self.assertEqual(tts.spoken, [])


if __name__ == "__main__":
    unittest.main()
