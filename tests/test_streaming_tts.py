import unittest
import threading
import time

from wheatley.tts.base import PreparedSpeech, SpeechResult, TTSBackend
from wheatley.tts.streaming import StreamingSpeaker


class RecordingTTS(TTSBackend):
    def __init__(self):
        self.spoken = []

    def speak(self, text: str) -> SpeechResult:
        self.spoken.append(text)
        return SpeechResult(audio_path=None, spoken=True)


class PipelinedRecordingTTS(TTSBackend):
    def __init__(self):
        self.prepare_started: dict[str, float] = {}
        self.prepare_finished: dict[str, float] = {}
        self.play_started: dict[str, float] = {}
        self.play_finished: dict[str, float] = {}
        self.played: list[str] = []

    def supports_stream_pipelining(self) -> bool:
        return True

    def prepare_for_playback(self, text: str) -> PreparedSpeech:
        self.prepare_started[text] = time.perf_counter()
        time.sleep(0.03)
        self.prepare_finished[text] = time.perf_counter()
        return PreparedSpeech(text=text, audio_path=None)

    def play_prepared(self, prepared: PreparedSpeech) -> bool:
        self.play_started[prepared.text] = time.perf_counter()
        if not self.played:
            time.sleep(0.18)
        else:
            time.sleep(0.02)
        self.play_finished[prepared.text] = time.perf_counter()
        self.played.append(prepared.text)
        return True


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

    def test_inter_chunk_wait_emits_feedback_chunk_without_reaching_min_words(self):
        tts = RecordingTTS()
        with StreamingSpeaker(
            tts,
            enabled=True,
            min_words=10,
            max_words=60,
            initial_min_words=2,
            feedback_min_words=3,
            max_initial_wait_seconds=0.0,
            max_inter_chunk_wait_seconds=0.0,
        ) as speaker:
            speaker.feed("one two three four five six")

        self.assertEqual(tts.spoken, ["one two three", "four five six"])

    def test_timeout_prefers_complete_short_sentence_over_mid_sentence_split(self):
        tts = RecordingTTS()
        with StreamingSpeaker(
            tts,
            enabled=True,
            min_words=24,
            max_words=60,
            initial_min_words=24,
            feedback_min_words=14,
            max_initial_wait_seconds=0.0,
        ) as speaker:
            speaker.feed("One plus one equals two. Then we keep going with more words.")

        self.assertEqual(tts.spoken[0], "One plus one equals two.")

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

    def test_pipeline_prepares_next_chunk_while_current_chunk_is_playing(self):
        tts = PipelinedRecordingTTS()
        with StreamingSpeaker(
            tts,
            enabled=True,
            min_words=2,
            max_words=2,
            initial_min_words=2,
            playback_prebuffer_chunks=1,
            playback_prebuffer_max_wait_seconds=0.0,
        ) as speaker:
            speaker.feed("one two three four")

        self.assertEqual(tts.played, ["one two", "three four"])
        self.assertLess(
            tts.prepare_started["three four"],
            tts.play_finished["one two"],
        )


if __name__ == "__main__":
    unittest.main()
