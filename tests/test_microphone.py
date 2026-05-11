import unittest

from wheatley.config import AudioConfig
from wheatley.audio.devices import (
    _find_input_device_by_name,
    input_stream_device_kwargs,
    normalize_sounddevice_devices,
)
from wheatley.stt.microphone import (
    _append_pre_roll_frame,
    _endpoint_timing_reached,
    _trim_trailing_silence,
    _trailing_silence_keep_samples,
)


class MicrophoneTests(unittest.TestCase):
    def test_configured_input_device_index_takes_priority(self):
        class SoundDevice:
            @staticmethod
            def query_devices():
                return []

        cfg = AudioConfig(input_device_name="Headset", input_device_index=7)

        self.assertEqual(input_stream_device_kwargs(cfg, SoundDevice), {"device": 7})

    def test_configured_input_device_name_matches_input_device(self):
        devices = [
            {
                "name": "Built-in Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "name": "Wireless Headset",
                "max_input_channels": 1,
                "max_output_channels": 2,
            },
        ]

        self.assertEqual(_find_input_device_by_name("wireless", devices), 1)

    def test_configured_input_device_name_ignores_output_only_devices(self):
        devices = [
            {
                "name": "Wireless Headset",
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
            {
                "name": "Wireless Headset Hands-Free",
                "max_input_channels": 1,
                "max_output_channels": 1,
            },
        ]

        self.assertEqual(_find_input_device_by_name("Wireless Headset", devices), 1)

    def test_auto_input_device_prefers_configured_names_then_falls_back(self):
        class SoundDevice:
            @staticmethod
            def query_devices():
                return [
                    {
                        "name": "Built-in Microphone",
                        "max_input_channels": 1,
                        "max_output_channels": 0,
                    },
                    {
                        "name": "USB Headset",
                        "max_input_channels": 1,
                        "max_output_channels": 2,
                    },
                    {
                        "name": "Wireless Headset Hands-Free",
                        "max_input_channels": 1,
                        "max_output_channels": 1,
                    },
                ]

        cfg = AudioConfig(
            input_device_mode="auto",
            input_device_preferred_names=["Wireless"],
        )

        self.assertEqual(input_stream_device_kwargs(cfg, SoundDevice), {"device": 2})

    def test_auto_input_device_uses_default_when_no_headset_is_available(self):
        class SoundDevice:
            @staticmethod
            def query_devices():
                return [
                    {
                        "name": "Built-in Microphone",
                        "max_input_channels": 1,
                        "max_output_channels": 0,
                    }
                ]

        cfg = AudioConfig(input_device_mode="auto")

        self.assertEqual(input_stream_device_kwargs(cfg, SoundDevice), {})

    def test_audio_device_normalization_includes_input_and_output_counts(self):
        devices = normalize_sounddevice_devices(
            [
                {
                    "name": "Mic",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                    "default_samplerate": 16000,
                }
            ]
        )

        self.assertEqual(
            devices,
            [
                {
                    "index": 0,
                    "name": "Mic",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                    "default_samplerate": 16000.0,
                }
            ],
        )

    def test_pre_roll_keeps_recent_audio_before_voice_start(self):
        frames = []
        sample_count = 0
        for index in range(5):
            frames, sample_count = _append_pre_roll_frame(
                frames,
                sample_count,
                [index],
                sample_limit=3,
            )

        self.assertEqual(frames, [[2], [3], [4]])
        self.assertEqual(sample_count, 3)

    def test_pre_roll_disabled_keeps_no_audio(self):
        frames, sample_count = _append_pre_roll_frame(
            [[1]],
            1,
            [2],
            sample_limit=0,
        )

        self.assertEqual(frames, [])
        self.assertEqual(sample_count, 0)

    def test_trim_trailing_silence_keeps_only_configured_tail(self):
        cfg = AudioConfig(sample_rate=10, trailing_silence_keep_seconds=0.3)
        frames = [[index] for index in range(8)]

        trimmed = _trim_trailing_silence(frames, last_voice_frame_count=2, cfg=cfg)

        self.assertEqual(trimmed, [[0], [1], [2], [3], [4]])

    def test_trim_trailing_silence_keeps_endpoint_tail(self):
        cfg = AudioConfig(
            sample_rate=10,
            silence_seconds=0.6,
            trailing_silence_keep_seconds=0.3,
        )
        frames = [[index] for index in range(12)]

        trimmed = _trim_trailing_silence(frames, last_voice_frame_count=2, cfg=cfg)

        self.assertEqual(trimmed, [[0], [1], [2], [3], [4], [5], [6], [7]])

    def test_trailing_silence_keep_samples_caps_long_endpoint_tail(self):
        cfg = AudioConfig(
            sample_rate=10,
            silence_seconds=7.0,
            trailing_silence_keep_seconds=0.3,
        )

        self.assertEqual(_trailing_silence_keep_samples(cfg), 20)

    def test_endpoint_silence_uses_audio_samples_not_wall_clock(self):
        cfg = AudioConfig(
            sample_rate=16000,
            min_speech_seconds=0.45,
            silence_seconds=1.0,
            max_utterance_seconds=60.0,
        )

        enough_speech, enough_silence, too_long = _endpoint_timing_reached(
            cfg,
            audio_position_samples=16000 + 1024,
            speech_started_sample_count=0,
            last_voice_sample_count=16000,
            has_voice=False,
        )

        self.assertTrue(enough_speech)
        self.assertFalse(enough_silence)
        self.assertFalse(too_long)

    def test_endpoint_silence_triggers_after_configured_audio_duration(self):
        cfg = AudioConfig(
            sample_rate=16000,
            min_speech_seconds=0.45,
            silence_seconds=1.0,
            max_utterance_seconds=60.0,
        )

        enough_speech, enough_silence, too_long = _endpoint_timing_reached(
            cfg,
            audio_position_samples=32000,
            speech_started_sample_count=0,
            last_voice_sample_count=16000,
            has_voice=False,
        )

        self.assertTrue(enough_speech)
        self.assertTrue(enough_silence)
        self.assertFalse(too_long)

    def test_max_utterance_does_not_cut_off_active_voice(self):
        cfg = AudioConfig(
            sample_rate=10,
            min_speech_seconds=0.2,
            silence_seconds=1.0,
            max_utterance_seconds=2.0,
        )

        enough_speech, enough_silence, too_long = _endpoint_timing_reached(
            cfg,
            audio_position_samples=25,
            speech_started_sample_count=0,
            last_voice_sample_count=25,
            has_voice=True,
        )

        self.assertTrue(enough_speech)
        self.assertFalse(enough_silence)
        self.assertFalse(too_long)

    def test_max_utterance_finishes_after_voice_stops(self):
        cfg = AudioConfig(
            sample_rate=10,
            min_speech_seconds=0.2,
            silence_seconds=1.0,
            max_utterance_seconds=2.0,
        )

        enough_speech, enough_silence, too_long = _endpoint_timing_reached(
            cfg,
            audio_position_samples=30,
            speech_started_sample_count=0,
            last_voice_sample_count=20,
            has_voice=False,
        )

        self.assertTrue(enough_speech)
        self.assertTrue(enough_silence)
        self.assertTrue(too_long)

    def test_max_utterance_waits_through_short_pause(self):
        cfg = AudioConfig(
            sample_rate=10,
            min_speech_seconds=0.2,
            silence_seconds=1.0,
            max_utterance_seconds=2.0,
        )

        enough_speech, enough_silence, too_long = _endpoint_timing_reached(
            cfg,
            audio_position_samples=25,
            speech_started_sample_count=0,
            last_voice_sample_count=20,
            has_voice=False,
        )

        self.assertTrue(enough_speech)
        self.assertFalse(enough_silence)
        self.assertFalse(too_long)


if __name__ == "__main__":
    unittest.main()
