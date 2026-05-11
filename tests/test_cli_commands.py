import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from wheatley.cli import (
    _PartialTranscriptPreview,
    RecordedUtterance,
    UserTextCapture,
    _announce_model_selection,
    _can_use_partial_as_final,
    _format_preview_block,
    _handle_text_turn,
    _idle_speech_available,
    _idle_speech_instruction,
    _is_exit_command,
    _is_new_chat_command,
    _is_pause_command,
    _is_resume_command,
    _listen_for_resume_decision,
    _maybe_resume_recent_turns_on_start,
    _startup_resume_mode,
    _load_recent_turns,
    _next_idle_speech_wait_seconds,
    _print_tool_start,
    _print_streamed_turn,
    _resume_startup_decision,
    _transcribe_with_status,
    _voice_loop,
    main,
)
from wheatley.config import Config
from wheatley.pipeline import TurnResult
from wheatley.stt.base import Transcription


class CliCommandTests(unittest.TestCase):
    def _strip_ansi(self, text: str) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def test_exit_command_ignores_case_and_punctuation(self):
        self.assertTrue(_is_exit_command("Quit."))
        self.assertTrue(_is_exit_command("EXIT!"))
        self.assertFalse(_is_exit_command("stop"))

    def test_new_chat_command_ignores_punctuation(self):
        self.assertTrue(_is_new_chat_command("Start a new chat."))
        self.assertTrue(_is_new_chat_command("new chat!"))

    def test_resume_startup_decision_matches_yes_no_only(self):
        self.assertTrue(_resume_startup_decision("Resume."))
        self.assertTrue(_resume_startup_decision("yeah"))
        self.assertFalse(_resume_startup_decision(""))
        self.assertFalse(_resume_startup_decision("start over"))
        self.assertIsNone(_resume_startup_decision("maybe continue later"))

    def test_load_recent_turns_uses_last_configured_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "turns.jsonl"
            rows = [
                {
                    "timestamp": f"2026-04-28T19:00:0{index}+02:00",
                    "model_name": "echo",
                    "user_text": f"user {index}",
                    "assistant_text": f"assistant {index}",
                    "tool_results": [],
                }
                for index in range(3)
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            loaded = _load_recent_turns(str(path), 2)

        self.assertEqual([row["user_text"] for row in loaded], ["user 1", "user 2"])

    def test_load_recent_turns_skips_idle_speech_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "turns.jsonl"
            rows = [
                {
                    "timestamp": "2026-04-28T19:00:00+02:00",
                    "source": "idle",
                    "user_text": "[idle silence]",
                    "assistant_text": "idle",
                    "tool_results": [],
                },
                {
                    "timestamp": "2026-04-28T19:00:01+02:00",
                    "user_text": "real user",
                    "assistant_text": "answer",
                    "tool_results": [],
                },
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            loaded = _load_recent_turns(str(path), 2)

        self.assertEqual([row["user_text"] for row in loaded], ["real user"])

    def test_pause_resume_commands_match_slovak_words(self):
        self.assertTrue(_is_pause_command("pockaj"))
        self.assertTrue(_is_pause_command("Počkaj!"))
        self.assertTrue(_is_resume_command("pokracuj"))
        self.assertTrue(_is_resume_command("Pokračuj."))

    def test_pause_resume_commands_do_not_match_other_words(self):
        self.assertFalse(_is_pause_command("pause now please"))
        self.assertFalse(_is_resume_command("resume later please"))

    def test_partial_transcript_preview_wraps_to_multiple_lines(self):
        size = os.terminal_size((24, 20))
        with patch.object(shutil, "get_terminal_size", return_value=size):
            lines = _format_preview_block(
                "you~",
                "yellow",
                "this partial transcript should wrap across several lines",
            )
        self.assertGreater(len(lines), 1)
        self.assertIn("you~> ", lines[0])
        self.assertTrue(lines[1].startswith("      "))

    def test_partial_transcript_is_used_as_final_only_when_fresh(self):
        cfg = Config()
        cfg.audio.partial_transcript_use_as_final = True
        cfg.audio.partial_transcript_final_max_age_seconds = 6.0
        fresh = RecordedUtterance(
            path=__file__,
            partial_text="hello from partial",
            partial_age_seconds=4.0,
        )
        stale = RecordedUtterance(
            path=__file__,
            partial_text="old partial",
            partial_age_seconds=7.0,
        )

        self.assertTrue(_can_use_partial_as_final(fresh, cfg))
        self.assertFalse(_can_use_partial_as_final(stale, cfg))

    def test_speaking_turn_uses_streaming_even_without_print_stream(self):
        class Agent:
            def __init__(self):
                self.cfg = Config()
                self.cfg.tts.stream_speech = True
                self.streamed = False
                self.nonstreamed = False

            def handle_text_stream(self, text, speak=True, on_token=None):
                del text, speak, on_token
                self.streamed = True
                return TurnResult("hello", "streamed", [])

            def handle_text(self, text, speak=True):
                del text, speak
                self.nonstreamed = True
                return TurnResult("hello", "nonstreamed", [])

        agent = Agent()

        with patch.object(sys, "stdout", StringIO()):
            _handle_text_turn(agent, "hello", speak=True, stream=False)

        self.assertTrue(agent.streamed)
        self.assertFalse(agent.nonstreamed)

    def test_streamed_turn_prints_prefix_before_first_token(self):
        out = StringIO()

        class Agent:
            def __init__(self):
                self.before_tokens = ""

            def handle_text_stream(self, text, speak=True, on_token=None):
                del text, speak, on_token
                self.before_tokens = out.getvalue()
                return TurnResult("hello", "final answer", [])

        agent = Agent()
        with patch.object(sys, "stdout", out):
            _print_streamed_turn(agent, "hello", speak=False, print_tokens=True)

        before_plain = self._strip_ansi(agent.before_tokens)
        rendered_plain = self._strip_ansi(out.getvalue())
        self.assertEqual(before_plain, "wheatley> ")
        self.assertEqual(rendered_plain, "wheatley> final answer\n")

    def test_streamed_turn_uses_profile_name_prefix(self):
        out = StringIO()

        class Agent:
            def __init__(self):
                self.cfg = Config()
                self.cfg.profile_dir = "profiles/demo"

            def handle_text_stream(self, text, speak=True, on_token=None):
                del text, speak, on_token
                return TurnResult("hello", "final answer", [])

        agent = Agent()
        with patch.object(sys, "stdout", out):
            _handle_text_turn(agent, "hello", speak=False, stream=True)

        rendered_plain = self._strip_ansi(out.getvalue())
        self.assertEqual(rendered_plain, "demo> final answer\n")

    def test_tool_start_prints_message_without_prefix(self):
        out = StringIO()

        with patch.object(sys, "stdout", out):
            _print_tool_start("web_search", "Searching...")

        rendered = out.getvalue()
        self.assertEqual(self._strip_ansi(rendered), "Searching...\n")
        self.assertNotIn("tool> ", rendered)
        self.assertIn("\033[36m", rendered)

    def test_tool_start_prints_audit_input_for_selected_tools(self):
        cases = [
            (
                "calculator",
                {"expression": "2 + 2"},
                "Calculating...\n2 + 2\n",
            ),
            (
                "remember",
                {"memory": "User likes short answers."},
                "Remembering...\nUser likes short answers.\n",
            ),
            (
                "python_interpreter",
                {"code": "items = input['items']\nresult = len(items)"},
                "Running Python...\nitems = input['items']\nresult = len(items)\n",
            ),
            (
                "run_safe_cli_tool",
                {"command": "status", "args": ["--verbose", "two words"]},
                "Running...\nstatus --verbose 'two words'\n",
            ),
        ]
        for tool_name, arguments, expected in cases:
            with self.subTest(tool_name=tool_name):
                out = StringIO()
                with patch.object(sys, "stdout", out):
                    _print_tool_start(tool_name, expected.splitlines()[0], arguments=arguments)
                rendered = out.getvalue()
                self.assertEqual(self._strip_ansi(rendered), expected)
                self.assertIn("\033[38;5;130m", rendered)

    def test_model_selection_prints_tools_when_memory_has_no_notice(self):
        class Selection:
            message = "using test model."

        class TTS:
            def __init__(self):
                self.spoken = []

            def speak(self, text):
                self.spoken.append(text)

        class Agent:
            def __init__(self):
                self.cfg = Config()
                self.cfg.tts.enabled = True
                self.tts = TTS()

            def reset_chat(self, refresh_memory=False):
                del refresh_memory
                return Selection()

            def refresh_auto_memory(
                self,
                notify_memory=None,
                speak_memory=False,
                start_messages=None,
            ):
                del notify_memory, speak_memory, start_messages

            def current_tools_message(self):
                return "Current tools are: calculator and memory."

        agent = Agent()
        out = StringIO()

        with patch.object(sys, "stdout", out):
            _announce_model_selection(agent, speak=True)

        plain = self._strip_ansi(out.getvalue())
        self.assertIn("system> using test model.\n", plain)
        self.assertIn("system> Current tools are: calculator and memory.\n", plain)
        self.assertEqual(
            agent.tts.spoken,
            ["using test model.", "Current tools are: calculator and memory."],
        )

    def test_model_selection_does_not_duplicate_tools_from_memory_notice(self):
        class Selection:
            message = "using test model."

        class Agent:
            def __init__(self):
                self.cfg = Config()
                self.tts = object()

            def reset_chat(self, refresh_memory=False):
                del refresh_memory
                return Selection()

            def refresh_auto_memory(
                self,
                notify_memory=None,
                speak_memory=False,
                start_messages=None,
            ):
                del speak_memory, start_messages
                notify_memory("wait, I'm updating my memory...")
                notify_memory("Current tools are: calculator and memory.")

            def current_tools_message(self):
                return "Current tools are: calculator and memory."

        out = StringIO()

        with patch.object(sys, "stdout", out):
            _announce_model_selection(Agent(), speak=False)

        plain = self._strip_ansi(out.getvalue())
        self.assertLess(
            plain.index("wait, I'm updating my memory..."),
            plain.index("using test model."),
        )
        self.assertLess(
            plain.index("using test model."),
            plain.index("Current tools are:"),
        )
        self.assertEqual(plain.count("using test model."), 1)
        self.assertEqual(plain.count("Current tools are:"), 1)

    def test_resume_prompt_prints_before_speaking(self):
        class Recorder:
            def __init__(self, audio):
                del audio

            def record_utterance(self, path):
                del path
                raise TimeoutError

        class TTS:
            def __init__(self, output):
                self.output = output
                self.before_speak = ""

            def speak(self, text):
                del text
                self.before_speak = self.output.getvalue()

        class Agent:
            def __init__(self, output):
                self.tts = TTS(output)

            def transcribe(self, path):
                del path
                return Transcription(text="")

        cfg = Config()
        cfg.tts.enabled = True
        cfg.chat.resume_countdown_seconds = 1
        out = StringIO()
        agent = Agent(out)

        with patch.object(sys, "stdout", out), patch(
            "wheatley.cli.MicrophoneRecorder",
            Recorder,
        ), patch(
            "wheatley.cli.dated_audio_path",
            return_value=Path("resume.wav"),
        ):
            text = _listen_for_resume_decision(agent, cfg, 10)

        self.assertEqual(text, "")
        self.assertIn("Resume last 10 turns?", self._strip_ansi(agent.tts.before_speak))
        self.assertIn("system> Resume last 10 turns?", self._strip_ansi(agent.tts.before_speak))
        self.assertIn("Resume last 10 turns?", agent.tts.before_speak)
        self.assertNotIn("\033[36mResume last 10 turns?", agent.tts.before_speak)

    def test_voice_loop_waits_indefinitely_for_speech(self):
        captured_audio = []

        class Recorder:
            def __init__(self, audio):
                captured_audio.append(audio)

        cfg = Config()
        cfg.audio.max_wait_seconds = 30.0

        with patch("wheatley.cli.MicrophoneRecorder", Recorder), patch(
            "wheatley.cli._record_user_text_result",
            return_value=UserTextCapture(text="quit"),
        ), patch(
            "wheatley.cli.dated_audio_path",
            return_value=Path("voice.wav"),
        ), patch.object(
            sys,
            "stdout",
            StringIO(),
        ):
            result = _voice_loop(object(), cfg, speak=False, turns=0, stream=False)

        self.assertEqual(result, 0)
        self.assertEqual(captured_audio[0].max_wait_seconds, 0.0)
        self.assertEqual(cfg.audio.max_wait_seconds, 30.0)

    def test_idle_speech_wait_uses_configured_random_multiplier(self):
        cfg = Config()
        cfg.idle_speech.interval_seconds = 100.0
        cfg.idle_speech.random_min_multiplier = 1.0
        cfg.idle_speech.random_max_multiplier = 5.0

        with patch("wheatley.cli.random.uniform", return_value=3.0) as uniform:
            wait = _next_idle_speech_wait_seconds(cfg)

        self.assertEqual(wait, 300.0)
        uniform.assert_called_once_with(1.0, 5.0)

    def test_idle_speech_requires_speaking_mode(self):
        cfg = Config()
        cfg.idle_speech.enabled = True
        cfg.idle_speech.interval_seconds = 100.0

        self.assertTrue(_idle_speech_available(cfg, speak=True))
        self.assertFalse(_idle_speech_available(cfg, speak=False))

    def test_voice_loop_uses_idle_timeout_when_enabled(self):
        captured_audio = []

        class Recorder:
            def __init__(self, audio):
                captured_audio.append(audio)

        cfg = Config()
        cfg.idle_speech.enabled = True
        cfg.idle_speech.interval_seconds = 100.0
        cfg.idle_speech.random_min_multiplier = 1.0
        cfg.idle_speech.random_max_multiplier = 5.0

        with patch("wheatley.cli.random.uniform", return_value=2.0), patch(
            "wheatley.cli.MicrophoneRecorder",
            Recorder,
        ), patch(
            "wheatley.cli._record_user_text_result",
            return_value=UserTextCapture(text="quit"),
        ), patch(
            "wheatley.cli.dated_audio_path",
            return_value=Path("voice.wav"),
        ), patch.object(
            sys,
            "stdout",
            StringIO(),
        ):
            result = _voice_loop(object(), cfg, speak=True, turns=0, stream=False)

        self.assertEqual(result, 0)
        self.assertAlmostEqual(captured_audio[0].max_wait_seconds, 200.0, places=3)

    def test_voice_loop_speaks_idle_only_after_silent_timeout(self):
        class Agent:
            def __init__(self, cfg):
                self.cfg = cfg
                self.idle_prompts = []

            def handle_idle_speech(self, instruction, speak=True, on_token=None):
                del speak, on_token
                self.idle_prompts.append(instruction)
                return TurnResult("[idle silence]", "Tiny idle remark.", [])

        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp)
            (profile / "idle.md").write_text(
                "Talk about recent topics.",
                encoding="utf-8",
            )
            cfg = Config()
            cfg.profile_dir = str(profile)
            cfg.idle_speech.enabled = True
            cfg.idle_speech.interval_seconds = 100.0
            agent = Agent(cfg)

            with patch("wheatley.cli.random.uniform", return_value=1.0), patch(
                "wheatley.cli.MicrophoneRecorder",
                lambda audio: object(),
            ), patch(
                "wheatley.cli._record_user_text_result",
                side_effect=[
                    UserTextCapture(timed_out=True),
                    UserTextCapture(text="quit"),
                ],
            ), patch(
                "wheatley.cli.dated_audio_path",
                return_value=Path("voice.wav"),
            ), patch.object(
                sys,
                "stdout",
                StringIO(),
            ):
                result = _voice_loop(agent, cfg, speak=True, turns=0, stream=False)

        self.assertEqual(result, 0)
        self.assertEqual(agent.idle_prompts, ["Talk about recent topics."])

    def test_idle_speech_instruction_reads_profile_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp)
            (profile / "idle.md").write_text("Idle markdown prompt.", encoding="utf-8")
            cfg = Config()
            cfg.profile_dir = str(profile)

            self.assertEqual(_idle_speech_instruction(cfg), "Idle markdown prompt.")

    def test_voice_loop_does_not_idle_when_timeout_has_preview_text(self):
        class Agent:
            def __init__(self, cfg):
                self.cfg = cfg
                self.idle_prompts = []

            def handle_idle_speech(self, instruction, speak=True, on_token=None):
                del speak, on_token
                self.idle_prompts.append(instruction)
                return TurnResult("[idle silence]", "Tiny idle remark.", [])

        cfg = Config()
        cfg.idle_speech.enabled = True
        cfg.idle_speech.interval_seconds = 100.0
        agent = Agent(cfg)

        with patch("wheatley.cli.random.uniform", return_value=1.0), patch(
            "wheatley.cli.MicrophoneRecorder",
            lambda audio: object(),
        ), patch(
            "wheatley.cli._record_user_text_result",
            side_effect=[
                UserTextCapture(timed_out=True, partial_text="partial text"),
                UserTextCapture(text="quit"),
            ],
        ), patch(
            "wheatley.cli.dated_audio_path",
            return_value=Path("voice.wav"),
        ), patch.object(
            sys,
            "stdout",
            StringIO(),
        ):
            result = _voice_loop(agent, cfg, speak=True, turns=0, stream=False)

        self.assertEqual(result, 0)
        self.assertEqual(agent.idle_prompts, [])

    def test_voice_loop_empty_no_text_capture_does_not_reset_idle_timer(self):
        class Agent:
            def __init__(self, cfg):
                self.cfg = cfg
                self.idle_prompts = []

            def handle_idle_speech(self, instruction, speak=True, on_token=None):
                del speak, on_token
                self.idle_prompts.append(instruction)
                return TurnResult("[idle silence]", "Tiny idle remark.", [])

        cfg = Config()
        cfg.idle_speech.enabled = True
        cfg.idle_speech.interval_seconds = 10.0
        agent = Agent(cfg)

        with patch("wheatley.cli.random.uniform", return_value=1.0), patch(
            "wheatley.cli.time.monotonic",
            side_effect=[0.0, 0.0, 11.0, 11.0, 12.0],
        ), patch(
            "wheatley.cli.MicrophoneRecorder",
            lambda audio: object(),
        ), patch(
            "wheatley.cli._record_user_text_result",
            side_effect=[
                UserTextCapture(text=""),
                UserTextCapture(text="quit"),
            ],
        ), patch(
            "wheatley.cli.dated_audio_path",
            return_value=Path("voice.wav"),
        ), patch.object(
            sys,
            "stdout",
            StringIO(),
        ):
            result = _voice_loop(agent, cfg, speak=True, turns=0, stream=False)

        self.assertEqual(result, 0)
        self.assertEqual(len(agent.idle_prompts), 1)

    def test_voice_loop_nonempty_text_is_user_turn_even_when_idle_due(self):
        class Agent:
            def __init__(self, cfg):
                self.cfg = cfg
                self.idle_prompts = []
                self.text_turns = []

            def handle_idle_speech(self, instruction, speak=True, on_token=None):
                del speak, on_token
                self.idle_prompts.append(instruction)
                return TurnResult("[idle silence]", "Tiny idle remark.", [])

            def handle_text(self, text, speak=True):
                del speak
                self.text_turns.append(text)
                return TurnResult(text, "Handled.", [])

        cfg = Config()
        cfg.idle_speech.enabled = True
        cfg.idle_speech.interval_seconds = 10.0
        cfg.tts.stream_speech = False
        agent = Agent(cfg)

        with patch("wheatley.cli.random.uniform", return_value=1.0), patch(
            "wheatley.cli.time.monotonic",
            side_effect=[0.0, 0.0, 11.0, 11.0],
        ), patch(
            "wheatley.cli.MicrophoneRecorder",
            lambda audio: object(),
        ), patch(
            "wheatley.cli._record_user_text_result",
            return_value=UserTextCapture(text="I don't know."),
        ), patch(
            "wheatley.cli.dated_audio_path",
            return_value=Path("voice.wav"),
        ), patch.object(sys, "stdout", StringIO()):
            result = _voice_loop(agent, cfg, speak=True, turns=1, stream=False)

        self.assertEqual(result, 0)
        self.assertEqual(agent.idle_prompts, [])
        self.assertEqual(agent.text_turns, ["I don't know."])

    def test_partial_preview_finish_keeps_text_and_adds_newline(self):
        preview = _PartialTranscriptPreview(lambda _: "partial")
        out = StringIO()
        with patch.object(sys, "stdout", out):
            preview.update("temporary transcript")
            preview.finish()

        rendered = out.getvalue()
        plain = rendered.replace("\033[33m", "").replace("\033[0m", "")
        self.assertIn("you> temporary transcript", plain)
        self.assertTrue(rendered.endswith("\n"))

    def test_transcribe_with_status_has_no_transcribing_log(self):
        class Agent:
            def transcribe_final(self, path):
                del path
                return Transcription(text="hi", language="en")

        cfg = Config()
        recorded = RecordedUtterance(path=Path(__file__))
        out = StringIO()
        with patch.object(sys, "stdout", out):
            result = _transcribe_with_status(Agent(), recorded, cfg)

        self.assertEqual(result.text, "hi")
        self.assertEqual(out.getvalue(), "")

    def test_main_loads_requested_profile(self):
        with patch("wheatley.cli.load_config") as load, patch(
            "wheatley.cli.diagnostics_json",
            return_value="{}",
        ), patch.object(sys, "stdout", StringIO()):
            self.assertEqual(main(["--profile", "test", "doctor"]), 0)

        load.assert_called_once_with(path=None, profile="test")

    def test_resume_mode_yes_restores_without_prompt(self):
        cfg = Config()
        cfg.chat.resume_on_start_mode = "yes"
        cfg.chat.resume_turns = 2
        turns = [{"user_text": "u", "assistant_text": "a"}]

        with patch("wheatley.cli._load_recent_turns", return_value=turns), patch(
            "wheatley.cli._ask_resume_recent_turns",
            side_effect=AssertionError("should not prompt in yes mode"),
        ):
            restored = _maybe_resume_recent_turns_on_start(object(), cfg)

        self.assertEqual(restored, turns)

    def test_resume_mode_no_skips_restore(self):
        cfg = Config()
        cfg.chat.resume_on_start_mode = "no"
        cfg.chat.resume_turns = 2

        with patch("wheatley.cli._load_recent_turns") as load_turns:
            restored = _maybe_resume_recent_turns_on_start(object(), cfg)

        self.assertEqual(restored, [])
        load_turns.assert_not_called()

    def test_resume_mode_auto_alias_maps_to_ask(self):
        cfg = Config()
        cfg.chat.resume_on_start_mode = "auto"
        self.assertEqual(_startup_resume_mode(cfg), "ask")


if __name__ == "__main__":
    unittest.main()
