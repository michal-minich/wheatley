import os
import shutil
import sys
import unittest
from io import StringIO
from unittest.mock import patch

from wheatly.cli import (
    RecordedUtterance,
    _can_use_partial_as_final,
    _format_preview_block,
    _handle_text_turn,
    _is_exit_command,
    _is_new_chat_command,
    _print_tool_start,
)
from wheatly.config import Config
from wheatly.pipeline import TurnResult


class CliCommandTests(unittest.TestCase):
    def test_exit_command_ignores_case_and_punctuation(self):
        self.assertTrue(_is_exit_command("Stop."))
        self.assertTrue(_is_exit_command("STOP!"))

    def test_new_chat_command_ignores_punctuation(self):
        self.assertTrue(_is_new_chat_command("Start a new chat."))
        self.assertTrue(_is_new_chat_command("new chat!"))

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

    def test_fresh_partial_transcript_can_be_used_as_final(self):
        cfg = Config()
        cfg.audio.partial_transcript_use_as_final = True
        cfg.audio.partial_transcript_final_max_age_seconds = 6.0
        recorded = RecordedUtterance(
            path=__file__,
            partial_text="hello from partial",
            partial_age_seconds=4.0,
        )

        self.assertTrue(_can_use_partial_as_final(recorded, cfg))

    def test_stale_partial_transcript_is_not_used_as_final(self):
        cfg = Config()
        cfg.audio.partial_transcript_final_max_age_seconds = 6.0
        recorded = RecordedUtterance(
            path=__file__,
            partial_text="old partial",
            partial_age_seconds=7.0,
        )

        self.assertFalse(_can_use_partial_as_final(recorded, cfg))

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

    def test_tool_start_print_is_colored(self):
        out = StringIO()
        with patch.object(sys, "stdout", out):
            _print_tool_start("web_search", "Hľadám...")

        rendered = out.getvalue()
        self.assertIn("\033[36m", rendered)
        self.assertIn("tool> Hľadám...", rendered)


if __name__ == "__main__":
    unittest.main()
