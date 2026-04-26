import os
import shutil
import unittest
from unittest.mock import patch

from wheatly.cli import _format_preview_block, _is_exit_command, _is_new_chat_command


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


if __name__ == "__main__":
    unittest.main()
