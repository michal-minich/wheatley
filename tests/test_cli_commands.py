import unittest

from wheatly.cli import _is_exit_command, _is_new_chat_command


class CliCommandTests(unittest.TestCase):
    def test_exit_command_ignores_case_and_punctuation(self):
        self.assertTrue(_is_exit_command("Stop."))
        self.assertTrue(_is_exit_command("STOP!"))

    def test_new_chat_command_ignores_punctuation(self):
        self.assertTrue(_is_new_chat_command("Start a new chat."))
        self.assertTrue(_is_new_chat_command("new chat!"))


if __name__ == "__main__":
    unittest.main()
