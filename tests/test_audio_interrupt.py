import unittest

from wheatly.audio.interrupt import is_stop_interrupt


class AudioInterruptTests(unittest.TestCase):
    def test_stop_interrupt_matches_single_stop_word(self):
        self.assertTrue(is_stop_interrupt("Stop."))
        self.assertTrue(is_stop_interrupt("stóp"))

    def test_stop_interrupt_rejects_long_transcripts(self):
        self.assertFalse(is_stop_interrupt("please stop talking now because this is long"))

    def test_stop_interrupt_rejects_other_words(self):
        self.assertFalse(is_stop_interrupt("keep going"))


if __name__ == "__main__":
    unittest.main()
