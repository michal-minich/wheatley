import unittest

from wheatly.llm.backends import _filter_reasoning_stream, _openai_endpoint_url, _strip_reasoning


class LLMBackendTests(unittest.TestCase):
    def test_openai_endpoint_accepts_base_with_v1(self):
        self.assertEqual(
            _openai_endpoint_url("http://host:1234/v1", "models"),
            "http://host:1234/v1/models",
        )

    def test_openai_endpoint_adds_v1_when_missing(self):
        self.assertEqual(
            _openai_endpoint_url("http://host:1234", "chat/completions"),
            "http://host:1234/v1/chat/completions",
        )

    def test_strip_reasoning_removes_think_tail(self):
        self.assertEqual(
            _strip_reasoning("The user wants ok.</think>\n\nOK!"),
            "OK!",
        )

    def test_stream_filter_hides_reasoning_prefix(self):
        chunks = ["The user wants ", "ok.</think>\n\n", "OK!"]
        self.assertEqual("".join(_filter_reasoning_stream(iter(chunks))), "OK!")

    def test_stream_filter_releases_normal_text_quickly(self):
        chunks = ["The r", "esult is ready."]
        self.assertEqual(
            "".join(_filter_reasoning_stream(iter(chunks))),
            "The result is ready.",
        )

    def test_stream_filter_releases_short_normal_text(self):
        chunks = ["Ok", "."]
        self.assertEqual("".join(_filter_reasoning_stream(iter(chunks))), "Ok.")


if __name__ == "__main__":
    unittest.main()
