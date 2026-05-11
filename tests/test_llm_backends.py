import unittest
import tempfile
from pathlib import Path

from wheatley.llm.backends import (
    _filter_reasoning_stream,
    _ollama_messages,
    _openai_endpoint_url,
    _openai_messages,
    _strip_reasoning,
    model_supports_images,
)
from wheatley.llm.base import LLMImage, LLMMessage


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

    def test_model_supports_images_uses_model_name_hints(self):
        self.assertTrue(model_supports_images("lmstudio-community/gemma-4-31b-it"))
        self.assertTrue(model_supports_images("qwen2.5-vl:7b"))
        self.assertFalse(model_supports_images("qwen3.5:4b"))
        self.assertFalse(model_supports_images("qwen3.6-35b-a3b-ud-mlx"))

    def test_ollama_messages_attach_base64_images_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "photo.jpg"
            image.write_bytes(b"abc")
            messages = [
                LLMMessage(
                    "user",
                    "what is this?",
                    images=[LLMImage(path=str(image), mime_type="image/jpeg")],
                )
            ]

            self.assertEqual(
                _ollama_messages(messages, include_images=True)[0]["images"],
                ["YWJj"],
            )
            self.assertNotIn("images", _ollama_messages(messages, include_images=False)[0])

    def test_openai_messages_attach_data_url_images_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "photo.jpg"
            image.write_bytes(b"abc")
            messages = [
                LLMMessage(
                    "user",
                    "what is this?",
                    images=[LLMImage(path=str(image), mime_type="image/jpeg")],
                )
            ]

            content = _openai_messages(messages, include_images=True)[0]["content"]

            self.assertEqual(content[0], {"type": "text", "text": "what is this?"})
            self.assertEqual(content[1]["type"], "image_url")
            self.assertEqual(
                content[1]["image_url"]["url"],
                "data:image/jpeg;base64,YWJj",
            )
            self.assertEqual(
                _openai_messages(messages, include_images=False)[0]["content"],
                "what is this?",
            )


if __name__ == "__main__":
    unittest.main()
