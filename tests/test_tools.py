import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wheatley.config import Config
from wheatley.tools.announcements import tool_start_message
from wheatley.tools.builtins import build_registry
from wheatley.tools.registry import ToolCall


class ToolTests(unittest.TestCase):
    def test_set_eye_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.runtime.state_dir = str(Path(tmp) / "state")
            cfg.ensure_dirs()
            registry = build_registry(cfg)
            result = registry.execute(
                ToolCall("set_eye_expression", {"expression": "thinking"})
            )
            self.assertTrue(result.ok)
            self.assertTrue((Path(cfg.runtime.state_dir) / "eye.json").exists())

    def test_unknown_tool(self):
        cfg = Config()
        registry = build_registry(cfg)
        result = registry.execute(ToolCall("missing", {}))
        self.assertFalse(result.ok)

    def test_calculator(self):
        cfg = Config()
        registry = build_registry(cfg)
        result = registry.execute(
            ToolCall("calculator", {"expression": "sqrt(5+5)+sin(4)**3+6/7"})
        )
        self.assertTrue(result.ok)
        self.assertIn("result", result.content)

    def test_calculator_rejects_unsafe_expression(self):
        cfg = Config()
        registry = build_registry(cfg)
        result = registry.execute(
            ToolCall("calculator", {"expression": "__import__('os').system('date')"})
        )
        self.assertFalse(result.ok)

    def test_tool_description_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.prompts.tools_path = str(Path(tmp) / "tools.jsonc")
            Path(cfg.prompts.tools_path).write_text(
                '{"tools":{"calculator":{"description":"Custom math tool.",'
                '"instructions":"Use exact math for everything."}}}',
                encoding="utf-8",
            )
            registry = build_registry(cfg)
            calculator = [spec for spec in registry.specs() if spec.name == "calculator"][0]
            self.assertIn("Custom math tool", calculator.description)
            self.assertIn("Use exact math", calculator.description)

    def test_markdown_tool_description_override_still_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.prompts.tools_path = str(Path(tmp) / "tools.md")
            Path(cfg.prompts.tools_path).write_text(
                "## calculator\n"
                "Description: Markdown math tool.\n"
                "Instructions:\n"
                "Use markdown fallback.\n",
                encoding="utf-8",
            )
            registry = build_registry(cfg)
            calculator = [spec for spec in registry.specs() if spec.name == "calculator"][0]
            self.assertIn("Markdown math tool", calculator.description)
            self.assertIn("Use markdown fallback", calculator.description)

    def test_web_search_brave_uses_api_provider(self):
        cfg = Config()
        cfg.tools.web_search_enabled = True
        registry = build_registry(cfg)
        body = (
            b'{"web":{"results":[{"title":"Example","url":"https://example.com",'
            b'"description":"Example result","extra_snippets":["More context"]}]}}'
        )
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test-key"}), patch(
            "wheatley.tools.web.urllib.request.urlopen",
            return_value=_FakeResponse(body, "application/json"),
        ) as urlopen:
            result = registry.execute(ToolCall("web_search", {"query": "example"}))
        self.assertTrue(result.ok)
        self.assertEqual(result.content["provider"], "brave")
        self.assertEqual(result.content["results"][0]["url"], "https://example.com")
        headers = {
            key.lower(): value
            for key, value in urlopen.call_args.args[0].headers.items()
        }
        self.assertEqual(headers["x-subscription-token"], "test-key")

    def test_fetch_url_blocks_private_networks(self):
        cfg = Config()
        cfg.tools.web_fetch_enabled = True
        registry = build_registry(cfg)
        with patch(
            "wheatley.tools.web.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("127.0.0.1", 80))],
        ):
            result = registry.execute(ToolCall("fetch_url", {"url": "http://localhost/"}))
        self.assertFalse(result.ok)
        self.assertIn("blocked", result.content["error"])

    def test_fetch_url_strips_html_to_readable_text(self):
        cfg = Config()
        cfg.tools.web_fetch_enabled = True
        registry = build_registry(cfg)
        html = b"""
        <html><head><title>Ignored</title><script>bad()</script></head>
        <body><main><h1>Title</h1><p>Hello <a href="/docs">docs</a>.</p></main></body>
        </html>
        """
        opener = _FakeOpener(_FakeResponse(html, "text/html", "https://example.com/page"))
        with patch(
            "wheatley.tools.web.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 443))],
        ), patch("wheatley.tools.web.urllib.request.build_opener", return_value=opener):
            result = registry.execute(
                ToolCall("fetch_url", {"url": "https://example.com/page"})
            )
        self.assertTrue(result.ok)
        self.assertIn("# Title", result.content["text"])
        self.assertIn("Hello docs (https://example.com/docs)", result.content["text"])
        self.assertNotIn("bad()", result.content["text"])

    def test_tool_start_messages_cover_english_defaults(self):
        cfg = Config()

        self.assertEqual(tool_start_message(cfg, "remember"), "Remembering...")
        self.assertEqual(tool_start_message(cfg, "run_safe_cli_tool"), "Running...")
        self.assertEqual(tool_start_message(cfg, "web_search"), "Searching...")
        self.assertEqual(tool_start_message(cfg, "fetch_url"), "Downloading...")


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        content_type: str,
        url: str = "https://example.com",
    ) -> None:
        self.body = body
        self.headers = {"Content-Type": content_type}
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]

    def geturl(self) -> str:
        return self.url


class _FakeOpener:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def open(self, request, timeout: float):
        return self.response


if __name__ == "__main__":
    unittest.main()
