import tempfile
import unittest
from pathlib import Path

from wheatly.config import Config
from wheatly.tools.builtins import build_registry
from wheatly.tools.registry import ToolCall


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
            cfg.prompts.tools_path = str(Path(tmp) / "tools.json")
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


if __name__ == "__main__":
    unittest.main()
