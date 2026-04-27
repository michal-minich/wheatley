import tempfile
import unittest
from pathlib import Path

from wheatly.config import load_config
from wheatly.jsonc import loads_jsonc


class ConfigTests(unittest.TestCase):
    def test_jsonc_parser_accepts_comments_and_trailing_commas(self):
        data = loads_jsonc(
            """
            {
              // comment
              "name": "wheatly",
              "items": [1, 2,],
            }
            """
        )
        self.assertEqual(data["items"], [1, 2])

    def test_profile_relative_prompt_paths_resolve_from_config_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "demo"
            profile.mkdir()
            (profile / "config.jsonc").write_text(
                """
                {
                  // profile-local files
                  "prompts": {
                    "system_path": "system.md",
                    "user_path": "user.md",
                    "tools_path": "tools.jsonc",
                    "memory_path": "memory.md",
                  }
                }
                """,
                encoding="utf-8",
            )
            cfg = load_config(str(profile / "config.jsonc"))
            self.assertEqual(Path(cfg.prompts.system_path), profile / "system.md")
            self.assertEqual(Path(cfg.prompts.tools_path), profile / "tools.jsonc")
            self.assertEqual(Path(cfg.profile_dir), profile)

    def test_default_profile_runtime_lives_under_profile_folder(self):
        cfg = load_config()
        self.assertEqual(
            Path(cfg.runtime.turn_log),
            Path("profiles/wheatly/runtime/logs/turns.jsonl"),
        )
        self.assertEqual(
            Path(cfg.runtime.tool_log),
            Path("profiles/wheatly/runtime/logs/tools.jsonl"),
        )
        self.assertEqual(
            Path(cfg.audio.utterance_dir),
            Path("profiles/wheatly/runtime/audio"),
        )


if __name__ == "__main__":
    unittest.main()
