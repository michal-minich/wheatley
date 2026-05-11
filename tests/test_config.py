import tempfile
import unittest
from pathlib import Path

from wheatley.config import load_config
from wheatley.jsonc import loads_jsonc
from wheatley.language import apply_configured_language


class ConfigTests(unittest.TestCase):
    def test_jsonc_parser_accepts_comments_and_trailing_commas(self):
        data = loads_jsonc(
            """
            {
              // comment
              "name": "wheatley",
              "items": [1, 2,],
            }
            """
        )
        self.assertEqual(data["items"], [1, 2])

    def test_profile_local_paths_are_derived_from_config_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "demo"
            profile.mkdir()
            (profile / "config.jsonc").write_text(
                """
                {
                  // These legacy path overrides are ignored.
                  "runtime": {
                    "data_dir": "profiles/wheatley/runtime",
                    "turn_log": "profiles/wheatley/runtime/logs/turns.jsonl",
                    "tool_log": "profiles/wheatley/runtime/logs/tools.jsonl",
                    "system_llm_log": "profiles/wheatley/runtime/logs/system_llm.jsonl",
                    "state_dir": "profiles/wheatley/runtime/state"
                  },
                  "audio": {
                    "utterance_dir": "profiles/wheatley/runtime/audio"
                  },
                  "tts": {
                    "output_dir": "profiles/wheatley/runtime/audio"
                  },
                  "prompts": {
                    "system_path": "somewhere/system.md",
                    "user_path": "somewhere/user.md",
                    "tools_path": "somewhere/tools.jsonc",
                    "memory_path": "somewhere/memory.md",
                  }
                }
                """,
                encoding="utf-8",
            )
            cfg = load_config(str(profile / "config.jsonc"))
            self.assertEqual(Path(cfg.prompts.system_path), profile / "system.md")
            self.assertEqual(Path(cfg.runtime.turn_log), profile / "runtime/logs/turns.jsonl")
            self.assertEqual(Path(cfg.profile_dir), profile)

    def test_tool_settings_load_from_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "demo"
            profile.mkdir()
            (profile / "config.jsonc").write_text(
                """
                {
                  "tools": {
                    "enabled": false,
                    "tool_settings": {
                      "get_time": {"enabled": true, "description": "x", "start_messages": {"en": "Checking time..."}},
                      "system_status": {"enabled": false, "description": "x", "start_messages": {"en": "Checking status..."}}
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            cfg = load_config(str(profile / "config.jsonc"))
            self.assertTrue(cfg.tools.tool_settings["get_time"]["enabled"])
            self.assertFalse(cfg.tools.tool_settings["system_status"]["enabled"])
            self.assertEqual(
                cfg.tools.tool_settings["get_time"]["start_messages"]["en"],
                "Checking time...",
            )

    def test_tool_localization_loads_from_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "demo"
            profile.mkdir()
            (profile / "config.jsonc").write_text(
                """
                {
                  "tools": {
                    "enabled": false,
                    "current_tools_message": {
                      "en": "Current tools are: {tools}.",
                      "sk": "Aktuálne nástroje sú: {tools}."
                    },
                    "tool_list_conjunction": {
                      "en": "and",
                      "sk": "a"
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            cfg = load_config(str(profile / "config.jsonc"))
            self.assertEqual(
                cfg.tools.current_tools_message["sk"],
                "Aktuálne nástroje sú: {tools}.",
            )
            self.assertEqual(cfg.tools.tool_list_conjunction["sk"], "a")

    def test_tool_settings_require_start_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "demo"
            profile.mkdir()
            (profile / "config.jsonc").write_text(
                """
                {
                  "tools": {
                    "enabled": true,
                    "tool_settings": {
                      "get_time": {"enabled": true, "description": "x"}
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "get_time.start_messages"):
                load_config(str(profile / "config.jsonc"))

    def test_language_overrides_can_use_same_names_in_nested_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "demo"
            profile.mkdir()
            (profile / "config.jsonc").write_text(
                """
                {
                  "language": {
                    "enabled": true,
                    "default": "en",
                    "languages": {
                      "en": {
                        "label": "English",
                        "response_language": "English",
                        "audio": {
                          "partial_transcript_enabled": true,
                          "partial_transcript_use_as_final": true
                        },
                        "stt": {
                          "model": "small.en",
                          "language": "en",
                          "remote_model": "small.en",
                          "preview_model": "small.en",
                          "preview_beam_size": 1,
                          "final_model": "medium.en",
                          "final_beam_size": 3
                        },
                        "tts": {
                          "backend": "piper",
                          "voice": "Test",
                          "stream_speech": true
                        }
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            cfg = load_config(str(profile / "config.jsonc"))

            apply_configured_language(cfg, "en")

            self.assertTrue(cfg.audio.partial_transcript_enabled)
            self.assertEqual(cfg.stt.model, "small.en")
            self.assertEqual(cfg.stt.remote_model, "small.en")
            self.assertEqual(cfg.stt.preview_model, "small.en")
            self.assertEqual(cfg.stt.preview_beam_size, 1)
            self.assertEqual(cfg.stt.final_model, "medium.en")
            self.assertEqual(cfg.stt.final_beam_size, 3)
            self.assertEqual(cfg.tts.backend, "piper")
            self.assertTrue(cfg.tts.stream_speech)


if __name__ == "__main__":
    unittest.main()
