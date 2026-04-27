import json
import tempfile
import unittest
from pathlib import Path

from wheatly.config import Config
from wheatly.llm.base import LLMBackend, LLMMessage, LLMResponse
from wheatly.memory import (
    auto_memory_path,
    memory_consolidate_instructions_path,
    memory_candidates_path,
    memory_state_path,
    memory_update_instructions_path,
    refresh_auto_memory,
)
from wheatly.pipeline import VoiceAgent, build_system_prompt
from wheatly.tts.base import SpeechResult, TTSBackend


class SequenceLLM(LLMBackend):
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        self.messages.append(messages)
        return LLMResponse(self.responses.pop(0))


class FailingLLM(LLMBackend):
    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        del messages
        raise TimeoutError("memory model timed out")


class NoCallLLM(LLMBackend):
    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        del messages
        raise AssertionError("LLM should not be called")


class SilentTTS(TTSBackend):
    def speak(self, text: str) -> SpeechResult:
        del text
        return SpeechResult(audio_path=None, spoken=False)


class AutoMemoryTests(unittest.TestCase):
    def test_incremental_update_writes_separate_auto_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _memory_cfg(Path(tmp))
            _write_turn(
                Path(cfg.runtime.turn_log),
                "2026-04-27T10:00:00+02:00",
                "Remember that I like Lua and Pico-8.",
                "Sure.",
            )
            llm = SequenceLLM(
                [
                    json.dumps(
                        {
                            "stable_user_facts": ["User likes Lua and Pico-8."],
                            "preferences": [],
                            "current_projects": [],
                            "recent_context": ["User discussed Lua and Pico-8."],
                        }
                    )
                ]
            )
            notices = []

            result = refresh_auto_memory(cfg, llm, mode="offline", notify=notices.append)

            self.assertTrue(result.updated)
            self.assertEqual(
                notices,
                ["wait, I'm updating my memory...", "my memory was updated."],
            )
            self.assertIn(
                "User likes Lua and Pico-8.",
                auto_memory_path(cfg).read_text(encoding="utf-8"),
            )
            self.assertTrue(memory_state_path(cfg).exists())
            self.assertTrue(memory_candidates_path(cfg).exists())
            self.assertFalse(Path(cfg.prompts.memory_path).exists())

    def test_online_consolidation_can_include_assistant_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _memory_cfg(Path(tmp))
            cfg.memory.full_rewrite_requires_online = True
            cfg.memory.include_assistant_text_online = True
            _write_turn(
                Path(cfg.runtime.turn_log),
                "2026-04-27T11:00:00+02:00",
                "I am building Wheatly a robot body.",
                "That robot body sounds useful.",
            )
            llm = SequenceLLM(
                [
                    json.dumps(
                        {
                            "current_projects": [
                                "User is building Wheatly a robot body."
                            ],
                            "stable_user_facts": [],
                            "preferences": [],
                            "recent_context": [],
                        }
                    ),
                    json.dumps(
                        {
                            "auto_memory_md": "# Wheatly Auto Memory\n\n"
                            "## Stable User Facts\n- None yet.\n\n"
                            "## Preferences\n- None yet.\n\n"
                            "## Current Projects\n"
                            "- User is building Wheatly a robot body.\n\n"
                            "## Recent Context\n- None yet.\n"
                        }
                    ),
                ]
            )
            notices = []

            result = refresh_auto_memory(cfg, llm, mode="online", notify=notices.append)

            self.assertTrue(result.consolidated)
            self.assertIn("wait, I'm consolidating my memory...", notices)
            prompts = "\n\n".join(message.content for call in llm.messages for message in call)
            self.assertIn("That robot body sounds useful.", prompts)
            self.assertIn(
                "User is building Wheatly a robot body.",
                auto_memory_path(cfg).read_text(encoding="utf-8"),
            )

    def test_memory_model_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _memory_cfg(Path(tmp))
            _write_turn(
                Path(cfg.runtime.turn_log),
                "2026-04-27T12:00:00+02:00",
                "I like quick memory updates.",
                "",
            )
            notices = []

            result = refresh_auto_memory(
                cfg, FailingLLM(), mode="offline", notify=notices.append
            )

            self.assertFalse(result.updated)
            self.assertEqual(notices, ["wait, I'm updating my memory..."])

    def test_prompt_injects_manual_and_auto_memory_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _memory_cfg(Path(tmp))
            Path(cfg.prompts.system_path).write_text("System.", encoding="utf-8")
            Path(cfg.prompts.user_path).write_text("", encoding="utf-8")
            Path(cfg.prompts.memory_path).write_text(
                "- Manual fact.", encoding="utf-8"
            )
            auto_memory_path(cfg).write_text(
                "# Wheatly Auto Memory\n\n## Stable User Facts\n- Auto fact.\n",
                encoding="utf-8",
            )
            agent = VoiceAgent(cfg, tts=SilentTTS())

            prompt = build_system_prompt(cfg, agent.tools)

            self.assertIn("# Persistent Memory\n- Manual fact.", prompt)
            self.assertIn("# Conversation-Derived Memory", prompt)
            self.assertIn("- Auto fact.", prompt)

    def test_update_uses_update_instruction_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _memory_cfg(Path(tmp))
            memory_update_instructions_path(cfg).write_text(
                "UPDATE ONLY", encoding="utf-8"
            )
            memory_consolidate_instructions_path(cfg).write_text(
                "CONSOLIDATE ONLY", encoding="utf-8"
            )
            _write_turn(
                Path(cfg.runtime.turn_log),
                "2026-04-27T13:00:00+02:00",
                "I am testing memory update instructions.",
                "",
            )
            llm = SequenceLLM(
                [
                    json.dumps(
                        {
                            "stable_user_facts": [],
                            "preferences": [],
                            "current_projects": [],
                            "recent_context": [],
                        }
                    )
                ]
            )

            refresh_auto_memory(cfg, llm, mode="offline")

            prompt = llm.messages[0][1].content
            self.assertIn("UPDATE ONLY", prompt)
            self.assertNotIn("CONSOLIDATE ONLY", prompt)

    def test_consolidation_uses_consolidate_instruction_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _memory_cfg(Path(tmp))
            memory_update_instructions_path(cfg).write_text(
                "UPDATE ONLY", encoding="utf-8"
            )
            memory_consolidate_instructions_path(cfg).write_text(
                "CONSOLIDATE ONLY", encoding="utf-8"
            )
            auto_memory_path(cfg).write_text(
                "# Wheatly Auto Memory\n\n## Stable User Facts\n- Old fact.\n",
                encoding="utf-8",
            )
            llm = SequenceLLM(
                [
                    json.dumps(
                        {
                            "auto_memory_md": "# Wheatly Auto Memory\n\n"
                            "## Stable User Facts\n- Old fact.\n\n"
                            "## Preferences\n- None yet.\n\n"
                            "## Current Projects\n- None yet.\n\n"
                            "## Recent Context\n- None yet.\n"
                        }
                    )
                ]
            )

            refresh_auto_memory(cfg, llm, mode="online")

            prompt = llm.messages[0][1].content
            self.assertIn("CONSOLIDATE ONLY", prompt)
            self.assertNotIn("UPDATE ONLY", prompt)

    def test_incremental_update_skips_turns_older_than_last_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _memory_cfg(Path(tmp))
            _write_turn(
                Path(cfg.runtime.turn_log),
                "2026-04-27T10:00:00+02:00",
                "Old backlog fact.",
                "",
            )
            _write_turn(
                Path(cfg.runtime.turn_log),
                "2026-04-27T10:10:00+02:00",
                "New useful fact.",
                "",
            )
            _write_state(
                cfg,
                {
                    "last_processed_offset": 0,
                    "last_processed_timestamp": "2026-04-27T09:00:00+02:00",
                    "last_incremental_update_at": "2026-04-27T10:05:00+02:00",
                    "last_full_rewrite_at": None,
                },
            )
            llm = SequenceLLM(
                [
                    json.dumps(
                        {
                            "stable_user_facts": ["New useful fact."],
                            "preferences": [],
                            "current_projects": [],
                            "recent_context": [],
                        }
                    )
                ]
            )

            result = refresh_auto_memory(cfg, llm, mode="offline")

            self.assertEqual(result.processed_turns, 1)
            prompt = llm.messages[0][1].content
            self.assertIn("New useful fact.", prompt)
            self.assertNotIn("Old backlog fact.", prompt)
            state = json.loads(memory_state_path(cfg).read_text(encoding="utf-8"))
            self.assertEqual(
                state["last_processed_offset"],
                Path(cfg.runtime.turn_log).stat().st_size,
            )

    def test_existing_auto_memory_bootstrap_skips_historical_incremental_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _memory_cfg(Path(tmp))
            auto_memory_path(cfg).write_text(
                "# Wheatly Auto Memory\n\n## Stable User Facts\n- Existing fact.\n",
                encoding="utf-8",
            )
            _write_turn(
                Path(cfg.runtime.turn_log),
                "2026-04-27T10:00:00+02:00",
                "Historical fact already covered.",
                "",
            )

            result = refresh_auto_memory(cfg, NoCallLLM(), mode="offline")

            self.assertEqual(result.processed_turns, 0)
            state = json.loads(memory_state_path(cfg).read_text(encoding="utf-8"))
            self.assertEqual(
                state["last_processed_offset"],
                Path(cfg.runtime.turn_log).stat().st_size,
            )

    def test_incremental_update_deduplicates_memory_and_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _memory_cfg(Path(tmp))
            auto_memory_path(cfg).write_text(
                "# Wheatly Auto Memory\n\n"
                "## Stable User Facts\n- User likes Lua and Pico-8.\n",
                encoding="utf-8",
            )
            memory_candidates_path(cfg).write_text(
                json.dumps(
                    {
                        "recorded_at": "2026-04-27T09:00:00+02:00",
                        "category": "stable_user_facts",
                        "fact": "User likes Lua and Pico-8.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_turn(
                Path(cfg.runtime.turn_log),
                "2026-04-27T10:00:00+02:00",
                "I still like Lua and Pico-8.",
                "",
            )
            _write_state(
                cfg,
                {
                    "last_processed_offset": 0,
                    "last_processed_timestamp": None,
                    "last_incremental_update_at": None,
                    "last_full_rewrite_at": None,
                },
            )
            llm = SequenceLLM(
                [
                    json.dumps(
                        {
                            "stable_user_facts": ["User likes Lua and Pico-8."],
                            "preferences": [],
                            "current_projects": [],
                            "recent_context": [],
                        }
                    )
                ]
            )

            refresh_auto_memory(cfg, llm, mode="offline")

            memory = auto_memory_path(cfg).read_text(encoding="utf-8")
            self.assertEqual(memory.count("User likes Lua and Pico-8."), 1)
            candidates = memory_candidates_path(cfg).read_text(encoding="utf-8")
            self.assertEqual(candidates.count("User likes Lua and Pico-8."), 1)


def _memory_cfg(root: Path) -> Config:
    profile = root / "profile"
    cfg = Config()
    cfg.profile_dir = str(profile)
    cfg.runtime.data_dir = str(profile / "runtime")
    cfg.runtime.state_dir = str(profile / "runtime/state")
    cfg.runtime.turn_log = str(profile / "runtime/logs/turns.jsonl")
    cfg.audio.utterance_dir = str(profile / "runtime/audio")
    cfg.tts.output_dir = str(profile / "runtime/audio")
    cfg.prompts.system_path = str(profile / "system.md")
    cfg.prompts.user_path = str(profile / "user.md")
    cfg.prompts.tools_path = str(profile / "tools.jsonc")
    cfg.prompts.memory_path = str(profile / "memory.md")
    cfg.memory.full_rewrite_requires_online = True
    cfg.memory.full_rewrite_interval_days = 1
    cfg.ensure_dirs()
    return cfg


def _write_turn(path: Path, timestamp: str, user_text: str, assistant_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "user_text": user_text,
                    "assistant_text": assistant_text,
                    "tool_results": [],
                }
            )
            + "\n"
        )


def _write_state(cfg: Config, data: dict) -> None:
    path = memory_state_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
