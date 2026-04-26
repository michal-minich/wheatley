import tempfile
import unittest
from pathlib import Path

from wheatly.config import Config
from wheatly.pipeline import VoiceAgent, build_system_prompt
from wheatly.tts.base import SpeechResult, TTSBackend


class SilentTTS(TTSBackend):
    def speak(self, text: str) -> SpeechResult:
        return SpeechResult(audio_path=None, spoken=False)


class PipelineTests(unittest.TestCase):
    def test_echo_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.runtime.data_dir = tmp
            cfg.runtime.state_dir = str(Path(tmp) / "state")
            cfg.runtime.turn_log = str(Path(tmp) / "turns.jsonl")
            cfg.tts.enabled = False
            cfg.ensure_dirs()
            agent = VoiceAgent(cfg, tts=SilentTTS())
            result = agent.handle_text("hello", speak=False)
            self.assertIn("I heard", result.assistant_text)

    def test_echo_tool_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.runtime.data_dir = tmp
            cfg.runtime.state_dir = str(Path(tmp) / "state")
            cfg.runtime.turn_log = str(Path(tmp) / "turns.jsonl")
            cfg.ensure_dirs()
            agent = VoiceAgent(cfg, tts=SilentTTS())
            result = agent.handle_text("what time is it?", speak=False)
            self.assertEqual(result.tool_results[0].name, "get_time")
            self.assertIn("Local time", result.assistant_text)

    def test_remember_command_writes_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.runtime.data_dir = tmp
            cfg.runtime.state_dir = str(Path(tmp) / "state")
            cfg.runtime.turn_log = str(Path(tmp) / "turns.jsonl")
            cfg.prompts.memory_path = str(Path(tmp) / "memory.md")
            cfg.ensure_dirs()
            agent = VoiceAgent(cfg, tts=SilentTTS())
            result = agent.handle_text("Remember this: I like quick answers.", speak=False)
            self.assertEqual(result.tool_results[0].name, "remember")
            self.assertIn("I'll remember", result.assistant_text)
            self.assertIn("I like quick answers", Path(cfg.prompts.memory_path).read_text())

    def test_prompt_injects_user_instructions_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.prompts.system_path = str(Path(tmp) / "system.md")
            cfg.prompts.user_path = str(Path(tmp) / "user.md")
            cfg.prompts.memory_path = str(Path(tmp) / "memory.md")
            cfg.ensure_dirs()
            Path(cfg.prompts.system_path).write_text("System for {{AGENT_NAME}}.", encoding="utf-8")
            Path(cfg.prompts.user_path).write_text("Prefer direct answers.", encoding="utf-8")
            Path(cfg.prompts.memory_path).write_text("- User likes fast starts.", encoding="utf-8")
            agent = VoiceAgent(cfg, tts=SilentTTS())
            prompt = build_system_prompt(cfg, agent.tools)
            self.assertIn("System for Wheatly.", prompt)
            self.assertIn("Prefer direct answers.", prompt)
            self.assertIn("User likes fast starts.", prompt)


if __name__ == "__main__":
    unittest.main()
