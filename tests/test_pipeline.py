import tempfile
import unittest
from pathlib import Path

from wheatly.config import Config
from wheatly.language import model_selection_message
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

    def test_explicit_language_switch_updates_runtime_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.runtime.data_dir = tmp
            cfg.runtime.state_dir = str(Path(tmp) / "state")
            cfg.runtime.turn_log = str(Path(tmp) / "turns.jsonl")
            cfg.language.enabled = True
            cfg.ensure_dirs()
            agent = VoiceAgent(cfg, tts=SilentTTS())
            result = agent.handle_text("switch to Slovak", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "sk")
            self.assertEqual(cfg.agent.default_response_language, "Slovak")
            self.assertEqual(cfg.stt.model, "medium")
            self.assertEqual(cfg.stt.language, "sk")
            self.assertEqual(cfg.tts.backend, "edge_tts")
            self.assertEqual(cfg.tts.edge_voice, "sk-SK-LukasNeural")
            self.assertEqual(cfg.tts.length_scale, 0.84)
            self.assertEqual(cfg.tts.leading_silence_ms, 80)
            self.assertEqual(cfg.tts.stream_min_words, 10)
            self.assertEqual(result.assistant_text, "Ahoj")

    def test_generic_language_switch_uses_phrase_language_then_toggles(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.runtime.data_dir = tmp
            cfg.runtime.state_dir = str(Path(tmp) / "state")
            cfg.runtime.turn_log = str(Path(tmp) / "turns.jsonl")
            cfg.language.enabled = True
            cfg.ensure_dirs()
            agent = VoiceAgent(cfg, tts=SilentTTS())

            result = agent.handle_text("switch language", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "sk")
            self.assertEqual(result.assistant_text, "Ahoj")

            result = agent.handle_text("switch language", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "en")
            self.assertEqual(result.assistant_text, "Hi")

            result = agent.handle_text("prepni jazyk", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "sk")
            self.assertEqual(result.assistant_text, "Ahoj")

            result = agent.handle_text("prepni jazyk", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "en")
            self.assertEqual(result.assistant_text, "Hi")

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

    def test_model_selection_message_matches_start_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.runtime.data_dir = tmp
            cfg.runtime.state_dir = str(Path(tmp) / "state")
            cfg.runtime.turn_log = str(Path(tmp) / "turns.jsonl")
            cfg.language.enabled = True
            cfg.language.default = "sk"
            cfg.ensure_dirs()

            agent = VoiceAgent(cfg, tts=SilentTTS())
            selection = agent.reset_chat()

            self.assertEqual(selection.message, "Používam offline model.")
            self.assertEqual(model_selection_message(cfg, "online"), "Používam múdrejší online model.")


if __name__ == "__main__":
    unittest.main()
