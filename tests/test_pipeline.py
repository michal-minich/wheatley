import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from wheatley.config import Config, load_config
from wheatley.llm.base import LLMBackend, LLMMessage, LLMResponse
from wheatley.pipeline import (
    VoiceAgent,
    _current_tools_message,
    _join_tool_names,
    build_system_prompt,
)
from wheatley.stt.base import STTBackend, Transcription
from wheatley.tools.registry import ToolRegistry, ToolResult, ToolSpec
from wheatley.tts.base import SpeechResult, TTSBackend


class SilentSTT(STTBackend):
    def transcribe(self, audio_path=None) -> Transcription:
        del audio_path
        return Transcription(text="")


class SilentTTS(TTSBackend):
    def speak(self, text: str) -> SpeechResult:
        return SpeechResult(audio_path=None, spoken=False)


class RecordingTTS(TTSBackend):
    def __init__(self):
        self.spoken = []

    def speak(self, text: str) -> SpeechResult:
        self.spoken.append(text)
        return SpeechResult(audio_path=None, spoken=True)


class BlockingToolsTTS(TTSBackend):
    def __init__(self):
        self.spoken = []
        self.tools_started = threading.Event()
        self.release_tools = threading.Event()

    def speak(self, text: str) -> SpeechResult:
        self.spoken.append(text)
        if text.startswith("Current tools are:"):
            self.tools_started.set()
            self.release_tools.wait(timeout=2)
        return SpeechResult(audio_path=None, spoken=True)


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size: int = -1) -> bytes:
        del size
        return b"{}"


class SequenceLLM(LLMBackend):
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        del messages
        return LLMResponse(self.responses.pop(0))

    def stream_complete(self, messages: list[LLMMessage]):
        text = self.complete(messages).content
        for part in text.split(" "):
            yield part
            yield " "


class RecordingVisionLLM(SequenceLLM):
    def __init__(self, responses, supports_images=True):
        super().__init__(responses)
        self.supports_image_input = supports_images
        self.calls: list[list[LLMMessage]] = []

    def supports_images(self) -> bool:
        return self.supports_image_input

    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        self.calls.append(messages)
        return super().complete(messages)


def _pipeline_cfg(tmp: str | None = None) -> Config:
    cfg = load_config()
    cfg.llm.backend = "echo"
    cfg.llm.model = "echo"
    cfg.llm.remote.enabled = False
    cfg.tts.enabled = False
    cfg.tts.playback = False
    cfg.memory.auto_enabled = False
    cfg.tools.enabled = True
    cfg.tools.tool_settings["web_search"] = {
        **cfg.tools.tool_settings.get("web_search", {}),
        "enabled": False,
    }
    if tmp:
        root = Path(tmp)
        cfg.runtime.data_dir = str(root)
        cfg.runtime.state_dir = str(root / "state")
        cfg.runtime.turn_log = str(root / "turns.jsonl")
        cfg.runtime.tool_log = str(root / "tools.jsonl")
        cfg.runtime.system_llm_log = str(root / "system_llm.jsonl")
        cfg.audio.utterance_dir = str(root / "audio")
        cfg.tts.output_dir = str(root / "audio")
    cfg.ensure_dirs()
    return cfg


def _write_turn(path: str, user_text: str, assistant_text: str = "") -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-04-29T16:30:00+02:00",
                    "user_text": user_text,
                    "assistant_text": assistant_text,
                    "tool_results": [],
                }
            )
            + "\n"
        )


class PipelineTests(unittest.TestCase):
    def test_llm_and_stt_online_selection_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.llm.remote.enabled = True
            cfg.llm.remote.backend = "openai_compat"
            cfg.llm.remote.base_url = "http://remote-llm.test/v1"
            cfg.llm.remote.api_key = "EMPTY"
            cfg.stt.backend = "remote_fallback"
            agent = VoiceAgent(cfg, llm=SequenceLLM([]), stt=SilentSTT(), tts=SilentTTS())

            with patch("wheatley.pipeline.remote_llm_available", return_value=True), patch(
                "wheatley.pipeline.remote_stt_available", return_value=False
            ), patch("wheatley.pipeline.build_llm", return_value=SequenceLLM([])):
                selection = agent.select_chat_model()

            self.assertEqual(selection.mode, "online")
            self.assertEqual(selection.stt_mode, "local")

            with patch("wheatley.pipeline.remote_llm_available", return_value=False), patch(
                "wheatley.pipeline.remote_stt_available", return_value=True
            ), patch("wheatley.pipeline.build_llm", return_value=SequenceLLM([])):
                selection = agent.select_chat_model()

            self.assertEqual(selection.mode, "offline")
            self.assertEqual(selection.stt_mode, "remote")

    def test_restore_turn_history_rebuilds_user_assistant_messages(self):
        cfg = _pipeline_cfg()
        agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())

        agent.restore_turn_history(
            [
                {"user_text": "first user", "assistant_text": "first assistant"},
                {"user_text": "second user", "assistant_text": "second assistant"},
            ]
        )

        self.assertEqual(
            [(message.role, message.content) for message in agent.history],
            [
                ("user", "first user"),
                ("assistant", "first assistant"),
                ("user", "second user"),
                ("assistant", "second assistant"),
            ],
        )

    def test_echo_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())
            result = agent.handle_text("hello", speak=False)
            self.assertIn("I heard", result.assistant_text)
            row = json.loads(Path(cfg.runtime.turn_log).read_text().splitlines()[0])
            self.assertEqual(row["model_name"], cfg.llm.model)

    def test_idle_speech_uses_instruction_without_fake_user_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            llm = RecordingVisionLLM(["Tiny idle fact."])
            agent = VoiceAgent(cfg, llm=llm, stt=SilentSTT(), tts=SilentTTS())

            result = agent.handle_idle_speech(
                "No one spoke; make an idle remark.",
                speak=False,
            )

            self.assertEqual(result.user_text, "")
            self.assertEqual(result.assistant_text, "Tiny idle fact.")
            self.assertEqual(llm.calls[-1][-1].content, "No one spoke; make an idle remark.")
            self.assertEqual(
                [(message.role, message.content) for message in agent.history],
                [("assistant", "Tiny idle fact.")],
            )
            row = json.loads(Path(cfg.runtime.turn_log).read_text().splitlines()[0])
            self.assertIsNone(row["user_text"])
            self.assertEqual(row["source"], "idle")

    def test_idle_speech_falls_back_instead_of_silence_for_tool_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.tts.enabled = True
            cfg.tts.stream_speech = True
            tts = RecordingTTS()
            llm = RecordingVisionLLM(
                ['{"tool_calls":[{"name":"get_time","arguments":{}}]}']
            )
            agent = VoiceAgent(cfg, llm=llm, stt=SilentSTT(), tts=tts)

            result = agent.handle_idle_speech(
                "No one spoke; make an idle remark.",
                speak=True,
            )

            self.assertIn("Tiny idle note:", result.assistant_text)
            self.assertEqual(tts.spoken, [result.assistant_text])
            row = json.loads(Path(cfg.runtime.turn_log).read_text().splitlines()[0])
            self.assertEqual(row["source"], "idle")
            self.assertIsNone(row["user_text"])

    def test_idle_speech_compacts_multiline_answer_before_speaking(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.tts.enabled = True
            tts = RecordingTTS()
            llm = RecordingVisionLLM(
                [
                    "- First idle idea.\n"
                    "- Second idle idea.\n"
                    "- Third idle idea that should not be spoken."
                ]
            )
            agent = VoiceAgent(cfg, llm=llm, stt=SilentSTT(), tts=tts)

            result = agent.handle_idle_speech(
                "No one spoke; make an idle remark.",
                speak=True,
            )

            self.assertEqual(result.assistant_text, "First idle idea. Second idle idea.")
            self.assertEqual(tts.spoken, [result.assistant_text])

    def test_idle_speech_streams_console_tokens_but_speaks_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.tts.enabled = True
            tts = RecordingTTS()
            llm = RecordingVisionLLM(["Short idle remark."])
            agent = VoiceAgent(cfg, llm=llm, stt=SilentSTT(), tts=tts)
            tokens = []

            result = agent.handle_idle_speech(
                "No one spoke; make an idle remark.",
                speak=True,
                on_token=tokens.append,
            )

            self.assertEqual("".join(tokens), result.assistant_text)
            self.assertGreater(len(tokens), 1)
            self.assertEqual(tts.spoken, [result.assistant_text])

    def test_echo_tool_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())
            result = agent.handle_text("what time is it?", speak=False)
            self.assertEqual(result.tool_results[0].name, "get_time")
            self.assertIn("Local time", result.assistant_text)
            audit = json.loads(Path(cfg.runtime.tool_log).read_text().splitlines()[0])
            self.assertEqual(audit["source"], "llm")
            self.assertEqual(audit["tool"], "get_time")
            self.assertEqual(audit["arguments"], {})
            self.assertTrue(audit["result"]["ok"])
            self.assertIn("duration_seconds", audit)

    def test_disabled_direct_route_tool_does_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.tools.tool_settings["system_status"] = {
                **cfg.tools.tool_settings.get("system_status", {}),
                "enabled": False,
            }
            agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())

            result = agent.handle_text("what is your status?", speak=False)

            self.assertEqual(result.tool_results, [])
            self.assertFalse(Path(cfg.runtime.tool_log).exists())

    def test_unavailable_web_search_is_not_executed_or_prompted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.tools.tool_settings["web_search"]["enabled"] = True
            llm = SequenceLLM(
                [
                    json.dumps(
                        {
                            "tool_calls": [
                                {"name": "web_search", "arguments": {"query": "x"}}
                            ]
                        }
                    ),
                    "No search available.",
                ]
            )
            with patch.dict("os.environ", {}, clear=True):
                agent = VoiceAgent(cfg, llm=llm, stt=SilentSTT(), tts=SilentTTS())

            self.assertNotIn("web_search", build_system_prompt(cfg, agent.tools))
            result = agent.handle_text("search for x", speak=False)

            self.assertEqual(result.tool_results, [])
            self.assertEqual(result.assistant_text, "No search available.")
            self.assertFalse(Path(cfg.runtime.tool_log).exists())

    def test_photo_tool_result_is_attached_for_vision_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.tools.tool_settings["take_photo"]["enabled"] = True
            photo = Path(tmp) / "photo.jpg"
            photo.write_bytes(b"fake-jpeg")
            tools = ToolRegistry()
            tools.register(
                ToolSpec("take_photo", "camera", {"type": "object"}),
                lambda args: ToolResult(
                    name="take_photo",
                    ok=True,
                    content={"path": str(photo), "mime_type": "image/jpeg"},
                ),
            )
            llm = RecordingVisionLLM(
                [
                    json.dumps({"tool_calls": [{"name": "take_photo", "arguments": {}}]}),
                    "I can see it.",
                ],
                supports_images=True,
            )
            agent = VoiceAgent(
                cfg,
                llm=llm,
                stt=SilentSTT(),
                tts=SilentTTS(),
                tools=tools,
            )

            result = agent.handle_text("look at this", speak=False)

            self.assertEqual(result.assistant_text, "I can see it.")
            self.assertEqual(llm.calls[1][-1].images[0].path, str(photo))
            self.assertIn("Attached image input", llm.calls[1][-1].content)

    def test_photo_tool_result_is_metadata_only_for_text_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.tools.tool_settings["take_photo"]["enabled"] = True
            photo = Path(tmp) / "photo.jpg"
            photo.write_bytes(b"fake-jpeg")
            tools = ToolRegistry()
            tools.register(
                ToolSpec("take_photo", "camera", {"type": "object"}),
                lambda args: ToolResult(
                    name="take_photo",
                    ok=True,
                    content={"path": str(photo), "mime_type": "image/jpeg"},
                ),
            )
            llm = RecordingVisionLLM(
                [
                    json.dumps({"tool_calls": [{"name": "take_photo", "arguments": {}}]}),
                    "I captured a photo.",
                ],
                supports_images=False,
            )
            agent = VoiceAgent(
                cfg,
                llm=llm,
                stt=SilentSTT(),
                tts=SilentTTS(),
                tools=tools,
            )

            agent.handle_text("look at this", speak=False)

            self.assertEqual(llm.calls[1][-1].images, [])
            self.assertIn("only photo metadata", llm.calls[1][-1].content)

    def test_remember_command_writes_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.prompts.memory_path = str(Path(tmp) / "memory.md")
            cfg.ensure_dirs()
            agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())
            result = agent.handle_text("Remember this: I like quick answers.", speak=False)
            self.assertEqual(result.tool_results[0].name, "remember")
            self.assertIn("I'll remember", result.assistant_text)
            self.assertIn("I like quick answers", Path(cfg.prompts.memory_path).read_text())

    def test_explicit_language_switch_updates_runtime_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            profile_cfg = load_config()
            cfg.language = profile_cfg.language
            cfg.stt = profile_cfg.stt
            cfg.language.enabled = True
            cfg.tools.tool_settings["set_language"]["enabled"] = True
            agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())
            result = agent.handle_text("switch to Slovak", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "sk")
            self.assertEqual(cfg.agent.default_response_language, "Slovak")
            self.assertEqual(cfg.stt.language, "sk")
            self.assertEqual(cfg.stt.model, "small")
            self.assertEqual(cfg.stt.preview_model, "small")
            self.assertEqual(
                cfg.stt.final_model,
                "models/whisper/whisper-large-v3-turbo-sk-ct2-int8",
            )
            self.assertEqual(cfg.stt.final_beam_size, 3)
            self.assertTrue(cfg.tts.backend)
            self.assertEqual(result.assistant_text, "Ahoj")

    def test_generic_language_switch_uses_phrase_language_then_toggles(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            profile_cfg = load_config()
            cfg.language = profile_cfg.language
            cfg.stt = profile_cfg.stt
            cfg.language.enabled = True
            cfg.tools.tool_settings["set_language"]["enabled"] = True
            agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())

            result = agent.handle_text("switch language", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "sk")
            self.assertEqual(result.assistant_text, "Ahoj")

            result = agent.handle_text("switch language", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "en")
            self.assertTrue(cfg.audio.partial_transcript_enabled)
            self.assertFalse(cfg.audio.partial_transcript_use_as_final)
            self.assertTrue(cfg.tts.stream_speech)
            self.assertEqual(result.assistant_text, "Hi")

            result = agent.handle_text("prepni jazyk", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "sk")
            self.assertEqual(result.assistant_text, "Ahoj")

            result = agent.handle_text("prepni jazyk", speak=False)
            self.assertEqual(result.tool_results[0].name, "set_language")
            self.assertEqual(cfg.runtime.default_language, "en")
            self.assertEqual(result.assistant_text, "Hi")

    def test_stt_phase_configs_use_preview_and_final_models(self):
        cfg = _pipeline_cfg()
        agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())

        preview = agent._stt_config_for_phase("preview")
        final = agent._stt_config_for_phase("final")

        self.assertEqual(preview.model, "small")
        self.assertEqual(preview.beam_size, 1)
        self.assertEqual(final.model, "distil-large-v3")
        self.assertEqual(final.beam_size, 3)

    def test_remote_phase_fallbacks_use_local_preview_model(self):
        cfg = _pipeline_cfg()
        cfg.stt.preview_use_remote = True
        cfg.stt.final_use_remote = True
        agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())

        with patch("wheatley.pipeline.remote_stt_available", return_value=False):
            preview = agent._stt_config_for_phase("preview")
            final = agent._stt_config_for_phase("final")

        self.assertEqual(preview.backend, "faster_whisper")
        self.assertEqual(preview.model, "small")
        self.assertEqual(final.backend, "faster_whisper")
        self.assertEqual(final.model, "small")

    def test_prompt_injects_user_instructions_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.prompts.system_path = str(Path(tmp) / "system.md")
            cfg.prompts.user_path = str(Path(tmp) / "user.md")
            cfg.prompts.memory_path = str(Path(tmp) / "memory.md")
            cfg.ensure_dirs()
            Path(cfg.prompts.system_path).write_text(
                "System. Reply in {{DEFAULT_RESPONSE_LANGUAGE}}.",
                encoding="utf-8",
            )
            Path(cfg.prompts.user_path).write_text("Prefer direct answers.", encoding="utf-8")
            Path(cfg.prompts.memory_path).write_text("- User likes fast starts.", encoding="utf-8")
            agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())
            prompt = build_system_prompt(cfg, agent.tools)
            self.assertIn("System. Reply in English.", prompt)
            self.assertIn("Prefer direct answers.", prompt)
            self.assertIn("User likes fast starts.", prompt)

    def test_stream_nonstream_tts_does_not_speak_internal_tool_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.tts.enabled = True
            cfg.tts.stream_speech = False
            cfg.tools.tool_settings["set_eye_expression"] = {
                **cfg.tools.tool_settings.get("set_eye_expression", {}),
                "enabled": True,
            }
            tts = RecordingTTS()
            llm = SequenceLLM(
                [
                    '{"tool_calls":[{"name":"set_eye_expression","arguments":{"expression":"happy"}}]}',
                    "Tu je normalna odpoved.",
                ]
            )
            agent = VoiceAgent(cfg, llm=llm, stt=SilentSTT(), tts=tts)
            tokens = []

            result = agent.handle_text_stream(
                "Povedz mi 10 vtipov.",
                speak=True,
                on_token=tokens.append,
            )

            self.assertEqual(tts.spoken, ["Setting expression...", "Tu je normalna odpoved."])
            self.assertEqual(result.tool_results[0].name, "set_eye_expression")
            self.assertNotIn("tool_calls", "".join(tokens))
            self.assertIn("Tu je normalna odpoved.", result.assistant_text)

    def test_stream_shows_python_interpreter_code_preview_before_tool_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.tools.tool_settings["python_interpreter"] = {
                **cfg.tools.tool_settings.get("python_interpreter", {}),
                "enabled": True,
            }
            llm = SequenceLLM(
                [
                    json.dumps(
                        {
                            "tool_calls": [
                                {
                                    "name": "python_interpreter",
                                    "arguments": {
                                        "code": "x = 2\nresult = x * 3",
                                    },
                                }
                            ]
                        }
                    ),
                    "Done.",
                ]
            )
            agent = VoiceAgent(cfg, llm=llm, stt=SilentSTT(), tts=SilentTTS())
            tokens: list[str] = []

            result = agent.handle_text_stream(
                "compute something",
                speak=False,
                on_token=tokens.append,
            )

            joined = "".join(tokens)
            self.assertIn("x = 2", joined)
            self.assertIn("result = x * 3", joined)
            self.assertNotIn("tool_calls", joined)
            self.assertEqual(result.tool_results[0].name, "python_interpreter")

    def test_tool_start_announcements_use_active_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.language = load_config().language
            cfg.language.enabled = True
            cfg.language.default = "sk"
            cfg.tts.enabled = True
            cfg.tools.tool_settings["run_safe_cli_tool"] = {
                **cfg.tools.tool_settings.get("run_safe_cli_tool", {}),
                "enabled": True,
            }
            cfg.tools.tool_settings["web_search"]["enabled"] = True

            tools = ToolRegistry()
            for name in ["remember", "run_safe_cli_tool", "web_search"]:
                tools.register(
                    ToolSpec(name=name, description=name, parameters={"type": "object"}),
                    lambda args, tool_name=name: ToolResult(
                        name=tool_name,
                        ok=True,
                        content={"args": args},
                    ),
                )
            llm = SequenceLLM(
                [
                    json.dumps(
                        {
                            "tool_calls": [
                                {"name": "remember", "arguments": {"memory": "x"}},
                                {"name": "run_safe_cli_tool", "arguments": {"name": "x"}},
                                {"name": "web_search", "arguments": {"query": "x"}},
                            ]
                        }
                    ),
                    "hotovo",
                ]
            )
            tts = RecordingTTS()
            events = []
            agent = VoiceAgent(
                cfg,
                llm=llm,
                stt=SilentSTT(),
                tts=tts,
                tools=tools,
                on_tool_start=lambda name, message: events.append((name, message)),
            )

            result = agent.handle_text("urob naradie", speak=True)

            expected_tools = [
                "remember",
                "run_safe_cli_tool",
                "web_search",
            ]
            expected_messages = [
                cfg.tools.tool_settings[name]["start_messages"]["sk"]
                for name in expected_tools
            ]
            self.assertEqual([name for name, _ in events], expected_tools)
            self.assertEqual([message for _, message in events], expected_messages)
            self.assertEqual(tts.spoken, [*expected_messages, "hotovo"])
            self.assertEqual([item.name for item in result.tool_results], expected_tools)

    def test_memory_refresh_speaks_startup_status_and_tools_before_done_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.profile_dir = str(Path(tmp) / "profile")
            cfg.tts.enabled = True
            cfg.memory.auto_enabled = True
            cfg.memory.full_rewrite_requires_online = True
            cfg.memory.full_rewrite_interval_days = 1
            cfg.memory.max_turns_per_update = 12
            cfg.memory.max_candidates_for_rewrite = 10
            cfg.memory.max_total_words = 100
            cfg.memory.max_stable_facts = 5
            cfg.memory.max_preferences = 5
            cfg.memory.max_current_projects = 5
            cfg.memory.max_recent_context = 5
            for setting in cfg.tools.tool_settings.values():
                setting["enabled"] = False
            for name in ("remember", "web_search", "calculator"):
                cfg.tools.tool_settings[name]["enabled"] = True
            cfg.tools.tool_settings["remember"]["labels"] = {
                "en": "memory",
                "sk": "pamäť",
            }
            cfg.tools.tool_settings["web_search"]["labels"] = {
                "en": "web search",
                "sk": "webové vyhľadávanie",
            }
            cfg.tools.tool_settings["calculator"]["labels"] = {
                "en": "calculator",
                "sk": "kalkulačka",
            }
            _write_turn(cfg.runtime.turn_log, "I like quiet memory updates.")
            tts = BlockingToolsTTS()
            with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test-key"}), patch(
                "wheatley.tools.web.urllib.request.urlopen",
                return_value=FakeHTTPResponse(),
            ):
                agent = VoiceAgent(
                    cfg,
                    llm=SequenceLLM(
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
                    ),
                    stt=SilentSTT(),
                    tts=tts,
                )
            notices = []

            worker = threading.Thread(
                target=agent.refresh_auto_memory,
                kwargs={
                    "notify_memory": notices.append,
                    "speak_memory": True,
                    "start_messages": [
                        "using smarter online model and local speech recognition."
                    ],
                },
            )
            worker.start()

            self.assertTrue(tts.tools_started.wait(timeout=1))
            time.sleep(0.05)
            self.assertEqual(
                notices,
                [
                    "wait, I'm updating my memory...",
                    "using smarter online model and local speech recognition.",
                    "Current tools are: calculator, memory, and web search.",
                ],
            )

            tts.release_tools.set()
            worker.join(timeout=1)

            self.assertFalse(worker.is_alive())
            self.assertEqual(
                notices,
                [
                    "wait, I'm updating my memory...",
                    "using smarter online model and local speech recognition.",
                    "Current tools are: calculator, memory, and web search.",
                    "my memory was updated.",
                ],
            )
            self.assertEqual(
                tts.spoken,
                [
                    "wait, I'm updating my memory...",
                    "using smarter online model and local speech recognition.",
                    "Current tools are: calculator, memory, and web search.",
                    "my memory was updated.",
                ],
            )

    def test_current_tools_message_uses_active_language_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            cfg.language = load_config().language
            cfg.language.enabled = True
            cfg.language.default = "sk"
            cfg.runtime.default_language = "sk"
            for setting in cfg.tools.tool_settings.values():
                setting["enabled"] = False
            for name in ("remember", "web_search", "calculator", "set_language"):
                cfg.tools.tool_settings[name]["enabled"] = True
            cfg.tools.tool_settings["remember"]["labels"] = {
                "en": "memory",
                "sk": "pamäť",
            }
            cfg.tools.tool_settings["web_search"]["labels"] = {
                "en": "web search",
                "sk": "webové vyhľadávanie",
            }
            cfg.tools.tool_settings["calculator"]["labels"] = {
                "en": "calculator",
                "sk": "kalkulačka",
            }
            cfg.tools.tool_settings["set_language"]["labels"] = {
                "en": "language switching",
                "sk": "prepínanie jazyka",
            }
            with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test-key"}), patch(
                "wheatley.tools.web.urllib.request.urlopen",
                return_value=FakeHTTPResponse(),
            ):
                agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())

            self.assertEqual(
                _current_tools_message(cfg, agent.tools),
                (
                    "Aktuálne nástroje sú: kalkulačka, pamäť, "
                    "prepínanie jazyka a webové vyhľadávanie."
                ),
            )

    def test_current_tools_message_keeps_duplicate_display_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _pipeline_cfg(tmp)
            for setting in cfg.tools.tool_settings.values():
                setting["enabled"] = False
            cfg.tools.tool_settings["calculator"]["enabled"] = True
            cfg.tools.tool_settings["remember"]["enabled"] = True
            cfg.tools.tool_settings["calculator"]["labels"] = {"en": "same"}
            cfg.tools.tool_settings["remember"]["labels"] = {"en": "same"}
            agent = VoiceAgent(cfg, stt=SilentSTT(), tts=SilentTTS())

            self.assertEqual(
                _current_tools_message(cfg, agent.tools),
                "Current tools are: same and same.",
            )

    def test_tool_name_joining_uses_localized_conjunction(self):
        self.assertEqual(_join_tool_names(["memory"], "en", "and"), "memory")
        self.assertEqual(
            _join_tool_names(["memory", "web search"], "en", "and"),
            "memory and web search",
        )
        self.assertEqual(
            _join_tool_names(["memory", "web search", "calculator"], "en", "and"),
            "memory, web search, and calculator",
        )
        self.assertEqual(
            _join_tool_names(["pamäť", "kalkulačka"], "sk", "a"),
            "pamäť a kalkulačka",
        )


if __name__ == "__main__":
    unittest.main()
