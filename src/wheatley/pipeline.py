from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from wheatley.audio.interrupt import SpeechInterruptMonitor
from wheatley.config import Config
from wheatley.language import (
    apply_configured_language,
    match_language_switch,
    model_selection_message,
    normalize_language_code,
    online_llm_model,
    read_language_state,
    set_language_state,
)
from wheatley.llm.backends import build_llm, remote_llm_available
from wheatley.llm.base import LLMBackend, LLMImage, LLMMessage
from wheatley.memory import refresh_auto_memory
from wheatley.prompting import build_system_prompt
from wheatley.runtime_stats import LatencyStats
from wheatley.stt.backends import build_stt, remote_stt_available
from wheatley.stt.base import STTBackend, Transcription
from wheatley.tools.audit import log_tool_event
from wheatley.tools.announcements import tool_start_message
from wheatley.tools.builtins import build_registry
from wheatley.tools.parser import parse_tool_calls
from wheatley.tools.registry import ToolCall, ToolRegistry, ToolResult
from wheatley.tts.backends import build_tts
from wheatley.tts.base import TTSBackend
from wheatley.tts.streaming import StreamingSpeaker


@dataclass
class TurnResult:
    user_text: str
    assistant_text: str
    tool_results: List[ToolResult]
    duration_seconds: float = 0.0


@dataclass
class ModelSelection:
    mode: str
    message: str
    stt_mode: str = "local"


MEMORY_NOTICE_START_KINDS = {"update_start", "consolidate_start"}
MEMORY_NOTICE_DONE_KINDS = {"update_done", "consolidate_done"}


class VoiceAgent:
    def __init__(
        self,
        cfg: Config,
        llm: Optional[LLMBackend] = None,
        stt: Optional[STTBackend] = None,
        tts: Optional[TTSBackend] = None,
        tools: Optional[ToolRegistry] = None,
        on_tool_start: Optional[Callable[..., None]] = None,
    ):
        self.cfg = cfg
        apply_configured_language(self.cfg, read_language_state(self.cfg))
        self.llm = llm or build_llm(cfg.llm)
        self.stt_lock = threading.Lock()
        self._external_stt = stt is not None
        self.stt = stt or build_stt(cfg.stt)
        self._stt_backends: dict[tuple, STTBackend] = {}
        self.tts = tts or build_tts(cfg)
        self.tools = tools or build_registry(cfg)
        self.on_tool_start = on_tool_start
        self.latency_stats = LatencyStats(Path(cfg.runtime.state_dir) / "latency_stats.json")
        self.history: List[LLMMessage] = []
        self.model_selection = ModelSelection(
            "offline",
            model_selection_message(self.cfg, "offline", "local"),
            "local",
        )

    def reset_chat(
        self,
        refresh_memory: bool = True,
        notify_memory: Optional[Callable[[str], None]] = None,
        speak_memory: bool = False,
    ) -> ModelSelection:
        self.history.clear()
        selection = self.select_chat_model()
        if refresh_memory:
            self.refresh_auto_memory(
                notify_memory=notify_memory,
                speak_memory=speak_memory,
            )
        return selection

    def select_chat_model(self) -> ModelSelection:
        remote = self.cfg.llm.remote
        stt_mode = self.select_stt_mode()
        if remote.enabled and remote_llm_available(remote):
            remote_cfg = replace(
                self.cfg.llm,
                backend=remote.backend,
                base_url=remote.base_url,
                model=online_llm_model(self.cfg),
                api_key=remote.api_key,
                timeout_seconds=remote.request_timeout_seconds,
            )
            self.llm = build_llm(remote_cfg)
            self.model_selection = ModelSelection(
                "online",
                model_selection_message(self.cfg, "online", stt_mode),
                stt_mode,
            )
            return self.model_selection
        self.llm = build_llm(self.cfg.llm)
        self.model_selection = ModelSelection(
            "offline",
            model_selection_message(self.cfg, "offline", stt_mode),
            stt_mode,
        )
        return self.model_selection

    def select_stt_mode(self) -> str:
        backend = self.cfg.stt.backend.lower().replace("-", "_")
        if backend not in {"remote", "remote_fallback"}:
            return "local"
        return "remote" if remote_stt_available(self.cfg.stt) else "local"

    def refresh_auto_memory(
        self,
        notify_memory: Optional[Callable[[str], None]] = None,
        speak_memory: bool = False,
        start_messages: Optional[List[str]] = None,
    ) -> None:
        status_speech: Optional[threading.Thread] = None
        startup_messages = [message for message in (start_messages or []) if message]
        startup_messages_announced = False
        tools_announced = False

        def speak_status(message: str) -> None:
            if not (speak_memory and self.cfg.tts.enabled and message):
                return
            try:
                self.tts.speak(message)
            except Exception:
                return

        def speak_status_sequence(messages: List[str]) -> None:
            for message in messages:
                speak_status(message)

        def speak_status_async(messages: List[str]) -> None:
            nonlocal status_speech
            messages = [message for message in messages if message]
            if not (speak_memory and self.cfg.tts.enabled and messages):
                return
            status_speech = threading.Thread(
                target=speak_status_sequence,
                args=(messages,),
                daemon=True,
            )
            status_speech.start()

        def wait_status_speech() -> None:
            nonlocal status_speech
            if status_speech:
                status_speech.join()
                status_speech = None

        def emit(message: str, kind: str = "") -> None:
            nonlocal startup_messages_announced, tools_announced
            if kind in MEMORY_NOTICE_DONE_KINDS:
                wait_status_speech()
            if notify_memory:
                notify_memory(message)
            speak_status(message)
            if kind in MEMORY_NOTICE_START_KINDS:
                background_messages = []
                if not startup_messages_announced:
                    startup_messages_announced = True
                    for startup_message in startup_messages:
                        if notify_memory:
                            notify_memory(startup_message)
                        background_messages.append(startup_message)
                if tools_announced:
                    speak_status_async(background_messages)
                    return
                tools_message = _current_tools_message(self.cfg, self.tools)
                if not tools_message:
                    speak_status_async(background_messages)
                    return
                tools_announced = True
                if notify_memory:
                    notify_memory(tools_message)
                background_messages.append(tools_message)
                speak_status_async(background_messages)

        try:
            refresh_auto_memory(
                self.cfg,
                self.llm,
                self.model_selection.mode,
                notify=emit,
            )
        finally:
            wait_status_speech()

    def current_tools_message(self) -> str:
        return _current_tools_message(self.cfg, self.tools)

    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        return self.transcribe_final(audio_path)

    def transcribe_preview(self, audio_path: Optional[Path] = None) -> Transcription:
        return self._stt_for_phase("preview").transcribe(audio_path)

    def transcribe_final(self, audio_path: Optional[Path] = None) -> Transcription:
        final_backend = self._stt_for_phase("final")
        try:
            return final_backend.transcribe(audio_path)
        except Exception:
            if self._final_uses_preview_fallback():
                raise
            return self._stt_for_phase("preview").transcribe(audio_path)

    def _stt_for_phase(self, phase: str) -> STTBackend:
        if self._external_stt:
            return self.stt
        cfg = self._stt_config_for_phase(phase)
        key = _stt_cache_key(cfg)
        with self.stt_lock:
            if key not in self._stt_backends:
                self._stt_backends[key] = build_stt(cfg)
            return self._stt_backends[key]

    def _stt_config_for_phase(self, phase: str):
        if phase == "preview":
            return _preview_stt_config(self.cfg)
        if phase == "final":
            return _final_stt_config(self.cfg)
        raise ValueError(f"Unsupported STT phase: {phase}")

    def _final_uses_preview_fallback(self) -> bool:
        return _stt_cache_key(_final_stt_config(self.cfg)) == _stt_cache_key(
            _preview_stt_config(self.cfg)
        )

    def restore_turn_history(self, turns: List[dict]) -> None:
        self.history.clear()
        for turn in turns:
            self.history.append(LLMMessage("user", str(turn["user_text"])))
            self.history.append(
                LLMMessage("assistant", str(turn["assistant_text"]))
            )

    def set_language(self, requested_language: str) -> ToolResult:
        ok, content = set_language_state(self.cfg, requested_language)
        if ok:
            with self.stt_lock:
                if not self._external_stt:
                    self.stt = build_stt(self.cfg.stt)
                self._stt_backends.clear()
            self.tts = build_tts(self.cfg)
            if self.model_selection.mode == "online":
                self.select_chat_model()
        return ToolResult(name="set_language", ok=ok, content=content)

    def handle_text(self, text: str, speak: bool = True) -> TurnResult:
        started_at = time.perf_counter()
        text = text.strip()
        if not text:
            return TurnResult(user_text="", assistant_text="", tool_results=[])

        routed_calls = _route_deterministic_tools(text, self.cfg)
        if routed_calls and self.cfg.tools.enabled:
            tool_results = self._execute_tool_calls(
                routed_calls,
                source="direct_route",
                speak=speak,
            )
            direct_answer = _format_direct_tool_answer(tool_results)
            if direct_answer:
                result = self._finish_turn(text, direct_answer, tool_results, started_at)
                if speak:
                    self._speak_text(direct_answer)
                return result
            messages = self._messages_for_turn(text)
            messages.append(self._tool_results_message(tool_results))
            final_text = self.llm.complete(messages).content.strip()
            result = self._finish_turn(text, final_text, tool_results, started_at)
            if speak and final_text:
                self._speak_text(final_text)
            return result

        messages = self._messages_for_turn(text)
        first = self.llm.complete(messages)
        tool_calls = parse_tool_calls(first.content) if self.cfg.tools.enabled else []
        tool_results: List[ToolResult] = []
        final_text = first.content.strip()

        if tool_calls:
            tool_results = self._execute_tool_calls(
                tool_calls,
                source="llm",
                speak=speak,
            )
            direct_answer = _format_direct_tool_answer(tool_results)
            if direct_answer:
                final_text = direct_answer
            else:
                messages.append(LLMMessage(role="assistant", content=first.content))
                messages.append(self._tool_results_message(tool_results))
                second = self.llm.complete(messages)
                final_text = second.content.strip()

        result = self._finish_turn(text, final_text, tool_results, started_at)
        if speak and final_text:
            self._speak_text(final_text)
        return result

    def handle_text_stream(
        self,
        text: str,
        speak: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> TurnResult:
        started_at = time.perf_counter()
        text = text.strip()
        if not text:
            return TurnResult(user_text="", assistant_text="", tool_results=[])

        routed_calls = _route_deterministic_tools(text, self.cfg)
        if routed_calls and self.cfg.tools.enabled:
            tool_results = self._execute_tool_calls(
                routed_calls,
                source="direct_route",
                speak=speak,
            )
            direct_answer = _format_direct_tool_answer(tool_results)
            if direct_answer:
                if on_token:
                    on_token(direct_answer)
                if speak:
                    self._speak_text(direct_answer)
                return self._finish_turn(text, direct_answer, tool_results, started_at)
            messages = self._messages_for_turn(text)
            messages.append(self._tool_results_message(tool_results))
            final_text = self._stream_final_answer(messages, speak, on_token)
            return self._finish_turn(text, final_text, tool_results, started_at)

        messages = self._messages_for_turn(text)
        first_text = self._stream_final_answer(messages, speak, on_token)
        tool_calls = parse_tool_calls(first_text) if self.cfg.tools.enabled else []
        tool_results: List[ToolResult] = []
        final_text = first_text

        if tool_calls:
            tool_results = self._execute_tool_calls(
                tool_calls,
                source="llm",
                speak=speak,
            )
            direct_answer = _format_direct_tool_answer(tool_results)
            if direct_answer:
                if on_token and not _is_language_switch_tool_result(tool_results):
                    on_token(direct_answer)
                if speak:
                    self._speak_text(direct_answer)
                final_text = direct_answer
            else:
                messages.append(LLMMessage(role="assistant", content=first_text))
                messages.append(self._tool_results_message(tool_results))
                final_text = self._stream_final_answer(messages, speak, on_token)

        return self._finish_turn(text, final_text, tool_results, started_at)

    def handle_idle_speech(
        self,
        instruction: str,
        speak: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> TurnResult:
        started_at = time.perf_counter()
        instruction = instruction.strip()
        if not instruction:
            return TurnResult(user_text="", assistant_text="", tool_results=[])

        messages = self._messages_for_idle_speech(instruction)
        final_text = self.llm.complete(messages).content.strip()
        fallback_needed = False
        if self.cfg.tools.enabled and parse_tool_calls(final_text):
            fallback_needed = True
        if not final_text:
            fallback_needed = True
        if fallback_needed:
            final_text = _idle_speech_fallback_text()
        else:
            final_text = _compact_idle_speech_text(final_text)
        _emit_idle_console_text(final_text, on_token)
        if speak and final_text:
            self._speak_text(final_text)
        return self._finish_idle_speech(final_text, started_at)

    def _messages_for_turn(self, text: str) -> List[LLMMessage]:
        system = build_system_prompt(self.cfg, self.tools)
        trimmed_history = self.history[-self.cfg.llm.context_turns * 2 :]
        return (
            [LLMMessage("system", system)]
            + trimmed_history
            + [LLMMessage("user", text)]
        )

    def _messages_for_idle_speech(self, instruction: str) -> List[LLMMessage]:
        system = build_system_prompt(self.cfg, self.tools)
        trimmed_history = self.history[-self.cfg.llm.context_turns * 2 :]
        return (
            [LLMMessage("system", system)]
            + trimmed_history
            + [LLMMessage("user", instruction)]
        )

    def _tool_results_message(self, tool_results: List[ToolResult]) -> LLMMessage:
        return _tool_results_message(
            tool_results,
            self.cfg,
            attach_images=self.llm.supports_images(),
        )

    def _execute_tool_calls(
        self,
        calls: List[ToolCall],
        *,
        source: str,
        speak: bool,
    ) -> List[ToolResult]:
        results: List[ToolResult] = []
        for index, call in enumerate(calls):
            if not self._tool_call_available(call):
                continue
            started_at = time.perf_counter()
            self._announce_tool_start(call, source=source, speak=speak)
            if call.name == "set_language":
                language = str(call.arguments.get("language", ""))
                result = self.set_language(language)
            else:
                result = self.tools.execute(call)
            results.append(result)
            log_tool_event(
                self.cfg.runtime.tool_log,
                call,
                result,
                source=source,
                duration_seconds=time.perf_counter() - started_at,
                call_index=index,
            )
        return results

    def _tool_call_available(self, call: ToolCall) -> bool:
        if call.name == "set_language":
            return self.cfg.language.enabled and self.cfg.tools.is_tool_enabled(
                call.name
            )
        return self.cfg.tools.is_tool_enabled(call.name) and self.tools.has_tool(
            call.name
        )

    def _announce_tool_start(self, call: ToolCall, source: str, speak: bool) -> None:
        tool_name = call.name
        message = tool_start_message(self.cfg, tool_name)
        if not message:
            return
        if self.on_tool_start:
            try:
                self.on_tool_start(tool_name, message, source, call.arguments)
            except TypeError:
                try:
                    self.on_tool_start(tool_name, message, source)
                except TypeError:
                    self.on_tool_start(tool_name, message)
        if speak:
            self.tts.speak(message)

    def _append_history(self, user_text: str, assistant_text: str) -> None:
        self.history.append(LLMMessage("user", user_text))
        self.history.append(LLMMessage("assistant", assistant_text))

    def _finish_turn(
        self,
        user_text: str,
        assistant_text: str,
        tool_results: List[ToolResult],
        started_at: float,
    ) -> TurnResult:
        self._append_history(user_text, assistant_text)
        result = TurnResult(
            user_text=user_text,
            assistant_text=assistant_text,
            tool_results=tool_results,
            duration_seconds=round(time.perf_counter() - started_at, 4),
        )
        self._log_turn(result)
        return result

    def _finish_idle_speech(
        self,
        assistant_text: str,
        started_at: float,
    ) -> TurnResult:
        result = TurnResult(
            user_text="",
            assistant_text=assistant_text,
            tool_results=[],
            duration_seconds=round(time.perf_counter() - started_at, 4),
        )
        if not assistant_text:
            return result
        self.history.append(LLMMessage("assistant", assistant_text))
        self._log_turn(result, source="idle")
        return result

    def _stream_final_answer(
        self,
        messages: List[LLMMessage],
        speak: bool,
        on_token: Optional[Callable[[str], None]],
    ) -> str:
        chunks: List[str] = []
        pending = ""
        visible_started = False
        hold_for_possible_tool_json = False
        python_preview_emitted_chars = 0
        first_chunk_at: Optional[float] = None
        initial_words = self.latency_stats.recommended_initial_words(
            min_words=self.cfg.tts.stream_initial_min_words,
            max_words=self.cfg.tts.stream_max_words,
            adaptive=self.cfg.tts.adaptive_streaming,
        )

        interrupt_event = threading.Event()
        monitor_enabled = speak and self._speech_interrupt_available()
        with SpeechInterruptMonitor(
            self.cfg.audio,
            self.transcribe,
            interrupt_event,
            enabled=monitor_enabled,
        ) as interrupt_monitor:
            with StreamingSpeaker(
                self.tts,
                enabled=speak and self.cfg.tts.stream_speech,
                min_words=self.cfg.tts.stream_min_words,
                max_words=self.cfg.tts.stream_max_words,
                initial_min_words=initial_words,
                feedback_min_words=self.cfg.tts.stream_feedback_min_words,
                max_initial_wait_seconds=self.cfg.tts.stream_max_initial_wait_seconds,
                max_inter_chunk_wait_seconds=(
                    self.cfg.tts.stream_max_inter_chunk_wait_seconds
                ),
                playback_prebuffer_chunks=self.cfg.tts.stream_playback_prebuffer_chunks,
                playback_prebuffer_max_wait_seconds=(
                    self.cfg.tts.stream_playback_prebuffer_max_wait_seconds
                ),
                on_spoken=self._record_spoken_segment,
                stop_event=interrupt_event,
                pause_event=interrupt_monitor.pause_event,
            ) as speaker:
                for chunk in self.llm.stream_complete(messages):
                    if interrupt_event.is_set():
                        break
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    chunks.append(chunk)
                    if not visible_started:
                        pending += chunk
                        stripped = pending.lstrip()
                        if not stripped:
                            continue
                        if (
                            stripped.startswith("{")
                            or stripped.startswith("[")
                            or stripped.startswith("```")
                        ):
                            hold_for_possible_tool_json = True
                            preview = _extract_python_interpreter_code_preview(pending)
                            if preview:
                                new_preview = preview[python_preview_emitted_chars:]
                                if new_preview:
                                    _emit_token(new_preview, on_token, speaker)
                                    python_preview_emitted_chars = len(preview)
                            continue
                        visible_started = True
                        _emit_token(pending, on_token, speaker)
                        pending = ""
                        continue
                    if hold_for_possible_tool_json:
                        preview = _extract_python_interpreter_code_preview(pending)
                        if preview:
                            new_preview = preview[python_preview_emitted_chars:]
                            if new_preview:
                                _emit_token(new_preview, on_token, speaker)
                                python_preview_emitted_chars = len(preview)
                        continue
                    _emit_token(chunk, on_token, speaker)

                final_text = "".join(chunks).strip()
                if (
                    hold_for_possible_tool_json
                    and not interrupt_event.is_set()
                    and not parse_tool_calls(final_text)
                ):
                    _emit_token(pending, on_token, speaker)

        stream_ended_at = time.perf_counter()
        self.latency_stats.record_llm(
            words=_count_words(final_text),
            duration_seconds=stream_ended_at - (first_chunk_at or stream_ended_at),
        )
        if (
            speak
            and not self.cfg.tts.stream_speech
            and final_text
            and not (self.cfg.tools.enabled and parse_tool_calls(final_text))
        ):
            started_at = time.perf_counter()
            self._speak_text(final_text)
            self._record_spoken_segment(final_text, time.perf_counter() - started_at)
        return final_text

    def _speak_text(self, text: str) -> bool:
        if not text:
            return False
        interrupt_event = threading.Event()
        monitor_enabled = self._speech_interrupt_available()
        with SpeechInterruptMonitor(
            self.cfg.audio,
            self.transcribe,
            interrupt_event,
            enabled=monitor_enabled,
        ):
            self.tts.speak(text)
        return interrupt_event.is_set()

    def _speech_interrupt_available(self) -> bool:
        return (
            self.cfg.tts.enabled
            and self.cfg.tts.playback
            and self.cfg.audio.speech_interrupt_enabled
            and self.cfg.stt.backend.lower().replace("-", "_") != "keyboard"
        )

    def _record_spoken_segment(self, segment: str, duration_seconds: float) -> None:
        self.latency_stats.record_tts(_count_words(segment), duration_seconds)

    def _log_turn(self, result: TurnResult, source: str = "user") -> None:
        path = Path(self.cfg.runtime.turn_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model_name": self._active_llm_model(),
            "user_text": None if source == "idle" else result.user_text,
            "assistant_text": result.assistant_text,
            "tool_results": [asdict(item) for item in result.tool_results],
        }
        if source != "user":
            record["source"] = source
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _active_llm_model(self) -> str:
        if self.model_selection.mode == "online":
            return online_llm_model(self.cfg)
        return self.cfg.llm.model


def _tool_results_message(
    tool_results: List[ToolResult],
    cfg: Config,
    attach_images: bool = False,
) -> LLMMessage:
    images = _tool_result_images(tool_results) if attach_images else []
    attachment_note = ""
    if _has_photo_result(tool_results):
        if images:
            attachment_note = "\nAttached image input: latest camera photo."
        else:
            attachment_note = (
                "\nPhoto note: the active LLM model is not recognized as "
                "image-capable, so only photo metadata is available."
            )
    return LLMMessage(
        role="user",
        content="Tool results: "
        + json.dumps([asdict(result) for result in tool_results])
        + attachment_note
        + "\nNow answer the user in natural language. Match the requested length."
        + f"\nCurrent reply language: {cfg.agent.default_response_language}.",
        images=images,
    )


def _has_photo_result(tool_results: List[ToolResult]) -> bool:
    return any(result.name == "take_photo" and result.ok for result in tool_results)


def _tool_result_images(tool_results: List[ToolResult]) -> List[LLMImage]:
    images: List[LLMImage] = []
    for result in tool_results:
        if result.name != "take_photo" or not result.ok:
            continue
        path = str(result.content.get("path", "")).strip()
        if not path:
            continue
        image_path = Path(path)
        try:
            if not image_path.exists() or image_path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        images.append(
            LLMImage(
                path=str(image_path),
                mime_type=str(result.content.get("mime_type") or "image/jpeg"),
                detail="low",
            )
        )
    return images


def _current_tools_message(cfg: Config, tools: ToolRegistry) -> str:
    names = {spec.name for spec in tools.specs()}
    if cfg.language.enabled and cfg.tools.is_tool_enabled("set_language"):
        names.add("set_language")
    if not names:
        return ""

    language = normalize_language_code(cfg, cfg.runtime.default_language) or "en"
    displayed = [
        _tool_display_name(name, setting, language)
        for name, setting in cfg.tools.tool_settings.items()
        if name in names
    ]
    if not displayed:
        return ""
    template = _localized_config_text(cfg.tools.current_tools_message, language)
    conjunction = _localized_config_text(cfg.tools.tool_list_conjunction, language)
    if not template or not conjunction:
        return ""
    return template.format(tools=_join_tool_names(displayed, language, conjunction))


def _join_tool_names(names: List[str], language: str, conjunction: str) -> str:
    if len(names) <= 2:
        return f" {conjunction} ".join(names)
    if language == "en":
        return ", ".join(names[:-1]) + f", {conjunction} {names[-1]}"
    return ", ".join(names[:-1]) + f" {conjunction} {names[-1]}"


def _localized_config_text(values: dict, language: str) -> str:
    for code in (language, "en"):
        value = values.get(code)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _tool_display_name(name: str, setting: dict, language: str) -> str:
    labels = setting.get("labels", {})
    if isinstance(labels, dict):
        for code in (language, "en"):
            label = labels.get(code)
            if isinstance(label, str) and label.strip():
                return label.strip()
    return name.replace("_", " ")


def _emit_token(
    text: str,
    on_token: Optional[Callable[[str], None]],
    speaker: StreamingSpeaker,
) -> None:
    if not text:
        return
    if on_token:
        on_token(text)
    speaker.feed(text)


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def _idle_speech_fallback_text() -> str:
    return "Tiny idle note: silence is just the room doing lossless compression."


def _compact_idle_speech_text(text: str) -> str:
    lines = [
        re.sub(r"^\s*[-*]\s+", "", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]
    text = " ".join(lines)
    if not text:
        return ""
    sentences = [
        part.strip()
        for part in re.findall(r".+?(?:[.!?](?=\s|$)|$)", text)
        if part.strip()
    ]
    compact = " ".join(sentences[:2]).strip() if sentences else text.strip()
    words = compact.split()
    max_words = 42
    if len(words) > max_words:
        compact = " ".join(words[:max_words]).rstrip(" ,;:") + "."
    return compact


def _emit_idle_console_text(
    text: str,
    on_token: Optional[Callable[[str], None]],
) -> None:
    if not text or not on_token:
        return
    for token in re.findall(r"\S+\s*", text):
        on_token(token)


def _route_deterministic_tools(text: str, cfg: Config) -> List[ToolCall]:
    calls: List[ToolCall] = []
    language = match_language_switch(cfg, text)
    if language and cfg.tools.is_tool_enabled("set_language"):
        calls.append(ToolCall("set_language", {"language": language}))
        return calls
    memory = _extract_memory_text(text)
    if memory and cfg.tools.is_tool_enabled("remember"):
        calls.append(ToolCall("remember", {"memory": memory}))
        return calls
    return calls


def _format_direct_tool_answer(tool_results: List[ToolResult]) -> str:
    if len(tool_results) != 1:
        return ""
    result = tool_results[0]
    content = result.content
    if result.name != "calculator":
        if result.name == "set_language":
            if result.ok:
                return str(content.get("confirmation") or "Language switched.")
            return f"I could not switch language: {content.get('error', 'unknown error')}."
        if result.name == "remember":
            if result.ok:
                return "I'll remember that."
            return f"I could not save that memory: {content.get('error', 'unknown error')}."
        return ""
    if not result.ok:
        return f"I could not calculate that: {content.get('error', 'unknown error')}."
    return f"The result is {content.get('result_display', content.get('result'))}."


def _is_language_switch_tool_result(tool_results: List[ToolResult]) -> bool:
    return (
        len(tool_results) == 1
        and tool_results[0].name == "set_language"
        and tool_results[0].ok
    )


def _extract_memory_text(text: str) -> str:
    cleaned = text.strip().strip("\"'").rstrip(".?!").strip()
    match = re.match(
        r"(?i)^(?:please\s+)?remember(?:\s+(?:this|that))?(?:\s*[:,]\s*|\s+)(.+)$",
        cleaned,
    )
    if not match:
        return ""
    memory = match.group(1).strip().strip("\"'")
    if memory.lower() in {"this", "that"}:
        return ""
    return memory


def _extract_python_interpreter_code_preview(text: str) -> str:
    tool_match = re.search(
        r'"name"\s*:\s*"python_interpreter"',
        text,
    )
    if not tool_match:
        return ""
    code_match = re.search(r'"code"\s*:\s*"', text[tool_match.start() :])
    if not code_match:
        return ""
    start = tool_match.start() + code_match.end()
    return _decode_json_string_fragment(text[start:])


def _decode_json_string_fragment(fragment: str) -> str:
    decoded: List[str] = []
    index = 0
    while index < len(fragment):
        char = fragment[index]
        if char == '"':
            break
        if char != "\\":
            decoded.append(char)
            index += 1
            continue
        if index + 1 >= len(fragment):
            break
        escape = fragment[index + 1]
        simple = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        if escape in simple:
            decoded.append(simple[escape])
            index += 2
            continue
        if escape == "u":
            if index + 6 > len(fragment):
                break
            hex_digits = fragment[index + 2 : index + 6]
            if not re.fullmatch(r"[0-9a-fA-F]{4}", hex_digits):
                break
            decoded.append(chr(int(hex_digits, 16)))
            index += 6
            continue
        break
    return "".join(decoded)


def _preview_stt_config(cfg: Config):
    stt = cfg.stt
    preview_model = stt.preview_model or stt.model
    preview_remote_model = stt.preview_remote_model or stt.remote_model or preview_model
    preview_beam_size = stt.preview_beam_size or stt.beam_size
    offline = replace(
        stt,
        backend=_local_stt_backend(stt.backend, stt.remote_fallback_backend),
        model=preview_model,
        remote_model=preview_remote_model,
        beam_size=preview_beam_size,
    )
    if not stt.preview_use_remote or not remote_stt_available(stt):
        return offline
    return replace(
        offline,
        backend="remote_fallback",
        remote_model=preview_remote_model,
    )


def _final_stt_config(cfg: Config):
    stt = cfg.stt
    preview = _preview_stt_config(cfg)
    final_model = stt.final_model or stt.model
    final_remote_model = stt.final_remote_model or stt.remote_model or final_model
    final_beam_size = stt.final_beam_size or stt.beam_size
    if stt.final_use_remote:
        if not remote_stt_available(stt):
            return preview
        return replace(
            preview,
            backend="remote_fallback",
            remote_model=final_remote_model,
        )
    return replace(
        stt,
        backend=_local_stt_backend(stt.backend, stt.remote_fallback_backend),
        model=final_model,
        remote_model=final_remote_model,
        beam_size=final_beam_size,
    )


def _local_stt_backend(backend: str, fallback_backend: str) -> str:
    normalized = backend.lower().replace("-", "_")
    if normalized in {"remote", "remote_fallback"}:
        return fallback_backend or "faster_whisper"
    return backend


def _stt_cache_key(cfg) -> tuple:
    return (
        cfg.backend,
        cfg.model,
        cfg.language,
        cfg.device,
        cfg.compute_type,
        cfg.beam_size,
        cfg.vad_filter,
        cfg.condition_on_previous_text,
        cfg.remote_base_url,
        cfg.remote_model,
        cfg.remote_probe_timeout_seconds,
        cfg.remote_request_timeout_seconds,
        cfg.remote_fallback_backend,
        tuple(cfg.whisper_cpp_args),
        cfg.whisper_cpp_binary,
        cfg.whisper_cpp_model,
    )
