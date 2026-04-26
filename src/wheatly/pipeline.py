from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from wheatly.config import Config
from wheatly.language import (
    apply_configured_language,
    match_language_switch,
    read_language_state,
    set_language_state,
)
from wheatly.llm.backends import build_llm, remote_llm_available
from wheatly.llm.base import LLMBackend, LLMMessage
from wheatly.prompting import build_system_prompt
from wheatly.runtime_stats import LatencyStats
from wheatly.stt.backends import build_stt
from wheatly.stt.base import STTBackend, Transcription
from wheatly.tools.builtins import build_registry
from wheatly.tools.parser import parse_tool_calls
from wheatly.tools.registry import ToolCall, ToolRegistry, ToolResult
from wheatly.tts.backends import build_tts
from wheatly.tts.base import TTSBackend
from wheatly.tts.streaming import StreamingSpeaker


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


class VoiceAgent:
    def __init__(
        self,
        cfg: Config,
        llm: Optional[LLMBackend] = None,
        stt: Optional[STTBackend] = None,
        tts: Optional[TTSBackend] = None,
        tools: Optional[ToolRegistry] = None,
    ):
        self.cfg = cfg
        apply_configured_language(self.cfg, read_language_state(self.cfg))
        self.llm = llm or build_llm(cfg.llm)
        self.stt_lock = threading.Lock()
        self.stt = stt or build_stt(cfg.stt)
        self.tts = tts or build_tts(cfg)
        self.tools = tools or build_registry(cfg)
        self.latency_stats = LatencyStats(Path(cfg.runtime.state_dir) / "latency_stats.json")
        self.history: List[LLMMessage] = []
        self.model_selection = ModelSelection("offline", cfg.llm.remote.offline_message)

    def reset_chat(self) -> ModelSelection:
        self.history.clear()
        return self.select_chat_model()

    def select_chat_model(self) -> ModelSelection:
        remote = self.cfg.llm.remote
        if remote.enabled and remote_llm_available(remote):
            remote_cfg = replace(
                self.cfg.llm,
                backend=remote.backend,
                base_url=remote.base_url,
                model=remote.model,
                api_key=remote.api_key,
                timeout_seconds=remote.request_timeout_seconds,
            )
            self.llm = build_llm(remote_cfg)
            self.model_selection = ModelSelection("online", remote.online_message)
            return self.model_selection
        self.llm = build_llm(self.cfg.llm)
        self.model_selection = ModelSelection("offline", remote.offline_message)
        return self.model_selection

    def transcribe(self, audio_path: Optional[Path] = None) -> Transcription:
        with self.stt_lock:
            return self.stt.transcribe(audio_path)

    def set_language(self, requested_language: str) -> ToolResult:
        ok, content = set_language_state(self.cfg, requested_language)
        if ok:
            with self.stt_lock:
                self.stt = build_stt(self.cfg.stt)
            self.tts = build_tts(self.cfg)
        return ToolResult(name="set_language", ok=ok, content=content)

    def handle_text(self, text: str, speak: bool = True) -> TurnResult:
        started_at = time.perf_counter()
        text = text.strip()
        if not text:
            return TurnResult(user_text="", assistant_text="", tool_results=[])

        direct_tool_calls = _route_direct_tools(text, self.cfg)
        if direct_tool_calls and self.cfg.tools.enabled:
            tool_results = self._execute_tool_calls(direct_tool_calls)
            direct_answer = _format_direct_tool_answer(tool_results)
            if direct_answer:
                self._remember(text, direct_answer)
                result = TurnResult(
                    user_text=text,
                    assistant_text=direct_answer,
                    tool_results=tool_results,
                    duration_seconds=round(time.perf_counter() - started_at, 4),
                )
                self._log_turn(result)
                if speak:
                    self.tts.speak(direct_answer)
                return result
            messages = self._messages_for_turn(text)
            messages.append(_tool_results_message(tool_results, self.cfg))
            final_text = self.llm.complete(messages).content.strip()
            self._remember(text, final_text)
            result = TurnResult(
                user_text=text,
                assistant_text=final_text,
                tool_results=tool_results,
                duration_seconds=round(time.perf_counter() - started_at, 4),
            )
            self._log_turn(result)
            if speak and final_text:
                self.tts.speak(final_text)
            return result

        messages = self._messages_for_turn(text)
        first = self.llm.complete(messages)
        tool_calls = parse_tool_calls(first.content) if self.cfg.tools.enabled else []
        tool_results: List[ToolResult] = []
        final_text = first.content.strip()

        if tool_calls:
            tool_results = self._execute_tool_calls(tool_calls)
            direct_answer = _format_direct_tool_answer(tool_results)
            if direct_answer:
                final_text = direct_answer
            else:
                messages.append(LLMMessage(role="assistant", content=first.content))
                messages.append(_tool_results_message(tool_results, self.cfg))
                second = self.llm.complete(messages)
                final_text = second.content.strip()

        self._remember(text, final_text)
        result = TurnResult(
            user_text=text,
            assistant_text=final_text,
            tool_results=tool_results,
            duration_seconds=round(time.perf_counter() - started_at, 4),
        )
        self._log_turn(result)
        if speak and final_text:
            self.tts.speak(final_text)
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

        direct_tool_calls = _route_direct_tools(text, self.cfg)
        if direct_tool_calls and self.cfg.tools.enabled:
            tool_results = self._execute_tool_calls(direct_tool_calls)
            direct_answer = _format_direct_tool_answer(tool_results)
            if direct_answer:
                if on_token:
                    on_token(direct_answer)
                if speak:
                    self.tts.speak(direct_answer)
                return self._finish_turn(text, direct_answer, tool_results, started_at)
            messages = self._messages_for_turn(text)
            messages.append(_tool_results_message(tool_results, self.cfg))
            final_text = self._stream_final_answer(messages, speak, on_token)
            return self._finish_turn(text, final_text, tool_results, started_at)

        messages = self._messages_for_turn(text)
        first_text = self._stream_final_answer(messages, speak, on_token)
        tool_calls = parse_tool_calls(first_text) if self.cfg.tools.enabled else []
        tool_results: List[ToolResult] = []
        final_text = first_text

        if tool_calls:
            tool_results = self._execute_tool_calls(tool_calls)
            direct_answer = _format_direct_tool_answer(tool_results)
            if direct_answer:
                if on_token and not _is_language_switch_tool_result(tool_results):
                    on_token(direct_answer)
                if speak:
                    self.tts.speak(direct_answer)
                final_text = direct_answer
            else:
                messages.append(LLMMessage(role="assistant", content=first_text))
                messages.append(_tool_results_message(tool_results, self.cfg))
                final_text = self._stream_final_answer(messages, speak, on_token)

        return self._finish_turn(text, final_text, tool_results, started_at)

    def _messages_for_turn(self, text: str) -> List[LLMMessage]:
        system = build_system_prompt(self.cfg, self.tools)
        trimmed_history = self.history[-self.cfg.llm.context_turns * 2 :]
        return (
            [LLMMessage("system", system)]
            + trimmed_history
            + [LLMMessage("user", text)]
        )

    def _execute_tool_calls(self, calls: List[ToolCall]) -> List[ToolResult]:
        results: List[ToolResult] = []
        for call in calls:
            if call.name == "set_language":
                language = str(call.arguments.get("language", ""))
                results.append(self.set_language(language))
            else:
                results.append(self.tools.execute(call))
        return results

    def _remember(self, user_text: str, assistant_text: str) -> None:
        self.history.append(LLMMessage("user", user_text))
        self.history.append(LLMMessage("assistant", assistant_text))

    def _finish_turn(
        self,
        user_text: str,
        assistant_text: str,
        tool_results: List[ToolResult],
        started_at: float,
    ) -> TurnResult:
        self._remember(user_text, assistant_text)
        result = TurnResult(
            user_text=user_text,
            assistant_text=assistant_text,
            tool_results=tool_results,
            duration_seconds=round(time.perf_counter() - started_at, 4),
        )
        self._log_turn(result)
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
        first_chunk_at: Optional[float] = None
        initial_words = self.latency_stats.recommended_initial_words(
            min_words=self.cfg.tts.stream_initial_min_words,
            max_words=self.cfg.tts.stream_max_words,
            adaptive=self.cfg.tts.adaptive_streaming,
        )

        with StreamingSpeaker(
            self.tts,
            enabled=speak and self.cfg.tts.stream_speech,
            min_words=self.cfg.tts.stream_min_words,
            max_words=self.cfg.tts.stream_max_words,
            initial_min_words=initial_words,
            feedback_min_words=self.cfg.tts.stream_feedback_min_words,
            max_initial_wait_seconds=self.cfg.tts.stream_max_initial_wait_seconds,
            on_spoken=self._record_spoken_segment,
        ) as speaker:
            for chunk in self.llm.stream_complete(messages):
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                chunks.append(chunk)
                if not visible_started:
                    pending += chunk
                    stripped = pending.lstrip()
                    if not stripped:
                        continue
                    if stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("```"):
                        hold_for_possible_tool_json = True
                        continue
                    visible_started = True
                    _emit_token(pending, on_token, speaker)
                    pending = ""
                    continue
                _emit_token(chunk, on_token, speaker)

            final_text = "".join(chunks).strip()
            if hold_for_possible_tool_json and not parse_tool_calls(final_text):
                _emit_token(pending, on_token, speaker)

        stream_ended_at = time.perf_counter()
        self.latency_stats.record_llm(
            words=_count_words(final_text),
            duration_seconds=stream_ended_at - (first_chunk_at or stream_ended_at),
        )
        if speak and not self.cfg.tts.stream_speech and final_text:
            started_at = time.perf_counter()
            self.tts.speak(final_text)
            self._record_spoken_segment(final_text, time.perf_counter() - started_at)
        return final_text

    def _record_spoken_segment(self, segment: str, duration_seconds: float) -> None:
        self.latency_stats.record_tts(_count_words(segment), duration_seconds)

    def _log_turn(self, result: TurnResult) -> None:
        path = Path(self.cfg.runtime.turn_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "user_text": result.user_text,
            "assistant_text": result.assistant_text,
            "tool_results": [asdict(item) for item in result.tool_results],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

def _tool_results_message(tool_results: List[ToolResult], cfg: Config) -> LLMMessage:
    return LLMMessage(
        role="user",
        content="Tool results: "
        + json.dumps([asdict(result) for result in tool_results])
        + "\nNow answer the user in natural language. Match the requested length."
        + f"\nCurrent reply language: {cfg.agent.default_response_language}.",
    )


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


def _route_direct_tools(text: str, cfg: Config) -> List[ToolCall]:
    lowered = text.lower()
    calls: List[ToolCall] = []
    language = match_language_switch(cfg, text)
    if language:
        calls.append(ToolCall("set_language", {"language": language}))
        return calls
    memory = _extract_memory_text(text)
    if memory:
        calls.append(ToolCall("remember", {"memory": memory}))
        return calls
    calculator_args = _extract_calculator_args(text)
    if calculator_args:
        calls.append(ToolCall("calculator", calculator_args))
        return calls
    if _has_word(lowered, "time") or _has_word(lowered, "date"):
        calls.append(ToolCall("get_time", {}))
    elif _has_word(lowered, "status") or _has_word(lowered, "battery"):
        calls.append(ToolCall("robot_status", {}))
    return calls


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


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


def _extract_calculator_args(text: str) -> dict:
    cleaned = text.strip().rstrip("?.!")
    match = re.search(r"\b(?:calculate|compute|evaluate|eval)\s+(.+)$", cleaned, re.I)
    if match:
        return _calculator_args_from_expression_text(match.group(1).strip())
    match = re.search(r"\bwhat(?:'s| is)\s+(.+)$", cleaned, re.I)
    if match and _looks_like_math(match.group(1)):
        return _calculator_args_from_expression_text(match.group(1).strip())
    if _looks_like_math(cleaned):
        return _calculator_args_from_expression_text(cleaned)
    return {}


def _calculator_args_from_expression_text(text: str) -> dict:
    round_digits = None
    round_match = re.search(r"\bround(?:ed)?\s+to\s+(\d+)\s+decimal", text, re.I)
    if round_match:
        round_digits = int(round_match.group(1))
        text = text[: round_match.start()].strip(" ,")
    args = {"expression": text.replace("^", "**")}
    if round_digits is not None:
        args["round_digits"] = round_digits
    return args


def _looks_like_math(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b(sqrt|sin|cos|tan|log|gcd|lcm|round|floor|ceil)\s*\(", lowered):
        return True
    if re.search(r"\d\s*[\+\-\*/%\^]", lowered) or "**" in lowered:
        return True
    return False
