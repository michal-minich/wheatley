from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Iterator, List

from wheatly.config import LLMConfig
from wheatly.llm.base import LLMBackend, LLMMessage, LLMResponse


class EchoLLM(LLMBackend):
    """Deterministic local backend for smoke tests without model downloads."""

    def complete(self, messages: List[LLMMessage]) -> LLMResponse:
        last = messages[-1].content if messages else ""
        lowered = last.lower()
        if "tool results" in lowered:
            return LLMResponse(_summarize_tool_results(last))
        if _has_word(lowered, "time") or _has_word(lowered, "date"):
            payload = {"tool_calls": [{"name": "get_time", "arguments": {}}]}
            return LLMResponse(json.dumps(payload))
        if "battery" in lowered or "status" in lowered:
            payload = {"tool_calls": [{"name": "robot_status", "arguments": {}}]}
            return LLMResponse(json.dumps(payload))
        return LLMResponse("I heard: " + _compact(last))

    def stream_complete(self, messages: List[LLMMessage]) -> Iterator[str]:
        text = self.complete(messages).content
        for index, part in enumerate(text.split(" ")):
            yield part if index == 0 else " " + part


class OllamaLLM(LLMBackend):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def complete(self, messages: List[LLMMessage]) -> LLMResponse:
        payload = {
            "model": self.cfg.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "think": bool(self.cfg.enable_thinking),
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": self.cfg.max_tokens,
            },
        }
        raw = _post_json(
            f"{self.cfg.base_url.rstrip('/')}/api/chat",
            payload,
            timeout=self.cfg.timeout_seconds,
        )
        content = raw.get("message", {}).get("content", "")
        return LLMResponse(content=content, raw=raw)

    def stream_complete(self, messages: List[LLMMessage]) -> Iterator[str]:
        payload = {
            "model": self.cfg.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "think": bool(self.cfg.enable_thinking),
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": self.cfg.max_tokens,
            },
        }
        yield from _post_json_lines(
            f"{self.cfg.base_url.rstrip('/')}/api/chat",
            payload,
            timeout=self.cfg.timeout_seconds,
        )


class OpenAICompatLLM(LLMBackend):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def complete(self, messages: List[LLMMessage]) -> LLMResponse:
        payload = {
            "model": self.cfg.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.backend.lower() in {"vllm", "sglang"} and not self.cfg.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }
        raw = _post_json(
            f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions",
            payload,
            headers=headers,
            timeout=self.cfg.timeout_seconds,
        )
        choices = raw.get("choices") or []
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "") or choices[0].get(
                "text", ""
            )
        return LLMResponse(content=content, raw=raw)

    def stream_complete(self, messages: List[LLMMessage]) -> Iterator[str]:
        yield self.complete(messages).content


def build_llm(cfg: LLMConfig) -> LLMBackend:
    backend = cfg.backend.lower()
    if backend == "echo":
        return EchoLLM()
    if backend == "ollama":
        return OllamaLLM(cfg)
    if backend in {"openai", "openai_compat", "llama_cpp", "vllm", "sglang"}:
        return OpenAICompatLLM(cfg)
    raise ValueError(f"Unsupported LLM backend: {cfg.backend}")


def _post_json(url: str, payload: dict, timeout: float, headers: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers or {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed for {url}: {exc}") from exc


def _post_json_lines(url: str, payload: dict, timeout: float) -> Iterator[str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("done"):
                    break
                content = event.get("message", {}).get("content", "")
                if content:
                    yield content
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM streaming request failed for {url}: {exc}") from exc


def _compact(text: str, limit: int = 180) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _summarize_tool_results(text: str) -> str:
    marker = "Tool results:"
    start = text.find(marker)
    if start < 0:
        return "I checked that locally."
    payload_text = text[start + len(marker) :].strip()
    if "\n" in payload_text:
        payload_text = payload_text.split("\n", 1)[0].strip()
    try:
        results = json.loads(payload_text)
    except json.JSONDecodeError:
        return "I checked that locally."
    if not results:
        return "I checked that locally, but there was no result."
    first = results[0]
    name = first.get("name")
    content = first.get("content", {})
    if name == "get_time" and "iso" in content:
        return f"Local time is {content['iso']}."
    if name == "robot_status":
        return (
            "Status is local and running. "
            f"LLM is {content.get('llm_backend')}, STT is {content.get('stt_backend')}."
        )
    return f"I checked {name or 'the tool'} locally."
