from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Iterator, List

from wheatly.config import LLMConfig, RemoteLLMConfig
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
            _openai_endpoint_url(self.cfg.base_url, "chat/completions"),
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
        if self.cfg.strip_reasoning:
            content = _strip_reasoning(content)
        return LLMResponse(content=content, raw=raw)

    def stream_complete(self, messages: List[LLMMessage]) -> Iterator[str]:
        payload = {
            "model": self.cfg.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
        }
        if self.cfg.backend.lower() in {"vllm", "sglang"} and not self.cfg.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }
        stream = _post_openai_stream(
            _openai_endpoint_url(self.cfg.base_url, "chat/completions"),
            payload,
            headers=headers,
            timeout=self.cfg.timeout_seconds,
        )
        if self.cfg.strip_reasoning:
            yield from _filter_reasoning_stream(stream)
        else:
            yield from stream


def build_llm(cfg: LLMConfig) -> LLMBackend:
    backend = cfg.backend.lower()
    if backend == "echo":
        return EchoLLM()
    if backend == "ollama":
        return OllamaLLM(cfg)
    if backend in {"openai", "openai_compat", "llama_cpp", "vllm", "sglang"}:
        return OpenAICompatLLM(cfg)
    raise ValueError(f"Unsupported LLM backend: {cfg.backend}")


def remote_llm_available(cfg: RemoteLLMConfig) -> bool:
    if not cfg.enabled:
        return False
    try:
        _get_json(
            _openai_endpoint_url(cfg.base_url, "models"),
            timeout=cfg.probe_timeout_seconds,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
        )
        return True
    except Exception:
        return False


def _get_json(url: str, timeout: float, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


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


def _post_openai_stream(
    url: str,
    payload: dict,
    headers: dict,
    timeout: float,
) -> Iterator[str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                event = json.loads(line)
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM streaming request failed for {url}: {exc}") from exc


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


def _openai_endpoint_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{endpoint.lstrip('/')}"
    return f"{base}/v1/{endpoint.lstrip('/')}"


def _strip_reasoning(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"(?is)<think\b[^>]*>.*?</think>", "", text)
    return text.strip()


def _filter_reasoning_stream(chunks: Iterator[str]) -> Iterator[str]:
    buffer = ""
    suppress_reasoning = False
    decided = False
    for chunk in chunks:
        if decided and not suppress_reasoning:
            yield chunk
            continue

        buffer += chunk
        stripped = buffer.lstrip()
        lowered = stripped.lower()
        if "</think>" in buffer:
            after = _strip_reasoning(buffer)
            buffer = ""
            suppress_reasoning = False
            decided = True
            if after:
                yield after
            continue
        if not decided:
            if lowered.startswith("<think"):
                suppress_reasoning = True
                decided = True
            elif _looks_like_reasoning_prefix(lowered):
                suppress_reasoning = True
                decided = True
            elif _could_be_reasoning_prefix(lowered) and len(stripped) < 24:
                continue
            else:
                decided = True
                yield buffer
                buffer = ""
                continue

    if buffer and not suppress_reasoning:
        yield buffer


_REASONING_PREFIXES = (
    "the user wants",
    "the user asks",
    "i need to",
    "i should",
    "we need",
    "analyze request",
    "analysis:",
)


def _looks_like_reasoning_prefix(text: str) -> bool:
    return any(text.startswith(prefix) for prefix in _REASONING_PREFIXES)


def _could_be_reasoning_prefix(text: str) -> bool:
    if not text:
        return True
    if text.startswith("<") and "<think".startswith(text):
        return True
    words = text.split()
    if len(words) >= 2:
        return any(prefix.startswith(text) for prefix in _REASONING_PREFIXES)
    return any(prefix.startswith(text) for prefix in _REASONING_PREFIXES)


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
