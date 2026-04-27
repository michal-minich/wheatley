from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from wheatley.tools.registry import ToolCall


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_tool_calls(text: str) -> List[ToolCall]:
    payload = _load_json_payload(text)
    if not payload:
        return []

    raw_calls: List[Dict[str, Any]]
    if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
        raw_calls = payload["tool_calls"]
    elif "tool" in payload or "name" in payload:
        raw_calls = [payload]
    else:
        return []

    calls: List[ToolCall] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool")
        args = item.get("arguments", item.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"value": args}
        if name and isinstance(args, dict):
            calls.append(ToolCall(name=str(name), arguments=args))
    return calls


def _load_json_payload(text: str) -> Dict[str, Any] | None:
    text = text.strip()
    for candidate in _candidate_json_strings(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _candidate_json_strings(text: str) -> List[str]:
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in FENCE_RE.finditer(text))
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    return candidates

