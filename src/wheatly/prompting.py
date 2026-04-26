from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from wheatly.config import Config
from wheatly.tools.registry import ToolRegistry


DEFAULT_SYSTEM_PROMPT = """You are {{AGENT_NAME}}, a {{AGENT_PERSONA}}.
Default response language: {{DEFAULT_RESPONSE_LANGUAGE}}.

Voice interaction rules:
- Be concise by default for ordinary chat.
- If the user asks for a story, poem, list, plan, detailed explanation, or a specific length, follow that request instead of forcing a short answer.
- Do not emit hidden reasoning, think tags, or long preambles.

Tool calling rules:
- If a tool is needed, reply only with JSON in this shape: {"tool_calls":[{"name":"calculator","arguments":{"expression":"sqrt(10)"}}]}.
- After tool results are provided, answer naturally and match the requested length.
"""

DEFAULT_USER_INSTRUCTIONS = """Use natural spoken English unless the user switches language.
Prefer short answers in voice mode, but do not shorten requested stories, lists, plans, or explanations.
"""


def build_system_prompt(cfg: Config, tools: ToolRegistry) -> str:
    system = _render_template(
        _read_text(Path(cfg.prompts.system_path), DEFAULT_SYSTEM_PROMPT),
        cfg,
    )
    user = _read_text(Path(cfg.prompts.user_path), DEFAULT_USER_INSTRUCTIONS).strip()
    memory = _read_text(Path(cfg.prompts.memory_path), "").strip()
    specs = [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        }
        for spec in tools.specs()
    ]

    parts = [system.strip()]
    if user:
        parts.append("# User Instructions\n" + _render_template(user, cfg))
    if memory:
        parts.append("# Persistent Memory\n" + memory)
    parts.append("# Available Tools\n" + json.dumps(specs, ensure_ascii=True))
    return "\n\n".join(parts)


def load_tool_overrides(path: str) -> Dict[str, Dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        return _load_json_tool_overrides(text, file_path)
    return _load_markdown_tool_overrides(text)


def _load_json_tool_overrides(text: str, file_path: Path) -> Dict[str, Dict[str, str]]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid tool override file {file_path}: expected a JSON object")
    tools = raw.get("tools", raw)
    if not isinstance(tools, dict):
        raise ValueError(f"Invalid tool override file {file_path}: expected 'tools' object")
    overrides: Dict[str, Dict[str, str]] = {}
    for name, value in tools.items():
        if str(name).startswith("$"):
            continue
        override = _normalize_tool_override(value)
        if override:
            overrides[str(name)] = override
    return overrides


def _normalize_tool_override(value: Any) -> Dict[str, str]:
    if isinstance(value, str):
        return {"description": value.strip(), "instructions": ""}
    if not isinstance(value, dict):
        return {}
    description = value.get("description", "")
    instructions = value.get("instructions", "")
    if not isinstance(description, str):
        description = ""
    if not isinstance(instructions, str):
        instructions = ""
    description = description.strip()
    instructions = instructions.strip()
    if not description and not instructions:
        return {}
    return {"description": description, "instructions": instructions}


def _load_markdown_tool_overrides(text: str) -> Dict[str, Dict[str, str]]:
    sections = re.split(r"(?m)^##\s+([A-Za-z0-9_]+)\s*$", text)
    overrides: Dict[str, Dict[str, str]] = {}
    for index in range(1, len(sections), 2):
        name = sections[index].strip()
        body = sections[index + 1]
        description = _extract_description(body)
        instructions = _extract_instructions(body)
        if description or instructions:
            overrides[name] = {
                "description": description,
                "instructions": instructions,
            }
    return overrides


def _read_text(path: Path, default: str) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return default


def _render_template(text: str, cfg: Config) -> str:
    replacements = {
        "{{AGENT_NAME}}": cfg.agent.name,
        "{{AGENT_PERSONA}}": cfg.agent.persona,
        "{{DEFAULT_RESPONSE_LANGUAGE}}": cfg.agent.default_response_language,
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    return text


def _extract_description(body: str) -> str:
    match = re.search(r"(?im)^Description:\s*(.+)$", body)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_instructions(body: str) -> str:
    match = re.search(r"(?ims)^Instructions:\s*(.+)$", body)
    if not match:
        return ""
    return match.group(1).strip()
