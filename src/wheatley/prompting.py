from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from wheatley.config import Config
from wheatley.jsonc import loads_jsonc
from wheatley.language import active_language_hint, language_status_payload
from wheatley.tools.registry import ToolRegistry


def build_system_prompt(cfg: Config, tools: ToolRegistry) -> str:
    system = _render_template(
        _read_text(Path(cfg.prompts.system_path)),
        cfg,
    )
    user = _read_text(Path(cfg.prompts.user_path)).strip()
    memory = _read_text(Path(cfg.prompts.memory_path)).strip()
    auto_memory = _read_text(Path(cfg.profile_dir) / "auto_memory.md").strip()
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
    if cfg.memory.auto_enabled and auto_memory:
        parts.append("# Conversation-Derived Memory\n" + auto_memory)
    parts.append("# Available Tools\n" + json.dumps(specs, ensure_ascii=True))
    return "\n\n".join(parts)


def load_tool_overrides(path: str) -> Dict[str, Dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in {".json", ".jsonc"}:
        return _load_json_tool_overrides(text, file_path)
    return _load_markdown_tool_overrides(text)


def _load_json_tool_overrides(text: str, file_path: Path) -> Dict[str, Dict[str, str]]:
    raw = loads_jsonc(text, str(file_path))
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


def _read_text(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return ""


def _render_template(text: str, cfg: Config) -> str:
    language = language_status_payload(cfg)
    replacements = {
        "{{AGENT_NAME}}": cfg.agent.name,
        "{{AGENT_PERSONA}}": cfg.agent.persona,
        "{{DEFAULT_RESPONSE_LANGUAGE}}": cfg.agent.default_response_language,
        "{{ACTIVE_LANGUAGE_HINT}}": active_language_hint(cfg),
        "{{CURRENT_LANGUAGE_CODE}}": str(language.get("language", "")),
        "{{CURRENT_LANGUAGE_LABEL}}": str(language.get("label", "")),
        "{{CURRENT_RESPONSE_LANGUAGE}}": str(language.get("response_language", "")),
        "{{CURRENT_STT_MODEL}}": str(language.get("stt_model", "")),
        "{{CURRENT_STT_LANGUAGE}}": str(language.get("stt_language", "")),
        "{{CURRENT_TTS_BACKEND}}": str(language.get("tts_backend", "")),
        "{{CURRENT_TTS_VOICE}}": str(language.get("tts_voice", "")),
        "{{CURRENT_TTS_PIPER_MODEL}}": str(language.get("tts_piper_model", "")),
        "{{CURRENT_TTS_EDGE_VOICE}}": str(language.get("tts_edge_voice", "")),
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
