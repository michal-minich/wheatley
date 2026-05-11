from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from wheatley.config import Config
from wheatley.language import language_status_payload
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


def load_tool_overrides_from_config(cfg: Config) -> Dict[str, Dict[str, str]]:
    overrides: Dict[str, Dict[str, str]] = {}
    for name, value in cfg.tools.tool_settings.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            continue
        description = str(value.get("description", "")).strip()
        instructions = str(value.get("instructions", "")).strip()
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
        "{{DEFAULT_RESPONSE_LANGUAGE}}": cfg.agent.default_response_language,
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
