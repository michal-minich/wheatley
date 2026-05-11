from __future__ import annotations

from typing import Optional

from wheatley.config import Config
from wheatley.language import normalize_language_code


def tool_start_message(cfg: Config, tool_name: str) -> Optional[str]:
    code = normalize_language_code(cfg, cfg.runtime.default_language)
    if code is None:
        return None
    setting = cfg.tools.tool_settings.get(tool_name, {})
    messages = setting.get("start_messages", {})
    if not isinstance(messages, dict):
        return None
    if code in messages:
        return messages[code] or None
    return messages.get("en") or None
