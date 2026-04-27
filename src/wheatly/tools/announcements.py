from __future__ import annotations

from typing import Optional

from wheatly.config import Config
from wheatly.language import normalize_language_code


_MESSAGES = {
    "en": {
        "remember": "Remembering...",
        "run_safe_cli_tool": "Running...",
        "web_search": "Searching...",
        "fetch_url": "Downloading...",
    },
    "sk": {
        "remember": "Zapamätávam...",
        "run_safe_cli_tool": "Spúšťam...",
        "web_search": "Hľadám...",
        "fetch_url": "Sťahujem...",
    },
}


def tool_start_message(cfg: Config, tool_name: str) -> Optional[str]:
    code = normalize_language_code(cfg, cfg.runtime.default_language) or "en"
    messages = _MESSAGES.get(code, _MESSAGES["en"])
    return messages.get(tool_name)
