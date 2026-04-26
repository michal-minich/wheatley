from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

from wheatly.config import Config, LanguageOptionConfig


def apply_configured_language(cfg: Config, language: Optional[str] = None) -> str:
    if not cfg.language.enabled:
        return cfg.runtime.default_language
    code = normalize_language_code(cfg, language or read_language_state(cfg))
    if code is None:
        code = normalize_language_code(cfg, cfg.language.default) or "en"
    option = cfg.language.languages.get(code)
    if option is None:
        return cfg.runtime.default_language

    cfg.runtime.default_language = code
    cfg.agent.default_response_language = option.response_language or option.label
    if option.stt_model is not None:
        cfg.stt.model = option.stt_model
    cfg.stt.language = option.stt_language
    if option.tts_backend:
        cfg.tts.backend = option.tts_backend
    if option.tts_voice:
        cfg.tts.voice = option.tts_voice
    if option.tts_piper_model:
        cfg.tts.piper_model = option.tts_piper_model
    cfg.tts.piper_config = option.tts_piper_config
    cfg.tts.piper_speaker = option.tts_piper_speaker
    return code


def set_language_state(cfg: Config, requested_language: str) -> tuple[bool, dict]:
    code = normalize_language_code(cfg, requested_language)
    if code is None:
        return (
            False,
            {
                "error": "unsupported_language",
                "requested": requested_language,
                "available": sorted(cfg.language.languages),
            },
        )
    apply_configured_language(cfg, code)
    if cfg.language.persist:
        path = language_state_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"language": code}, indent=2), encoding="utf-8")
    option = cfg.language.languages[code]
    return True, _language_payload(code, option)


def read_language_state(cfg: Config) -> str:
    if not cfg.language.persist:
        return cfg.language.default
    path = language_state_path(cfg)
    if not path.exists():
        return cfg.language.default
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return cfg.language.default
    return str(raw.get("language") or cfg.language.default)


def language_state_path(cfg: Config) -> Path:
    return Path(cfg.runtime.state_dir) / cfg.language.state_file


def normalize_language_code(cfg: Config, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = _normalize_text(value)
    if normalized in cfg.language.languages:
        return normalized
    aliases = {
        "eng": "en",
        "english": "en",
        "anglictina": "en",
        "anglicky": "en",
        "po anglicky": "en",
        "slovak": "sk",
        "slovencina": "sk",
        "slovensky": "sk",
        "po slovensky": "sk",
    }
    if normalized in aliases and aliases[normalized] in cfg.language.languages:
        return aliases[normalized]
    for code, option in cfg.language.languages.items():
        if normalized == _normalize_text(option.label):
            return code
    return None


def match_language_switch(cfg: Config, text: str) -> Optional[str]:
    if not cfg.language.enabled:
        return None
    normalized = _normalize_text(text)
    if not normalized:
        return None
    for code, option in cfg.language.languages.items():
        for phrase in option.switch_phrases:
            phrase_text = _normalize_text(phrase)
            if phrase_text and _phrase_matches(normalized, phrase_text):
                return code
    return None


def active_language_hint(cfg: Config) -> str:
    if not cfg.language.enabled:
        return ""
    code = normalize_language_code(cfg, cfg.runtime.default_language)
    if code is None:
        return ""
    option = cfg.language.languages.get(code)
    if option is None:
        return ""
    return (
        f"Current language mode: {option.label} ({code}). "
        f"Reply in {option.response_language or option.label}. "
        "Do not switch language unless the user explicitly asks."
    )


def language_status_payload(cfg: Config) -> dict:
    code = normalize_language_code(cfg, cfg.runtime.default_language) or cfg.runtime.default_language
    option = cfg.language.languages.get(code)
    if option is None:
        return {"language": code}
    return _language_payload(code, option)


def _language_payload(code: str, option: LanguageOptionConfig) -> dict:
    return {
        "language": code,
        "label": option.label,
        "response_language": option.response_language,
        "stt_model": option.stt_model,
        "stt_language": option.stt_language,
        "tts_voice": option.tts_voice,
        "tts_piper_model": option.tts_piper_model,
        "confirmation": option.confirmation,
    }


def _phrase_matches(text: str, phrase: str) -> bool:
    return text == phrase or re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())
