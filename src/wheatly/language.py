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
    if option.tts_edge_voice:
        cfg.tts.edge_voice = option.tts_edge_voice
    if option.tts_edge_rate:
        cfg.tts.edge_rate = option.tts_edge_rate
    if option.tts_edge_pitch:
        cfg.tts.edge_pitch = option.tts_edge_pitch
    if option.tts_edge_volume:
        cfg.tts.edge_volume = option.tts_edge_volume
    if option.tts_length_scale is not None:
        cfg.tts.length_scale = option.tts_length_scale
    if option.tts_noise_scale is not None:
        cfg.tts.noise_scale = option.tts_noise_scale
    if option.tts_noise_w_scale is not None:
        cfg.tts.noise_w_scale = option.tts_noise_w_scale
    if option.tts_sentence_silence is not None:
        cfg.tts.sentence_silence = option.tts_sentence_silence
    if option.tts_volume is not None:
        cfg.tts.volume = option.tts_volume
    if option.tts_leading_silence_ms is not None:
        cfg.tts.leading_silence_ms = option.tts_leading_silence_ms
    if option.tts_stream_initial_min_words is not None:
        cfg.tts.stream_initial_min_words = option.tts_stream_initial_min_words
    if option.tts_stream_min_words is not None:
        cfg.tts.stream_min_words = option.tts_stream_min_words
    if option.tts_stream_max_words is not None:
        cfg.tts.stream_max_words = option.tts_stream_max_words
    if option.tts_stream_feedback_min_words is not None:
        cfg.tts.stream_feedback_min_words = option.tts_stream_feedback_min_words
    if option.tts_stream_max_initial_wait_seconds is not None:
        cfg.tts.stream_max_initial_wait_seconds = (
            option.tts_stream_max_initial_wait_seconds
        )
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
    current_code = normalize_language_code(cfg, cfg.runtime.default_language)
    apply_configured_language(cfg, code)
    if cfg.language.persist:
        path = language_state_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"language": code}
        if current_code and current_code != code:
            payload["previous_language"] = current_code
        else:
            previous = read_previous_language_state(cfg)
            if previous:
                payload["previous_language"] = previous
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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


def read_previous_language_state(cfg: Config) -> str:
    if not cfg.language.persist:
        return ""
    path = language_state_path(cfg)
    if not path.exists():
        return ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(raw.get("previous_language") or "")


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
    for code, option in cfg.language.languages.items():
        for phrase in option.switch_language_phrases:
            phrase_text = _normalize_text(phrase)
            if phrase_text and _phrase_matches(normalized, phrase_text):
                return _target_for_generic_switch(cfg, phrase_language=code)
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


def model_selection_message(cfg: Config, mode: str) -> str:
    code = normalize_language_code(cfg, cfg.runtime.default_language)
    option = cfg.language.languages.get(code or "") if code else None
    if mode == "online":
        if option and option.online_model_message:
            return option.online_model_message
        return cfg.llm.remote.online_message
    if option and option.offline_model_message:
        return option.offline_model_message
    return cfg.llm.remote.offline_message


def _language_payload(code: str, option: LanguageOptionConfig) -> dict:
    return {
        "language": code,
        "label": option.label,
        "response_language": option.response_language,
        "stt_model": option.stt_model,
        "stt_language": option.stt_language,
        "tts_backend": option.tts_backend,
        "tts_voice": option.tts_voice,
        "tts_piper_model": option.tts_piper_model,
        "tts_edge_voice": option.tts_edge_voice,
        "tts_length_scale": option.tts_length_scale,
        "tts_leading_silence_ms": option.tts_leading_silence_ms,
        "tts_stream_initial_min_words": option.tts_stream_initial_min_words,
        "tts_stream_min_words": option.tts_stream_min_words,
        "tts_stream_feedback_min_words": option.tts_stream_feedback_min_words,
        "confirmation": option.confirmation,
        "online_model_message": option.online_model_message,
        "offline_model_message": option.offline_model_message,
    }


def _target_for_generic_switch(cfg: Config, phrase_language: str) -> str:
    active = normalize_language_code(cfg, cfg.runtime.default_language)
    if phrase_language != active:
        return phrase_language
    previous = normalize_language_code(cfg, read_previous_language_state(cfg))
    if previous and previous != active:
        return previous
    for code in cfg.language.languages:
        if code != active:
            return code
    return phrase_language


def _phrase_matches(text: str, phrase: str) -> bool:
    return text == phrase or re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())
