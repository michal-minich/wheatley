from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

from wheatley.config import Config, LanguageOptionConfig
from wheatley.text import normalize_words

LANGUAGE_STATE_FILENAME = "language.json"


def apply_configured_language(cfg: Config, language: Optional[str] = None) -> str:
    if not cfg.language.enabled:
        return cfg.runtime.default_language
    code = normalize_language_code(cfg, language or read_language_state(cfg))
    if code is None:
        code = normalize_language_code(cfg, cfg.language.default)
    if code is None:
        return cfg.runtime.default_language
    option = cfg.language.languages[code]

    cfg.runtime.default_language = code
    cfg.agent.default_response_language = option.response_language
    _apply_present_fields(cfg.audio, option.audio)
    _apply_present_fields(cfg.stt, option.stt)
    cfg.stt.language = option.stt.language
    _apply_present_fields(cfg.tts, option.tts, skip_empty_strings=True)
    cfg.tts.piper_config = option.tts.piper_config
    cfg.tts.piper_speaker = option.tts.piper_speaker
    return code


def _apply_present_fields(
    target: object,
    source: object,
    *,
    skip_empty_strings: bool = False,
) -> None:
    for item in fields(source):
        value = getattr(source, item.name)
        if value is None:
            continue
        if skip_empty_strings and value == "":
            continue
        setattr(target, item.name, value)


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
    return Path(cfg.runtime.state_dir) / LANGUAGE_STATE_FILENAME


def normalize_language_code(cfg: Config, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = _normalize_text(value)
    if normalized in cfg.language.languages:
        return normalized
    for code, option in cfg.language.languages.items():
        if normalized == _normalize_text(option.label):
            return code
        for alias in option.aliases:
            if normalized == _normalize_text(alias):
                return code
    return None


def match_language_switch(cfg: Config, text: str) -> Optional[str]:
    if not cfg.language.enabled:
        return None
    normalized = _normalize_text(text)
    if not normalized:
        return None
    for code, option in cfg.language.languages.items():
        for phrase in option.target_switch_phrases:
            phrase_text = _normalize_text(phrase)
            if phrase_text and _phrase_matches(normalized, phrase_text):
                return code
    for code, option in cfg.language.languages.items():
        for phrase in option.toggle_switch_phrases:
            phrase_text = _normalize_text(phrase)
            if phrase_text and _phrase_matches(normalized, phrase_text):
                return _target_for_generic_switch(cfg, phrase_language=code)
    return None


def language_status_payload(cfg: Config) -> dict:
    code = normalize_language_code(cfg, cfg.runtime.default_language)
    if code is None:
        return {"language": cfg.runtime.default_language}
    return _language_payload(code, cfg.language.languages[code])


def model_selection_message(cfg: Config, mode: str, stt_mode: str = "local") -> str:
    code = normalize_language_code(cfg, cfg.runtime.default_language)
    option = cfg.language.languages.get(code or "") if code else None
    if mode == "online":
        if option and option.online_model_message:
            base = option.online_model_message
        else:
            base = _localized_remote_message(cfg.llm.remote.online_message, cfg)
    elif option and option.offline_model_message:
        base = option.offline_model_message
    else:
        base = _localized_remote_message(cfg.llm.remote.offline_message, cfg)
    return _model_stt_message(base, stt_mode, option)


def online_llm_model(cfg: Config) -> str:
    code = normalize_language_code(cfg, cfg.runtime.default_language)
    option = cfg.language.languages.get(code or "") if code else None
    if option and option.online_llm_model is not None:
        return option.online_llm_model
    return cfg.llm.remote.model


def _model_stt_message(
    base: str,
    stt_mode: str,
    option: Optional[LanguageOptionConfig],
) -> str:
    base = base.strip().rstrip(".!")
    if option is None:
        return _finish_sentence(base)
    stt = option.remote_stt_message if stt_mode == "remote" else option.local_stt_message
    template = option.model_selection_message_template
    if not template or not stt:
        return _finish_sentence(base)
    return _finish_sentence(template.format(model=base, stt=stt))


def _localized_remote_message(value: Any, cfg: Config) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    code = normalize_language_code(cfg, cfg.runtime.default_language)
    if code and isinstance(value.get(code), str):
        return value[code]
    default_code = normalize_language_code(cfg, cfg.language.default)
    if default_code and isinstance(value.get(default_code), str):
        return value[default_code]
    if isinstance(value.get("en"), str):
        return value["en"]
    for message in value.values():
        if isinstance(message, str):
            return message
    return ""


def _finish_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


def _language_payload(code: str, option: LanguageOptionConfig) -> dict:
    return {
        "language": code,
        "label": option.label,
        "response_language": option.response_language,
        "audio_partial_transcript_enabled": option.audio.partial_transcript_enabled,
        "audio_partial_transcript_use_as_final": (
            option.audio.partial_transcript_use_as_final
        ),
        "stt_model": option.stt.model,
        "stt_language": option.stt.language,
        "remote_stt_model": option.stt.remote_model,
        "stt_preview_model": option.stt.preview_model,
        "stt_preview_beam_size": option.stt.preview_beam_size,
        "stt_preview_use_remote": option.stt.preview_use_remote,
        "stt_final_model": option.stt.final_model,
        "stt_final_beam_size": option.stt.final_beam_size,
        "stt_final_use_remote": option.stt.final_use_remote,
        "tts_backend": option.tts.backend,
        "tts_voice": option.tts.voice,
        "tts_piper_model": option.tts.piper_model,
        "tts_edge_voice": option.tts.edge_voice,
        "tts_length_scale": option.tts.length_scale,
        "tts_leading_silence_ms": option.tts.leading_silence_ms,
        "tts_stream_speech": option.tts.stream_speech,
        "tts_stream_initial_min_words": option.tts.stream_initial_min_words,
        "tts_stream_min_words": option.tts.stream_min_words,
        "tts_stream_feedback_min_words": option.tts.stream_feedback_min_words,
        "tts_stream_max_inter_chunk_wait_seconds": (
            option.tts.stream_max_inter_chunk_wait_seconds
        ),
        "tts_stream_playback_prebuffer_chunks": (
            option.tts.stream_playback_prebuffer_chunks
        ),
        "tts_stream_playback_prebuffer_max_wait_seconds": (
            option.tts.stream_playback_prebuffer_max_wait_seconds
        ),
        "confirmation": option.confirmation,
        "online_model_message": option.online_model_message,
        "offline_model_message": option.offline_model_message,
        "online_llm_model": option.online_llm_model,
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
    return normalize_words(text)
