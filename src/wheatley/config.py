from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from wheatley.jsonc import load_jsonc


REQUIRED_TOOL_SETTINGS = {
    "get_time",
    "system_status",
    "set_eye_expression",
    "calculator",
    "remember",
    "set_language",
    "take_photo",
    "run_safe_cli_tool",
    "web_search",
    "python_interpreter",
}


@dataclass
class RuntimeConfig:
    data_dir: str = ""
    turn_log: str = ""
    tool_log: str = ""
    system_llm_log: str = ""
    state_dir: str = ""
    default_language: str = ""


@dataclass
class ChatConfig:
    resume_on_start_mode: str = "ask"
    resume_on_start: bool = False
    resume_turns: int = 0
    resume_countdown_seconds: int = 0


@dataclass
class IdleSpeechConfig:
    enabled: bool = False
    interval_seconds: float = 0.0
    random_min_multiplier: float = 1.0
    random_max_multiplier: float = 1.0


@dataclass
class AudioConfig:
    sample_rate: int = 0
    channels: int = 0
    input_device_mode: str = "default"
    input_device_preferred_names: List[str] = field(default_factory=list)
    input_device_name: Optional[str] = None
    input_device_index: Optional[int] = None
    vad_threshold: float = 0.0
    min_speech_seconds: float = 0.0
    silence_seconds: float = 0.0
    pre_roll_seconds: float = 0.0
    trailing_silence_keep_seconds: float = 0.0
    max_utterance_seconds: float = 0.0
    max_wait_seconds: float = 0.0
    utterance_dir: str = ""
    partial_transcript_enabled: bool = False
    partial_transcript_interval_seconds: float = 0.0
    partial_transcript_min_audio_seconds: float = 0.0
    partial_transcript_use_as_final: bool = False
    partial_transcript_final_max_age_seconds: float = 0.0
    listening_chimes_enabled: bool = False
    listening_chime_volume: float = 0.0
    speech_interrupt_enabled: bool = False
    speech_interrupt_phrase: str = ""
    speech_interrupt_min_rms: float = 0.0
    speech_interrupt_vad_multiplier: float = 0.0
    speech_interrupt_baseline_multiplier: float = 0.0
    speech_interrupt_grace_seconds: float = 0.0
    speech_interrupt_pre_roll_seconds: float = 0.0
    speech_interrupt_record_seconds: float = 0.0
    speech_interrupt_max_words: int = 0
    speech_interrupt_pause_tts_while_verifying: bool = False


@dataclass
class STTConfig:
    backend: str = ""
    model: str = ""
    language: Optional[str] = None
    device: str = ""
    compute_type: str = ""
    beam_size: int = 0
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    preview_model: str = ""
    preview_remote_model: str = ""
    preview_use_remote: bool = False
    preview_beam_size: int = 0
    final_model: str = ""
    final_remote_model: str = ""
    final_use_remote: bool = False
    final_beam_size: int = 0
    whisper_cpp_binary: str = ""
    whisper_cpp_model: str = ""
    whisper_cpp_args: List[str] = field(default_factory=list)
    remote_base_url: str = ""
    remote_api_key: str = ""
    remote_model: str = ""
    remote_probe_timeout_seconds: float = 0.0
    remote_request_timeout_seconds: float = 0.0
    remote_fallback_backend: str = ""


@dataclass
class LanguageAudioOverrides:
    partial_transcript_enabled: Optional[bool] = None
    partial_transcript_use_as_final: Optional[bool] = None


@dataclass
class LanguageSTTOverrides:
    model: Optional[str] = None
    language: Optional[str] = None
    remote_model: Optional[str] = None
    preview_model: Optional[str] = None
    preview_remote_model: Optional[str] = None
    preview_use_remote: Optional[bool] = None
    preview_beam_size: Optional[int] = None
    final_model: Optional[str] = None
    final_remote_model: Optional[str] = None
    final_use_remote: Optional[bool] = None
    final_beam_size: Optional[int] = None


@dataclass
class LanguageTTSOverrides:
    backend: Optional[str] = None
    voice: Optional[str] = None
    piper_model: Optional[str] = None
    piper_config: Optional[str] = None
    piper_speaker: Optional[int] = None
    edge_voice: Optional[str] = None
    edge_rate: Optional[str] = None
    edge_pitch: Optional[str] = None
    edge_volume: Optional[str] = None
    length_scale: Optional[float] = None
    noise_scale: Optional[float] = None
    noise_w_scale: Optional[float] = None
    sentence_silence: Optional[float] = None
    volume: Optional[float] = None
    leading_silence_ms: Optional[int] = None
    stream_speech: Optional[bool] = None
    stream_initial_min_words: Optional[int] = None
    stream_min_words: Optional[int] = None
    stream_max_words: Optional[int] = None
    stream_feedback_min_words: Optional[int] = None
    stream_max_initial_wait_seconds: Optional[float] = None
    stream_max_inter_chunk_wait_seconds: Optional[float] = None
    stream_playback_prebuffer_chunks: Optional[int] = None
    stream_playback_prebuffer_max_wait_seconds: Optional[float] = None


@dataclass
class LanguageOptionConfig:
    label: str = ""
    response_language: str = ""
    aliases: List[str] = field(default_factory=list)
    audio: LanguageAudioOverrides = field(default_factory=LanguageAudioOverrides)
    stt: LanguageSTTOverrides = field(default_factory=LanguageSTTOverrides)
    tts: LanguageTTSOverrides = field(default_factory=LanguageTTSOverrides)
    confirmation: str = ""
    online_model_message: Optional[str] = None
    offline_model_message: Optional[str] = None
    model_selection_message_template: str = ""
    remote_stt_message: str = ""
    local_stt_message: str = ""
    online_llm_model: Optional[str] = None
    target_switch_phrases: List[str] = field(default_factory=list)
    toggle_switch_phrases: List[str] = field(default_factory=list)


@dataclass
class LanguageConfig:
    enabled: bool = False
    default: str = ""
    persist: bool = False
    state_file: str = ""
    languages: Dict[str, LanguageOptionConfig] = field(default_factory=dict)


@dataclass
class RemoteLLMConfig:
    enabled: bool = False
    backend: str = ""
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    probe_timeout_seconds: float = 0.0
    request_timeout_seconds: float = 0.0
    online_message: Any = ""
    offline_message: Any = ""


@dataclass
class LLMConfig:
    backend: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    top_p: float = 0.0
    max_tokens: int = 0
    timeout_seconds: float = 0.0
    context_turns: int = 0
    enable_thinking: bool = False
    strip_reasoning: bool = False
    remote: RemoteLLMConfig = field(default_factory=RemoteLLMConfig)


@dataclass
class FilterConfig:
    enabled: bool = False
    ffmpeg_binary: str = ""
    preset: str = ""


@dataclass
class TTSConfig:
    backend: str = ""
    enabled: bool = False
    voice: str = ""
    output_dir: str = ""
    playback: bool = False
    playback_command: Optional[List[str]] = None
    piper_binary: str = ""
    piper_model: str = ""
    piper_config: Optional[str] = None
    piper_speaker: Optional[int] = None
    piper_pronunciation_replacements: Dict[str, str] = field(default_factory=dict)
    edge_voice: str = ""
    edge_rate: str = ""
    edge_pitch: str = ""
    edge_volume: str = ""
    length_scale: float = 0.0
    noise_scale: float = 0.0
    noise_w_scale: float = 0.0
    sentence_silence: float = 0.0
    volume: float = 0.0
    leading_silence_ms: int = 0
    stream_speech: bool = False
    adaptive_streaming: bool = False
    stream_initial_min_words: int = 0
    stream_min_words: int = 0
    stream_max_words: int = 0
    stream_feedback_min_words: int = 0
    stream_max_initial_wait_seconds: float = 0.0
    stream_max_inter_chunk_wait_seconds: float = 0.0
    stream_playback_prebuffer_chunks: int = 0
    stream_playback_prebuffer_max_wait_seconds: float = 0.0
    external_command: Optional[List[str]] = None
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class ToolConfig:
    enabled: bool = False
    current_tools_message: Dict[str, str] = field(default_factory=dict)
    tool_list_conjunction: Dict[str, str] = field(default_factory=dict)
    tool_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    allowed_commands: Dict[str, List[str]] = field(default_factory=dict)
    photo_command: Optional[List[str]] = None
    photo_short_side: int = 640
    photo_quality: int = 75
    photo_timeout_seconds: float = 8.0
    web_search_max_results: int = 0
    web_search_timeout_seconds: float = 0.0
    python_interpreter_timeout_seconds: float = 30.0
    python_interpreter_max_stdout_chars: int = 4000
    python_interpreter_max_stderr_chars: int = 2000
    python_interpreter_max_result_chars: int = 8000
    python_interpreter_memory_limit_mb: int = 256
    python_interpreter_file_size_limit_mb: int = 1
    python_interpreter_read_roots: List[str] = field(default_factory=lambda: ["files"])

    def is_tool_enabled(self, name: str) -> bool:
        if not self.enabled:
            return False
        setting = self.tool_settings.get(name)
        if not setting or not bool(setting.get("enabled")):
            return False
        return True


@dataclass
class PromptConfig:
    system_path: str = ""
    user_path: str = ""
    tools_path: str = ""
    memory_path: str = ""


@dataclass
class AgentConfig:
    default_response_language: str = ""


@dataclass
class MemoryConfig:
    auto_enabled: bool = False
    consolidation_enabled: bool = False
    include_assistant_text_online: bool = False
    include_assistant_text_offline: bool = False
    full_rewrite_interval_days: float = 0.0
    full_rewrite_requires_online: bool = False
    full_rewrite_recent_days: int = 0
    max_turns_per_update: int = 0
    max_candidates_for_rewrite: int = 0
    max_total_words: int = 0
    max_stable_facts: int = 0
    max_preferences: int = 0
    max_current_projects: int = 0
    max_recent_context: int = 0


@dataclass
class Config:
    profile_dir: str = "profiles/wheatley"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    idle_speech: IdleSpeechConfig = field(default_factory=IdleSpeechConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def __post_init__(self) -> None:
        _derive_profile_paths(self)

    def ensure_dirs(self) -> None:
        for path in [
            self.runtime.data_dir,
            self.runtime.state_dir,
            self.audio.utterance_dir,
            self.tts.output_dir,
            str(Path(self.runtime.turn_log).parent),
            str(Path(self.runtime.tool_log).parent),
            str(Path(self.runtime.system_llm_log).parent),
            str(Path(self.prompts.system_path).parent),
            str(Path(self.prompts.user_path).parent),
            str(Path(self.prompts.tools_path).parent),
            str(Path(self.prompts.memory_path).parent),
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def resolved_tts_backend(self) -> str:
        if self.tts.backend != "auto":
            return self.tts.backend
        if shutil.which("say"):
            return "macos_say"
        if shutil.which(self.tts.piper_binary) and Path(self.tts.piper_model).exists():
            return "piper"
        return "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str] = None, profile: Optional[str] = None) -> Config:
    cfg = Config()
    config_path = path
    if not config_path:
        profile_name = profile or "wheatley"
        profile_config = profile_config_path(profile_name)
        if profile_config.exists():
            config_path = str(profile_config)

    if config_path:
        config_file = Path(config_path)
        raw = load_jsonc(config_file)
        cfg = _apply_dict(cfg, raw)
        cfg.profile_dir = str(config_file.parent)
        _derive_profile_paths(cfg)

    cfg.ensure_dirs()
    return cfg


def profile_config_path(profile: str) -> Path:
    return Path("profiles") / profile / "config.jsonc"


def _apply_dict(cfg: Config, raw: Dict[str, Any]) -> Config:
    data = cfg.to_dict()
    _deep_update(data, raw)
    return Config(
        profile_dir=data.get("profile_dir", cfg.profile_dir),
        runtime=RuntimeConfig(**data["runtime"]),
        chat=ChatConfig(**data.get("chat", {})),
        idle_speech=_idle_speech_config_from_data(data.get("idle_speech", {})),
        audio=AudioConfig(**data["audio"]),
        stt=STTConfig(**data["stt"]),
        language=_language_config_from_data(data["language"]),
        llm=LLMConfig(
            **{
                **data["llm"],
                "remote": RemoteLLMConfig(**data["llm"].get("remote", {})),
            }
        ),
        tts=TTSConfig(
            **{
                **data["tts"],
                "filter": FilterConfig(**data["tts"].get("filter", {})),
            }
        ),
        tools=_tool_config_from_data(data["tools"]),
        prompts=PromptConfig(**data["prompts"]),
        agent=_agent_config_from_data(data.get("agent", {})),
        memory=MemoryConfig(**data.get("memory", {})),
    )


def _agent_config_from_data(data: Dict[str, Any]) -> AgentConfig:
    known = {item.name for item in fields(AgentConfig)}
    return AgentConfig(**{key: value for key, value in data.items() if key in known})


def _idle_speech_config_from_data(data: Dict[str, Any]) -> IdleSpeechConfig:
    if not isinstance(data, dict):
        data = {}
    known = {item.name for item in fields(IdleSpeechConfig)}
    return IdleSpeechConfig(
        **{key: value for key, value in data.items() if key in known}
    )


def _language_config_from_data(data: Dict[str, Any]) -> LanguageConfig:
    languages = {
        str(code): _language_option_from_data(option)
        for code, option in data.get("languages", {}).items()
    }
    return LanguageConfig(
        enabled=data.get("enabled", False),
        default=data.get("default", ""),
        persist=data.get("persist", False),
        state_file="language.json",
        languages=languages,
    )


def _language_option_from_data(raw: Dict[str, Any]) -> LanguageOptionConfig:
    if not isinstance(raw, dict):
        raise ValueError("Invalid language option: expected object")
    data = dict(raw)
    audio = _language_override_from_data(
        data,
        "audio",
        LanguageAudioOverrides,
        legacy_prefix="audio_",
    )
    stt = _language_override_from_data(
        data,
        "stt",
        LanguageSTTOverrides,
        legacy_prefix="stt_",
        legacy_names={"remote_model": "remote_stt_model"},
    )
    tts = _language_override_from_data(
        data,
        "tts",
        LanguageTTSOverrides,
        legacy_prefix="tts_",
    )
    return LanguageOptionConfig(**data, audio=audio, stt=stt, tts=tts)


def _language_override_from_data(
    data: Dict[str, Any],
    key: str,
    config_type: type,
    *,
    legacy_prefix: str,
    legacy_names: Optional[Dict[str, str]] = None,
):
    raw = data.pop(key, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid language.{key}: expected object")
    values = dict(raw)
    legacy_names = legacy_names or {}
    for item in fields(config_type):
        legacy_key = legacy_names.get(item.name, legacy_prefix + item.name)
        if legacy_key in data:
            values[item.name] = data.pop(legacy_key)
    return config_type(**values)


def _tool_config_from_data(data: Dict[str, Any]) -> ToolConfig:
    known = {item.name for item in fields(ToolConfig)}
    values = {key: value for key, value in data.items() if key in known}
    strict = bool(values.get("enabled", False))
    values["current_tools_message"] = _normalize_localized_strings(
        "tools.current_tools_message",
        data.get("current_tools_message", {}),
    )
    values["tool_list_conjunction"] = _normalize_localized_strings(
        "tools.tool_list_conjunction",
        data.get("tool_list_conjunction", {}),
    )
    tool_settings = _normalize_tool_settings(data.get("tool_settings"), strict=strict)
    values["tool_settings"] = tool_settings
    return ToolConfig(**values)


def _normalize_tool_settings(raw: Any, strict: bool) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        if strict:
            raise ValueError("Missing tools.tool_settings")
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Invalid tools.tool_settings: expected object")
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"Invalid tools.tool_settings.{name}: expected object with enabled/description"
            )
        item: Dict[str, Any] = {}
        if "enabled" not in value:
            raise ValueError(f"Missing tools.tool_settings.{name}.enabled")
        if not isinstance(value["enabled"], bool):
            raise ValueError(
                f"Invalid tools.tool_settings.{name}.enabled: expected boolean"
            )
        if "description" not in value:
            raise ValueError(f"Missing tools.tool_settings.{name}.description")
        description = value["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Invalid tools.tool_settings.{name}.description: expected non-empty string"
            )
        labels = _normalize_optional_localized_strings(
            name,
            "labels",
            value.get("labels", {}),
        )
        if "start_messages" not in value:
            raise ValueError(f"Missing tools.tool_settings.{name}.start_messages")
        start_messages = _normalize_tool_setting_start_messages(
            name,
            value["start_messages"],
        )
        instructions = value.get("instructions", "")
        if not isinstance(instructions, str):
            raise ValueError(
                f"Invalid tools.tool_settings.{name}.instructions: expected string"
            )
        item["enabled"] = value["enabled"]
        item["description"] = description.strip()
        item["labels"] = labels
        item["start_messages"] = start_messages
        item["instructions"] = instructions.strip()
        normalized[name] = item
    missing = sorted(REQUIRED_TOOL_SETTINGS.difference(normalized))
    if strict and missing:
        raise ValueError(
            "Missing tools.tool_settings entries: " + ", ".join(missing)
        )
    return normalized


def _normalize_tool_setting_start_messages(
    tool_name: str,
    raw: Any,
) -> Dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Invalid tools.tool_settings.{tool_name}.start_messages: expected object"
        )
    normalized: Dict[str, str] = {}
    for language, message in raw.items():
        if not isinstance(language, str):
            continue
        if not isinstance(message, str):
            raise ValueError(
                "Invalid "
                f"tools.tool_settings.{tool_name}.start_messages.{language}: "
                "expected string"
            )
        normalized[language.strip().lower()] = message.strip()
    return normalized


def _normalize_optional_localized_strings(
    tool_name: str,
    field_name: str,
    raw: Any,
) -> Dict[str, str]:
    return _normalize_localized_strings(
        f"tools.tool_settings.{tool_name}.{field_name}",
        raw,
    )


def _normalize_localized_strings(path: str, raw: Any) -> Dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid {path}: expected object")
    normalized: Dict[str, str] = {}
    for language, value in raw.items():
        if not isinstance(language, str):
            continue
        if not isinstance(value, str):
            raise ValueError(f"Invalid {path}.{language}: expected string")
        value = value.strip()
        if value:
            normalized[language.strip().lower()] = value
    return normalized


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _derive_profile_paths(cfg: Config) -> None:
    profile = Path(cfg.profile_dir)
    runtime = profile / "runtime"
    logs = runtime / "logs"
    audio = runtime / "audio"
    cfg.prompts.system_path = str(profile / "system.md")
    cfg.prompts.user_path = str(profile / "user.md")
    cfg.prompts.tools_path = str(profile / "tools.jsonc")
    cfg.prompts.memory_path = str(profile / "memory.md")
    cfg.runtime.data_dir = str(runtime)
    cfg.runtime.turn_log = str(logs / "turns.jsonl")
    cfg.runtime.tool_log = str(logs / "tools.jsonl")
    cfg.runtime.system_llm_log = str(logs / "system_llm.jsonl")
    cfg.runtime.state_dir = str(runtime / "state")
    cfg.audio.utterance_dir = str(audio)
    cfg.tts.output_dir = str(audio)
