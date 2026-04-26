from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from wheatly.jsonc import load_jsonc


@dataclass
class RuntimeConfig:
    data_dir: str = "runtime"
    turn_log: str = "runtime/logs/turns.jsonl"
    state_dir: str = "runtime/state"
    default_language: str = "en"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    vad_threshold: float = 0.018
    min_speech_seconds: float = 0.45
    silence_seconds: float = 0.75
    max_utterance_seconds: float = 14.0
    max_wait_seconds: float = 30.0
    utterance_dir: str = "runtime/audio"


@dataclass
class STTConfig:
    backend: str = "keyboard"
    model: str = "small.en"
    language: Optional[str] = "en"
    device: str = "cpu"
    compute_type: str = "int8"
    whisper_cpp_binary: str = "whisper-cli"
    whisper_cpp_model: str = "models/whisper/ggml-small.en.bin"
    whisper_cpp_args: List[str] = field(default_factory=lambda: ["--no-timestamps"])


@dataclass
class LLMConfig:
    backend: str = "echo"
    model: str = "qwen3.5:4b"
    base_url: str = "http://localhost:11434"
    api_key: str = "EMPTY"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 96
    timeout_seconds: float = 60.0
    context_turns: int = 8
    enable_thinking: bool = False


@dataclass
class FilterConfig:
    enabled: bool = True
    ffmpeg_binary: str = "ffmpeg"
    preset: str = "wheatley_light"


@dataclass
class TTSConfig:
    backend: str = "auto"
    enabled: bool = False
    voice: str = "Daniel"
    output_dir: str = "runtime/audio"
    playback: bool = True
    playback_command: Optional[List[str]] = None
    piper_binary: str = "piper"
    piper_model: str = "models/piper/en_GB-alan-medium.onnx"
    piper_config: Optional[str] = None
    piper_speaker: Optional[int] = None
    length_scale: float = 0.82
    noise_scale: float = 0.78
    noise_w_scale: float = 0.95
    sentence_silence: float = 0.08
    volume: float = 1.0
    stream_speech: bool = True
    adaptive_streaming: bool = True
    stream_initial_min_words: int = 8
    stream_min_words: int = 14
    stream_max_words: int = 34
    stream_feedback_min_words: int = 8
    stream_max_initial_wait_seconds: float = 2.0
    external_command: Optional[List[str]] = None
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class ToolConfig:
    enabled: bool = True
    allowed_commands: Dict[str, List[str]] = field(default_factory=dict)
    photo_command: Optional[List[str]] = None


@dataclass
class PromptConfig:
    system_path: str = "profiles/wheatly/system.md"
    user_path: str = "profiles/wheatly/user.md"
    tools_path: str = "profiles/wheatly/tools.jsonc"
    memory_path: str = "profiles/wheatly/memory.md"


@dataclass
class AgentConfig:
    name: str = "Wheatly"
    persona: str = "compact, fast, slightly nervous, helpful robot"
    default_response_language: str = "English"


@dataclass
class Config:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def ensure_dirs(self) -> None:
        for path in [
            self.runtime.data_dir,
            self.runtime.state_dir,
            self.audio.utterance_dir,
            self.tts.output_dir,
            str(Path(self.runtime.turn_log).parent),
            str(Path(self.prompts.system_path).parent),
            str(Path(self.prompts.user_path).parent),
            str(Path(self.prompts.tools_path).parent),
            str(Path(self.prompts.memory_path).parent),
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def resolved_tts_backend(self) -> str:
        if self.tts.backend != "auto":
            return self.tts.backend
        if platform.system() == "Darwin" and shutil.which("say"):
            return "macos_say"
        if shutil.which(self.tts.piper_binary) and Path(self.tts.piper_model).exists():
            return "piper"
        return "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str] = None, profile: Optional[str] = None) -> Config:
    cfg = Config()
    config_path = path or os.getenv("WHEATLY_CONFIG")
    if not config_path:
        profile_name = profile or os.getenv("WHEATLY_PROFILE", "wheatly")
        profile_config = profile_config_path(profile_name)
        if profile_config.exists():
            config_path = str(profile_config)

    if config_path:
        config_file = Path(config_path)
        raw = load_jsonc(config_file)
        cfg = _apply_dict(cfg, raw)
        _resolve_profile_paths(cfg, config_file.parent)

    cfg.ensure_dirs()
    return cfg


def profile_config_path(profile: str) -> Path:
    return Path("profiles") / profile / "config.jsonc"


def _apply_dict(cfg: Config, raw: Dict[str, Any]) -> Config:
    data = cfg.to_dict()
    _deep_update(data, raw)
    return Config(
        runtime=RuntimeConfig(**data["runtime"]),
        audio=AudioConfig(**data["audio"]),
        stt=STTConfig(**data["stt"]),
        llm=LLMConfig(**data["llm"]),
        tts=TTSConfig(
            **{
                **data["tts"],
                "filter": FilterConfig(**data["tts"].get("filter", {})),
            }
        ),
        tools=ToolConfig(**data["tools"]),
        prompts=PromptConfig(**data["prompts"]),
        agent=AgentConfig(**data["agent"]),
    )


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _resolve_profile_paths(cfg: Config, base_dir: Path) -> None:
    cfg.prompts.system_path = _resolve_relative(cfg.prompts.system_path, base_dir)
    cfg.prompts.user_path = _resolve_relative(cfg.prompts.user_path, base_dir)
    cfg.prompts.tools_path = _resolve_relative(cfg.prompts.tools_path, base_dir)
    cfg.prompts.memory_path = _resolve_relative(cfg.prompts.memory_path, base_dir)


def _resolve_relative(path: str, base_dir: Path) -> str:
    raw = Path(path)
    if raw.is_absolute():
        return str(raw)
    if len(raw.parts) > 1 and raw.parts[0] in {"profiles", "prompts", "memory"}:
        return str(raw)
    return str(base_dir / raw)
