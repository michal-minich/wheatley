from __future__ import annotations

import argparse
import json
import random
import shlex
import shutil
import sys
import threading
import textwrap
import time
from dataclasses import dataclass, replace
from pathlib import Path

from wheatley.audio.chimes import play_listening_chime
from wheatley.audio.devices import list_audio_devices
from wheatley.audio.log_paths import dated_audio_path
from wheatley.config import Config, load_config
from wheatley.doctor import diagnostics_json
from wheatley.language import (
    apply_configured_language,
    match_language_switch,
    read_language_state,
)
from wheatley.pipeline import VoiceAgent
from wheatley.runtime_stats import LatencyStats
from wheatley.stt.microphone import MicrophoneRecorder
from wheatley.stt.base import Transcription
from wheatley.text import normalize_words
from wheatley.tools.audit import log_tool_event
from wheatley.tools.announcements import tool_start_message
from wheatley.tools.builtins import build_registry
from wheatley.tts.backends import build_tts


@dataclass
class RecordedUtterance:
    path: Path
    partial_text: str = ""
    partial_age_seconds: float | None = None
    preview_rendered_lines: int = 0


@dataclass
class UserTextCapture:
    text: str = ""
    timed_out: bool = False
    partial_text: str = ""


EXIT_COMMANDS = {"quit", "exit"}
PAUSE_COMMANDS = {"pockaj", "pauza", "pause"}
RESUME_COMMANDS = {"pokracuj", "resume"}
RESUME_STARTUP_YES = {"yes", "yeah", "yep", "resume", "continue"}
RESUME_STARTUP_NO = {"no", "nope", "new", "start over", "start normally"}
NEW_CHAT_COMMANDS = {
    "new chat",
    "start new chat",
    "start a new chat",
    "clear chat",
    "reset chat",
    "reset conversation",
    "start a fresh chat",
}
IDLE_SPEECH_PROMPT_FILENAME = "idle.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wheatley")
    parser.add_argument(
        "--profile",
        help="Profile name under profiles/ to load. Defaults to wheatley.",
    )
    parser.add_argument(
        "--config",
        help="Explicit config.jsonc path. Defaults to the selected profile.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Show environment diagnostics.")
    sub.add_parser("audio-devices", help="List PortAudio input/output devices.")
    sub.add_parser("stats", help="Show adaptive latency stats.")
    sub.add_parser("tools", help="List available tool specs.")

    tool = sub.add_parser("tool", help="Execute one tool directly.")
    tool.add_argument("name")
    tool.add_argument("--args", default="{}", help="JSON object with tool arguments.")

    once = sub.add_parser("once", help="Run one text turn.")
    once.add_argument("--text", required=True)
    once.add_argument("--speak", action="store_true", help="Speak the response.")
    once.add_argument("--stream", action="store_true", help="Stream text as it is generated.")

    bench = sub.add_parser("bench", help="Run repeated text turns and print timing.")
    bench.add_argument("--text", default="Give me a short status update.")
    bench.add_argument("--repeat", type=int, default=3)

    chat = sub.add_parser("chat", help="Interactive text chat loop.")
    chat.add_argument("--speak", action="store_true", help="Speak responses.")
    chat.add_argument("--stream", action="store_true", help="Stream text as it is generated.")

    speak = sub.add_parser("speak", help="Speak text through configured TTS.")
    speak.add_argument("text")

    transcribe = sub.add_parser("transcribe", help="Transcribe an audio file.")
    transcribe.add_argument("audio_path")

    stt_server = sub.add_parser("stt-server", help="Serve remote STT over HTTP.")
    stt_server.add_argument("--host", default="0.0.0.0")
    stt_server.add_argument("--port", type=int, default=8765)
    stt_server.add_argument("--backend", default="faster_whisper")
    stt_server.add_argument(
        "--default-model",
        default="small",
    )
    stt_server.add_argument("--model", action="append", default=[], help="language=model")
    stt_server.add_argument("--device", default="cpu")
    stt_server.add_argument("--compute-type", default="int8")
    stt_server.add_argument("--beam-size", type=int, default=1)
    stt_server.add_argument("--no-vad-filter", action="store_true")
    stt_server.add_argument("--condition-on-previous-text", action="store_true")

    listen = sub.add_parser("listen", help="Record one utterance, transcribe and answer.")
    listen.add_argument("--speak", action="store_true", help="Speak the response.")

    voice = sub.add_parser("voice", help="Continuous microphone voice loop.")
    voice.add_argument("--turns", type=int, default=0, help="Stop after N turns; 0 means forever.")
    voice.add_argument("--no-speak", action="store_true", help="Do not speak responses.")
    voice.add_argument("--no-stream", action="store_true", help="Disable token streaming.")

    args = parser.parse_args(argv)

    if args.command == "stt-server":
        from wheatley.stt.server import main as stt_server_main

        server_args = [
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--backend",
            args.backend,
            "--default-model",
            args.default_model,
            "--device",
            args.device,
            "--compute-type",
            args.compute_type,
            "--beam-size",
            str(args.beam_size),
        ]
        if args.no_vad_filter:
            server_args.append("--no-vad-filter")
        if args.condition_on_previous_text:
            server_args.append("--condition-on-previous-text")
        for model in args.model:
            server_args.extend(["--model", model])
        return stt_server_main(server_args)

    cfg = load_config(path=args.config, profile=args.profile)

    if args.command == "doctor":
        print(diagnostics_json(cfg))
        return 0

    if args.command == "audio-devices":
        print(json.dumps(list_audio_devices(), indent=2, ensure_ascii=True))
        return 0

    if args.command == "stats":
        stats = LatencyStats(Path(cfg.runtime.state_dir) / "latency_stats.json")
        payload = dict(stats.data.__dict__)
        payload["recommended_initial_words"] = stats.recommended_initial_words(
            min_words=cfg.tts.stream_initial_min_words,
            max_words=cfg.tts.stream_max_words,
            adaptive=cfg.tts.adaptive_streaming,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "tools":
        registry = build_registry(cfg)
        print(json.dumps([spec.__dict__ for spec in registry.specs()], indent=2))
        return 0

    if args.command == "tool":
        from wheatley.tools.registry import ToolCall

        apply_configured_language(cfg, read_language_state(cfg))
        registry = build_registry(cfg)
        try:
            tool_args = json.loads(args.args)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for --args: {exc}") from exc
        call = ToolCall(args.name, tool_args)
        message = tool_start_message(cfg, call.name)
        if message:
            _print_tool_start(call.name, message, "cli", call.arguments)
        started_at = time.perf_counter()
        result = registry.execute(call)
        log_tool_event(
            cfg.runtime.tool_log,
            call,
            result,
            source="cli",
            duration_seconds=time.perf_counter() - started_at,
        )
        print(json.dumps(result.__dict__, indent=2, default=str))
        return 0

    if args.command == "speak":
        cfg.tts.enabled = True
        build_tts(cfg).speak(args.text)
        return 0

    agent = VoiceAgent(cfg, on_tool_start=_print_tool_start)

    if args.command == "once":
        if args.speak:
            cfg.tts.enabled = True
        _announce_model_selection(agent, speak=args.speak)
        if args.stream:
            _handle_text_turn(agent, args.text, speak=args.speak, stream=True)
        else:
            _handle_text_turn(agent, args.text, speak=args.speak, stream=False)
        return 0

    if args.command == "bench":
        _announce_model_selection(agent, speak=False)
        return _bench(agent, args.text, args.repeat)

    if args.command == "chat":
        if args.speak:
            cfg.tts.enabled = True
        _announce_model_selection(agent, speak=args.speak)
        return _chat_loop(agent, speak=args.speak, stream=args.stream)

    if args.command == "transcribe":
        result = agent.transcribe(Path(args.audio_path))
        print(result.text)
        return 0

    if args.command == "listen":
        if args.speak:
            cfg.tts.enabled = True
        _announce_model_selection(agent, speak=args.speak)
        recorder = MicrophoneRecorder(cfg.audio)
        audio_path = dated_audio_path(Path(cfg.audio.utterance_dir), "user")
        text = _record_user_text(agent, cfg, recorder, audio_path)
        if not text:
            return 0
        _handle_text_turn(agent, text, speak=args.speak, stream=False)
        return 0

    if args.command == "voice":
        cfg.tts.enabled = not args.no_speak
        resume_turns = _maybe_resume_recent_turns_on_start(agent, cfg)
        selection = agent.reset_chat(refresh_memory=False)
        if resume_turns:
            agent.restore_turn_history(resume_turns)
            _replay_turns(resume_turns, cfg)
        _refresh_memory_then_announce_tools(
            agent,
            speak=not args.no_speak,
            status_message=selection.message,
        )
        return _voice_loop(
            agent,
            cfg,
            speak=not args.no_speak,
            turns=args.turns,
            stream=not args.no_stream,
        )

    parser.print_help()
    return 2


def _chat_loop(agent: VoiceAgent, speak: bool, stream: bool) -> int:
    print("Wheatley text chat. Ctrl-D or empty line exits.")
    while True:
        try:
            text = input(_prefix("you", "yellow")).strip()
        except EOFError:
            print()
            return 0
        if not text:
            return 0
        if _is_exit_command(text):
            return 0
        if _is_new_chat_command(text):
            _start_new_chat(agent, speak=speak)
            continue
        _handle_text_turn(agent, text, speak=speak, stream=stream)


def _handle_text_turn(agent: VoiceAgent, text: str, speak: bool, stream: bool) -> None:
    assistant_name = _assistant_prompt_name(agent.cfg)
    speech_stream = speak and agent.cfg.tts.stream_speech
    if (stream or speech_stream) and not match_language_switch(agent.cfg, text):
        result = _print_streamed_turn(
            agent,
            text,
            speak=speak,
            print_tokens=stream,
            name=assistant_name,
        )
        if _is_language_switch_result(result):
            _print_language_turn(result.assistant_text)
        return
    result = agent.handle_text(text, speak=speak)
    if _is_language_switch_result(result):
        _print_language_turn(result.assistant_text)
    else:
        _print_turn(result.assistant_text, name=assistant_name)


def _print_turn(text: str, name: str = "wheatley", color: str = "orange") -> None:
    sys.stdout.write(f"{_prefix(name, color)}{text}\n")
    sys.stdout.flush()


def _print_language_turn(text: str) -> None:
    _print_turn(text, name="language", color="blue")


def _print_tool_start(
    tool_name: str,
    message: str,
    source: str = "llm",
    arguments: dict | None = None,
) -> None:
    del source
    sys.stdout.write(f"{_color(message, 'cyan')}\n")
    input_text = _tool_input_preview(tool_name, arguments or {})
    if input_text:
        sys.stdout.write(f"{_color(input_text, 'dark_orange')}\n")
    sys.stdout.flush()


def _tool_input_preview(tool_name: str, arguments: dict) -> str:
    if tool_name == "python_interpreter":
        return str(arguments.get("code", "")).strip()
    if tool_name == "calculator":
        return str(arguments.get("expression", "")).strip()
    if tool_name == "remember":
        return str(arguments.get("memory", "")).strip()
    if tool_name == "run_safe_cli_tool":
        command = str(arguments.get("command") or arguments.get("name") or "").strip()
        extra = arguments.get("args", [])
        if not isinstance(extra, list):
            extra = []
        parts = [command] if command else []
        parts.extend(str(part) for part in extra)
        return " ".join(shlex.quote(part) for part in parts)
    return ""


def _print_streamed_turn(
    agent: VoiceAgent,
    text: str,
    speak: bool,
    print_tokens: bool = True,
    name: str = "wheatley",
):
    printed_prefix = False
    emitted_tokens = False

    if print_tokens:
        sys.stdout.write(_prefix(name, "orange"))
        sys.stdout.flush()
        printed_prefix = True

    def on_token(token: str) -> None:
        nonlocal emitted_tokens
        if not print_tokens:
            return
        emitted_tokens = True
        sys.stdout.write(token)
        sys.stdout.flush()

    result = agent.handle_text_stream(
        text,
        speak=speak,
        on_token=on_token if print_tokens else None,
    )
    if printed_prefix:
        if (
            not emitted_tokens
            and result.assistant_text
            and not _is_language_switch_result(result)
        ):
            sys.stdout.write(result.assistant_text)
        sys.stdout.write("\n")
    elif result.assistant_text and not _is_language_switch_result(result):
        sys.stdout.write(f"{_prefix(name, 'orange')}{result.assistant_text}\n")
    sys.stdout.flush()
    return result


def _handle_idle_speech(
    agent: VoiceAgent,
    cfg: Config,
    speak: bool,
    stream: bool,
) -> None:
    instruction = _idle_speech_instruction(cfg)
    assistant_name = _assistant_prompt_name(cfg)
    speech_stream = speak and agent.cfg.tts.stream_speech
    if stream or speech_stream:
        _print_streamed_idle_speech(
            agent,
            instruction,
            speak=speak,
            print_tokens=stream,
            name=assistant_name,
        )
        return
    result = agent.handle_idle_speech(instruction, speak=speak)
    if result.assistant_text:
        _print_turn(result.assistant_text, name=assistant_name)


def _print_streamed_idle_speech(
    agent: VoiceAgent,
    instruction: str,
    speak: bool,
    print_tokens: bool = True,
    name: str = "wheatley",
):
    printed_prefix = False
    emitted_tokens = False

    if print_tokens:
        sys.stdout.write(_prefix(name, "orange"))
        sys.stdout.flush()
        printed_prefix = True

    def on_token(token: str) -> None:
        nonlocal emitted_tokens
        if not print_tokens:
            return
        emitted_tokens = True
        sys.stdout.write(token)
        sys.stdout.flush()

    result = agent.handle_idle_speech(
        instruction,
        speak=speak,
        on_token=on_token if print_tokens else None,
    )
    if printed_prefix:
        if not emitted_tokens and result.assistant_text:
            sys.stdout.write(result.assistant_text)
        sys.stdout.write("\n")
    elif result.assistant_text:
        sys.stdout.write(f"{_prefix(name, 'orange')}{result.assistant_text}\n")
    sys.stdout.flush()
    return result


def _idle_speech_available(cfg: Config, speak: bool) -> bool:
    return (
        speak
        and cfg.idle_speech.enabled
        and cfg.idle_speech.interval_seconds > 0
    )


def _next_idle_speech_wait_seconds(cfg: Config) -> float:
    base = max(0.0, float(cfg.idle_speech.interval_seconds))
    if base <= 0:
        return 0.0
    low = max(0.0, float(cfg.idle_speech.random_min_multiplier))
    high = max(0.0, float(cfg.idle_speech.random_max_multiplier))
    if low <= 0 and high <= 0:
        low = high = 1.0
    if high < low:
        low, high = high, low
    return base * random.uniform(low, high)


def _idle_speech_instruction(cfg: Config) -> str:
    prompt = _read_idle_speech_prompt(cfg)
    if prompt:
        return prompt
    return (
        "No one has spoken for a while and preview speech recognition produced no "
        "text. Make one brief, speech-ready idle remark. Use the recent "
        "conversation or memory if useful. Be interesting, varied, non-demanding, "
        "and not too long. Do not use tools."
    )


def _read_idle_speech_prompt(cfg: Config) -> str:
    path = Path(cfg.profile_dir) / IDLE_SPEECH_PROMPT_FILENAME
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _bench(agent: VoiceAgent, text: str, repeat: int) -> int:
    repeat = max(1, repeat)
    rows = []
    for index in range(repeat):
        started = time.perf_counter()
        result = agent.handle_text(text, speak=False)
        wall = round(time.perf_counter() - started, 4)
        approx_tokens = max(1, len(result.assistant_text.split()))
        row = {
            "run": index + 1,
            "wall_seconds": wall,
            "agent_seconds": result.duration_seconds,
            "approx_output_words": approx_tokens,
            "approx_words_per_second": round(approx_tokens / wall, 2),
            "tool_calls": [item.name for item in result.tool_results],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True))
    avg = sum(row["wall_seconds"] for row in rows) / len(rows)
    print(json.dumps({"average_wall_seconds": round(avg, 4)}, ensure_ascii=True))
    return 0


def _voice_loop(
    agent: VoiceAgent,
    cfg: Config,
    speak: bool,
    turns: int,
    stream: bool,
) -> int:
    idle_enabled = _idle_speech_available(cfg, speak)
    idle_due_at: float | None = None

    def reset_idle_wait() -> None:
        nonlocal idle_due_at
        if idle_enabled:
            idle_due_at = time.monotonic() + _next_idle_speech_wait_seconds(cfg)

    def idle_wait_remaining_seconds() -> float:
        if not idle_enabled or idle_due_at is None:
            return 0.0
        return max(0.05, idle_due_at - time.monotonic())

    def idle_wait_reached() -> bool:
        return bool(
            idle_enabled
            and idle_due_at is not None
            and time.monotonic() >= idle_due_at
        )

    reset_idle_wait()

    print("Say 'quit' or 'exit' or press Ctrl-C to end.")
    count = 0
    paused = False
    while True:
        if turns and count >= turns:
            return 0
        try:
            wait_seconds = idle_wait_remaining_seconds() if not paused else 0.0
            voice_audio = replace(cfg.audio, max_wait_seconds=wait_seconds)
            recorder = MicrophoneRecorder(voice_audio)
            audio_path = dated_audio_path(
                Path(cfg.audio.utterance_dir),
                "user",
                extra=f"{count + 1:04d}",
            )
            capture = _record_user_text_result(agent, cfg, recorder, audio_path)
            if capture.timed_out:
                if idle_enabled and not paused and not capture.partial_text:
                    _handle_idle_speech(agent, cfg, speak=speak, stream=stream)
                    reset_idle_wait()
                continue
            text = capture.text
            if not text:
                if (
                    idle_enabled
                    and not paused
                    and not capture.partial_text
                    and idle_wait_reached()
                ):
                    _handle_idle_speech(agent, cfg, speak=speak, stream=stream)
                    reset_idle_wait()
                elif capture.partial_text:
                    reset_idle_wait()
                continue
            if _is_exit_command(text):
                return 0
            if _is_pause_command(text):
                paused = True
                _print_turn(
                    _paused_prompt_text(cfg),
                    name=_assistant_prompt_name(cfg),
                    color="red",
                )
                count += 1
                reset_idle_wait()
                continue
            if paused:
                if _is_resume_command(text):
                    paused = False
                    reset_idle_wait()
                count += 1
                continue
            if _is_new_chat_command(text):
                _start_new_chat(agent, speak=speak)
                count += 1
                reset_idle_wait()
                continue
            if stream:
                _handle_text_turn(agent, text, speak=speak, stream=True)
            else:
                _handle_text_turn(agent, text, speak=speak, stream=False)
            count += 1
            reset_idle_wait()
        except KeyboardInterrupt:
            print()
            return 0
        except Exception as exc:
            print(f"error> {exc}", file=sys.stderr)
            return 1


def _maybe_resume_recent_turns_on_start(agent: VoiceAgent, cfg) -> list[dict]:
    mode = _startup_resume_mode(cfg)
    if mode == "no":
        return []
    if mode == "ask" and not cfg.chat.resume_on_start:
        return []
    turns = _load_recent_turns(cfg.runtime.turn_log, cfg.chat.resume_turns)
    if not turns:
        return []
    if mode == "yes":
        return turns
    if not _ask_resume_recent_turns(agent, cfg, len(turns)):
        return []
    return turns


def _startup_resume_mode(cfg: Config) -> str:
    mode = str(getattr(cfg.chat, "resume_on_start_mode", "ask")).strip().lower()
    if mode == "auto":
        return "ask"
    if mode in {"ask", "yes", "no"}:
        return mode
    return "ask"


def _load_recent_turns(path: str, limit: int) -> list[dict]:
    if limit <= 0:
        return []
    log_path = Path(path)
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    turns = [
        row
        for row in (json.loads(line) for line in lines)
        if _is_restorable_turn(row)
    ]
    return turns[-limit:]


def _is_restorable_turn(row: dict) -> bool:
    if str(row.get("source", "")).strip().lower() == "idle":
        return False
    return bool(str(row.get("user_text", "")).strip())


def _ask_resume_recent_turns(agent: VoiceAgent, cfg, turn_count: int) -> bool:
    while True:
        text = _listen_for_resume_decision(agent, cfg, turn_count).strip()
        decision = _resume_startup_decision(text)
        if decision is not None:
            return decision


def _listen_for_resume_decision(agent: VoiceAgent, cfg, turn_count: int) -> str:
    seconds = max(1, int(cfg.chat.resume_countdown_seconds))
    prompt = f"Resume last {turn_count} turns? "
    _print_resume_countdown(prompt, seconds)
    if cfg.tts.enabled:
        agent.tts.speak(prompt.strip())
    stop_countdown = threading.Event()
    countdown = threading.Thread(
        target=_run_resume_countdown,
        args=(prompt, seconds, stop_countdown),
        daemon=True,
    )
    countdown.start()
    try:
        resume_audio = replace(
            cfg.audio,
            max_wait_seconds=float(seconds),
            silence_seconds=min(cfg.audio.silence_seconds, 0.8),
            max_utterance_seconds=min(cfg.audio.max_utterance_seconds, 5.0),
            partial_transcript_enabled=False,
            partial_transcript_use_as_final=False,
        )
        recorder = MicrophoneRecorder(resume_audio)
        audio_path = dated_audio_path(
            Path(cfg.audio.utterance_dir),
            "resume_decision",
            subdir="resume",
        )
        recorded = recorder.record_utterance(audio_path)
    except TimeoutError:
        return ""
    finally:
        stop_countdown.set()
        countdown.join(timeout=0.2)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return agent.transcribe(recorded).text


def _run_resume_countdown(prompt: str, seconds: int, stop_event: threading.Event) -> None:
    for remaining in range(seconds - 1, 0, -1):
        if stop_event.wait(1):
            return
        _print_resume_countdown(prompt, remaining)


def _print_resume_countdown(prompt: str, remaining: int) -> None:
    sys.stdout.write(f"\r\033[2K{_prefix('system', 'light_blue')}{prompt}{remaining}")
    sys.stdout.flush()


def _resume_startup_decision(text: str) -> bool | None:
    normalized = _normalize_voice_command(text)
    if not normalized:
        return False
    if normalized in RESUME_STARTUP_YES:
        return True
    if normalized in RESUME_STARTUP_NO:
        return False
    return None


def _replay_turns(turns: list[dict], cfg: Config) -> None:
    assistant_name = _assistant_prompt_name(cfg)
    for turn in turns:
        _print_user(str(turn["user_text"]))
        _print_turn(str(turn["assistant_text"]), name=assistant_name)


def _assistant_prompt_name(cfg: Config) -> str:
    name = Path(cfg.profile_dir).name.strip()
    return name or "wheatley"


def _color(text: str, color: str) -> str:
    codes = {
        "green": "32",
        "red": "31",
        "yellow": "33",
        "orange": "38;5;208",
        "dark_orange": "38;5;130",
        "cyan": "36",
        "light_blue": "94",
        "magenta": "35",
        "blue": "34",
    }
    code = codes.get(color)
    if not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def _prefix(name: str, color: str) -> str:
    return _color(f"{name}> ", color)


def _print_user(text: str) -> None:
    sys.stdout.write(f"{_prefix('you', 'yellow')}{text}\n")
    sys.stdout.flush()


def _record_user_text(
    agent: VoiceAgent,
    cfg: Config,
    recorder: MicrophoneRecorder,
    audio_path: Path,
) -> str:
    return _record_user_text_result(agent, cfg, recorder, audio_path).text


def _record_user_text_result(
    agent: VoiceAgent,
    cfg: Config,
    recorder: MicrophoneRecorder,
    audio_path: Path,
) -> UserTextCapture:
    print(_color("listening...", "green"))
    play_listening_chime("start", cfg.audio)
    try:
        recorded = _record_with_partial_transcript(
            recorder,
            audio_path,
            _build_partial_transcriber(agent, cfg),
        )
    except TimeoutError:
        _clear_previous_terminal_lines(1)
        return UserTextCapture(timed_out=True)
    print(_color("stopped listening.", "red"))
    play_listening_chime("stop", cfg.audio)
    transcription = _transcribe_with_status(agent, recorded, cfg)
    text = transcription.text.strip()
    _clear_previous_terminal_lines(1 + recorded.preview_rendered_lines)
    if text:
        _print_user(text)
    return UserTextCapture(
        text=text,
        timed_out=False,
        partial_text=recorded.partial_text,
    )


def _transcribe_with_status(agent: VoiceAgent, recorded: RecordedUtterance, cfg):
    if _can_use_partial_as_final(recorded, cfg):
        return Transcription(
            text=recorded.partial_text,
            language=cfg.stt.language,
            duration_seconds=None,
        )
    return agent.transcribe_final(recorded.path)


def _clear_previous_terminal_line() -> None:
    sys.stdout.write("\033[1A\r\033[2K")
    sys.stdout.flush()


def _clear_previous_terminal_lines(line_count: int) -> None:
    for _ in range(max(0, line_count)):
        _clear_previous_terminal_line()


def _can_use_partial_as_final(recorded: RecordedUtterance, cfg) -> bool:
    if not cfg.audio.partial_transcript_use_as_final:
        return False
    if not recorded.partial_text:
        return False
    if recorded.partial_age_seconds is None:
        return False
    return recorded.partial_age_seconds <= cfg.audio.partial_transcript_final_max_age_seconds


def _is_language_switch_result(result) -> bool:
    return any(item.name == "set_language" and item.ok for item in result.tool_results)


def _record_with_partial_transcript(
    recorder: MicrophoneRecorder, audio_path: Path, partial_transcriber
) -> RecordedUtterance:
    preview = _PartialTranscriptPreview(partial_transcriber)
    try:
        path = recorder.record_utterance(
            audio_path,
            partial_transcriber=preview.transcribe if preview.enabled else None,
            on_partial_transcript=preview.update if preview.enabled else None,
        )
    finally:
        preview.finish()
    return RecordedUtterance(
        path=path,
        partial_text=preview.last_text,
        partial_age_seconds=preview.partial_age_seconds(),
        preview_rendered_lines=preview.rendered_lines,
    )


class _PartialTranscriptPreview:
    def __init__(self, transcriber):
        self.enabled = transcriber is not None
        self.closed = False
        self.used = False
        self.last_text = ""
        self.last_update_at = None
        self.rendered_lines = 0
        self.lock = threading.Lock()
        self.transcriber = transcriber

    def transcribe(self, audio_path: Path) -> str:
        if not self.transcriber:
            return ""
        return self.transcriber(audio_path)

    def update(self, text: str) -> None:
        text = " ".join(text.split())
        if not text:
            return
        with self.lock:
            if self.closed:
                return
            lines = _format_preview_block("you", "yellow", text)
            self._clear_rendered_lines()
            self.used = True
            self.last_text = text
            self.last_update_at = time.monotonic()
            self.rendered_lines = len(lines)
            sys.stdout.write("\n".join(lines))
            sys.stdout.flush()

    def finish(self) -> None:
        with self.lock:
            self.closed = True
            if self.used:
                # Keep the latest partial transcript visible until final STT replaces it.
                sys.stdout.write("\n")
                sys.stdout.flush()

    def _clear_rendered_lines(self) -> None:
        if self.rendered_lines <= 0:
            return
        sys.stdout.write("\r\033[2K")
        for _ in range(self.rendered_lines - 1):
            sys.stdout.write("\033[1A\r\033[2K")
        self.rendered_lines = 0

    def partial_age_seconds(self) -> float | None:
        if self.last_update_at is None:
            return None
        return time.monotonic() - self.last_update_at


def _build_partial_transcriber(agent: VoiceAgent, cfg):
    backend = cfg.stt.backend.lower().replace("-", "_")
    if not cfg.audio.partial_transcript_enabled or backend == "keyboard":
        return None
    return lambda audio_path: agent.transcribe_preview(audio_path).text


def _format_preview_block(prefix_name: str, color: str, text: str) -> list[str]:
    columns = shutil.get_terminal_size((120, 20)).columns
    plain_prefix = f"{prefix_name}> "
    width = max(20, columns - len(plain_prefix))
    parts = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    continuation = " " * len(plain_prefix)
    lines = [f"{_prefix(prefix_name, color)}{parts[0]}"]
    lines.extend(f"{continuation}{part}" for part in parts[1:])
    return lines


def _announce_model_selection(
    agent: VoiceAgent,
    speak: bool,
    refresh_memory: bool = True,
) -> None:
    selection = agent.reset_chat(refresh_memory=False)
    if not refresh_memory:
        _announce_system_status(agent, selection.message, speak=speak)
        return
    _refresh_memory_then_announce_tools(
        agent,
        speak=speak,
        status_message=selection.message,
    )


def _start_new_chat(agent: VoiceAgent, speak: bool) -> None:
    selection = agent.reset_chat(refresh_memory=False)
    message = "Starting a new chat."
    _print_turn(message)
    if speak:
        agent.tts.speak(message)
    _refresh_memory_then_announce_tools(
        agent,
        speak=speak,
        status_message=selection.message,
    )


def _refresh_memory_then_announce_tools(
    agent: VoiceAgent,
    speak: bool,
    status_message: str = "",
) -> None:
    tools_message = agent.current_tools_message()
    status_announced = False
    tools_announced = False

    def announce_status() -> None:
        nonlocal status_announced
        if not status_message or status_announced:
            return
        status_announced = True
        _announce_system_status(agent, status_message, speak=speak)

    def notify(message: str) -> None:
        nonlocal status_announced, tools_announced
        if status_message and message == status_message:
            status_announced = True
            _print_memory_status(message)
            return
        if tools_message and message == tools_message:
            announce_status()
            tools_announced = True
        _print_memory_status(message)

    agent.refresh_auto_memory(
        notify_memory=notify,
        speak_memory=speak,
        start_messages=[status_message] if status_message else None,
    )
    announce_status()
    if not tools_message or tools_announced:
        return
    _print_memory_status(tools_message)
    if not (speak and agent.cfg.tts.enabled):
        return
    try:
        agent.tts.speak(tools_message)
    except Exception:
        return


def _announce_system_status(agent: VoiceAgent, message: str, speak: bool) -> None:
    _print_memory_status(message)
    if not speak:
        return
    try:
        agent.tts.speak(message)
    except Exception:
        return


def _print_memory_status(message: str) -> None:
    sys.stdout.write(f"{_prefix('system', 'light_blue')}{message}\n")
    sys.stdout.flush()


def _paused_prompt_text(cfg) -> str:
    language = read_language_state(cfg) if cfg.language.enabled else cfg.language.default
    if language == "sk":
        return "Pozastavené, povedz 'pokračuj' pre pokračovanie."
    return "Paused, say 'resume' to continue."


def _is_exit_command(text: str) -> bool:
    return _normalize_voice_command(text) in EXIT_COMMANDS


def _is_pause_command(text: str) -> bool:
    return _normalize_voice_command(text) in PAUSE_COMMANDS


def _is_resume_command(text: str) -> bool:
    return _normalize_voice_command(text) in RESUME_COMMANDS


def _is_new_chat_command(text: str) -> bool:
    return _normalize_voice_command(text) in NEW_CHAT_COMMANDS


def _normalize_voice_command(text: str) -> str:
    return normalize_words(text)


if __name__ == "__main__":
    raise SystemExit(main())
