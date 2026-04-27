from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import threading
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from wheatley.audio.chimes import play_listening_chime
from wheatley.config import load_config
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
from wheatley.tools.audit import log_tool_event
from wheatley.tools.announcements import tool_start_message
from wheatley.tools.builtins import build_registry
from wheatley.tts.backends import build_tts


@dataclass
class RecordedUtterance:
    path: Path
    partial_text: str = ""
    partial_age_seconds: float | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wheatley")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Show environment diagnostics.")
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
    stt_server.add_argument("--default-model", default="small.en")
    stt_server.add_argument("--model", action="append", default=[], help="language=model")
    stt_server.add_argument("--device", default="cpu")
    stt_server.add_argument("--compute-type", default="int8")

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
        ]
        for model in args.model:
            server_args.extend(["--model", model])
        return stt_server_main(server_args)

    cfg = load_config()

    if args.command == "doctor":
        print(diagnostics_json(cfg))
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
            _print_tool_start(call.name, message)
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
        partial_transcriber = _build_partial_transcriber(agent, cfg)
        audio_path = (
            Path(cfg.audio.utterance_dir)
            / f"utterance_{time.time_ns()}.wav"
        )
        print(_color("listening...", "green"))
        play_listening_chime("start", cfg.audio)
        recorded = _record_with_partial_transcript(
            recorder, audio_path, partial_transcriber
        )
        print(_color("stopped listening.", "red"))
        play_listening_chime("stop", cfg.audio)
        transcription = _transcribe_with_status(agent, recorded, cfg)
        _print_user(transcription.text)
        _handle_text_turn(
            agent, transcription.text, speak=args.speak, stream=False
        )
        return 0

    if args.command == "voice":
        cfg.tts.enabled = not args.no_speak
        _announce_model_selection(agent, speak=not args.no_speak)
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
    speech_stream = speak and agent.cfg.tts.stream_speech
    if (stream or speech_stream) and not match_language_switch(agent.cfg, text):
        result = _print_streamed_turn(
            agent,
            text,
            speak=speak,
            print_tokens=stream,
        )
        if _is_language_switch_result(result):
            _print_language_turn(result.assistant_text)
        return
    result = agent.handle_text(text, speak=speak)
    if _is_language_switch_result(result):
        _print_language_turn(result.assistant_text)
    else:
        _print_turn(result.assistant_text)


def _print_turn(text: str, name: str = "wheatley", color: str = "orange") -> None:
    sys.stdout.write(f"{_prefix(name, color)}{text}\n")
    sys.stdout.flush()


def _print_language_turn(text: str) -> None:
    _print_turn(text, name="language", color="blue")


def _print_tool_start(tool_name: str, message: str) -> None:
    del tool_name
    sys.stdout.write(f"{_color(f'tool> {message}', 'cyan')}\n")
    sys.stdout.flush()


def _print_streamed_turn(agent: VoiceAgent, text: str, speak: bool, print_tokens: bool = True):
    printed_prefix = False

    def on_token(token: str) -> None:
        nonlocal printed_prefix
        if not print_tokens:
            return
        if not printed_prefix:
            sys.stdout.write(_prefix("wheatley", "orange"))
            printed_prefix = True
        sys.stdout.write(token)
        sys.stdout.flush()

    result = agent.handle_text_stream(
        text,
        speak=speak,
        on_token=on_token if print_tokens else None,
    )
    if printed_prefix:
        sys.stdout.write("\n")
    elif result.assistant_text and not _is_language_switch_result(result):
        sys.stdout.write(f"{_prefix('wheatley', 'orange')}{result.assistant_text}\n")
    sys.stdout.flush()
    return result


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


def _voice_loop(agent: VoiceAgent, cfg, speak: bool, turns: int, stream: bool) -> int:
    recorder = MicrophoneRecorder(cfg.audio)
    print("Wheatley voice loop. Say 'stop', 'quit', or press Ctrl-C to exit.")
    count = 0
    while True:
        if turns and count >= turns:
            return 0
        try:
            audio_path = (
                Path(cfg.audio.utterance_dir)
                / f"utterance_{time.time_ns()}_{count + 1}.wav"
            )
            print(_color("listening...", "green"))
            play_listening_chime("start", cfg.audio)
            partial_transcriber = _build_partial_transcriber(agent, cfg)
            recorded = _record_with_partial_transcript(
                recorder, audio_path, partial_transcriber
            )
            print(_color("stopped listening.", "red"))
            play_listening_chime("stop", cfg.audio)
            transcription = _transcribe_with_status(agent, recorded, cfg)
            text = transcription.text.strip()
            _print_user(text)
            if _is_exit_command(text):
                return 0
            if _is_new_chat_command(text):
                _start_new_chat(agent, speak=speak)
                count += 1
                continue
            if stream:
                _handle_text_turn(agent, text, speak=speak, stream=True)
            else:
                _handle_text_turn(agent, text, speak=speak, stream=False)
            count += 1
        except KeyboardInterrupt:
            print()
            return 0
        except TimeoutError as exc:
            print(f"timeout> {exc}")
        except Exception as exc:
            print(f"error> {exc}", file=sys.stderr)
            return 1


def _color(text: str, color: str) -> str:
    codes = {
        "green": "32",
        "red": "31",
        "yellow": "33",
        "orange": "38;5;208",
        "cyan": "36",
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


def _transcribe_with_status(agent: VoiceAgent, recorded: RecordedUtterance, cfg):
    if _can_use_partial_as_final(recorded, cfg):
        return Transcription(
            text=recorded.partial_text,
            language=cfg.stt.language,
            duration_seconds=None,
        )
    print(_color("transcribing...", "cyan"))
    return agent.transcribe(recorded.path)


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
            lines = _format_preview_block("you~", "yellow", text)
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
                self._clear_rendered_lines()
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
    return lambda audio_path: agent.transcribe(audio_path).text


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


def _announce_model_selection(agent: VoiceAgent, speak: bool) -> None:
    selection = agent.reset_chat(refresh_memory=False)
    _print_model_selection(selection)
    if speak:
        agent.tts.speak(selection.message)
    agent.refresh_auto_memory(
        notify_memory=_print_memory_status,
        speak_memory=speak,
    )


def _start_new_chat(agent: VoiceAgent, speak: bool) -> None:
    selection = agent.reset_chat(refresh_memory=False)
    message = "Starting a new chat."
    _print_turn(message)
    _print_model_selection(selection)
    if speak:
        agent.tts.speak(message)
        agent.tts.speak(selection.message)
    agent.refresh_auto_memory(
        notify_memory=_print_memory_status,
        speak_memory=speak,
    )


def _print_model_selection(selection) -> None:
    color = "cyan" if selection.mode == "online" else "magenta"
    sys.stdout.write(f"{_color('model> ', color)}{selection.message}\n")
    sys.stdout.flush()


def _print_memory_status(message: str) -> None:
    sys.stdout.write(f"{_color('memory> ', 'blue')}{message}\n")
    sys.stdout.flush()


def _is_exit_command(text: str) -> bool:
    return _normalize_voice_command(text) in {"stop", "quit", "exit", "goodbye"}


def _is_new_chat_command(text: str) -> bool:
    return _normalize_voice_command(text) in {
        "new chat",
        "start new chat",
        "start a new chat",
        "clear chat",
        "reset chat",
        "reset conversation",
        "start a fresh chat",
    }


def _normalize_voice_command(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


if __name__ == "__main__":
    raise SystemExit(main())
