from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import threading
import textwrap
import time
from pathlib import Path

from wheatly.config import load_config, profile_config_path
from wheatly.doctor import diagnostics_json
from wheatly.pipeline import VoiceAgent
from wheatly.runtime_stats import LatencyStats
from wheatly.stt.microphone import MicrophoneRecorder
from wheatly.tools.builtins import build_registry
from wheatly.tts.backends import build_tts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wheatly")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--profile", help="Profile folder under profiles/. Defaults to WHEATLY_PROFILE or wheatly.")
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

    listen = sub.add_parser("listen", help="Record one utterance, transcribe and answer.")
    listen.add_argument("--speak", action="store_true", help="Speak the response.")

    voice = sub.add_parser("voice", help="Continuous microphone voice loop.")
    voice.add_argument("--turns", type=int, default=0, help="Stop after N turns; 0 means forever.")
    voice.add_argument("--no-speak", action="store_true", help="Do not speak responses.")
    voice.add_argument("--no-stream", action="store_true", help="Disable token streaming.")

    args = parser.parse_args(argv)
    if args.config and args.profile:
        raise SystemExit("Use either --config or --profile, not both.")
    if args.profile and not profile_config_path(args.profile).exists():
        raise SystemExit(f"Missing profile config: {profile_config_path(args.profile)}")
    cfg = load_config(args.config, profile=args.profile)

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
        from wheatly.tools.registry import ToolCall

        registry = build_registry(cfg)
        try:
            tool_args = json.loads(args.args)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for --args: {exc}") from exc
        result = registry.execute(ToolCall(args.name, tool_args))
        print(json.dumps(result.__dict__, indent=2, default=str))
        return 0

    if args.command == "speak":
        cfg.tts.enabled = True
        build_tts(cfg).speak(args.text)
        return 0

    agent = VoiceAgent(cfg)

    if args.command == "once":
        if args.speak:
            cfg.tts.enabled = True
        _announce_model_selection(agent, speak=args.speak)
        if args.stream:
            _print_streamed_turn(agent, args.text, speak=args.speak)
        else:
            result = agent.handle_text(args.text, speak=args.speak)
            _print_turn(result.assistant_text)
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
        partial_stt = _enable_partial_transcript_stt(agent, cfg)
        audio_path = (
            Path(cfg.audio.utterance_dir)
            / f"utterance_{int(__import__('time').time())}.wav"
        )
        print(_color("listening...", "green"))
        recorded = _record_with_partial_transcript(recorder, audio_path, partial_stt)
        print(_color("stopped listening.", "red"))
        transcription = agent.transcribe(recorded)
        _print_user(transcription.text)
        result = agent.handle_text(transcription.text, speak=args.speak)
        _print_turn(result.assistant_text)
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
    print("Wheatly text chat. Ctrl-D or empty line exits.")
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
            selection = agent.reset_chat()
            message = "Starting a new chat."
            _print_turn(message)
            _print_model_selection(selection)
            if speak:
                agent.tts.speak(message)
                agent.tts.speak(selection.message)
            continue
        if stream:
            _print_streamed_turn(agent, text, speak=speak)
        else:
            result = agent.handle_text(text, speak=speak)
            _print_turn(result.assistant_text)


def _print_turn(text: str) -> None:
    sys.stdout.write(f"{_prefix('wheatly', 'orange')}{text}\n")
    sys.stdout.flush()


def _print_streamed_turn(agent: VoiceAgent, text: str, speak: bool):
    printed_prefix = False

    def on_token(token: str) -> None:
        nonlocal printed_prefix
        if not printed_prefix:
            sys.stdout.write(_prefix("wheatly", "orange"))
            printed_prefix = True
        sys.stdout.write(token)
        sys.stdout.flush()

    result = agent.handle_text_stream(text, speak=speak, on_token=on_token)
    if not printed_prefix:
        sys.stdout.write(_prefix("wheatly", "orange"))
    sys.stdout.write("\n")
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
    partial_stt = _enable_partial_transcript_stt(agent, cfg)
    print("Wheatly voice loop. Say 'stop', 'quit', or press Ctrl-C to exit.")
    count = 0
    while True:
        if turns and count >= turns:
            return 0
        try:
            audio_path = (
                Path(cfg.audio.utterance_dir)
                / f"utterance_{int(time.time())}_{count + 1}.wav"
            )
            print(_color("listening...", "green"))
            recorded = _record_with_partial_transcript(recorder, audio_path, partial_stt)
            print(_color("stopped listening.", "red"))
            transcription = agent.transcribe(recorded)
            text = transcription.text.strip()
            _print_user(text)
            if _is_exit_command(text):
                return 0
            if _is_new_chat_command(text):
                selection = agent.reset_chat()
                message = "Starting a new chat."
                _print_turn(message)
                _print_model_selection(selection)
                if speak:
                    agent.tts.speak(message)
                    agent.tts.speak(selection.message)
                count += 1
                continue
            if stream:
                _print_streamed_turn(agent, text, speak=speak)
            else:
                result = agent.handle_text(text, speak=speak)
                _print_turn(result.assistant_text)
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


def _record_with_partial_transcript(
    recorder: MicrophoneRecorder, audio_path: Path, partial_stt
) -> Path:
    preview = _PartialTranscriptPreview(partial_stt)
    try:
        return recorder.record_utterance(
            audio_path,
            partial_transcriber=preview.transcribe if preview.enabled else None,
            on_partial_transcript=preview.update if preview.enabled else None,
        )
    finally:
        preview.finish()


class _PartialTranscriptPreview:
    def __init__(self, stt):
        self.enabled = stt is not None
        self.closed = False
        self.used = False
        self.rendered_lines = 0
        self.lock = threading.Lock()
        self.stt = stt

    def transcribe(self, audio_path: Path) -> str:
        if not self.stt:
            return ""
        return self.stt.transcribe(audio_path).text

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


def _enable_partial_transcript_stt(agent: VoiceAgent, cfg):
    backend = cfg.stt.backend.lower().replace("-", "_")
    if not cfg.audio.partial_transcript_enabled or backend == "keyboard":
        return None
    locked_stt = _LockedSTT(agent.stt)
    agent.stt = locked_stt
    return locked_stt


class _LockedSTT:
    def __init__(self, stt):
        self.stt = stt
        self.lock = threading.Lock()

    def transcribe(self, audio_path: Path):
        with self.lock:
            return self.stt.transcribe(audio_path)


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
    selection = agent.reset_chat()
    _print_model_selection(selection)
    if speak:
        agent.tts.speak(selection.message)


def _print_model_selection(selection) -> None:
    color = "cyan" if selection.mode == "online" else "magenta"
    sys.stdout.write(f"{_color('model> ', color)}{selection.message}\n")
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
