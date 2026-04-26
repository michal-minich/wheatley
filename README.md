# Wheatly

Offline-first voice agent foundation for a small Wheatley-style robot.

The goal is a fast, debuggable `audio -> text -> LLM -> TTS` pipeline that can run on an 8 GB class machine first, then move to the final robot hardware.

## Current Status

Implemented now:

- Text agent loop with persistent turn logs.
- STT adapters: keyboard, `faster-whisper`, `whisper.cpp`.
- LLM adapters: echo smoke-test backend, Ollama, OpenAI-compatible local servers.
- TTS adapters: macOS `say`, Piper command backend, no-op backend.
- Optional microphone recording with simple RMS VAD through `sounddevice`.
- Whitelisted tools: time, robot status, eye expression state, calculator, memory, photo placeholder, safe CLI command wrapper.
- Wheatley-style audio post-filter through `ffmpeg` when available.
- Editable prompt, tool-description, and memory files under `prompts/` and `memory/`.

## Quick Start

From the repo root:

```bash
PYTHONPATH=src python3 -m wheatly doctor
PYTHONPATH=src python3 -m wheatly once --text "hello"
PYTHONPATH=src python3 -m wheatly once --text "what time is it?"
```

Or with `make`:

```bash
make test doctor smoke
```

On macOS, test speech with the built-in voice:

```bash
PYTHONPATH=src python3 -m wheatly speak "Right, tiny offline robot brain online."
```

Interactive text loop:

```bash
PYTHONPATH=src python3 -m wheatly chat
```

## Local Config

Copy one example and edit it:

```bash
cp configs/wheatly.example.json configs/wheatly.local.json
```

The app automatically loads `configs/wheatly.local.json` if it exists. You can also pass a config explicitly:

```bash
PYTHONPATH=src python3 -m wheatly --config configs/wheatly.ollama-qwen35-4b.example.json once --text "status please"
```

## Recommended First Real Stack

For the 8 GB target:

```text
VAD: simple RMS VAD first, Silero/openWakeWord later
STT: faster-whisper small.en int8, then test medium or multilingual
LLM: Qwen3.5-4B Q4/Q5 through Ollama or llama.cpp
TTS: Kokoro or Piper, with ffmpeg Wheatley filter
Tools: whitelist only, no raw shell
```

Qwen3.6-35B-A3B remains a future 32 GB+ experiment, not an 8 GB robot default.

## Voice Mode

Install optional audio and STT dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[audio,stt]'
```

Then configure `configs/wheatly.local.json` with:

```json
{
  "stt": {"backend": "faster_whisper", "model": "small.en"},
  "llm": {"backend": "ollama", "model": "qwen3.5:4b"},
  "tts": {"backend": "piper", "enabled": true}
}
```

Run one recorded turn:

```bash
PYTHONPATH=src python3 -m wheatly listen --speak
```

Run the continuous voice loop:

```bash
./scripts/run_voice_default.sh
```

The default voice loop streams text tokens and starts speaking before the full answer is generated. The ffmpeg voice filter is disabled by default because chunked filtered playback caused static tails on macOS.

## Documentation

- [Architecture](docs/architecture.md)
- [Decision Log](docs/decisions.md)
- [Runbook](docs/runbook.md)
- [Runtime Options](docs/runtime-options.md)
- [Hardware and Models](docs/hardware-and-models.md)
- [Agent Instructions](AGENTS.md)
