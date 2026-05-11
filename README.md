# Local Talking Assistant

This project is a local-first talking assistant. It listens to spoken English, transcribes it locally, asks a local LLM for the next response or tool call, and speaks back through local TTS. The project favors reliability, low latency, and clear logs over maximum benchmark intelligence.

## Tools

The assistant currently has a small, practical tool set:

- **Web search**: optional Brave Search support for current public facts, when enabled and configured with an API key.
- **Camera**: captures a small local photo by configured short side and sends it to the next LLM call when the active model name is known to support image input.
- **Calculator**: handles safe math without using Python `eval`.
- **Python scratchpad**: optional bounded Python snippets with trusted profile preamble imports and read-only configured file roots.
- **Memory**: `Remember this: ...` appends explicit facts or preferences to profile memory.
- **Time:** answers current local time, date, and timezone questions.

## Capabilities

- Runs a local `microphone -> speech-to-text -> LLM -> text-to-speech -> speaker` loop.
- Also works as a text chat or one-shot command runner for debugging without a microphone.
- Uses whitelisted tools for safe actions such as current time, calculator, memory, camera photos, optional Python scratchpad, and optional device state.
- Keeps assistant instructions, tool wording, memory, logs, and generated audio inside profile folders.
- Targets 8 GB class machines first, while still being usable on a Mac or PC with more RAM.
- Provides smoke tests and diagnostics so humans and agents can validate setup quickly.

## What It Is For

Use it as a practical foundation for an offline talking assistant. The current goal is not a fully autonomous computer operator. The goal is a fast, inspectable assistant that can hear, answer, remember explicit facts, use a few safe tools, and keep working when internet services are unavailable.

## Start Here

For a new clone, use the local English startup guide:

- [Local English Startup](docs/local-english-startup.md)

That guide covers the intended first path only: local Ollama LLM, local faster-whisper STT, and local Piper TTS.

## Common Commands

```bash
PYTHONPATH=src python3 -m wheatley doctor
PYTHONPATH=src python3 -m wheatley --profile test once --text "what time is it?"
PYTHONPATH=src python3 -m wheatley once --stream --text "hello"
PYTHONPATH=src python3 -m wheatley chat --stream
PYTHONPATH=src python3 -m wheatley voice
```

With `make`:

```bash
make test
make doctor
make smoke
make voice
```

## Project Layout

```text
src/wheatley/          Python package and CLI
profiles/wheatley/    Main editable profile
profiles/test/        Isolated echo profile for smoke tests
docs/                 Human and agent documentation
scripts/              Startup helpers
tests/                stdlib unittest coverage
models/               Local model files, ignored by git
```

## More Docs

- [Runbook](docs/runbook.md)
- [Architecture](docs/architecture.md)
- [Hardware and Models](docs/hardware-and-models.md)
- [Decision Log](docs/decisions.md)

## Technical Notes

The runtime is intentionally cascaded instead of speech-to-speech: STT, LLM, tools, and TTS are separate pieces. That makes failures easier to diagnose and keeps tool use whitelisted. Model weights and generated runtime files are not committed; keep them under `models/` and `profiles/<profile>/runtime/`.
