# AGENTS.md

This repo is the local-first Wheatly voice-agent project. Keep changes pragmatic, portable, and documented.

## Project Intent

Build a fast offline `audio -> text -> LLM -> TTS` robot assistant that fits an 8 GB class machine first. The first target is reliable latency and tool use, not maximum benchmark intelligence.

## Current Architecture

- Python package under `src/wheatly`.
- CLI entrypoint: `python3 -m wheatly` or installed `wheatly`.
- Config format: JSON, with examples in `configs/`.
- Editable prompts live in `prompts/system.md`, `prompts/user.md`, and `prompts/tools.json`.
- Persistent memory lives in `memory/wheatly.md` and is injected into the system prompt.
- Runtime files live under `runtime/`.
- Tests use stdlib `unittest` and should run without external model downloads.

## Hard Rules

- Do not give the model unrestricted shell access.
- Tool use must stay whitelisted through `src/wheatly/tools`.
- Do not add web/search/internet tools unless explicitly requested again.
- Do not re-add notes search without a clear product reason.
- Keep 8 GB RAM as the default design constraint.
- Keep Qwen3.6-35B-A3B as a documented future experiment, not a default runtime.
- Keep docs updated when model, hardware, latency, or tool decisions change.
- Do not commit downloaded model weights into this repo. Use `models/`, which is ignored.

## Preferred Defaults

- LLM: Qwen3.5-4B Q4/Q5 for first real local model.
- STT: faster-whisper `small.en` int8 first, then test `medium` or multilingual for Slovak-accented English.
- TTS: Kokoro or Piper; apply a light post-filter instead of hard voice cloning.
- Runtime: Ollama or llama.cpp/OpenAI-compatible server.
- Calculator: use the local AST-based `calculator` tool, never raw Python `eval`.

## Validation

Before handing back changes, run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m wheatly doctor
PYTHONPATH=src python3 -m wheatly once --text "what time is it?"
```

If external dependencies or models are unavailable, note that clearly and keep the echo backend smoke tests passing.
