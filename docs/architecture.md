# Architecture

This project is a cascaded local voice agent. Each stage does one job, records enough state to debug failures, and can be replaced without rewriting the whole assistant.

```text
microphone -> STT -> LLM -> whitelisted tools -> final answer -> TTS -> speaker
```

## Design Goals

- Work locally first on 8 GB class hardware.
- Keep voice latency practical.
- Keep model behavior inspectable through logs.
- Keep tool use small, explicit, and safe.
- Keep assistant instructions and memory editable without touching Python code.

## Runtime Flow

1. The microphone records one utterance and saves it under the active profile.
2. STT turns the audio into text.
3. The agent builds a prompt from profile instructions, user preferences, memory, tool descriptions, and recent chat history.
4. The LLM either answers directly or emits a JSON tool request.
5. Only whitelisted tools can run.
6. Tool results are sent back to the LLM when needed. Successful camera captures are attached as image input only when the active model name is recognized as vision-capable.
7. The final answer is printed and optionally spoken.
8. Turn logs, tool logs, generated speech, and state are saved under the active profile.

## Profiles

Profiles are the main editing surface.

```text
profiles/wheatley/
  config.jsonc
  system.md
  user.md
  memory.md
  auto_memory.md
  python_preamble.py
  files/
  memory_update.md
  memory_consolidate.md
  runtime/
```

- `config.jsonc`: model choices, audio behavior, TTS, tools, memory, and runtime settings.
- `system.md`: assistant behavior.
- `user.md`: always-on user preferences.
- `memory.md`: explicit manual memory.
- `auto_memory.md`: generated memory from previous turns.
- `python_preamble.py`: trusted imports and helper functions for the optional Python scratchpad.
- `files/`: default read-only file root for the optional Python scratchpad.
- `runtime/`: logs, state, recorded audio, and generated speech.

Use `profiles/test/` for smoke checks. It uses an echo backend so validation does not depend on downloaded models.

## Tools

The LLM never gets raw shell access. It can only ask for tools that are registered and enabled by config.

Common enabled tools:

- `get_time`
- `calculator`
- `remember`
- `take_photo`

Optional tools exist for device state, display expression, approved local commands, web search, and a bounded Python scratchpad, but they must be explicitly enabled and configured.

The camera tool is still whitelisted: it runs only configured or built-in local capture commands, never unrestricted shell. The default output uses a small configured short side for latency while preserving the camera's returned aspect ratio. Image attachment is automatic from the active model name; names containing hints such as `llava`, `moondream`, `minicpm-v`, `minicpm-o`, `pixtral`, `paligemma`, `qwen-vl`, `qwen2-vl`, `qwen2.5-vl`, `llama-3.2-vision`, `gemma3`, `gemma-3`, `gemma-4`, `gpt-4o`, `gpt-4.1`, `gpt-5`, or `claude-3` receive image payloads. Current/default text-only names such as `qwen3.5:4b` and `qwen3.6-35b-a3b-ud-mlx` receive only photo metadata.

The Python scratchpad is not unrestricted Python. Model code cannot write imports; trusted imports and helper functions live in `python_preamble.py`. Scratchpad file helpers are read-only and limited to configured roots such as profile-local `files/`. It runs in a subprocess with a timeout, output caps, and best-effort memory/file-size resource caps, which is useful for local calculations and data transforms but is not a hostile-code OS sandbox.

## Memory

Manual memory and automatic memory are separate.

- `memory.md` is edited by the user or by the explicit `remember` command.
- `auto_memory.md` is generated from conversation logs.

Both are injected into the system prompt. Starting a new chat clears recent conversation history but keeps profile instructions and memory.

## Streaming

Text can stream to the terminal as the LLM generates it. Speech can also start before the full answer is complete by splitting the answer into short TTS chunks.

This is not true token-level speech synthesis. It is a practical latency optimization that works with normal local TTS engines.

## Two-Phase STT

Live partial transcription and endpoint final transcription can use different models. The checked-in profiles use multilingual `small` for local preview and `distil-large-v3` for local final transcription. Each phase has independent remote toggles and remote model names. If configured remote preview or remote final STT is unavailable, the fallback is always the local preview model so voice input remains usable.

Microphone endpointing measures speech and silence from captured audio samples, not from wall-clock loop time. That keeps CPU-heavy preview transcription from making a short low-energy block look like a long pause. `audio.max_utterance_seconds` is a soft budget: after it is reached, recording continues until the configured silence endpoint so active speech and short pauses are not cut off. Final WAV trimming preserves the endpoint tail, capped at 2s, so quiet syllables already captured by the recorder are not removed before final STT. Preview and final STT backends are also cached and locked per backend/model, so a stale preview job does not block the final model from starting after the utterance endpoint.

Continuous voice mode can optionally generate idle speech after silent listening timeouts. `idle_speech.interval_seconds` is multiplied by a random factor between `idle_speech.random_min_multiplier` and `idle_speech.random_max_multiplier`; the checked-in profiles use 100s with a 1x-5x multiplier. The idle prompt is loaded from hardcoded profile-relative `idle.md`. Empty captures with no live preview text do not reset the idle timer; any non-empty transcript is treated as a user turn.

## Technical Details

The cascaded design is intentional. End-to-end speech-to-speech models may become useful later, but they make tool safety, logging, memory, and debugging harder.

Tool calls use plain JSON shaped like:

```json
{"tool_calls":[{"name":"get_time","arguments":{}}]}
```

The agent executes at most one normal LLM tool round before asking for a natural-language answer. Deterministic commands such as explicit memory writes can be routed before the LLM so they behave predictably.

Online services are checked independently at startup. The assistant uses the online LLM only when that configured endpoint is reachable, uses remote STT only when the remote STT health check passes, and registers `web_search` only when the profile enables it, `BRAVE_SEARCH_API_KEY` is set, and a short internet probe succeeds. If the search check fails, `web_search` is omitted from the tool registry and is not shown to the LLM.

Profile-local paths are derived from the profile folder. Copied profiles remain portable because `system.md`, `user.md`, `memory.md`, and `runtime/` always live beside that profile's `config.jsonc`.
