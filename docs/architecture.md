# Architecture

The first implementation is a cascaded, offline-first voice pipeline:

```text
microphone
  -> VAD / utterance recorder
  -> STT
  -> conversation agent
  -> tool parser and whitelist executor
  -> LLM final answer
  -> streaming text output
  -> streaming TTS segment queue
  -> optional Wheatley voice filter
  -> speaker
```

## Why Cascaded Pipeline

We are intentionally not starting with an end-to-end speech-to-speech model.

Reasons:

- Better tool-use control.
- Easier debugging because every step is visible in logs.
- Lower RAM pressure on 8 GB machines.
- Easier model swaps for STT, LLM, and TTS independently.
- Easier support for English now and Slovak later.

End-to-end multimodal/audio models remain interesting for later experiments, but they make interruption handling, logging, voice control, and safe tools harder.

## Main Modules

- `wheatley.cli`: command-line interface.
- `wheatley.pipeline`: turn orchestration and conversation history.
- `wheatley.stt`: keyboard, faster-whisper, whisper.cpp, remote STT fallback, and the bundled STT HTTP server.
- `wheatley.llm`: echo, Ollama, and OpenAI-compatible adapters.
- `wheatley.tts`: no-op, macOS say, Piper, Edge TTS, and external command adapters.
- `wheatley.audio`: playback and ffmpeg post-filter.
- `wheatley.tools`: deterministic whitelist tools and parser.

## Profiles, Instructions And Memory

The system prompt is assembled by `wheatley.prompting.build_system_prompt()`.

The active persona is the profile folder under `profiles/wheatley/`:

- `config.jsonc`: runtime, model, voice, tool and path settings.
- `system.md`: main assistant behavior and tool-calling rules.
- `user.md`: user preferences and extra always-on instructions.
- `tools.jsonc`: editable tool descriptions and tool-specific instructions.
- `memory.md`: manual persistent memory injected into context.
- `auto_memory.md`: generated conversation-derived memory injected after manual memory.
- `memory_update.md`: editable instructions for incremental automatic memory updates.
- `memory_consolidate.md`: editable instructions for full automatic memory consolidation.
- `runtime/`: profile-local logs, state, generated audio, and memory state.

Profile text files can use template markers such as `{{AGENT_NAME}}`, `{{AGENT_PERSONA}}`, `{{DEFAULT_RESPONSE_LANGUAGE}}`, `{{ACTIVE_LANGUAGE_HINT}}`, `{{CURRENT_LANGUAGE_CODE}}`, `{{CURRENT_LANGUAGE_LABEL}}`, `{{CURRENT_STT_MODEL}}`, `{{CURRENT_TTS_BACKEND}}`, `{{CURRENT_TTS_VOICE}}`, and `{{CURRENT_TTS_EDGE_VOICE}}`.

The live transcribed utterance or text is passed as the final user message to `VoiceAgent.handle_text*()`. Persistent memory is not a search tool; manual memory and generated auto-memory are injected automatically into the system prompt at the start of every turn, including after a new chat reset.

`memory.md` remains explicit and tool-managed. The `remember` tool only appends to that file. Automatic history-derived memory is stored separately in `auto_memory.md` and is built from profile-local `runtime/logs/turns.jsonl`. Incremental updates use only turns newer than the last memory update, while full consolidation can use the current auto-memory, compact candidate evidence, and selected recent log turns.

## Remote STT

STT and LLM fallback are independent. The profile can use remote STT while keeping the local LLM, local STT while using the remote LLM, both remote services, or both local services.

The startup/new-chat `model>` status line also treats them independently. It probes the remote LLM and remote STT endpoints, keeps the LLM mode as `online` or `offline` for memory behavior, and adds a separate STT mode of `remote` or `local` for the user-facing announcement.

The default `stt.backend` is `remote_fallback`:

1. POST the recorded WAV to `stt.remote_base_url` using an OpenAI-compatible `/v1/audio/transcriptions` request.
2. Send `language` and the active `remote_stt_model`.
3. If the remote request fails, load the configured local fallback backend from `stt.remote_fallback_backend` and transcribe locally.

Language switching keeps local and remote STT model choices separate. English uses `small.en` for both remote and local quality. Slovak requests the server-side CTranslate2 conversion of `NaiveNeuron/whisper-large-v3-sk` at `models/whisper/whisper-large-v3-sk-ct2-int8` and keeps the existing local `models/whisper/whisper-large-v3-turbo-sk-ct2-int8` fallback.

The remote server lives in this repo as `wheatley.stt.server` and can be started with:

```bash
PYTHONPATH=src python3 -m wheatley stt-server \
  --host 0.0.0.0 \
  --port 8765 \
  --default-model small.en \
  --model en=small.en \
  --model sk=models/whisper/whisper-large-v3-sk-ct2-int8
```

The server intentionally does not download or commit model weights. Models belong under `models/` or the normal Hugging Face cache on the serving machine.

## Tool Protocol

Local models are asked to emit tool calls as plain JSON:

```json
{"tool_calls":[{"name":"get_time","arguments":{}}]}
```

The agent executes at most one tool round by default, then sends the tool results back to the LLM and asks for a brief natural-language answer.

This avoids giving the LLM direct shell access and keeps tool behavior inspectable.

For basic local facts, the agent also has a deterministic pre-router. Questions containing whole-word `time`, `date`, `status`, or `battery` are routed to local tools before the LLM answers. Calculator-style requests are routed to the local calculator. This prevents small models from inventing local state or arithmetic.

Current tools:

- `get_time`
- `robot_status`
- `set_eye_expression`
- `calculator`
- `remember`
- `take_photo`, only if a camera command is configured
- `run_safe_cli_tool`, only for commands explicitly listed in config
- `web_search`, only if enabled and backed by a configured provider
- `fetch_url`, only if enabled; blocks private/local network targets by default

Web access stays in explicit tools. Search uses a configured provider API instead of scraping search-engine result pages. URL fetching is separate from search, returns cleaned text/markdown-like content, and is not a shell or browser tool. The notes-search tool was removed from the active registry.

## Streaming

The Ollama adapter uses `/api/chat` with `stream: true` and `think: false`. Text tokens are printed as they arrive.

For speech, `StreamingSpeaker` buffers the generated text and queues short sentence-like segments to TTS. This lets Piper or Edge TTS start speaking before the full LLM response is complete. It is not true token-level speech synthesis, but it reduces perceived delay without changing the model.

Streaming TTS now uses a two-stage pipeline when the backend supports it:

- Stage 1 prepares audio files for upcoming text chunks.
- Stage 2 plays prepared chunks in-order.

This allows chunk `N+1` to synthesize while chunk `N` is playing, which reduces the "first words, then pause" effect from single-thread chunk processing. A small startup prebuffer (`stream_playback_prebuffer_chunks` with `stream_playback_prebuffer_max_wait_seconds`) helps avoid immediate underruns without waiting for the full answer.

The first spoken chunk is adaptive and has a separate threshold from later chunks. The default profile now uses a low first-chunk target, short timeout, and moderate later chunk size so voice starts quickly while avoiding one-word fragments.

There is also an upper bound on the first wait. If `stream_max_initial_wait_seconds` expires, the speaker queues the first complete sentence. If no sentence boundary exists yet, it queues `stream_feedback_min_words` words. This gives audible feedback on slow hardware even when the next chunk may need to wait for more generated text.

Later chunks have their own timeout (`stream_max_inter_chunk_wait_seconds`). If the next segment would otherwise wait too long for `stream_min_words`, the speaker emits a shorter `stream_feedback_min_words` chunk to keep speech continuous.

When speech output is enabled, CLI turns use the streaming path even if token-by-token text printing is disabled. `--stream` controls terminal text streaming; speech streaming follows `tts.stream_speech`.

Speech-interrupt verification can run without pausing playback (`speech_interrupt_pause_tts_while_verifying: false`), which avoids regular chunk-boundary stalls when interrupt candidates are false positives.

Tool JSON is held back until complete so partial JSON is not printed or spoken.

## Latency Strategy

For live voice, optimize in this order:

1. Keep replies short, usually 40-80 tokens.
2. Disable thinking mode for normal conversation.
3. Stream LLM tokens.
4. Start TTS after the first complete phrase or short chunk.
5. Use a 4B-class model before trying larger MoE models.
6. Use scripted instant reactions for acknowledgements if needed.

## Runtime State

Runtime outputs are intentionally separated from source:

- `profiles/<profile>/runtime/logs/turns.jsonl`: append-only conversation log with per-turn active LLM `model_name` and tool results.
- `profiles/<profile>/runtime/logs/tools.jsonl`: append-only tool audit log with request arguments, results, source, and duration.
- `profiles/<profile>/runtime/state/eye.json`: current eye expression state.
- `profiles/<profile>/runtime/state/memory_state.json`: automatic memory refresh state.
- `profiles/<profile>/runtime/state/memory_candidates.jsonl`: compact evidence for later memory consolidation.
- `profiles/<profile>/runtime/audio/`: recorded utterances, partial STT snapshots, and generated speech. Runtime audio files are kept and never auto-deleted by the app.

These files are ignored by git.
