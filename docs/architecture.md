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

- `wheatly.cli`: command-line interface.
- `wheatly.pipeline`: turn orchestration and conversation history.
- `wheatly.stt`: keyboard, faster-whisper, and whisper.cpp adapters.
- `wheatly.llm`: echo, Ollama, and OpenAI-compatible adapters.
- `wheatly.tts`: no-op, macOS say, Piper, Edge TTS, and external command adapters.
- `wheatly.audio`: playback and ffmpeg post-filter.
- `wheatly.tools`: deterministic whitelist tools and parser.

## Profiles, Instructions And Memory

The system prompt is assembled by `wheatly.prompting.build_system_prompt()`.

The active persona is the profile folder under `profiles/wheatly/`:

- `config.jsonc`: runtime, model, voice, tool and path settings.
- `system.md`: main assistant behavior and tool-calling rules.
- `user.md`: user preferences and extra always-on instructions.
- `tools.jsonc`: editable tool descriptions and tool-specific instructions.
- `memory.md`: persistent memory injected into context.

Profile text files can use template markers such as `{{AGENT_NAME}}`, `{{AGENT_PERSONA}}`, `{{DEFAULT_RESPONSE_LANGUAGE}}`, `{{ACTIVE_LANGUAGE_HINT}}`, `{{CURRENT_LANGUAGE_CODE}}`, `{{CURRENT_LANGUAGE_LABEL}}`, `{{CURRENT_STT_MODEL}}`, `{{CURRENT_TTS_BACKEND}}`, `{{CURRENT_TTS_VOICE}}`, and `{{CURRENT_TTS_EDGE_VOICE}}`.

The live transcribed utterance or text is passed as the final user message to `VoiceAgent.handle_text*()`. Persistent memory is not a search tool; it is injected automatically into the system prompt at the start of every turn, including after a new chat reset.

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

There is no web-search tool in the current local default. The notes-search tool was removed from the active registry.

## Streaming

The Ollama adapter uses `/api/chat` with `stream: true` and `think: false`. Text tokens are printed as they arrive.

For speech, `StreamingSpeaker` buffers the generated text and queues short sentence-like segments to TTS. This lets Piper start speaking before the full LLM response is complete. It is not true token-level speech synthesis, but it reduces perceived delay without changing the model.

The first spoken chunk is adaptive and has a separate threshold from later chunks. The agent records recent LLM generation speed and TTS consumption speed in `runtime/state/latency_stats.json`, then chooses how many words to buffer before speaking. If TTS is faster than generation, it waits for a larger buffer; if generation is faster, it speaks earlier.

There is also an upper bound on the first wait. If `stream_max_initial_wait_seconds` expires, the speaker queues the first complete sentence. If no sentence boundary exists yet, it queues `stream_feedback_min_words` words. This gives audible feedback on slow hardware even when the next chunk may need to wait for more generated text.

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

- `runtime/<profile>/logs/turns.jsonl`: conversation and tool traces.
- `runtime/<profile>/state/eye.json`: current eye expression state.
- `runtime/<profile>/audio/`: recorded utterances and generated speech.

These files are ignored by git.
