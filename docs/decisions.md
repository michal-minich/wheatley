# Decision Log

This is the short version of the project decisions. Keep it high signal so new agents and people can understand why the repo is shaped this way.

## Start With A Cascaded Voice Pipeline

Decision: use separate STT, LLM, tools, and TTS stages.

Why:

- Easier to debug what was heard, decided, and spoken.
- Easier to keep tool use safe.
- Easier to swap one model without replacing the whole system.
- More practical for 8 GB class hardware.

## Optimize For Reliable Latency First

Decision: the first target is a responsive local assistant, not maximum model intelligence.

Why:

- Voice interaction feels broken if every answer is slow.
- A smaller model with short answers is more useful on modest local hardware.
- Tool use and memory can cover many practical tasks without a huge model.

## Use Small Local Models First

Decision: start with a 4B-class local LLM and two-phase faster-whisper STT.

Why:

- They are realistic on small machines.
- They reduce setup friction for people cloning the repo.
- They leave room for TTS, audio recording, and the operating system.
- Multilingual `small` is the live preview model because it has the best observed quality/latency tradeoff for the current recordings.
- `distil-large-v3` is the endpoint final model when running locally; if configured remote STT is unavailable, final falls back to the local preview model.
- Non-English profile modes can override the English final model: they may keep preview on multilingual `small` with a language hint, then use a stronger local final model at beam 3.
- The main profile uses a longer endpoint silence target and a higher soft utterance budget so users can pause and think mid-sentence. Lower-latency profiles can use shorter endpoint silence. Once the soft budget is reached, the recorder still waits for the configured silence endpoint instead of truncating the user mid-sentence or on a short pause. Profiles use the lower RMS VAD threshold and sample-based endpointing so background preview transcription should not cut off active speech.
- Continuous voice mode can enable idle speech after a randomized silent wait. The base interval is 100s and the random multiplier is 1x-5x, so the assistant does not speak on a mechanical cadence. Idle remarks use profile-local `idle.md`, must be non-demanding, varied, recent-topic-aware, mostly short to medium, and generated only when preview listening recognized no text. Empty captures are idle-neutral; any non-empty transcript is treated as a user turn.
- The first TTS chunk can be fixed per profile instead of adaptive, so perceived speech start latency does not drift from runtime stats.

## Keep A Stronger Smart-Model Path

Decision: document `lmstudio-community/gemma-4-31b-it` as a strong optional model.

Why:

- It works very well in LM Studio.
- It has been good for both English and Slovak in local testing.
- It belongs on stronger hardware, not in the minimum local startup path.

## Keep Tool Use Whitelisted

Decision: do not expose unrestricted shell or filesystem access to the model.

Why:

- Configured tools must be predictable.
- Local command execution needs explicit allowlists.
- Calculator uses a safe AST evaluator, not raw Python `eval`.
- The optional Python scratchpad is disabled by default, rejects model-written imports, uses trusted profile preamble imports, and limits file reads to configured roots.
- The Python scratchpad is a best-effort bounded helper, not a hostile-code OS sandbox.
- Camera capture is a whitelisted tool, not shell access. It uses a configured command or built-in local capture command choices, keeps images small by a configured short side while preserving aspect ratio, and attaches images to the LLM only when the active model name looks vision-capable.

## Keep Profile Files Editable

Decision: prompts, user preferences, tool descriptions, and memory live beside the profile config.

Why:

- Users can tune the assistant without changing Python.
- Profiles can be copied as self-contained assistant configurations.
- Runtime logs stay next to the profile that produced them.

## Separate Manual And Generated Memory

Decision: `memory.md` is explicit memory, while `auto_memory.md` is generated from prior turns.

Why:

- User-approved facts remain easy to inspect.
- Generated summaries can be rebuilt or deleted without losing manual memory.
- New chats can clear short-term history while preserving useful context.
- Automatic memory runs incremental updates by default. Full consolidation is gated
  by `memory.consolidation_enabled` and is disabled in the checked-in profiles
  unless deliberately re-enabled.

## Technical Details

The default runtime uses an Ollama-compatible local LLM adapter and a faster-whisper STT adapter. Piper is the preferred local TTS path for the first English setup.

Larger models, remote services, multilingual modes, vision models, and device integrations can be tested later, but they should not become required for the basic local English startup path.
