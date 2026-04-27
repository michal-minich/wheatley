# Decision Log

Date: 2026-04-26

This file preserves the main conclusions from the initial brainstorming so we do not re-litigate them later.

## D1: Start With STT -> LLM -> TTS

Decision: build a cascaded pipeline first.

Reasoning:

- It is easier to debug what the robot heard, thought, and said.
- Tool use can be made deterministic and safe.
- Individual models can be replaced without rewriting the whole robot.
- It fits 8 GB hardware better than large end-to-end voice models.

Expected behavior: it will feel like a responsive local voice assistant, not a fully natural full-duplex person. That is acceptable for phase 1.

## D2: 8 GB Default Means No Qwen3.6-35B-A3B Runtime

Decision: Qwen3.6-35B-A3B is not the 8 GB default.

Reasoning:

- It has about 35B total parameters and 3B activated per token.
- Active parameters help compute cost, but the weights still need memory.
- Q4 GGUF variants are roughly 18-22 GB, so 8 GB is not a good fit.
- It becomes interesting on 32 GB+, especially 64 GB or GPU/UMA systems.

Use later for smart mode on stronger hardware, not phase 1.

## D3: Qwen3.5-4B Is The First Main LLM

Decision: use Qwen3.5-4B Q4/Q5 as the first real LLM target.

Reasoning:

- Fits 8 GB class hardware.
- Strong multilingual and agentic behavior for its size.
- Has long context and vision-family compatibility for later camera work.
- Works through Ollama, llama.cpp/GGUF, and OpenAI-compatible local servers.

Operational settings:

- Disable thinking for voice chat.
- Keep output to 40-80 tokens.
- Use one tool round by default.

## D4: Gemma 4 E2B/E4B Are Edge Experiments

Decision: keep Gemma 4 E2B/E4B as an alternate track.

Reasoning:

- They are designed for edge/mobile use.
- E2B/E4B have native audio input for speech recognition and understanding.
- They may become useful for direct audio or vision experiments.

Risk: local runtime support and tool behavior may be less predictable than the cascaded Qwen stack.

## D5: STT Quality Matters More Than LLM Size

Decision: spend testing effort on STT with Slovak-accented English.

Initial candidates:

- `faster-whisper small.en int8`: first speed baseline.
- `faster-whisper medium`: quality comparison.
- multilingual Whisper: test if accent/code-switching improves.
- NVIDIA Nemotron Speech Streaming 0.6B: strong English streaming option on NVIDIA.
- NVIDIA Parakeet TDT 0.6B v3: multilingual option, includes Slovak.

Acceptance test: natural speech with Slovak accent, not dictation-style speech.

## D6: TTS Should Be Separate From Character Filter

Decision: use a normal TTS voice first, then add a light post-filter.

Initial candidates:

- Kokoro for English quality and speed.
- Piper for reliability and low resource use.
- Piper `sk_SK-lili-medium` for local Slovak fallback.
- Edge TTS `sk-SK-LukasNeural` for the current male Slovak voice.

Voice direction: lightly compressed, narrow-band, slightly synthetic, still intelligible. Avoid heavy robot effects.

## D7: Tool Use Is Whitelisted Only

Decision: no raw shell tool for the model.

Tools should be small JSON functions:

- `get_time`
- `robot_status`
- `set_eye_expression`
- `calculator`
- `remember`
- `take_photo`
- `run_safe_cli_tool`

`run_safe_cli_tool` can only call commands explicitly listed in config.

## D8: Hardware Direction

For the 8 GB/low-power line:

- N100/N150 mini-PC: simplest x86 Linux path.
- Raspberry Pi 5 8/16 GB: good GPIO, weaker AI value.
- Jetson Orin Nano 8 GB: good camera/CUDA path, tight RAM for LLM.
- Orange Pi/RK3588: interesting, but more runtime friction.

For Qwen3.6-35B-A3B later:

- 32 GB minimum.
- 64 GB preferred.
- Ryzen 5700U/5800U/7840U class mini-PC or stronger.

## D9: Development Runtime On M1 Pro

Decision: use Ollama first on the current Mac.

Observed local setup:

- Apple M1 Pro, 16 GB unified memory.
- Ollama reports Metal compute with 11.8 GiB available.
- `qwen3.5:4b` downloads as a 3.4 GB Q4_K_M model.
- The model supports thinking, tools and vision according to `ollama show`.

Important implementation detail: thinking must be disabled through the Ollama chat API with top-level `"think": false`. The raw `ollama run` CLI defaults to thinking and is not representative of the voice agent path.

Current default:

- LLM: `qwen3.5:4b` through Ollama.
- STT: `faster-whisper small.en`, int8 CPU.
- TTS: Piper `en_GB-alan-medium`.
- Run command: `./scripts/start_wheatly.sh`.

## D10: Stream Text And Start TTS Early

Decision: stream Ollama text tokens and queue TTS by phrase.

Reasoning:

- The user should see the response immediately, not only after completion.
- Piper is not token-streaming TTS, but it can synthesize short chunks while the LLM continues.
- Tool JSON is held back until complete, so partial tool calls are not printed or spoken.

Current behavior:

- `listening...` is green.
- `stopped listening.` is red after recording stops.
- `listening...` plays a rising chime; `stopped listening.` plays a falling chime.
- Partial microphone transcript is shown as a rewriting yellow `you~>` preview when enabled.
- Normal answer text streams to terminal.
- TTS begins after the first usable phrase or chunk.
- Speech uses the streaming path whenever speaking is enabled and `tts.stream_speech` is true, even if terminal token streaming is off.
- English and Slovak both stream speech; Slovak Edge TTS may have small chunk gaps, but early audible feedback is preferred.
- During TTS playback, a separate high-threshold interrupt monitor can stop speech when the user loudly says `stop`.
- Interrupt candidate verification now defaults to background mode (`speech_interrupt_pause_tts_while_verifying: false`) so false positives do not pause chunk playback.
- The first TTS chunk is adaptive, based on persisted LLM/TTS speed stats.
- The first chunk has its own lower word threshold so startup can be quicker than mid-answer chunking.
- `stream_max_initial_wait_seconds` prevents slow hardware from waiting too long before audible feedback.
- `stream_max_inter_chunk_wait_seconds` does the same for later chunks: if chunking would wait too long for `stream_min_words`, it emits a shorter `stream_feedback_min_words` chunk.
- For Piper/Edge/external TTS, streaming now runs as a prepare+play pipeline so the next chunk can synthesize while the current chunk is playing.
- Startup playback can use a small prebuffer (`stream_playback_prebuffer_chunks`, bounded by `stream_playback_prebuffer_max_wait_seconds`) to reduce immediate underruns without waiting for the whole answer.

## D11: Keep Tools Whitelisted

Decision: internet access is allowed only through explicit, whitelisted web tools. Notes search remains removed from the active registry.

Active tools:

- `get_time`
- `robot_status`
- `set_eye_expression`
- `calculator`
- `remember`
- `set_language`
- `take_photo` only if configured
- `run_safe_cli_tool` only for explicitly allowed commands
- `web_search` only if enabled and backed by a configured provider
- `fetch_url` only if enabled; private/local network addresses are blocked by default

Calculator uses an AST-based math evaluator, not Python `eval`.

`web_search` uses provider APIs such as Brave Search, SearXNG, or Tavily rather than scraping search engine HTML. `fetch_url` directly fetches a known HTTP(S) page and strips unnecessary markup into readable text/markdown-like output. Search and fetch are separate so the robot can search quickly without downloading full pages, then inspect only a chosen source when needed.

Before selected tools start, the CLI prints a colored `tool>` status and speech mode says a short active-language cue: `Remembering...` / `Zapamätávam...`, `Running...` / `Spúšťam...`, `Searching...` / `Hľadám...`, or `Downloading...` / `Sťahujem...`.

## D12: Editable Prompts And Injected Memory

Decision: keep assistant instructions, user preferences, tool descriptions, and memory in editable project files.

Files:

- `profiles/wheatly/system.md`
- `profiles/wheatly/user.md`
- `profiles/wheatly/tools.jsonc`
- `profiles/wheatly/memory.md`

The `remember` tool appends short facts to the active profile memory. Manual memory is injected into the system prompt on every turn, so the model does not need a separate retrieval command to use it. `Start a new chat.` clears conversation history but keeps the editable prompts and persistent memory.

Automatic conversation-derived memory is separate:

- `memory.md` remains manual and `remember`-managed.
- `auto_memory.md` is generated from `runtime/logs/turns.jsonl`.
- `memory_update.md` contains editable incremental update instructions.
- `memory_consolidate.md` contains editable full consolidation instructions.
- `runtime/state/memory_state.json` tracks processed log offsets and rewrite cadence.
- `runtime/state/memory_candidates.jsonl` stores compact evidence for full consolidation.
- `runtime/logs/tools.jsonl` records each tool request, result, source, and duration for debugging.

Quick updates run from turns newer than the last memory update at startup/new chat. Full consolidation is interval-based and requires the online model by default. Raw turn logs are append-only, include per-turn active LLM `model_name`, and are never deleted by memory maintenance.

## D13: Profile Folder Layout

Decision: group all persona-specific editable files under `profiles/wheatly/`.

Reasoning:

- The active profile doubles as the working example.
- Config, prompts, tool wording, voice settings and memory travel together.
- The active `profiles/wheatly/` config is the canonical example and runtime config.
- Main configs use `.jsonc` because comments belong next to settings.

## D14: Explicit Language Switching

Decision: use explicit English/Slovak switching instead of automatic language detection.

Reasoning:

- Small STT and LLM models behave better with a strong language hint.
- English mode can keep `small.en` for latency.
- Slovak mode uses a stronger Slovak STT model than English.
- The LLM model stays language-selectable only through the existing online/offline model selection; language switching also updates prompt hint, STT model/language, remote STT model, and TTS voice.

Current behavior:

- `switch to Slovak`, `speak Slovak`, `hovor po slovensky`, and `prepni na slovencinu` switch to Slovak.
- `switch to English`, `speak English`, `hovor po anglicky`, and `prepni na anglictinu` switch to English.
- `switch language` and `prepni jazyk` switch based on the language of the command; if already active, they toggle to the previous or next configured language.
- Switching prints a blue `language>` line and speaks only `Ahoj` or `Hi`.

## D15: Remote STT Is Independent From Remote LLM

Decision: support remote STT fallback separately from remote LLM fallback.

Reasoning:

- A robot body can stay within an 8 GB class memory budget by offloading large Slovak STT when Janka Mac is reachable.
- English STT quality is already sufficient with `small.en`, so remote English STT should optimize for speed rather than larger models.
- Slovak STT quality benefits from using a CTranslate2 conversion of the full `NaiveNeuron/whisper-large-v3-sk` model remotely, while keeping the existing turbo model as local fallback.
- The robot should still work if only one remote service is available: STT remote with local LLM, local STT with remote LLM, both remote, or both local.

Current behavior:

- `stt.backend` is `remote_fallback`.
- The remote endpoint is `http://jankas-mac-mini.local:8765/v1`.
- English remote STT requests `small.en`.
- Slovak remote STT requests `models/whisper/whisper-large-v3-sk-ct2-int8`.
- Local fallback uses `stt.remote_fallback_backend` and the active language's local `stt_model`.
- The `model>` line announces all LLM/STT availability combinations in the active language, for example online LLM plus remote STT, online LLM plus local STT, offline LLM plus remote STT, or fully local fallback.
