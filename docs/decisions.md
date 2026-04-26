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
- Piper `sk_SK-lili-medium` for Slovak later.

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
- Run command: `./scripts/run_voice_default.sh`.

## D10: Stream Text And Start TTS Early

Decision: stream Ollama text tokens and queue TTS by phrase.

Reasoning:

- The user should see the response immediately, not only after completion.
- Piper is not token-streaming TTS, but it can synthesize short chunks while the LLM continues.
- Tool JSON is held back until complete, so partial tool calls are not printed or spoken.

Current behavior:

- `listening...` is green.
- `answering...` is red after recording stops.
- Normal answer text streams to terminal.
- TTS begins after the first usable phrase or chunk.
- The first TTS chunk is adaptive, based on persisted LLM/TTS speed stats.
- The first chunk has its own lower word threshold so startup can be quicker than mid-answer chunking.
- `stream_max_initial_wait_seconds` prevents slow hardware from waiting too long before audible feedback.

## D11: Keep Tools Local

Decision: no internet tool in the current default, and notes search is removed from the active registry.

Active tools:

- `get_time`
- `robot_status`
- `set_eye_expression`
- `calculator`
- `remember`
- `take_photo` only if configured
- `run_safe_cli_tool` only for explicitly allowed commands

Calculator uses an AST-based math evaluator, not Python `eval`.

## D12: Editable Prompts And Injected Memory

Decision: keep assistant instructions, user preferences, tool descriptions, and memory in editable project files.

Files:

- `prompts/system.md`
- `prompts/user.md`
- `prompts/tools.json`
- `memory/wheatly.md`

The `remember` tool appends short facts to `memory/wheatly.md`. Memory is injected into the system prompt on every turn, so the model does not need a separate retrieval command to use it. `Start a new chat.` clears conversation history but keeps the editable prompts and persistent memory.
