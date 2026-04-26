# Runtime Options

Date: 2026-04-26

Current development machine:

```text
Apple M1 Pro
10 CPU cores
16 GB unified memory
Ollama Metal reports 11.8 GiB available VRAM
```

## Chosen Default For This Mac

```text
LLM runtime: Ollama 0.21.2
LLM model: qwen3.5:4b
Quantization: Q4_K_M
Model size: 3.4 GB
STT: faster-whisper small.en, int8 CPU
TTS: Piper en_GB-alan-medium
Voice filter: disabled by default; ffmpeg wheatley_bright remains available
```

Reasoning:

- Ollama is fastest to get working and already uses Metal on this Mac.
- `qwen3.5:4b` fits both this Mac and the intended 8 GB class target better than 9B or 35B-A3B.
- The Ollama model reports tools, vision and thinking capabilities, but voice mode disables thinking.
- `faster-whisper small.en` is the first accent-tolerance baseline.
- Piper is reliable and low-latency enough for a first spoken loop.
- Text output streams immediately; TTS starts after the first sentence-like chunk.

## Ollama

Pros:

- Easiest install and model management.
- Good Apple Silicon path through Metal.
- Built-in model tags and quantizations.
- Chat API supports `think: false`, which is required for voice latency.

Cons:

- Less exact control over quant choice than raw GGUF.
- Some thinking models need explicit `think: false` on every API call.
- Tool-call formatting still needs empirical testing with each model.

Use as default until there is a concrete reason to switch.

## llama.cpp

Pros:

- Best control over GGUF quantization, context, Metal offload, KV cache and server options.
- More portable to Linux SBC/mini-PC installs.
- Good for measuring exact quant tradeoffs such as Q4_K_M vs Q5_K_M.

Cons:

- More setup and model-file management.
- We need to pick and download exact GGUF files manually.

Use when tuning the final target hardware.

## MLX / mlx-lm

Pros:

- Very strong on Apple Silicon.
- Often faster than generic CPU paths on this Mac.

Cons:

- Apple-only; less useful for the final 8 GB Linux robot.
- Adds a runtime split between development machine and deployment machine.

Use for Mac-only experiments, not the default project path.

## Model Candidates

| Model | Runtime | Quant | Role |
| --- | --- | --- | --- |
| `qwen3.5:4b` | Ollama | Q4_K_M | default fast brain |
| Qwen3.5-4B GGUF | llama.cpp | Q4_K_M / Q5_K_M | final-target tuning |
| `qwen3.5:9b` | Ollama | likely Q4 | quality comparison on 16 GB+ |
| Gemma 4 E2B/E4B | Ollama/llama.cpp when stable | Q4 | edge/multimodal experiment |
| Qwen3.6-35B-A3B | llama.cpp/Ollama on 32 GB+ | Q3/Q4 | future smart mode |

## Current Run Command

```bash
./scripts/run_voice_default.sh
```

Equivalent:

```bash
. .venv/bin/activate
PYTHONPATH=src python3 -m wheatly --config configs/wheatly.local.json voice
```

## Local Defaults

`configs/wheatly.local.json`

The higher `max_tokens` is intentional. Short ordinary answers are enforced by the system prompt, but stories and requested long-form answers need room to finish.

Editable instruction and memory files:

```text
prompts/system.md
prompts/user.md
prompts/tools.json
memory/wheatly.md
```
