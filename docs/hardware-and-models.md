# Hardware And Models

This project is designed to become useful on modest local hardware before chasing large models. The first target is a responsive English assistant that runs locally with small models and clear fallbacks.

## First Local English Setup

| Role | Recommended first choice | Why |
| --- | --- | --- |
| LLM | `qwen3.5:4b` through Ollama | Small enough to start on 8 GB class machines, capable enough for tools and short answers. |
| STT | faster-whisper preview `small` + final `distil-large-v3` int8 | Fast live preview with higher quality endpoint transcription. |
| TTS | Piper `en_GB-alan-medium` | Local speech output with simple files and no service dependency. |
| Playback | `afplay`, `aplay`, or `ffplay` | Uses common OS playback tools. |

The default profile may contain stronger or experimental settings. For a new local-only setup, follow [Local English Startup](local-english-startup.md) and start conservative.

## Hardware Expectations

8 GB RAM:

- Use a small LLM.
- Use multilingual `small` for STT preview first.
- Keep answers short.
- Expect CPU-only STT to be the main latency cost.

16 GB RAM:

- More comfortable for the same setup.
- Lets you test larger STT models or larger LLM quantizations.

32 GB+ RAM:

- Useful for "smart mode" experiments.
- Better fit for larger local LLMs and heavier STT.

## Good Development Machines

| Hardware | Fit |
| --- | --- |
| Apple Silicon Mac, 16 GB+ | Good development machine with fast local LLM support through Ollama. |
| Intel N100/N150 mini PC, 8-16 GB | Good cheap Linux prototype for local English mode. |
| Ryzen mini PC, 32 GB+ | Good local smart-mode test box. |
| Raspberry Pi 5 | Better as a device controller than as the main LLM machine. |
| Jetson Orin Nano 8 GB | Useful for camera/CUDA experiments, but RAM is still tight for a full local voice stack. |

## Model Selection Rules

- Prefer a model that answers quickly over a model that benchmarks higher.
- Disable thinking/reasoning output for live voice.
- Keep normal voice answers short.
- Test STT with natural speech, not only clean dictation.
- Do not commit model weights. Put them under `models/`.

## Observed Model Notes

The current STT path is two-phase. Live partial preview uses multilingual `small` with CPU int8 and beam size 1. Endpoint final transcription uses `distil-large-v3` with CPU int8 and beam size 3. Remote preview/final can be enabled independently, but both phases fall back to the local preview model when the configured remote endpoint is unavailable.

Current local benchmark on 16 saved utterances: preview averages about 1.14s warm transcription time, final averages about 3.47s, and sequential preview+final totals about 4.61s. In real voice use the preview work happens during recording; perceived post-speech STT latency is dominated by the active profile's endpoint silence plus the final `distil-large-v3` pass. Profiles can trade off longer pause-and-think endpointing against lower latency.

Non-English modes can keep STT local: preview can use multilingual `small` with a language hint and beam 1; final can use a stronger local Whisper model with beam 3. `lmstudio-community/gemma-4-31b-it` is a useful smart-mode candidate for stronger hardware.

The first streamed TTS chunk can be fixed per profile. Adaptive first-chunk sizing is disabled by default because it can drift from per-profile runtime stats and make startup latency feel inconsistent.

`lmstudio-community/gemma-4-31b-it` works very well as a stronger chat model in LM Studio. It has been good for both English and Slovak. Treat it as a smart-model option for stronger hardware, not as the required local startup model.

## Technical Details

Qwen3.6-35B-A3B remains a future experiment, not a default. Even with sparse activation, the weights still need much more memory than an 8 GB target can comfortably provide.

Larger Whisper models can improve accuracy in some non-English, accented, or noisy cases, but they increase startup time, RAM use, and CPU latency. Keep them in the endpoint final path or background correction path, not in live preview.
