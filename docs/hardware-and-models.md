# Hardware And Model Notes

This project is being built on a development machine first, then copied to the target robot hardware.

## 8 GB Target Profile

Expected constraints:

- RAM: 8 GB total.
- Power: power bank or battery.
- Priority: response latency and stability.
- Secondary: future camera and tool use.

## First Model Matrix

| Role | Default | Reason |
| --- | --- | --- |
| VAD | simple RMS VAD | zero dependency baseline |
| STT | remote_fallback to Janka Mac, local small.en/turbo fallback | lower robot RAM while keeping offline degradation |
| LLM | Qwen3.5-4B Q4/Q5 | best first 8 GB quality/latency tradeoff |
| TTS | Kokoro or Piper | local, fast, simple |
| Filter | ffmpeg light preset | character voice without model coupling |

## Current STT Split

| Language | Remote STT target | Local fallback | Reason |
| --- | --- | --- | --- |
| English | `small.en` | `small.en` int8 | current quality is enough; optimize for latency |
| Slovak | `models/whisper/whisper-large-v3-sk-ct2-int8`, converted from `NaiveNeuron/whisper-large-v3-sk` | `whisper-large-v3-turbo-sk-ct2-int8` | use the best larger Slovak model when Janka Mac is reachable |

Remote STT and remote LLM are independent. The robot can use either service, both services, or neither service depending on what is reachable.

Memory planning:

| Robot mode | Practical RAM target | Notes |
| --- | --- | --- |
| Remote STT + remote LLM | 4-8 GB | robot mostly records audio, routes requests, and speaks |
| Remote STT + local 4B LLM | 8-16 GB | good 8 GB target if local STT fallback stays small |
| Local STT + local 4B LLM | 16 GB minimum | Slovak fallback can be tight on 8 GB |
| Strong local Slovak STT + local 4B LLM | 32 GB comfortable | avoids relying on Janka Mac |

## Alternatives To Test

| Role | Candidate | Why |
| --- | --- | --- |
| STT | Whisper medium | better accent tolerance |
| STT | multilingual Whisper | code switching and Slovak-accented English |
| STT | whisper.cpp Metal/CoreML small.en | macOS-only acceleration for English with similar quality |
| STT | MLX Whisper | Apple Silicon experiment, especially for stock Whisper models |
| STT | Nemotron Speech Streaming 0.6B | low-latency English on NVIDIA |
| STT | Parakeet TDT 0.6B v3 | multilingual, includes Slovak |
| LLM | Gemma 4 E2B | very low memory edge mode |
| LLM | Gemma 4 E4B | multimodal edge experiment |
| LLM | Phi-4-mini | English reasoning fallback |
| TTS | Piper sk_SK-lili-medium | Slovak mode later |

## Not A Default On 8 GB

Qwen3.6-35B-A3B:

- Good future smart-mode candidate.
- Not an 8 GB default because quantized weights exceed practical RAM.
- Consider on 32 GB+ hardware.

Voxtral Mini Realtime:

- Interesting realtime STT.
- Too heavy to combine casually with a 4B LLM and TTS on 8 GB.

End-to-end speech-to-speech:

- Interesting for natural interruption and prosody.
- Deferred until the cascaded pipeline is stable.

## Hardware Shortlist

| Hardware | Fit |
| --- | --- |
| Intel N100/N150 8-16 GB | best cheap x86 prototype |
| Intel N100/N150 32 GB | useful bridge, still not ideal for 35B MoE |
| Ryzen 5700U/5800U 32 GB | budget smart-mode experiment |
| Ryzen 7840U/8845HS 64 GB | strong local AI mini-PC |
| Jetson Orin Nano 8 GB | camera/CUDA body, not big LLM brain |
| Raspberry Pi 5 | controller or minimal mode |
