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
| STT | faster-whisper small.en int8 | speed and accent robustness baseline |
| LLM | Qwen3.5-4B Q4/Q5 | best first 8 GB quality/latency tradeoff |
| TTS | Kokoro or Piper | local, fast, simple |
| Filter | ffmpeg light preset | character voice without model coupling |

## Alternatives To Test

| Role | Candidate | Why |
| --- | --- | --- |
| STT | Whisper medium | better accent tolerance |
| STT | multilingual Whisper | code switching and Slovak-accented English |
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

