# STT Benchmark Results

Generated: `2026-04-30T19:49:05+0200`

Reference model: `distil-large-v3` beam `3`.

Samples: `16` user WAV files from profile runtime logs.

Raw JSON: `docs/benchmarks/stt-benchmark-20260430-194650.json`

Raw CSV: `docs/benchmarks/stt-benchmark-20260430-194650.csv`

## Summary

| Provider | Phase | Model | Beam | Mean wall s | Median wall s | Mean RTF | Mean WER vs reference | Mean CER | Words/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `local` | `final` | `distil-large-v3` | 3 | 3.474 | 3.661 | 0.805 | 0.000 | 0.000 | 3.098 |
| `local` | `preview` | `small` | 1 | 1.135 | 0.972 | 0.209 | 0.327 | 0.249 | 9.558 |
| `local` | `reference` | `distil-large-v3` | 3 | 3.426 | 3.590 | 0.797 | 0.000 | 0.000 | 3.128 |

## Two-Phase Simulation

| Provider | Preview model | Final model | Mean preview s | Mean final s | Mean total s | Mean total RTF |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `local` | `small` | `distil-large-v3` | 1.135 | 3.474 | 4.609 | 1.014 |

## Interpretation

- The configured local model pair is still sensible: `small@1` is fast enough for preview, while `distil-large-v3@3` is the quality anchor for final text.
- The sequential preview+final number is not the normal perceived wait. Preview runs while the user is speaking; after the endpoint, perceived STT wait is mostly the final pass, about `3.47s` warm on these samples.
- The main profile uses a longer endpoint silence for pause-and-think voice turns; lower-latency profiles can use shorter endpointing. With sample-based endpointing, that silence is based on actual captured audio rather than CPU scheduling delay from preview transcription.
- Final WAV trimming now preserves the endpoint tail, capped at 2s, so low-energy final syllables captured before endpointing are not discarded before final STT.
- The next meaningful latency improvement is not another small Whisper-family model swap; it is either a faster final backend/hardware path or changing UX so the assistant can start from preview text and correct with final text later.

## Notes

- WER/CER are measured against the configured reference model transcript, not human labels.
- Empty-reference samples count for timing but are excluded from WER/CER averages.
- RTF is wall transcription seconds divided by audio duration; lower is better.
- `text_similarity` in raw outputs is `1 - CER` after simple normalization.
- This benchmark uses saved endpoint WAVs, so it measures final transcription latency, not live partial cadence.

## Sample Details

Sample-level details are intentionally omitted from the public summary because
they can include private runtime paths, timestamps, and real utterance text.
