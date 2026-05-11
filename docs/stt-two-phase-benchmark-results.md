# STT Benchmark Results

Generated: `2026-04-30T19:00:34+0200`

Reference model: `distil-large-v3` beam `3`.

Samples: `16` user WAV files from profile runtime logs.

Raw JSON: `docs/benchmarks/stt-benchmark-20260430-185616.json`

Raw CSV: `docs/benchmarks/stt-benchmark-20260430-185616.csv`

## Summary

| Provider | Phase | Model | Beam | Mean wall s | Median wall s | Mean RTF | Mean WER vs reference | Mean CER | Words/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `remote` | `final` | `distil-large-v3` | 3 | 5.510 | 5.783 | 1.342 | 0.000 | 0.000 | 1.984 |
| `remote` | `preview` | `small` | 1 | 2.258 | 2.130 | 0.557 | 0.307 | 0.240 | 4.624 |
| `local` | `final` | `distil-large-v3` | 3 | 3.411 | 3.577 | 0.806 | 0.000 | 0.000 | 3.178 |
| `local` | `preview` | `small` | 1 | 1.113 | 0.943 | 0.208 | 0.311 | 0.244 | 9.671 |
| `local` | `reference` | `distil-large-v3` | 3 | 3.427 | 3.615 | 0.800 | 0.000 | 0.000 | 3.143 |

## Two-Phase Simulation

| Provider | Preview model | Final model | Mean preview s | Mean final s | Mean total s | Mean total RTF |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `remote` | `small` | `distil-large-v3` | 2.258 | 5.510 | 7.769 | 1.899 |
| `local` | `small` | `distil-large-v3` | 1.113 | 3.411 | 4.524 | 1.014 |

## Notes

- WER/CER are measured against the configured reference model transcript, not human labels.
- Empty-reference samples count for timing but are excluded from WER/CER averages.
- RTF is wall transcription seconds divided by audio duration; lower is better.
- `text_similarity` in raw outputs is `1 - CER` after simple normalization.
- This benchmark uses saved endpoint WAVs, so it measures final transcription latency, not live partial cadence.

## Sample Details

Sample-level details are intentionally omitted from the public summary because
they can include private runtime paths, timestamps, and real utterance text.
