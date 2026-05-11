# STT Benchmark

Run a repeatable faster-whisper benchmark over saved user utterance WAVs:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_stt.py
```

Useful faster run while tuning:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_stt.py \
  --max-samples 8 \
  --model Systran/faster-distil-whisper-small.en \
  --model tiny.en \
  --model small.en \
  --model small \
  --reference-model distil-large-v3
```

The generated Markdown summary is written to `docs/stt-benchmark-results.md`.
Raw JSON/CSV files are written under ignored `docs/benchmarks/` local artifacts.

Current local two-phase run:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_stt.py \
  --min-duration 0.4 \
  --max-duration 90 \
  --duration-bucket short:0.4:4:4 \
  --duration-bucket medium:4:12:5 \
  --duration-bucket long:12:25:5 \
  --duration-bucket very_long:25:90:2 \
  --phase-model preview:small@1 \
  --phase-model final:distil-large-v3@3 \
  --reference-model distil-large-v3 \
  --reference-beam-size 3 \
  --markdown docs/stt-benchmark-results.md
```

## Two-Phase Local vs Remote

To compare local STT with a stronger machine, start the STT server on that
machine and point the benchmark at its OpenAI-compatible endpoint.

Run the two-phase benchmark with an anonymized remote label:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_stt.py \
  --min-duration 0.4 \
  --max-duration 90 \
  --duration-bucket short:0.4:4:4 \
  --duration-bucket medium:4:12:5 \
  --duration-bucket long:12:25:5 \
  --duration-bucket very_long:25:90:2 \
  --phase-model preview:small@1 \
  --phase-model final:distil-large-v3@3 \
  --reference-model distil-large-v3 \
  --reference-beam-size 3 \
  --include-remote \
  --remote-base-url http://remote-stt.example:8765/v1 \
  --remote-label remote \
  --markdown docs/stt-two-phase-benchmark-results.md
```

Example remote STT server command:

```bash
ssh user@remote-stt.example \
  'cd /path/to/wheatley && nohup env PYTHONPATH=src python3 -m wheatley stt-server --host 0.0.0.0 --port 8765 --default-model small --model en=small --model sk=small --beam-size 1 > stt-server.log 2>&1 & echo $! > stt-server.pid'
```

Metrics:

- Wall seconds: local transcription time for one saved WAV.
- RTF: wall seconds divided by audio duration; lower is better.
- WER/CER: text distance against the configured reference model transcript, not human labels.
- Words/s: transcript word count divided by wall seconds.

The default reference is `distil-large-v3`. That is useful for consistency, but it is not ground truth. For a serious quality pass, manually label a fixed small set of WAVs and compare models against those labels.
