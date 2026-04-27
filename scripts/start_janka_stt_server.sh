#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
  . ".venv/bin/activate"
fi

exec env PYTHONPATH=src python3 -m wheatley stt-server \
  --host 0.0.0.0 \
  --port 8765 \
  --default-model small.en \
  --model en=small.en \
  --model sk=models/whisper/whisper-large-v3-sk-ct2-int8
