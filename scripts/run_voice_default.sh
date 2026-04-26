#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${WHEATLY_CONFIG:-configs/wheatly.local.json}"

if [ ! -f "$CONFIG" ]; then
  echo "Missing $CONFIG. Copy configs/wheatly.voice-stack.example.json first." >&2
  echo "  cp configs/wheatly.voice-stack.example.json configs/wheatly.local.json" >&2
  exit 1
fi

if command -v ollama >/dev/null 2>&1; then
  if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting ollama serve..."
    ollama serve >/tmp/wheatly-ollama.log 2>&1 &
    sleep 3
  fi
fi

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . ".venv/bin/activate"
fi

PYTHONPATH=src python3 -m wheatly --config "$CONFIG" voice "$@"
