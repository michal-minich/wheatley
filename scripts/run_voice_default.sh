#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${WHEATLY_CONFIG:-}"
PROFILE="${WHEATLY_PROFILE:-wheatly}"

if [ -n "$CONFIG" ]; then
  if [ ! -f "$CONFIG" ]; then
    echo "Missing config: $CONFIG" >&2
    exit 1
  fi
  WHEATLY_ARGS=(--config "$CONFIG")
else
  if [ ! -f "profiles/$PROFILE/config.jsonc" ]; then
    echo "Missing profile: profiles/$PROFILE/config.jsonc" >&2
    echo "Copy one from examples/profiles/ or set WHEATLY_PROFILE." >&2
    exit 1
  fi
  WHEATLY_ARGS=(--profile "$PROFILE")
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

PYTHONPATH=src python3 -m wheatly "${WHEATLY_ARGS[@]}" voice "$@"
