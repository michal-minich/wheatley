#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="python3"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [ -f ".env" ]; then
  set -a
  . ".env"
  set +a
fi

PROFILE="${WHEATLEY_PROFILE:-wheatley}"

exec env PYTHONPATH=src "$PYTHON_BIN" -m wheatley --profile "$PROFILE" voice
