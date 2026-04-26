#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[audio,stt]'

echo "Bootstrap complete."
echo "Run: PYTHONPATH=src python3 -m wheatly doctor"

