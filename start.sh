#!/usr/bin/env bash
# ClayMirror launcher for Linux / macOS (Intel NPU works on Linux with the intel-npu driver)
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  echo "[ClayMirror] creating virtual environment ..."
  "$PY" -m venv .venv
fi
if [ ! -f .venv/deps-ok.txt ]; then
  echo "[ClayMirror] installing packages (first run only) ..."
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
  echo ok > .venv/deps-ok.txt
fi
exec .venv/bin/python server/server.py "$@"
