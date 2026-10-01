#!/usr/bin/env bash
# deploy/senseless-ui.sh -- start the touchscreen app full screen, logging to ~/senseless.log
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python -m senseless.ui "$@" >> "$HOME/senseless.log" 2>&1
