#!/bin/sh
exec "$(dirname "$0")/.venv/bin/python" "$(dirname "$0")/paralax_3d.py" "$@"
