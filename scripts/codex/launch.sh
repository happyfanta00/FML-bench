#!/bin/bash
# Enter the sandbox of a Codex run. See launch.py.
exec python3 "$(dirname "$0")/launch.py" "$@"
