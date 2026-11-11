#!/bin/bash
# Enter the sandbox of a Claude Code run. See launch.py.
exec python3 "$(dirname "$0")/launch.py" "$@"
