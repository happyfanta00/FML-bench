#!/bin/bash
# Start the visualization server (127.0.0.1 only). From the Mac:
#   ssh -L 8765:localhost:8765 ec2-user@<host>   then open http://localhost:8765
# Env: FMLBENCH_VIZ_RUN (run dir, default $FMLBENCH_CC_RUNS/codex/Generalization_domainbed/viz), VIZ_PORT.
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../env_l4.sh"
cd "$HERE"
exec "${FMLBENCH_UV_ENVS:-$HOME/fml_envs}/viz/bin/uvicorn" server:app --host 127.0.0.1 --port "${VIZ_PORT:-8765}" --log-level warning
