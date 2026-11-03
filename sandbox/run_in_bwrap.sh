#!/bin/bash
# Run run_agent_benchmark.py inside a bubblewrap sandbox.
#
# Usage (from anywhere):
#   sandbox/run_in_bwrap.sh [--output-dir DIR] <run_agent_benchmark.py args...>
#   sandbox/run_in_bwrap.sh --shell [-c CMD]   # shell (or one command) inside the same sandbox
#
# What the sandbox sees:
#   - repo root           read-only   (agents/, benchmark/, ml_tasks/, configs/, ...)
#   - repo/workspace      read-write  (the runner copies task templates here, agents edit the copy)
#   - output dir          read-write  (default repo/benchmark_results; --output-dir is honoured)
#   - $FMLBENCH_UV_ENVS   read-only   (harness env "fmlbench" + per-task envs, picked by
#                                      benchmark.utils.env_command from config.json's conda_env)
#   - uv-managed CPython  read-only   (the venvs symlink their interpreter here)
#   - ~/.cache            read-write  (pretrained-weight caches: torch hub, HF, ...)
#   - /usr, /etc, /sys    read-only;  fresh /proc, /dev, /tmp, /dev/shm and an empty $HOME
#   - NVIDIA device nodes; CUDA_VISIBLE_DEVICES is passed through (default: all GPUs)
#   - network is shared: Bedrock is reached over HTTPS with the instance-role credentials
#     from IMDS (a separate network namespace would cut IMDS off)
# Everything else on the host (other home dirs, ~/.ssh, ~/.aws unless AWS_PROFILE is set,
# conda, the EFS data outside this repo) is not visible.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
UV_ENVS=${FMLBENCH_UV_ENVS:-$HOME/fml_envs}
UV_PYTHON=${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}
CACHE=${FMLBENCH_SANDBOX_CACHE:-$HOME/.cache}
PY=$UV_ENVS/fmlbench/bin/python

# Resolve the output dir the same way run_agent_benchmark.py will (relative to repo root).
OUT=benchmark_results
prev=""
for a in "$@"; do
  [[ $prev == --output-dir ]] && OUT=$a
  [[ $a == --output-dir=* ]] && OUT=${a#--output-dir=}
  prev=$a
done
[[ $OUT = /* ]] || OUT=$ROOT/$OUT
mkdir -p "$OUT" "$CACHE"

for p in "$PY" "$UV_PYTHON"; do
  [[ -e $p ]] || { echo "missing $p (build envs with setup.py first)" >&2; exit 1; }
done

args=(
  --die-with-parent --new-session
  --unshare-all --share-net
  --ro-bind /usr /usr
  --symlink usr/bin /bin --symlink usr/sbin /sbin
  --symlink usr/lib /lib --symlink usr/lib64 /lib64
  --ro-bind /etc /etc
  --ro-bind-try /run/systemd/resolve /run/systemd/resolve  # /etc/resolv.conf points here
  --ro-bind /sys /sys
  --proc /proc
  --dev /dev
  --tmpfs /dev/shm
  --tmpfs /tmp
  --tmpfs "$HOME"
  --ro-bind "$UV_ENVS" "$UV_ENVS"
  --ro-bind "$UV_PYTHON" "$UV_PYTHON"
  --bind "$CACHE" "$HOME/.cache"
  --ro-bind "$ROOT" "$ROOT"
  --bind "$ROOT/workspace" "$ROOT/workspace"
  --bind "$OUT" "$OUT"
  --chdir "$ROOT"
  --clearenv
  --setenv HOME "$HOME"
  --setenv USER "$(id -un)"
  --setenv PATH "$UV_ENVS/fmlbench/bin:/usr/local/bin:/usr/bin"
  --setenv VIRTUAL_ENV "$UV_ENVS/fmlbench"
  --setenv LANG C.UTF-8
  --setenv TMPDIR /tmp
  --setenv PYTHONDONTWRITEBYTECODE 1
  --setenv PYTHONUNBUFFERED 1
  --setenv FMLBENCH_UV_ENVS "$UV_ENVS"
  --setenv CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
)
for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia[0-9]*; do
  [[ -e $d ]] && args+=(--dev-bind "$d" "$d")
done
# Pass through AWS region/credential settings and the agent knobs if the caller set them.
for v in AWS_REGION AWS_DEFAULT_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
         AWS_PROFILE FMLBENCH_CLAUDE_EFFORT; do
  [[ -n ${!v:-} ]] && args+=(--setenv "$v" "${!v}")
done
[[ -n ${AWS_PROFILE:-} && -d $HOME/.aws ]] && args+=(--ro-bind "$HOME/.aws" "$HOME/.aws")

if [[ ${1:-} == --shell ]]; then
  exec bwrap "${args[@]}" bash "${@:2}"
fi
exec bwrap "${args[@]}" "$PY" run_agent_benchmark.py "$@"
