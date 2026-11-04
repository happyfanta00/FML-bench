#!/bin/bash
# Smoke-test ai_scientist_v2 on the 7 Lite tasks (all but Privacy_opacus): 3 steps each, all
# tasks in parallel inside sandbox/run_in_bwrap.sh, with GPU memory / host PSS sampled by
# resource_monitor.py. Runs in the foreground; takes ~1.5 h on 2 GPUs.
#
# Usage (from anywhere):
#   scripts/reproduction/run_smoke_lite.sh [LOG_DIR]       # default docs/reproduction_logs/smoke_lite7
# Env: GPU_A / GPU_B (default 0 / 1) pick the two GPUs; STEPS (default 3);
#      OUTPUT_DIR (default benchmark_results_smoke);
#      SMOKE_TASKS: space-separated "task-config|CUDA_VISIBLE_DEVICES" pairs replacing the default
#        7-task / 2-GPU layout, e.g. "privacy_opacus|2 robustness_and_reliability_art|";
#      EXTRA_OVERRIDES: extra key=value overrides for run_agent_benchmark.py, e.g.
#        "agent.ai_scientist_v2.execute_timeout=7200" on slower GPUs.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
LOG=${1:-docs/reproduction_logs/smoke_lite7}
GPU_A=${GPU_A:-0} GPU_B=${GPU_B:-1} STEPS=${STEPS:-3} OUTPUT_DIR=${OUTPUT_DIR:-benchmark_results_smoke}
mkdir -p "$LOG"

# task-config -> CUDA_VISIBLE_DEVICES ("" = CPU only; ART's TF 2.10 can't use recent GPUs)
TASKS=(
  "continual_learning_pycil|$GPU_A"
  "generalization|$GPU_A"
  "privacy_privacymeter|$GPU_A"
  "data_efficiency_usb|$GPU_B"
  "generalization_officehome|$GPU_B"
  "robustness_openood|$GPU_B"
  "robustness_and_reliability_art|"
)
[[ -n ${SMOKE_TASKS:-} ]] && read -r -a TASKS <<< "$SMOKE_TASKS"
read -r -a EXTRA <<< "${EXTRA_OVERRIDES:-}"

python3 scripts/reproduction/resource_monitor.py "$LOG/resources.csv" 5 > "$LOG/monitor.out" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

pids=()
for t in "${TASKS[@]}"; do
  cfg=${t%%|*} gpu=${t#*|}
  CUDA_VISIBLE_DEVICES=$gpu sandbox/run_in_bwrap.sh --output-dir "$OUTPUT_DIR" \
    --agent-config configs/agents/ai_scientist_v2.yaml \
    --task-config "configs/tasks/$cfg.yaml" \
    --model us.anthropic.claude-opus-5-5 --provider AnthropicBedrock \
    --workspace-label smoke agent.ai_scientist_v2.max_steps=$STEPS "${EXTRA[@]}" \
    > "$LOG/$cfg.log" 2>&1 &
  pids+=($!)
  echo "$(date '+%F %T') started $cfg gpu=[$gpu] pid=$!"
done

for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; echo "$(date '+%F %T') ${TASKS[$i]%%|*} exited rc=$?"
done

echo; echo "== results"
for t in "${TASKS[@]}"; do
  cfg=${t%%|*}
  echo "$cfg: $(grep -c 'Running command' "$LOG/$cfg.log") commands, $(grep -ci 'bug execution' "$LOG/$cfg.log") buggy;" \
       "$(grep -hE 'Best Val Metric|Test Metric' "$LOG/$cfg.log" | cut -c1-40 | tr '\n' ' ')"
done
echo; echo "== resources"
python3 scripts/reproduction/resource_monitor.py --summary "$LOG/resources.csv"
