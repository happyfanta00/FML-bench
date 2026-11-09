#!/bin/bash
# Run an agent (default ai_scientist_v2) on the Lite tasks in GPU "slots": tasks within a slot
# run one after another, slots run in parallel. Each run goes through sandbox/run_in_bwrap.sh.
#
# Usage (from anywhere):
#   scripts/reproduction/run_lite_queue.sh LOG_DIR
# Env: AGENT (default ai_scientist_v2; must match configs/agents/$AGENT.yaml);
#      QUEUE_SLOTS: ";"-separated "<CUDA_VISIBLE_DEVICES>|<task-config> ..." slots replacing the
#        default layout below, e.g. "0|generalization;1|privacy_opacus;|robustness_and_reliability_art";
#      EXTRA: extra run_agent_benchmark.py overrides ("key=value ...").
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG=${1:?usage: run_lite_queue.sh LOG_DIR}
AGENT=${AGENT:-ai_scientist_v2}
mkdir -p "$LOG"
cd "$ROOT"

# slot = "<CUDA_VISIBLE_DEVICES>|<task-config> <task-config> ..."   ("" = CPU only)
SLOTS=(
  "0|generalization_officehome"
  "0|privacy_privacymeter continual_learning_pycil"
  "1|data_efficiency_usb"
  "1|robustness_openood generalization"
  "|robustness_and_reliability_art"
)
[[ -n ${QUEUE_SLOTS:-} ]] && IFS=';' read -r -a SLOTS <<< "$QUEUE_SLOTS"

run_slot() {
  local gpu=$1; shift
  for cfg in "$@"; do
    echo "$(date '+%F %T') start $cfg gpu=[$gpu]" >> "$LOG/queue.log"
    CUDA_VISIBLE_DEVICES=$gpu sandbox/run_in_bwrap.sh \
      --agent-config "configs/agents/$AGENT.yaml" \
      --task-config "configs/tasks/$cfg.yaml" \
      --model us.anthropic.claude-opus-5-5 --provider AnthropicBedrock \
      --workspace-label opus55 \
      agent.$AGENT.max_steps=100 agent.$AGENT.execute_timeout=4800 ${EXTRA:-} \
      > "$LOG/$cfg.log" 2>&1
    echo "$(date '+%F %T') end   $cfg rc=$?" >> "$LOG/queue.log"
  done
}

for s in "${SLOTS[@]}"; do
  run_slot "${s%%|*}" ${s#*|} &
done
wait
echo "$(date '+%F %T') all slots done" >> "$LOG/queue.log"
