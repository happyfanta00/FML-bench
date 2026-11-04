#!/bin/bash
# Re-measure the baselines with at most one task per GPU at a time, so each task's wall-clock
# time is not inflated by another task sharing its GPU.
#
# Usage (from anywhere):
#   scripts/reproduction/rebaseline_queue.sh <LOG_DIR> [TAG]
#
# QUEUES: space-separated "GPU:TASK,TASK,..." entries, one serial queue per GPU (the queues run
#         in parallel). Default: the 7 GPU tasks of FML-bench-Lite on 4 GPUs, balanced by the
#         val+test times measured on the first machine (docs/reproduction_log.md, section 3).
# CPU_TASKS: tasks run with CUDA_VISIBLE_DEVICES="" after all GPU queues have finished, alone,
#            because TF-on-CPU (ART) uses every core and would slow the CPU-bound GPU tasks
#            (domainbed, pycil, usb) down. Default: Robustness_and_Reliability_art.
# TAG: suffix for rebaseline_results/<TASK><TAG> (default: empty).
#
# Each task is run by rebaseline_task.sh (val then test in one workspace copy). The script also
# records `nvidia-smi dmon` (5 s) per GPU in LOG_DIR and prints a timing summary at the end.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG=${1:?usage: rebaseline_queue.sh <LOG_DIR> [TAG]}
TAG=${2:-}
QUEUES=${QUEUES:-"0:Continual_Learning_pycil 1:Generalization_domainbed_officehome,Privacy_opacus 2:Data_Efficiency_usb,Generalization_domainbed 3:Robustness_openood,Privacy_privacymeter"}
CPU_TASKS=${CPU_TASKS:-Robustness_and_Reliability_art}
mkdir -p "$LOG"
LOG=$(cd "$LOG" && pwd)
cd "$ROOT"

echo "$(date '+%F %T') start  QUEUES='$QUEUES' CPU_TASKS='$CPU_TASKS' TAG='$TAG'" | tee -a "$LOG/queue.txt"

dmon_pids=()
gpus=$(for q in $QUEUES; do echo "${q%%:*}"; done | sort -u | paste -sd,)
nvidia-smi dmon -i "$gpus" -s pucm -d 5 -o T > "$LOG/dmon.txt" 2>&1 &
dmon_pids+=($!)

run_one() {  # <TASK> <GPU or "">
  local t=$1 g=$2
  echo "$(date '+%F %T') begin  $t gpu='$g'" >> "$LOG/queue.txt"
  scripts/reproduction/rebaseline_task.sh "$t" "$g" "$TAG" > "$LOG/$t.out" 2>&1
  echo "$(date '+%F %T') end    $t gpu='$g'" >> "$LOG/queue.txt"
}

pids=()
for q in $QUEUES; do
  g=${q%%:*}
  (
    IFS=, read -ra ts <<< "${q#*:}"
    for t in "${ts[@]}"; do run_one "$t" "$g"; done
  ) &
  pids+=($!)
done
wait "${pids[@]}"
kill "${dmon_pids[@]}" 2>/dev/null

for t in $CPU_TASKS; do run_one "$t" ""; done
echo "$(date '+%F %T') done" | tee -a "$LOG/queue.txt"

# Summary: wall-clock (bash `time` real) of val/test and the metric vs the adopted baseline.
echo
for q in $QUEUES "cpu:${CPU_TASKS// /,}"; do
  IFS=, read -ra ts <<< "${q#*:}"
  for t in "${ts[@]}"; do
    [[ -z $t ]] && continue
    d=rebaseline_results/$t$TAG
    v=$(grep -m1 '^real' "$d/val.log" 2>/dev/null | awk '{print $2}')
    s=$(grep -m1 '^real' "$d/test.log" 2>/dev/null | awk '{print $2}')
    rc=$(grep -o 'rc=[0-9]*' "$d/run_info.txt" 2>/dev/null | paste -sd' ')
    printf '%-40s gpu=%-4s val=%-10s test=%-10s %s\n' "$t" "${q%%:*}" "${v:-?}" "${s:-?}" "$rc"
    python3 scripts/reproduction/compare_baseline.py "$t" "$d" 2>&1 | sed 's/^/    /'
  done
done | tee "$LOG/summary.txt"
