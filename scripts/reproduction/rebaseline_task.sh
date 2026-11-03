#!/bin/bash
# Usage: scripts/reproduction/rebaseline_task.sh <TASK> <CUDA_VISIBLE_DEVICES value, "" for CPU> [TAG]
# Copies workspace/<TASK> to a fresh workspace/<TASK>__rebaseline_<ts>, then runs the task's
# val_command followed by test_command (same copy, so test can reuse val checkpoints) inside
# the task's uv env. Results/logs go to rebaseline_results/<TASK>/.
set -u
T=$1; GPU=$2; TAG=${3:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd $ROOT
CFG=ml_tasks/$T/config.json
ENV=$(python3 -c "import json;print(json.load(open('$CFG'))['conda_env'])")
SUB=$(python3 -c "import json;print(json.load(open('$CFG'))['repo_dir'].split('/',2)[2])")
TS=$(date +%Y%m%d_%H%M%S)
COPY=workspace/${T}__rebaseline_$TS
OUT=$ROOT/rebaseline_results/$T$TAG
mkdir -p $OUT
cp -a workspace/$T $COPY
REPO=$ROOT/$COPY/$SUB
{
  echo "task=$T env=$ENV gpu='$GPU'"
  echo "copy=$COPY"
  echo "repo HEAD: $(git -C $REPO log --oneline -1); changes: $(git -C $REPO status --short | wc -l)"
} > $OUT/run_info.txt
VIRTUAL_ENV=$HOME/fml_envs/$ENV uv pip freeze > $OUT/pip_freeze.txt 2>/dev/null
VAL_CMD=$(python3 -c "import json;print(json.load(open('$CFG'))['val_command'])")
TEST_CMD=$(python3 -c "import json;print(json.load(open('$CFG'))['test_command'])")

cd $REPO
export VIRTUAL_ENV=$HOME/fml_envs/$ENV PATH=$HOME/fml_envs/$ENV/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU
echo "== VAL start $(date)" >> $OUT/run_info.txt
( time bash -c "$VAL_CMD" ) > $OUT/val.log 2>&1; rc=$?
echo "== VAL end $(date) rc=$rc" >> $OUT/run_info.txt
cp results_tmp/val_info.json $OUT/ 2>/dev/null || echo "VAL: no val_info.json" >> $OUT/run_info.txt
echo "== TEST start $(date)" >> $OUT/run_info.txt
( time bash -c "$TEST_CMD" ) > $OUT/test.log 2>&1; rc=$?
echo "== TEST end $(date) rc=$rc" >> $OUT/run_info.txt
cp results_tmp/test_info.json $OUT/ 2>/dev/null || echo "TEST: no test_info.json" >> $OUT/run_info.txt
echo "DONE $T"
