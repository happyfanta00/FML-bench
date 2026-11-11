"""Run a task's val/test commands on a code snapshot inside a non-interactive sandbox.

Used by verify_task_view.py (agent view = masked data) and finalize.py (real data). The
sandbox has the same layout as the agent's (common.bwrap_args), without Claude Code; inside,
/opt/fml/bin/fml-exec copies the template, writes the snapshot's target files and runs the
phases in order in one copy (test reuses the checkpoints written by val).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import new_run  # noqa: E402


def baseline_snapshot(task: str) -> dict:
    repo = common.template_dir(task) / common.repo_subdir(task)
    return {tf: (repo / tf).read_text() for tf in common.load_config(task)["target_files"]}


def run_exec(task: str, snapshot: dict, *, view: str, gpu, cpus: str, work: Path,
             phases=("val", "test"), timeouts: dict | None = None) -> dict:
    """Run the phases; returns {"phases": [...], "val_info": ..., "test_info": ..., "out": dir}."""
    cfg = common.load_config(task)
    tdir = work / "task"
    (tdir / "workspace").mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(exist_ok=True)
    for rel in common.TASK_CACHE_FILES.get(task, []):
        src = common.HOME / ".cache" / rel
        if src.exists() and not (cache / rel).exists():
            (cache / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, cache / rel)
    meta = new_run.make_meta(task, 0, 0)
    (tdir / "meta.json").write_text(json.dumps(meta, indent=1))
    cmds = {"val": (cfg["val_command"], "results_tmp/val_info.json"),
            "test": (cfg["test_command"], "results_tmp/test_info.json")}
    spec = {"snapshot": snapshot, "out": "/task/out",
            "phases": [{"name": p, "command": cmds[p][0], "output": cmds[p][1],
                        "timeout": (timeouts or {}).get(p)} for p in phases]}
    (tdir / "exec_spec.json").write_text(json.dumps(spec, indent=1))
    refs = common.ml_tasks_refs(task, [cmds[p][0] for p in phases])
    args = common.bwrap_args(task=task, task_dir=tdir, view=view, gpu=gpu, cache_dir=cache,
                             claude_config=None, ml_tasks_paths=refs)
    argv = ["taskset", "-c", cpus, "bwrap", *args, "/opt/fml/bin/fml-exec"]
    t0 = time.time()
    with open(work / "sandbox.log", "w") as log:
        rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT).returncode
    out = tdir / "out"
    res = json.loads((out / "exec.json").read_text()) if (out / "exec.json").exists() else {"phases": []}
    res.update(sandbox_returncode=rc, wall_s=round(time.time() - t0, 1), out=str(out))
    for p in ("val", "test"):
        f = out / f"{p}_info.json"
        res[f"{p}_info"] = json.loads(f.read_text()) if f.exists() else None
    return res
