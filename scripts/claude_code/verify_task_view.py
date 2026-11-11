#!/usr/bin/env python3
"""Check a task's agent data view and measure the baseline val time under the run's CPU slot.

    python3 scripts/claude_code/verify_task_view.py <TASK> [--gpu N] --cpus SLOT

Runs the baseline code's val and test commands in the agent sandbox (masked data) and checks:
  1. val_info.json is byte-identical to ml_tasks/<TASK>/baseline_results/val_info.json
     (tasks that are not deterministic: same metric within --tol);
  2. the test metric on the masked data, next to the real baseline test value (it should
     collapse; see the per-task expectation printed with it).
Writes $FMLBENCH_CC_RUNS/_verify/<TASK>/<timestamp>/verify.json and records the val time in
scripts/claude_code/val_times.json (used for the runtime hint in CLAUDE.md).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import exec_host  # noqa: E402

NONDETERMINISTIC = {"Robustness_openood": 0.02, "Privacy_privacymeter": 0.01}
EXPECT = {
    "Privacy_opacus": "~0.10 (chance, 10 classes)",
    "Robustness_openood": "~0.5 (blank SVHN vs blank CIFAR-10 test images)",
    "Data_Efficiency_usb": "~0.01 (chance, 100 classes)",
    "Continual_Learning_pycil": "~0.01-0.02 (chance on blank images)",
    "Generalization_domainbed": "near the label-noise/colour-only level, far from the real value",
    "Generalization_domainbed_officehome": "can stay high: the dataset has only 2 classes (train/val folder), so blank images get the majority class",
    "Privacy_privacymeter": "unchanged (nothing can be masked in this task)",
    "Robustness_and_Reliability_art": "lower clean accuracy on blank images",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--cpus", required=True)
    a = ap.parse_args()
    task, cfg = a.task, common.load_config(a.task)
    work = common.RUNS_ROOT / "_verify" / task / time.strftime("%Y%m%d_%H%M%S")
    work.mkdir(parents=True)
    res = exec_host.run_exec(task, exec_host.baseline_snapshot(task), view="agent", gpu=a.gpu,
                             cpus=common.CPU_SLOTS[a.cpus], work=work)
    base_dir = common.REPO / "ml_tasks" / task / "baseline_results"
    out = Path(res["out"])
    metric = cfg["metric"]
    bval = common.primary_metric(json.loads((base_dir / "val_info.json").read_text()), metric)
    btest = common.primary_metric(json.loads((base_dir / "test_info.json").read_text()), metric)
    val = common.primary_metric(res["val_info"], metric)
    test = common.primary_metric(res["test_info"], metric)
    identical = (out / "val_info.json").exists() and \
        (out / "val_info.json").read_bytes() == (base_dir / "val_info.json").read_bytes()
    tol = NONDETERMINISTIC.get(task)
    if identical:
        val_ok = True
    elif tol is not None and val is not None:
        val_ok = abs(val - bval) <= tol
    else:
        val_ok = False
    phases = {p["name"]: p for p in res["phases"]}
    report = {
        "task": task, "cpus": a.cpus, "gpu": a.gpu, "work": str(work),
        "val_identical_bytes": identical, "val_ok": val_ok,
        "val": val, "baseline_val": bval, "test_masked": test, "baseline_test": btest,
        "expected_masked_test": EXPECT[task],
        "val_seconds": phases.get("val", {}).get("duration_s"),
        "test_seconds": phases.get("test", {}).get("duration_s"),
        "returncodes": {k: v["returncode"] for k, v in phases.items()},
    }
    (work / "verify.json").write_text(json.dumps(report, indent=1))
    if val_ok and report["val_seconds"]:
        times = Path(__file__).resolve().parent / "val_times.json"
        with open(times.with_suffix(".lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = json.loads(times.read_text()) if times.exists() else {}
            d[task] = report["val_seconds"]
            times.write_text(json.dumps(dict(sorted(d.items())), indent=1) + "\n")
    fmt = lambda v: "None" if v is None else f"{v:.4f}"
    print(f"{task}: val {'OK' if val_ok else 'MISMATCH'} ({'byte-identical' if identical else 'metric'}"
          f" {fmt(val)} vs baseline {fmt(bval)}); masked test {fmt(test)} vs real {fmt(btest)} "
          f"[expected {EXPECT[task]}]; val {report['val_seconds']} s on {a.cpus}; "
          f"returncodes {report['returncodes']}")
    return 0 if val_ok else 1


if __name__ == "__main__":
    sys.exit(main())
