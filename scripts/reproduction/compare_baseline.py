#!/usr/bin/env python3
"""Compare a rebaseline run against the upstream baseline of a task.

Usage (from repo root):
    python3 scripts/reproduction/compare_baseline.py <TASK> [RESULT_DIR]

Upstream values are read from ml_tasks/<TASK>/baseline_results/original/ (the files shipped
with the repo); new values from RESULT_DIR (default: rebaseline_results/<TASK>). Prints the
task's config metric per dataset for val and test.
"""
import json
import sys

t = sys.argv[1]
new_dir = sys.argv[2] if len(sys.argv) > 2 else f"rebaseline_results/{t}"
cfg = json.load(open(f"ml_tasks/{t}/config.json"))
m = cfg["metric"]


def get(p):
    try:
        d = json.load(open(p))
    except Exception:
        return None
    return {ds: v.get("means", {}).get(m) for ds, v in d.items() if isinstance(v, dict)}


for s in ("val", "test"):
    o = get(f"ml_tasks/{t}/baseline_results/original/{s}_info.json")
    n = get(f"{new_dir}/{s}_info.json")
    for ds in (o or {}):
        ov, nv = o[ds], (n or {}).get(ds)
        diff = f" diff={nv - ov:+.4f}" if None not in (ov, nv) else ""
        print(f"{t} {s:4s} {ds} {m} ({cfg['metric_direction']}): orig={ov} new={nv}{diff}")
