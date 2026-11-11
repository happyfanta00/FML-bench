#!/usr/bin/env python3
"""Summary table of finalized Claude Code runs.

    python3 scripts/claude_code/summarize.py [RUN_DIR ...] [--out DIR]

Default: every run under $FMLBENCH_CC_RUNS/<TASK>/<run_id>/ that has final/report.json
(directories starting with "_" are skipped). Writes summary.md and summary.csv to --out
(default $FMLBENCH_CC_RUNS) and prints the Markdown table. The normalized improvement uses the
same definition and ranges as compute_agent_metrics.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

FMLBENCH_PY = common.UV_ENVS / "fmlbench" / "bin" / "python"
if Path(sys.prefix).resolve() != FMLBENCH_PY.parent.parent.resolve() and FMLBENCH_PY.exists():
    os.execv(str(FMLBENCH_PY), [str(FMLBENCH_PY), *sys.argv])          # needs numpy/pandas
sys.path.insert(0, str(common.REPO))
import compute_agent_metrics as cam  # noqa: E402

COLUMNS = ["task", "run_id", "metric", "baseline_val", "baseline_test", "submitted_run",
           "submitted_val_logged", "submitted_val_rerun", "submitted_test", "ni_test",
           "best_val_run", "best_val_test", "fml_val_runs", "timeouts", "hours", "tokens_M",
           "cost_usd", "audit_flags"]


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def row_for(run: Path) -> dict:
    rep = json.loads((run / "final" / "report.json").read_text())
    task = rep["task"]
    summ = json.loads((Path(rep["harness_dir"]) / "summary.json").read_text())
    base_dir = common.REPO / "ml_tasks" / task / "baseline_results"
    cfg = common.load_config(task)
    bval = common.primary_metric(json.loads((base_dir / "val_info.json").read_text()), cfg["metric"])
    btest = common.primary_metric(json.loads((base_dir / "test_info.json").read_text()), cfg["metric"])
    sub = rep["finals"].get("submitted") or {}
    best = rep["finals"].get("best_val") or {}
    direction, p_best, p_worst = cam.RANGE_META[task]
    p_worst = btest if p_worst == "baseline" else p_worst
    test = sub.get("test")
    if sub.get("constraint_violated"):
        test = btest                    # same rule as compute_agent_metrics: violation -> baseline
    ni = cam.normalized_improvement(test, btest, direction, p_best, p_worst) if test is not None else None
    steps = summ.get("val_steps") or []
    tok = (summ.get("token_usage") or {}).get("total_tokens")
    dur = summ.get("total_duration_seconds")
    return {
        "task": task, "run_id": run.name, "metric": f"{cfg['metric']} ({cfg['metric_direction']})",
        "baseline_val": bval, "baseline_test": btest,
        "submitted_run": sub.get("run"), "submitted_val_logged": sub.get("logged_val"),
        "submitted_val_rerun": sub.get("val"), "submitted_test": sub.get("test"),
        "ni_test": None if ni is None or ni != ni else float(ni),
        "best_val_run": best.get("run"), "best_val_test": best.get("test"),
        "fml_val_runs": len(steps), "timeouts": sum(1 for s in steps if s.get("timed_out")),
        "hours": None if dur is None else dur / 3600,
        "tokens_M": None if tok is None else tok / 1e6,
        "cost_usd": summ.get("cost_usd"),
        "audit_flags": ", ".join(k for k, v in rep["audit"]["flags"].items() if v) or "none",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out", default=str(common.RUNS_ROOT))
    a = ap.parse_args()
    if a.runs:
        runs = [Path(r).resolve() for r in a.runs]
    else:
        runs = sorted(p for t in common.RUNS_ROOT.iterdir() if t.is_dir() and not t.name.startswith("_")
                      for p in t.iterdir() if (p / "final" / "report.json").exists())
    rows = [row_for(r) for r in runs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join(
            fmt(r[c], 2 if c in ("hours", "tokens_M", "cost_usd") else 4) for c in COLUMNS) + " |")
    nis = [r["ni_test"] for r in rows if r["ni_test"] is not None]
    if nis:
        lines.append(f"\nMean normalized improvement (test) over {len(nis)} runs: {sum(nis) / len(nis):.4f}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out / 'summary.md'}, {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
