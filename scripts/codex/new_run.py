#!/usr/bin/env python3
"""Create a Codex (goal mode) run for one FML-bench task (see docs/codex_agent_design.md).

    python3 scripts/codex/new_run.py <TASK> --gpu N --cpus SLOT [--budget-hours 1] [--max-runs 20]
        [--baseline-dir rebaseline_results/<TASK>__l4x1] [--model openai.gpt-6-astra] [--effort high]

Same layout and tools as the Claude Code runs (scripts/claude_code/new_run.py), except:
  codex_home/            CODEX_HOME (~/.codex in the sandbox): config.toml, sessions/ (rollouts)
    AGENTS.md            task brief, mounted read-only as Codex's global instructions. Codex only
                         reads AGENTS.md from the git root down to the cwd, so /task/AGENTS.md
                         (above the repo) would not be loaded.
  task/goal.txt          the `/goal ...` line to type in the Codex TUI
Run directories: $FMLBENCH_CC_RUNS/codex/<TASK>/<run_id>/. Start with scripts/codex/launch.sh <run_dir>.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "claude_code"))
import common  # noqa: E402
import new_run as cc_new_run  # noqa: E402

MODEL = "openai.gpt-6-astra"
EFFORT = "high"
AWS_PROFILE = "codex-bedrock"     # ~/.aws/config: credential_process -> default chain (instance role)
AWS_REGION = "us-west-2"
VAL_TIMES = HERE / "val_times_l4.json"   # baseline val seconds measured in the sandbox on this machine
RUNS_ROOT = common.RUNS_ROOT / "codex"


def gpu_name(gpu: int | None) -> str | None:
    if gpu is None:
        return None
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    name, mib = [x.strip() for x in out.split(",")]
    return f"1 × {name} ({round(int(mib) * 1.048576 / 1000)} GB)"   # decimal GB, as on spec sheets


def build_agents_md(task: str, meta: dict, n_cpus: int, gpu: int | None) -> str:
    # The Claude Code brief, with this machine's runtime hint and GPU.
    orig = cc_new_run.baseline_val_seconds
    cc_new_run.baseline_val_seconds = lambda t: (json.loads(VAL_TIMES.read_text()).get(t)
                                                 if VAL_TIMES.exists() else None)
    try:
        text = cc_new_run.build_claude_md(task, meta, n_cpus)
    finally:
        cc_new_run.baseline_val_seconds = orig
    hw = gpu_name(gpu)
    if hw:
        text = text.replace("1 × NVIDIA RTX PRO 6000 Blackwell (96 GB)", hw)
    # Codex runs commands in PTY sessions; say how to wait for fml-val.
    text = text.replace(
        "It blocks until the run finishes, so start it in the background and wait for it to finish.",
        "It blocks until the run finishes (a few minutes or more): run it in a command session, keep "
        "polling that session until the command exits, and only then read the result.")
    if meta.get("viz"):
        text = viz_brief(text, meta)
    return text


def viz_brief(text: str, meta: dict) -> str:
    """Visualization runs (new_run.py --viz): every fml-val carries a short card, results get a note,
    and the baseline has already been run by the host (runs/baseline/, shown as run 0)."""
    base = f"{meta['baseline_val']:.4f}" if meta.get("baseline_val") is not None else "?"
    old = "- `fml-val` — the official validation run."
    assert old in text
    text = text.replace(old, (
        "- `fml-val --title \"<short name, at most 60 characters>\" --idea \"<what you changed and why>\" "
        "[--expect \"<what you expect>\"]` — the official validation run. `--title` and `--idea` are required "
        "(without them the run is refused and not counted); they are recorded with the run, so write them for "
        "a reader who follows your research: the idea in one to three sentences, the expectation as a "
        "concrete prediction if you have one."))
    old = "- `fml-status` —"
    text = text.replace(old, (
        "- `fml-note <run id> \"<conclusion>\"` — record your conclusion about a finished run (what the result "
        "tells you and what you will do next), in one or two sentences. Do this after every `fml-val` run.\n"
        + old), 1)
    old = f"- Baseline (unmodified code): validation `{meta['metric']}` = "
    i = text.index(old)
    j = text.index("\n", i)
    text = text[:j] + (f" The baseline has already been run once for you (run 0, validation {base}; "
                       "`fml-status` shows it): do not spend an `fml-val` run on the unmodified code.") + text[j:]
    return text


def build_goal(task: str, meta: dict) -> str:
    """The objective typed after /goal. Codex has no separate judge: the agent itself calls
    update_goal(status="complete"), so the completion conditions are stated as a self-check."""
    return (
        f"/goal Solve the FML-bench task {task} described in your AGENTS.md instructions (also at "
        f"~/.codex/AGENTS.md): improve the validation metric {meta['metric']} ({meta['metric_direction']} "
        f"is better) by editing only the target files and measuring every candidate with `fml-val` (at most "
        f"{meta['max_runs']} runs in total), and keep working until the budget is used up. Mark this goal "
        "complete only after ALL of the following hold, each shown by command output in this session: "
        "(1) `fml-status`, run after your final `fml-val` run, reports phase WRAP-UP or BUDGET EXHAUSTED; "
        "(2) `fml-submit` printed \"SUBMITTED run <id>\" for a run that finished with returncode 0; "
        "(3) /task/scratch/FINAL_REPORT.md exists and you printed its contents. A good result or stalled "
        "progress is not completion: while `fml-status` reports SEARCH, keep forming and testing new ideas. "
        "While an `fml-val` run is in progress, keep polling it until it finishes. Mark the goal blocked "
        "only if `fml-val` cannot complete even the unmodified baseline for reasons outside your control.")


def build_config_toml(meta: dict, repo_in_sandbox: str, model: str, effort: str) -> str:
    # Visualization runs keep reasoning summaries in the rollouts (shown in the process view).
    summary = 'model_reasoning_summary = "auto"\n' if meta.get("viz") else ""
    return f"""# Codex config for an FML-bench run (generated by scripts/codex/new_run.py).
model_provider = "amazon-bedrock"
model = "{model}"
model_reasoning_effort = "{effort}"
{summary}
# The bwrap sandbox is the isolation; Codex's own sandbox would nest inside it.
approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "disabled"
check_for_update_on_startup = false

[features]
goals = true
memories = false

[model_providers.amazon-bedrock.aws]
profile = "{AWS_PROFILE}"
region = "{AWS_REGION}"

[projects."{repo_in_sandbox}"]
trust_level = "trusted"
"""


def create_run(task: str, *, gpu: int | None, cpu_slot: str, budget_hours: float, max_runs: int,
               baseline_dir: Path | None, model: str, effort: str, run_id: str | None = None,
               viz: bool = False) -> Path:
    """viz: visualization run (scripts/codex/viz): fml-val requires --title/--idea, keeps the model and
    the training curve, reasoning summaries on. The baseline is run by the host first (viz/jobs.py),
    which sets meta.baseline_val and only then writes AGENTS.md and goal.txt (write_prompts)."""
    if task not in common.TASK_DATA:
        raise SystemExit(f"unknown task {task}")
    if (gpu is None) != (task in common.CPU_ONLY_TASKS):
        raise SystemExit(f"{task}: " + ("CPU-only task, do not pass --gpu" if gpu is not None
                                        else "--gpu is required"))
    if cpu_slot not in common.CPU_SLOTS:
        raise SystemExit(f"--cpus must be one of {list(common.CPU_SLOTS)}")
    common.view_info(task)                      # masked data must exist
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run = RUNS_ROOT / task / run_id
    if run.exists():
        raise SystemExit(f"{run} exists")
    tdir = run / "task"
    for d in (tdir / "workspace", tdir / "scratch", tdir / "runs", run / "codex_home", run / "cache"):
        d.mkdir(parents=True)
    common.copy_template(task, tdir / "workspace" / task)
    for rel in common.TASK_CACHE_FILES.get(task, []):
        src = common.HOME / ".cache" / rel
        if src.exists():
            (run / "cache" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, run / "cache" / rel)
    meta = cc_new_run.make_meta(task, budget_hours, max_runs)
    meta["model"] = model
    if viz:
        meta.update(viz=True, require_card=True, keep_artifacts=True)
    if baseline_dir:
        # Baseline re-measured on this hardware: the agent's first fml-val reproduces it exactly.
        info = json.loads((baseline_dir / "val_info.json").read_text())
        meta["baseline_val"] = common.primary_metric(info, meta["metric"])
        meta["baseline_source"] = str(baseline_dir)
    if viz:
        # The reference (re-measured on this machine) is what the host-run baseline is checked against.
        meta["baseline_reference"] = meta["baseline_val"]
        meta["baseline_val"] = None
    (tdir / "meta.json").write_text(json.dumps(meta, indent=1))
    cpus = common.CPU_SLOTS[cpu_slot]
    n_cpus = sum(int(b) - int(a) + 1 for a, b in
                 ((r.split("-") + r.split("-"))[:2] for r in cpus.split(",")))
    repo_in_sandbox = f"/task/workspace/{task}/{meta['repo_subdir']}"
    (run / "codex_home" / "config.toml").write_text(build_config_toml(meta, repo_in_sandbox, model, effort))
    version = subprocess.run([str(common.CODEX_BIN_DIR / "codex"), "--version"], capture_output=True,
                             text=True).stdout.strip()
    (run / "host.json").write_text(json.dumps({
        "agent": "codex", "task": task, "run_id": run_id, "gpu": gpu, "cpu_slot": cpu_slot, "cpus": cpus,
        "model": model, "reasoning_effort": effort, "provider": "amazon-bedrock",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "codex_version": version,
        "codex_bin_dir": str(common.CODEX_BIN_DIR),
        "baseline_source": str(baseline_dir) if baseline_dir else "ml_tasks baseline_results",
        "n_cpus": n_cpus, "viz": viz}, indent=1))
    if not viz:
        write_prompts(run)
    return run


def write_prompts(run: Path) -> None:
    """AGENTS.md and goal.txt from the current meta.json (for viz runs: after the baseline)."""
    host = json.loads((run / "host.json").read_text())
    meta = json.loads((run / "task" / "meta.json").read_text())
    if meta.get("baseline_val") is None:
        raise SystemExit("baseline_val is not set yet (run the baseline first)")
    (run / "codex_home" / "AGENTS.md").write_text(build_agents_md(host["task"], meta, host["n_cpus"], host["gpu"]))
    (run / "task" / "goal.txt").write_text(build_goal(host["task"], meta) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--gpu", type=int, help="GPU index (omit for the CPU-only ART task)")
    ap.add_argument("--cpus", required=True, help=f"CPU slot: {', '.join(common.CPU_SLOTS)}")
    ap.add_argument("--budget-hours", type=float, default=1.0)
    ap.add_argument("--max-runs", type=int, default=20)
    ap.add_argument("--baseline-dir", help="val_info.json of a baseline re-measured on this machine "
                    "(e.g. rebaseline_results/<TASK>__l4x1); default: ml_tasks/<TASK>/baseline_results")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default=EFFORT, help="model_reasoning_effort (none/low/medium/high/xhigh)")
    ap.add_argument("--run-id")
    ap.add_argument("--viz", action="store_true", help="visualization run (normally created by scripts/codex/viz)")
    a = ap.parse_args()
    run = create_run(a.task, gpu=a.gpu, cpu_slot=a.cpus, budget_hours=a.budget_hours, max_runs=a.max_runs,
                     baseline_dir=Path(a.baseline_dir).resolve() if a.baseline_dir else None,
                     model=a.model, effort=a.effort, run_id=a.run_id, viz=a.viz)
    print(f"created {run}")
    print(f"start it with:  {common.REPO}/scripts/codex/launch.sh {run}")


if __name__ == "__main__":
    main()
