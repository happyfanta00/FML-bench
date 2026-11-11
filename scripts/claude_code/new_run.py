#!/usr/bin/env python3
"""Create a Claude Code run for one FML-bench task (see docs/claude_code_agent_design.md).

    python3 scripts/claude_code/new_run.py <TASK> --gpu N --cpus G0|G1|G2|G3|ART [--budget-hours 24]

Creates $FMLBENCH_CC_RUNS/<TASK>/<run_id>/ (default /mnt/nvme/cc_runs):
  task/                 mounted as /task in the sandbox
    meta.json           run parameters read by fml-val/fml-status (read-only in the sandbox)
    CLAUDE.md           task brief (read-only); Claude Code loads it from the working dir's parent
    goal.txt            the /goal condition (read-only)
    workspace/<TASK>/   the agent's copy of the task template
    scratch/  runs/     agent notes; fml-val records
  claude_config/        CLAUDE_CONFIG_DIR (session transcripts)
  cache/                ~/.cache (pretrained weights the baseline needs)
  host.json             GPU / CPU assignment (host only)
Then start it with scripts/claude_code/launch.sh <run_dir>.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Per-task notes the generic template cannot derive.
DATA_NOTES = {
    "Privacy_opacus": "The CIFAR-10 training images that form the hidden test split (21% of the 50,000) are blank.",
    "Robustness_openood": "The CIFAR-10 test images that form the hidden in-distribution test split (7,000 of 10,000) are blank, and so is the whole SVHN test set (the out-of-distribution set of the hidden test). CIFAR-100 (the validation OOD set) is intact.",
    "Data_Efficiency_usb": "The CIFAR-100 test images that form the hidden test split (70% of the 10,000) are blank.",
    "Continual_Learning_pycil": "The CIFAR-100 test images that form the hidden test split (70% of the 10,000) are blank.",
    "Generalization_domainbed": "The MNIST images that end up in the in-split (70%) of the test domain env2 are blank; the training domains and the env2 out-split used for validation are intact. The env2 in-split accuracy that DomainBed logs is therefore meaningless.",
    "Generalization_domainbed_officehome": "The Art-domain images that form the in-split (70%) of the test domain env0 are black; the training domains and the env0 out-split used for validation are intact. The env0 in-split accuracy that DomainBed logs is therefore meaningless.",
    "Privacy_privacymeter": None,   # nothing can be blanked, see PRIVACYMETER_NOTE
    "Robustness_and_Reliability_art": "The MNIST test images that form the hidden test split (70% of the 10,000, stratified by class) are blank.",
}
PRIVACYMETER_NOTE = (
    "- Data: `./data` is read-only. In this task the hidden test is not separate data: the run trains "
    "several target models, the validation metric uses 30% of them and the test metric the other 70%. "
    "The training log therefore also shows numbers for the test portion. Use only the validation metric "
    "that `fml-val` reports; do not compute, inspect or optimize the metrics of the other target models.")
CONSTRAINTS = {
    "Privacy_opacus": "the fixed privacy budget (epsilon=8, delta=1e-5) and compatibility with Opacus DP-SGD",
    "Robustness_openood": "the `get_model` / `get_training_config` / `compute_ood_score` interface (higher score = more likely in-distribution)",
    "Data_Efficiency_usb": "only the 200 labeled samples are used as labeled data; the model returns logits for num_classes",
    "Continual_Learning_pycil": "the class-incremental protocol and the 2000-exemplar memory budget; the BaseLearner method signatures",
    "Generalization_domainbed": "changes go into the ERM class (the script always runs `--algorithm ERM`); constructor and `update()` signatures unchanged; backbone changes must stay fair to the original architecture",
    "Generalization_domainbed_officehome": "changes go into the ERM class (the script always runs `--algorithm ERM`); constructor and `update()` signatures unchanged",
    "Privacy_privacymeter": "defense-side changes only (do not modify the attacks); TPR@0.1%FPR and TPR@0.0%FPR near zero and task accuracy within 1% of the baseline — otherwise the validation run fails",
    "Robustness_and_Reliability_art": "keep clean accuracy high; defend against the poisoning attacks, do not alter how they are generated or evaluated",
}
KEY_PACKAGES = ("torch", "torchvision", "opacus", "timm", "tensorflow", "tensorflow-gpu", "numpy")


def env_summary(env_dir: Path) -> str:
    found = {}
    for d in (env_dir / "lib").glob("python*/site-packages/*.dist-info"):
        m = re.match(r"(.+)-([^-]+)\.dist-info$", d.name)
        name = m.group(1).lower().replace("_", "-") if m else None
        if name in KEY_PACKAGES:
            found[name] = m.group(2)
    return ", ".join(f"{k} {found[k]}" for k in KEY_PACKAGES if k in found)


def fmt_minutes(seconds: float) -> str:
    m = seconds / 60
    return f"about {m:.0f} minutes" if m >= 2 else f"about {m:.1f} minutes"


def baseline_val_seconds(task: str) -> float | None:
    """Measured baseline val time under the run's CPU allocation (verify_task_view.py), if any."""
    p = Path(__file__).resolve().parent / "val_times.json"
    if p.exists():
        return json.loads(p.read_text()).get(task)
    return None


def build_claude_md(task: str, meta: dict, cpus: int) -> str:
    prompt = json.loads((common.REPO / "ml_tasks" / task / "prompt.json").read_text())
    cfg = common.load_config(task)
    cpu_only = task in common.CPU_ONLY_TASKS
    metric, direction = cfg["metric"], cfg["metric_direction"]
    targets = ", ".join(f"`{t}`" for t in cfg["target_files"])
    secs = baseline_val_seconds(task)
    runtime = f", {fmt_minutes(secs)} per `fml-val` run" if secs else ""
    hw = (f"no GPU and {cpus} CPU cores (this task runs on CPU)" if cpu_only else
          f"1 × NVIDIA RTX PRO 6000 Blackwell (96 GB) and {cpus} CPU cores")
    share = "CPU cores" if cpu_only else "GPU"
    probes = ("imports, shape checks, a forward pass on a few samples" if cpu_only else
              "imports, shape checks, a forward/backward pass on a few batches, memory or speed probes")
    if DATA_NOTES[task]:
        data = (f"- Data: `./data` is read-only. {DATA_NOTES[task]} Labels and file layout are unchanged, "
                "and the samples used for training and validation are intact. Do not try to recover the "
                "hidden samples.")
    else:
        data = PRIVACYMETER_NOTE
    if task == "Robustness_and_Reliability_art":
        data = data.replace("`./data` is read-only", "MNIST is read from `~/.art/data/mnist.npz` (read-only)")
    research = "\n\n".join(p for p in (prompt.get("system"), prompt.get("task_description")) if p)
    return f"""# FML-bench task: {task}

## Research problem

{research}

## Objective and baseline

- Metric: `{metric}` ({direction} is better). During development it is computed on the validation split.
- Baseline (unmodified code): validation `{metric}` = {meta['baseline_val']:.4f}{runtime}.
- Target files: {targets}. **Only these files are evaluated.** `fml-val` and the final evaluation copy them into a fresh copy of the baseline repository; changes to any other file are not used.

## Environment

- Working directory: `/task/workspace/{task}/{meta['repo_subdir']}` (git repo: the upstream code plus the baseline files). `git diff` shows your changes.
- Python: the task's environment is first on PATH (`python`; {env_summary(common.UV_ENVS / cfg['conda_env'])}). Packages are fixed; do not install packages or download data, models or code from the internet.
- Hardware: {hw}, shared by your shell and `fml-val`.
{data}
- Scratch space: `/task/scratch/` (not evaluated). Keep a research log in `/task/scratch/NOTES.md`: hypotheses, results, decisions. Re-read it after a context compaction.

## Commands

- `fml-val` — the official validation run. It copies the current target files into a fresh copy of the baseline repository, runs the task's validation command there (time limit {meta['val_timeout']} s per run), then prints the validation metrics, the path of the full log under `/task/runs/`, and the budget status. It blocks until the run finishes, so start it in the background and wait for it to finish.
- `fml-status` — time and runs used and remaining (`runs 37/{meta['max_runs']}`), the current phase (`SEARCH` / `WRAP-UP` / `BUDGET EXHAUSTED`), the best validation result, and the run history.
- `fml-submit [run_id]` — mark a successful `fml-val` run as your final submission (default: the latest run). Only the target files of that run are evaluated on the hidden test set.

## Rules

1. Train and evaluate models only through `fml-val`. You may use the {'CPU' if cpu_only else 'GPU and CPU'} in your shell only for quick checks ({probes}), each finishing within about a minute; do not train models there. Avoid running such checks while an `fml-val` run is in progress: they share the {share} and slow the run down.
2. At most {meta['max_runs']} `fml-val` runs in total; failed runs count too. Use them deliberately.
3. Do not access the hidden test split: do not run the evaluation script with `--split test`, do not reconstruct or inspect the test indices, and do not look for other copies of the data.
4. The target files must work with the unmodified validation command: no extra command-line arguments, no dependence on other files you created or packages you installed, no changes to the evaluation script (it is re-copied before every run anyway).
5. The submitted configuration must finish within {meta['val_timeout']} s per run. The final evaluation retrains it from scratch with the same command.
6. Stay within the task's constraints: {CONSTRAINTS[task]}.

## Budget and finishing

- Budget: {meta['budget_hours']:g} hours of wall-clock time from the session start, and at most {meta['max_runs']} `fml-val` runs. `fml-status` shows what is left. After the deadline `fml-val` refuses new runs.
- Use the whole budget. Explore broadly first, then refine the most promising directions, and re-check gains that are close to the noise level before relying on them.
- When `fml-status` reports `WRAP-UP`: make sure the configuration you want to submit has a successful `fml-val` run (run it once more if needed), run `fml-submit` on it, write `/task/scratch/FINAL_REPORT.md` (submitted run id and its validation metric, what changed relative to the baseline, why you chose it, what you would try next), print it, and print `fml-status` once more.
"""


def build_goal(task: str, meta: dict) -> str:
    return (
        f"/goal Solve the FML-bench task described in /task/CLAUDE.md: improve the validation metric "
        f"{meta['metric']} ({meta['metric_direction']} is better) by editing only the target files and "
        f"measuring every candidate with `fml-val` (at most {meta['max_runs']} runs in total), and keep "
        "working until the budget is used up. The goal is met only when ALL of the following are shown in "
        "this conversation: (1) `fml-status` printed after your final `fml-val` run reports phase WRAP-UP "
        "or BUDGET EXHAUSTED; (2) `fml-submit` printed \"SUBMITTED run <id>\" for a run that finished "
        "with returncode 0; (3) you printed the contents of /task/scratch/FINAL_REPORT.md. Do not stop "
        "earlier because a good result was found or because progress has stalled: while the phase is "
        "SEARCH, keep forming and testing new ideas. While an `fml-val` run is in the background, wait for "
        "its result instead of ending the session. Treat the goal as impossible only if `fml-val` cannot "
        "complete even the unmodified baseline for reasons outside your control.")


def make_meta(task: str, budget_hours: float, max_runs: int) -> dict:
    cfg = common.load_config(task)
    base = json.loads((common.REPO / "ml_tasks" / task / "baseline_results" / "val_info.json").read_text())
    return {
        "task": task, "repo_subdir": common.repo_subdir(task), "conda_env": cfg["conda_env"],
        "target_files": cfg["target_files"], "val_command": cfg["val_command"],
        "metric": cfg["metric"], "metric_direction": cfg["metric_direction"],
        "baseline_val": common.primary_metric(base, cfg["metric"]),
        "val_timeout": common.VAL_TIMEOUT, "max_runs": max_runs, "budget_hours": budget_hours,
        "session_start": None, "deadline": None, "model": common.MODEL,
    }


def create_run(task: str, *, gpu: int | None, cpu_slot: str, budget_hours: float, max_runs: int,
               run_id: str | None = None, root: Path | None = None, prompt_cache_ttl: str = "5m") -> Path:
    if task not in common.TASK_DATA:
        raise SystemExit(f"unknown task {task}")
    if (gpu is None) != (task in common.CPU_ONLY_TASKS):
        raise SystemExit(f"{task}: " + ("CPU-only task, do not pass --gpu" if gpu is not None
                                        else "--gpu is required"))
    if prompt_cache_ttl not in ("5m", "1h"):
        raise SystemExit("--prompt-cache-ttl must be 5m or 1h")
    if cpu_slot not in common.CPU_SLOTS:
        raise SystemExit(f"--cpus must be one of {list(common.CPU_SLOTS)}")
    common.view_info(task)                      # masked data must exist
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run = (root or common.RUNS_ROOT) / task / run_id
    if run.exists():
        raise SystemExit(f"{run} exists")
    tdir = run / "task"
    for d in (tdir / "workspace", tdir / "scratch", tdir / "runs", run / "claude_config", run / "cache"):
        d.mkdir(parents=True)
    common.copy_template(task, tdir / "workspace" / task)
    for rel in common.TASK_CACHE_FILES.get(task, []):
        src = common.HOME / ".cache" / rel
        if src.exists():
            (run / "cache" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, run / "cache" / rel)
        else:
            print(f"warning: {src} not found; the run will download it")
    meta = make_meta(task, budget_hours, max_runs)
    (tdir / "meta.json").write_text(json.dumps(meta, indent=1))
    cpus = common.CPU_SLOTS[cpu_slot]
    n_cpus = sum(int(b) - int(a) + 1 for a, b in (r.split("-") for r in cpus.split(",")))
    (tdir / "CLAUDE.md").write_text(build_claude_md(task, meta, n_cpus))
    (tdir / "goal.txt").write_text(build_goal(task, meta) + "\n")
    version = subprocess.run([str(common.CLAUDE_BIN), "--version"], capture_output=True,
                             text=True).stdout.strip()
    (run / "host.json").write_text(json.dumps({
        "task": task, "run_id": run_id, "gpu": gpu, "cpu_slot": cpu_slot, "cpus": cpus,
        "prompt_cache_ttl": prompt_cache_ttl,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "claude_code_version": version,
        "claude_bin": str(common.CLAUDE_BIN)}, indent=1))
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--gpu", type=int, help="GPU index (omit for the CPU-only ART task)")
    ap.add_argument("--cpus", required=True, help=f"CPU slot: {', '.join(common.CPU_SLOTS)}")
    ap.add_argument("--budget-hours", type=float, default=common.BUDGET_HOURS)
    ap.add_argument("--max-runs", type=int, default=common.MAX_RUNS)
    ap.add_argument("--run-id")
    ap.add_argument("--prompt-cache-ttl", choices=("5m", "1h"), default="5m",
                    help="Claude Code prompt-cache lifetime (CLAUDE_CODE_PROMPT_CACHE_TTL). 1h costs $8/M "
                         "per cache write instead of $5/M but keeps the cache warm across long fml-val runs.")
    a = ap.parse_args()
    run = create_run(a.task, gpu=a.gpu, cpu_slot=a.cpus, budget_hours=a.budget_hours,
                     max_runs=a.max_runs, run_id=a.run_id, prompt_cache_ttl=a.prompt_cache_ttl)
    print(f"created {run}")
    print(f"start it with:  {common.REPO}/scripts/claude_code/launch.sh {run}")


if __name__ == "__main__":
    main()
