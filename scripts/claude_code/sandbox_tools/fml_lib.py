"""Core of fml-val / fml-status / fml-submit (runs inside the task sandbox, stdlib only).

Layout inside the sandbox (see scripts/claude_code/common.py::bwrap_args):
  /task/meta.json                  run parameters (read-only)
  /task/baseline/<TASK>/           task template (read-only)
  /task/workspace/<TASK>/<repo>/   the agent's working copy
  /task/runs/NNNN/                 one directory per fml-val run
  /task/runs/submission.json       written by fml-submit
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

TASK_ROOT = Path(os.environ.get("FMLBENCH_TASK_ROOT", "/task"))
RUNS = TASK_ROOT / "runs"


def load_meta() -> dict:
    return json.loads((TASK_ROOT / "meta.json").read_text())


def fmt_dur(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def primary_metric(info, metric):
    """Same rule as benchmark/executor.py::_extract_primary_metric (no dataset filter)."""
    if not isinstance(info, dict):
        return None
    vals = [float(v["means"][metric]) for v in info.values()
            if isinstance(v, dict) and isinstance(v.get("means"), dict) and metric in v["means"]]
    if vals:
        return sum(vals) / len(vals)
    try:
        return float(info[metric])
    except (KeyError, TypeError, ValueError):
        return None


def list_runs() -> list[dict]:
    runs = []
    if RUNS.is_dir():
        for d in sorted(RUNS.iterdir()):
            f = d / "run.json"
            if d.is_dir() and d.name.isdigit() and f.exists():
                try:
                    runs.append(json.loads(f.read_text()))
                except ValueError:
                    pass
    return runs


def better(a, b, direction) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    return a > b if direction == "higher" else a < b


def best_run(runs, direction):
    best = None
    for r in runs:
        if r.get("success") and better(r.get("metric"), best and best.get("metric"), direction):
            best = r
    return best


def phase(meta, runs, now=None) -> tuple[str, float]:
    """(phase, seconds left). WRAP-UP leaves room for one more run plus the submission."""
    now = time.time() if now is None else now
    left = meta["deadline"] - now
    n = len(runs)
    if left <= 0 or n >= meta["max_runs"]:
        return "BUDGET EXHAUSTED", left
    last = max([r.get("duration_s") or 0 for r in runs[-3:]] or [0])
    window = max(min(3600, 0.1 * meta["budget_hours"] * 3600), 2 * last)
    if left < window or n >= meta["max_runs"] - 5:
        return "WRAP-UP", left
    return "SEARCH", left


def status_line(meta, runs) -> str:
    ph, left = phase(meta, runs)
    elapsed = time.time() - meta["session_start"]
    b = best_run(runs, meta["metric_direction"])
    best = f"best {meta['metric']} = {b['metric']:.4f} (run {b['run']})" if b else "best: none yet"
    return (f"[fml] runs {len(runs)}/{meta['max_runs']} | time {fmt_dur(elapsed)} used, "
            f"{fmt_dur(left)} left of {meta['budget_hours']:g}h | phase {ph} | {best} | "
            f"baseline {meta['baseline_val']:.4f} ({meta['metric_direction']} is better)")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot_targets(meta, src_repo: Path) -> dict:
    """{target file: content} of the agent's working copy (missing files are reported)."""
    snap, missing = {}, []
    for tf in meta["target_files"]:
        p = src_repo / tf
        if p.is_file():
            snap[tf] = p.read_text()
        else:
            missing.append(tf)
    return {"files": snap, "missing": missing}


def make_exec_copy(meta, dst_task_dir: Path, files: dict) -> Path:
    """Fresh copy of the task template with the given target files written into its repo."""
    shutil.copytree(TASK_ROOT / "baseline" / meta["task"], dst_task_dir, symlinks=True,
                    ignore=shutil.ignore_patterns(".git"))
    repo = dst_task_dir / meta["repo_subdir"]
    for tf, content in files.items():
        p = repo / tf
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return repo


def run_command(cmd: str, cwd: Path, log: Path, timeout) -> dict:
    """Run a shell command in its own process group; kill the group on timeout."""
    t0 = time.time()
    with open(log, "ab") as f:
        f.write(f"$ {cmd}\n".encode())
        f.flush()
        p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        timed_out = False
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    p.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    continue
            rc = p.returncode if p.returncode is not None else -9
            f.write(f"\n[fml] killed after {timeout} s time limit\n".encode())
    return {"returncode": rc, "timed_out": timed_out, "duration_s": round(time.time() - t0, 1)}


def tail(p: Path, n=40) -> str:
    try:
        return "\n".join(p.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# --------------------------------------------------------------------------- commands

def cmd_val() -> int:
    meta = load_meta()
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[fml] another fml-val run is in progress; waiting for it to finish ...", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        runs = list_runs()
        if time.time() >= meta["deadline"]:
            print("[fml] REFUSED: the wall-clock budget is used up; no new runs can be started.")
            print(status_line(meta, runs))
            return 3
        n = len(runs) + 1
        rid = f"{n:04d}"
        rdir = RUNS / rid
        rdir.mkdir()
        work_repo = TASK_ROOT / "workspace" / meta["task"] / meta["repo_subdir"]
        snap = snapshot_targets(meta, work_repo)
        (rdir / "code.json").write_text(json.dumps(snap["files"], indent=1))
        rec = {"run": n, "start": time.time(), "start_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
               "target_sha256": {k: hashlib.sha256(v.encode()).hexdigest()
                                 for k, v in snap["files"].items()},
               "missing_target_files": snap["missing"]}
        log = rdir / "output.log"
        exec_dir = TASK_ROOT / "workspace" / f"{meta['task']}__exec_{rid}"
        print(f"[fml] run {rid}: started {rec['start_iso']} (time limit {meta['val_timeout']} s); "
              f"log: {log}", flush=True)
        try:
            if snap["missing"]:
                res = {"returncode": None, "timed_out": False, "duration_s": 0.0}
                log.write_text(f"[fml] missing target files: {snap['missing']}\n")
            else:
                repo = make_exec_copy(meta, exec_dir, snap["files"])
                res = run_command(meta["val_command"], repo, log, meta["val_timeout"])
                info_p = repo / "results_tmp" / "val_info.json"
                if info_p.exists():
                    shutil.copy(info_p, rdir / "val_info.json")
        finally:
            shutil.rmtree(exec_dir, ignore_errors=True)
        info = None
        if (rdir / "val_info.json").exists():
            try:
                info = json.loads((rdir / "val_info.json").read_text())
            except ValueError:
                info = None
        rec.update(res)
        rec["end"] = time.time()
        rec["success"] = res["returncode"] == 0 and info is not None   # same rule as the harness
        rec["metric"] = primary_metric(info, meta["metric"]) if rec["success"] else None
        (rdir / "run.json").write_text(json.dumps(rec, indent=1))

    runs = list_runs()
    if rec["success"]:
        m = rec["metric"]
        delta = "" if m is None else f" ({m - meta['baseline_val']:+.4f} vs baseline)"
        print(f"[fml] run {rid}: OK in {fmt_dur(rec['duration_s'])}  {meta['metric']} = "
              f"{m if m is None else round(m, 6)}{delta}")
        print(f"[fml] full results: {rdir / 'val_info.json'}")
    else:
        why = ("missing target files" if snap["missing"] else
               f"time limit of {meta['val_timeout']} s exceeded" if rec["timed_out"] else
               f"returncode {rec['returncode']}" + ("" if rec["returncode"] else ", no val_info.json"))
        print(f"[fml] run {rid}: FAILED ({why}) after {fmt_dur(rec['duration_s'])}. Last lines of {log}:")
        print(tail(log))
    print(status_line(meta, runs))
    return 0 if rec["success"] else 1


def cmd_status() -> int:
    meta = load_meta()
    runs = list_runs()
    print(status_line(meta, runs))
    sub = RUNS / "submission.json"
    if sub.exists():
        print(f"[fml] submitted: run {json.loads(sub.read_text())['run']}")
    if runs:
        print(f"{'run':>4}  {'start':19}  {'duration':>8}  {'result':10}  {meta['metric']}")
        for r in runs:
            res = "ok" if r.get("success") else ("timeout" if r.get("timed_out") else "failed")
            m = "" if r.get("metric") is None else f"{r['metric']:.4f}"
            print(f"{r['run']:>4}  {r.get('start_iso', ''):19}  {fmt_dur(r.get('duration_s') or 0):>8}  "
                  f"{res:10}  {m}")
    return 0


def cmd_submit(argv) -> int:
    meta = load_meta()
    runs = {r["run"]: r for r in list_runs()}
    if not runs:
        print("[fml] no runs yet")
        return 1
    rid = int(argv[0]) if argv else max(runs)
    r = runs.get(rid)
    if r is None:
        print(f"[fml] run {rid} does not exist")
        return 1
    if not r.get("success"):
        print(f"[fml] run {rid} did not succeed; only successful runs can be submitted")
        return 1
    (RUNS / "submission.json").write_text(json.dumps(
        {"run": rid, "metric": r.get("metric"), "time": time.time(),
         "time_iso": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"SUBMITTED run {rid} ({meta['metric']} = {r.get('metric')})")
    print(status_line(meta, list(runs.values())))
    return 0


def cmd_exec() -> int:
    """Host-driven batch run (verification / final test); not used by the agent.

    /task/exec_spec.json (written by the host): {"snapshot": {target file: content},
    "phases": [{"name": "val", "command": "...", "timeout": null, "output": "results_tmp/val_info.json"}, ...],
    "out": "/task/out"}. Phases run in order in one fresh copy (test reuses val's checkpoints).
    """
    meta = load_meta()
    spec = json.loads((TASK_ROOT / "exec_spec.json").read_text())
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    exec_dir = TASK_ROOT / "workspace" / f"{meta['task']}__exec_final"
    shutil.rmtree(exec_dir, ignore_errors=True)
    results = {"phases": []}
    try:
        repo = make_exec_copy(meta, exec_dir, spec["snapshot"])
        for ph in spec["phases"]:
            log = out / f"{ph['name']}.log"
            res = run_command(ph["command"], repo, log, ph.get("timeout"))
            src = repo / ph["output"]
            res.update(name=ph["name"], output_found=src.exists())
            if src.exists():
                shutil.copy(src, out / Path(ph["output"]).name)
            results["phases"].append(res)
            print(f"[fml-exec] {ph['name']}: returncode {res['returncode']} in {res['duration_s']} s",
                  flush=True)
            if res["returncode"] != 0:
                break
    finally:
        shutil.rmtree(exec_dir, ignore_errors=True)
    (out / "exec.json").write_text(json.dumps(results, indent=1))
    return 0
