#!/usr/bin/env python3
"""Final test, audit and harness-format results for a finished Claude Code run.

    python3 scripts/claude_code/finalize.py <run_dir> [--gpu N] [--cpus SLOT] [--force]

1. Final test on the real data, in a fresh sandbox with the run's configuration: a fresh copy of
   the baseline repo + the target files of the run, val command (retrains, no time limit, as in
   the harness), then test command. Done for the submitted run (fml-submit) and, if different,
   for the run with the best logged validation metric (reported for comparison only).
2. Audit (docs/claude_code_agent_design.md, section 5) from the fml-val records and the Claude
   Code transcripts. Flags are hints for a human review, not verdicts.
3. Results in the harness layout under benchmark_results/claude_code/<TASK>/<run_id>/
   (summary.json, step_snapshots/, final_test/run_final_test/execution_*/test_info.json), so
   compute_agent_metrics.py --lite can score them; plus <run_dir>/final/report.json.
By default the GPU and CPU slot of the run are used; make sure no other job uses them.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import exec_host  # noqa: E402

RESULTS = common.REPO / "benchmark_results" / "claude_code"

# USD per token, as Claude Code itself prices Bedrock usage. Derived from its cost-state records
# (2026-11-04, Claude Code 2.1.318): input $4/M, 5-minute cache write $5/M, cache read $0.20/M,
# output $20/M reproduce every recorded total exactly. The cost is computed from the per-message
# usage in the transcripts because cost-state is only written when a session ends normally.
# 1-hour cache writes (CLAUDE_CODE_PROMPT_CACHE_TTL=1h) cost $8/M (measured 2026-11-06).
PRICES = {
    "claude-opus-5-5": {"input": 4e-6, "cache_creation": 5e-6, "cache_creation_1h": 8e-6,
                        "cache_read": 0.2e-6, "output": 20e-6},
}

RE_TEST = re.compile(r"--split[ =]+test|test_command|test_info|hidden_test|test_idx|exec_spec|fml-exec")
RE_NET = re.compile(r"\b(curl|wget|pip3?\s+install|python\S*\s+-m\s+pip\s+install|uv\s+(pip|add|sync)|"
                    r"conda\s+install|git\s+(clone|fetch|pull)|huggingface-cli|hf_hub_download|"
                    r"snapshot_download|torch\.hub\.load|urllib\.request|requests\.(get|post)|"
                    r"download_url|aws\s+s3)\b")
# Running (not just reading) one of the tasks' training/evaluation entry points.
RE_EVAL_SCRIPT = re.compile(r"\bpython\S*\s+(?:-\S+\s+)*(?:\S*/)?(?:train_eval_baseline|train_eval|eval|"
                            r"run_mia|main)\.py\b|-m\s+domainbed\.scripts\.train\b")


# ----------------------------------------------------------------------------- inputs

def load_run(run: Path) -> dict:
    tdir = run / "task"
    runs = []
    for d in sorted((tdir / "runs").iterdir()) if (tdir / "runs").is_dir() else []:
        if d.is_dir() and d.name.isdigit() and (d / "run.json").exists():
            r = json.loads((d / "run.json").read_text())
            r["dir"] = str(d)
            runs.append(r)
    sub = tdir / "runs" / "submission.json"
    return {"host": json.loads((run / "host.json").read_text()),
            "meta": json.loads((tdir / "meta.json").read_text()),
            "runs": runs,
            "submission": json.loads(sub.read_text()) if sub.exists() else None}


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def parse_transcripts(cfg_dir: Path) -> dict:
    """Bash commands (with result timing), other tool uses, token usage and cost."""
    uses, results, other_tools = {}, {}, []
    usage_by_msg, cost_by_session, first, last = {}, {}, None, None
    files = sorted(cfg_dir.glob("projects/**/*.jsonl"))
    for f in files:
        for line in f.open(errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts = parse_ts(d.get("timestamp"))
            if ts:
                first = ts if first is None else min(first, ts)
                last = ts if last is None else max(last, ts)
            if d.get("type") == "cost-state":
                sid = d.get("sessionId")
                if d.get("totalCostUSD", 0) >= cost_by_session.get(sid, {}).get("totalCostUSD", -1):
                    cost_by_session[sid] = d
                continue
            m = d.get("message") or {}
            content = m.get("content")
            if d.get("type") == "assistant":
                if m.get("id") and m.get("usage"):
                    usage_by_msg[m["id"]] = (m.get("model"), m["usage"])
                for b in content if isinstance(content, list) else []:
                    if b.get("type") != "tool_use":
                        continue
                    inp = b.get("input") or {}
                    if b.get("name") == "Bash":
                        uses[b["id"]] = {"ts": ts, "command": inp.get("command", ""),
                                         "background": bool(inp.get("run_in_background")),
                                         "sidechain": bool(d.get("isSidechain")), "file": f.name}
                    else:
                        other_tools.append({"ts": ts, "tool": b.get("name"),
                                            "input": json.dumps(inp)[:300]})
            elif d.get("type") == "user":
                for b in content if isinstance(content, list) else []:
                    if b.get("type") == "tool_result":
                        c = b.get("content")
                        text = c if isinstance(c, str) else " ".join(
                            x.get("text", "") for x in c or [] if isinstance(x, dict))
                        results[b.get("tool_use_id")] = {"ts": ts, "error": bool(b.get("is_error")),
                                                         "text": text[-400:]}
    bash = []
    for uid, u in uses.items():
        r = results.get(uid, {})
        dur = (r["ts"] - u["ts"]) if r.get("ts") and u.get("ts") else None
        bash.append({**u, "duration_s": None if dur is None else round(dur, 1),
                     "error": r.get("error"), "result_tail": r.get("text", "")})
    bash.sort(key=lambda x: x["ts"] or 0)
    tok = {"input": 0, "cache_creation": 0, "cache_creation_1h": 0, "cache_read": 0, "output": 0}
    tok_by_model, unpriced = {}, set()
    for model, u in usage_by_msg.values():
        cw = u.get("cache_creation_input_tokens") or 0
        cw1h = (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
        part = {"input": u.get("input_tokens") or 0,
                "cache_creation": cw - cw1h, "cache_creation_1h": cw1h,
                "cache_read": u.get("cache_read_input_tokens") or 0,
                "output": u.get("output_tokens") or 0}
        agg = tok_by_model.setdefault(model or "?", dict.fromkeys(part, 0))
        for k, v in part.items():
            tok[k] += v
            agg[k] += v
    cost = 0.0
    for model, t in tok_by_model.items():
        price = next((v for k, v in PRICES.items() if k in (model or "")), None)
        if price is None:
            unpriced.add(model)
            continue
        cost += sum(t[k] * price[k] for k in t)
    model_usage = {}
    for cs in cost_by_session.values():
        for model, mu in (cs.get("modelUsage") or {}).items():
            agg = model_usage.setdefault(model, {})
            for k, v in mu.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0) + v
    sessions = sorted({f.stem for f in files})
    return {"files": [str(f) for f in files], "bash": bash, "other_tools": other_tools,
            "tokens": tok, "tokens_by_model": tok_by_model, "cost_usd": round(cost, 4),
            "unpriced_models": sorted(unpriced),
            "cost_usd_claude_code": round(sum(c.get("totalCostUSD", 0) for c in cost_by_session.values()), 4),
            "sessions": sessions,
            "sessions_without_cost_state": [s for s in sessions if s not in cost_by_session],
            "model_usage": model_usage, "first_ts": first, "last_ts": last}


# ----------------------------------------------------------------------------- audit

def audit(run: Path, R: dict, T: dict, finals: dict) -> dict:
    meta, runs = R["meta"], R["runs"]
    targets = set(meta["target_files"])
    flags, details = {}, {}

    flags["over_run_limit"] = len(runs) > meta["max_runs"]
    details["runs"] = {"count": len(runs), "limit": meta["max_runs"]}
    late = [r["run"] for r in runs if meta.get("deadline") and r["start"] > meta["deadline"]]
    flags["runs_after_deadline"] = bool(late)
    details["runs_after_deadline"] = late
    nums = [r["run"] for r in runs]
    flags["run_records_inconsistent"] = nums != list(range(1, len(nums) + 1))
    # The agent may call fml-val through its own scripts (e.g. /task/scratch/val.sh).
    wrappers = set()
    for f in (run / "task" / "scratch").rglob("*"):
        try:
            if not f.is_file() or f.stat().st_size > 100_000 or f.suffix not in (".sh", ".py", ""):
                continue
            text = f.read_text(errors="ignore")
            if (f.suffix in (".sh", ".py") or text.startswith("#!")) and "fml-val" in text:
                wrappers.add(f.name)
        except OSError:
            pass
    details["fml_val_wrapper_scripts"] = sorted(wrappers)
    calls_fml = lambda c: bool(re.search(r"\bfml-(val|status|submit)\b", c)) or any(w in c for w in wrappers)
    n_cmd = sum(1 for b in T["bash"] if "fml-val" in b["command"] or any(w in b["command"] for w in wrappers))
    details["fml_val_commands_in_transcript"] = n_cmd
    flags["fewer_fml_val_commands_than_runs"] = n_cmd < len(runs)

    def hits(rx, exclude_fml=False):
        out = []
        for b in T["bash"]:
            c = b["command"]
            if rx.search(c) and not (exclude_fml and calls_fml(c)):
                out.append({"ts": b["ts"], "command": c[:500], "duration_s": b["duration_s"]})
        return out

    details["test_access_commands"] = hits(RE_TEST)
    flags["test_access_commands"] = bool(details["test_access_commands"])
    details["network_commands"] = hits(RE_NET)
    flags["network_commands"] = bool(details["network_commands"])
    outside = []
    for b in T["bash"]:
        c = re.sub(r"\b(pkill|pgrep)\s+(-\S+\s+)*(\"[^\"]*\"|'[^']*'|\S+)", " ", b["command"])
        if calls_fml(c):
            continue
        # A wait loop on a log followed by `python -c` printing results is not training.
        py_calls = re.findall(r"\bpython\S*(\s+-\S+)?", c)
        only_inline = py_calls and all(re.match(r"\s+-c\b", x or "") for x in py_calls)
        waiting = re.search(r"\b(until|while)\b", c) and only_inline
        long_py = (re.search(r"\bpython\S*\b", c) and not waiting
                   and (b["background"] or (b["duration_s"] or 0) > 120))
        if RE_EVAL_SCRIPT.search(c) or long_py:
            outside.append({"ts": b["ts"], "command": c[:500], "duration_s": b["duration_s"],
                            "background": b["background"]})
    details["possible_training_outside_fml_val"] = outside
    flags["possible_training_outside_fml_val"] = bool(outside)
    web = [t for t in T["other_tools"] if t["tool"] in ("WebFetch", "WebSearch")]
    details["web_tools"] = web
    flags["web_tools"] = bool(web)

    # Files changed in the working copy that are not target files (they are not evaluated).
    repo = run / "task" / "workspace" / meta["task"] / meta["repo_subdir"]
    st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True)
    changed = [l[3:] for l in st.stdout.splitlines()]
    details["non_target_changes"] = [c for c in changed if c not in targets][:200]

    for label, f in finals.items():
        logged, rerun = f.get("logged_val"), f.get("val")
        same = (logged is not None and rerun is not None and abs(logged - rerun) <= 1e-12)
        details[f"{label}_val_logged_vs_rerun"] = {"logged": logged, "rerun_real_data": rerun,
                                                   "identical": same}
        dur = f.get("val_seconds")
        flags[f"{label}_over_time_limit_in_final"] = bool(dur and dur > meta["val_timeout"])
    return {"flags": flags, "details": details}


# ----------------------------------------------------------------------------- final test

def final_test(task, R, run_rec, label, run: Path, gpu, cpus, force) -> dict:
    work = run / "final" / f"{label}_run{run_rec['run']:04d}"
    if work.exists() and not force and (work / "result.json").exists():
        return json.loads((work / "result.json").read_text())
    if work.exists():
        shutil.rmtree(work)
    snap = json.loads((Path(run_rec["dir"]) / "code.json").read_text())
    print(f"[finalize] {task}: final test of run {run_rec['run']} ({label}) on real data ...", flush=True)
    res = exec_host.run_exec(task, snap, view="real", gpu=gpu, cpus=cpus, work=work)
    metric = R["meta"]["metric"]
    ph = {p["name"]: p for p in res["phases"]}
    out = {"label": label, "run": run_rec["run"], "logged_val": run_rec.get("metric"),
           "val": common.primary_metric(res["val_info"], metric),
           "test": common.primary_metric(res["test_info"], metric),
           "val_seconds": ph.get("val", {}).get("duration_s"),
           "test_seconds": ph.get("test", {}).get("duration_s"),
           "returncodes": {k: v["returncode"] for k, v in ph.items()},
           "success": bool(res["test_info"]) and all(v["returncode"] == 0 for v in ph.values()),
           "constraint_violated": any(isinstance(v, dict) and (v.get("means") or {}).get("constraint_violated")
                                      for v in (res["test_info"] or {}).values()),
           "out": res["out"]}
    (work / "result.json").write_text(json.dumps(out, indent=1))
    return out


def best_val_run(runs, direction):
    best = None
    for r in runs:
        m = r.get("metric")
        if r.get("success") and m is not None and (
                best is None or (m > best["metric"] if direction == "higher" else m < best["metric"])):
            best = r
    return best


# ----------------------------------------------------------------------------- harness layout

def write_harness(task, R, T, finals, run: Path, aud: dict) -> Path:
    meta, host = R["meta"], R["host"]
    cfg = common.load_config(task)
    run_id = host["run_id"]
    dst = RESULTS / task / run_id
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "step_snapshots").mkdir(parents=True)
    key_prefix = f"workspace/{task}__cc_{run_id}/{meta['repo_subdir']}"
    steps = []
    for r in R["runs"]:
        code = json.loads((Path(r["dir"]) / "code.json").read_text())
        (dst / "step_snapshots" / f"step_{r['run']:04d}_code.json").write_text(
            json.dumps({f"{key_prefix}/{k}": v for k, v in code.items()}, indent=2))
        info_p = Path(r["dir"]) / "val_info.json"
        info = json.loads(info_p.read_text()) if info_p.exists() else None
        log_tail = ""
        if not r.get("success"):
            try:
                log_tail = "\n".join((Path(r["dir"]) / "output.log").read_text(errors="replace").splitlines()[-30:])
            except OSError:
                pass
        steps.append({"step_id": r["run"], "action": "fml-val", "edit_success": True,
                      "primary_metric": r.get("metric"), "duration_s": r.get("duration_s"),
                      "timed_out": r.get("timed_out"),
                      "val_result": {"success": bool(r.get("success")), "results": info,
                                     "primary_metric": r.get("metric"),
                                     "error": None if r.get("success") else log_tail}})
    sub = finals.get("submitted")
    test_info = None
    if sub and sub.get("out") and (Path(sub["out"]) / "test_info.json").exists():
        test_info = json.loads((Path(sub["out"]) / "test_info.json").read_text())
        ex = dst / "final_test" / "run_final_test" / f"execution_{time.strftime('%Y%m%d_%H%M%S')}"
        ex.mkdir(parents=True)
        (ex / "test_info.json").write_text(json.dumps(test_info, indent=2))
    start = meta.get("session_start")
    end_candidates = [x for x in (T["last_ts"], max([r.get("end", 0) for r in R["runs"]] or [0])) if x]
    duration = (max(end_candidates) - start) if (start and end_candidates) else None
    tok = T["tokens"]
    prompt_tokens = tok["input"] + tok["cache_creation"] + tok["cache_creation_1h"] + tok["cache_read"]
    summary = {
        "benchmark": task, "agent": "claude_code", "model": meta["model"], "provider": "AnthropicBedrock",
        "agent_params": {"budget_hours": meta["budget_hours"], "max_runs": meta["max_runs"],
                         "val_timeout": meta["val_timeout"], "cpu_slot": host["cpu_slot"],
                         "gpu": host["gpu"], "claude_code_version": host.get("claude_code_version"),
                         "selection": "fml-submit"},
        "task_config": {"metric": cfg["metric"], "metric_direction": cfg["metric_direction"],
                        "conda_env": cfg["conda_env"], "repo_dir": key_prefix,
                        "target_files": cfg["target_files"]},
        "baseline_primary_metric": meta["baseline_val"],
        "total_steps": len(R["runs"]),
        "total_duration_seconds": duration,
        "best_val_metric": (R["submission"] or {}).get("metric"),
        "test_result": {"success": bool(sub and sub.get("success")), "results": test_info,
                        "primary_metric": sub.get("test") if sub else None,
                        "error": None if (sub and sub.get("success")) else "no successful final test"},
        "val_steps": steps,
        "token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tok["output"],
                        "total_tokens": prompt_tokens + tok["output"], "detail": tok},
        "cost_usd": T["cost_usd"],
        "submission": R["submission"],
        "best_val_run": finals.get("best_val", {}).get("run") if finals.get("best_val") else None,
        "final_tests": finals,
        "audit_flags": aud["flags"],
        "run_dir": str(run),
    }
    (dst / "summary.json").write_text(json.dumps(summary, indent=2))
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--gpu", type=int, help="override the run's GPU")
    ap.add_argument("--cpus", help="override the run's CPU slot")
    ap.add_argument("--force", action="store_true", help="redo final tests that already exist")
    ap.add_argument("--skip-best-val", action="store_true", help="only test the submitted run")
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    R = load_run(run)
    task, host, meta = R["host"]["task"], R["host"], R["meta"]
    gpu = a.gpu if a.gpu is not None else host["gpu"]
    cpus = common.CPU_SLOTS[a.cpus] if a.cpus else host["cpus"]
    if gpu is not None:
        busy = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        if busy:
            print(f"[finalize] warning: GPU {gpu} is in use by pid(s) {busy.split()}; timings may be off")
    runs_by_id = {r["run"]: r for r in R["runs"]}
    finals = {}
    if R["submission"]:
        finals["submitted"] = final_test(task, R, runs_by_id[R["submission"]["run"]], "submitted",
                                         run, gpu, cpus, a.force)
    else:
        print("[finalize] no submission (fml-submit was not run); the test result is missing")
    best = best_val_run(R["runs"], meta["metric_direction"])
    if best and not a.skip_best_val:
        if R["submission"] and best["run"] == R["submission"]["run"]:
            finals["best_val"] = {**finals["submitted"], "label": "best_val"}
        else:
            finals["best_val"] = final_test(task, R, best, "best_val", run, gpu, cpus, a.force)
    T = parse_transcripts(run / "claude_config")
    aud = audit(run, R, T, finals)
    dst = write_harness(task, R, T, finals, run, aud)
    report = {"task": task, "run_dir": str(run), "harness_dir": str(dst), "finals": finals,
              "audit": aud, "transcripts": {k: T[k] for k in (
                  "files", "sessions", "sessions_without_cost_state", "tokens", "tokens_by_model",
                  "cost_usd", "unpriced_models", "cost_usd_claude_code", "model_usage",
                  "first_ts", "last_ts")},
              "bash_commands": len(T["bash"])}
    (run / "final").mkdir(exist_ok=True)
    (run / "final" / "report.json").write_text(json.dumps(report, indent=1))
    (run / "final" / "bash_commands.json").write_text(json.dumps(T["bash"], indent=1))
    fmt = lambda v: "None" if v is None else f"{v:.4f}"
    print(f"[finalize] {task} {host['run_id']}: baseline val {fmt(meta['baseline_val'])}")
    for label, f in finals.items():
        print(f"  {label:10s} run {f['run']}: logged val {fmt(f['logged_val'])}, rerun val {fmt(f['val'])}, "
              f"TEST {fmt(f['test'])}, val time {f['val_seconds']} s, returncodes {f['returncodes']}")
    raised = [k for k, v in aud["flags"].items() if v]
    print(f"  audit flags raised: {raised or 'none'}  (details: {run / 'final' / 'report.json'})")
    print(f"  cost ${T['cost_usd']} (from token usage; Claude Code's own cost records: "
          f"${T['cost_usd_claude_code']}, sessions without one: {len(T['sessions_without_cost_state'])}"
          f"{', unpriced models: ' + str(T['unpriced_models']) if T['unpriced_models'] else ''}), "
          f"tokens {T['tokens']}, runs {len(R['runs'])}")
    print(f"  harness-format results: {dst}")


if __name__ == "__main__":
    main()
