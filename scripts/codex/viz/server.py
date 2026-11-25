"""Web server of the Codex × ColoredMNIST visualization (docs/codex_viz_design.md).

    scripts/codex/viz/serve.sh            # listens on 127.0.0.1:8765; from the Mac:
    ssh -L 8765:localhost:8765 ec2-user@<host>   then open http://localhost:8765

One run directory (FMLBENCH_VIZ_RUN, default $FMLBENCH_CC_RUNS/codex/Generalization_domainbed/viz).
"New run" deletes it and creates a fresh one. The server only reads the run, except for the jobs it
starts (baseline, predictions: scripts/codex/viz/jobs.py) and that reset.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent.parent / "claude_code"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import jobs  # noqa: E402
import rollout_index  # noqa: E402

TASK = "Generalization_domainbed"
RUN = Path(os.environ.get("FMLBENCH_VIZ_RUN", common.RUNS_ROOT / "codex" / TASK / "viz")).resolve()
REF_BASELINE = common.REPO / "rebaseline_results" / f"{TASK}__l4x1"
RUN_DEFAULTS = {"gpu": 0, "cpu_slot": "L0", "budget_hours": 1.0, "max_runs": 20}
N_STEPS = 5000          # ColoredMNIST: DomainBed's N_STEPS (5001 steps, 0..5000)

app = FastAPI()
_lock = threading.Lock()
_jobs: dict[str, threading.Thread] = {}
_pred_failed: set[str] = set()
_index = {"obj": None, "run": None}


# ----------------------------------------------------------------------------- reading the run

def rj(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def index() -> rollout_index.RolloutIndex:
    if _index["run"] != str(RUN) or _index["obj"] is None:
        _index.update(obj=rollout_index.RolloutIndex(RUN / "codex_home"), run=str(RUN))
    return _index["obj"].refresh()


def run_dirs() -> list[tuple[str, Path]]:
    rdir = RUN / "task" / "runs"
    out = []
    if (rdir / "baseline").is_dir():
        out.append(("baseline", rdir / "baseline"))
    if rdir.is_dir():
        out += [(d.name, d) for d in sorted(rdir.iterdir()) if d.is_dir() and d.name.isdigit()]
    return out


def live_results(rid: str, d: Path, meta: dict) -> Path | None:
    """results.jsonl: kept copy after the run, the execution copy's while it trains."""
    if (d / "results.jsonl").exists():
        return d / "results.jsonl"
    name = "baseline" if rid == "baseline" else rid
    p = RUN / "task" / "workspace" / f"{meta['task']}__exec_{name}" / meta["repo_subdir"] / "results_tmp" / "results.jsonl"
    return p if p.exists() else None


def train_rows(p: Path | None) -> list[dict]:
    rows = []
    if p is None:
        return rows
    for line in p.read_text(errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows.append({k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return rows


def node_summary(rid: str, d: Path, meta: dict) -> dict:
    rec = rj(d / "run.json")
    card = rj(d / "card.json") or {}
    notes = rj(d / "note.json") or []
    n = {"id": rid, "num": 0 if rid == "baseline" else int(rid), "title": card.get("title"),
         "has_card": bool(card), "has_note": bool(notes)}
    if rec is None:
        rows = train_rows(live_results(rid, d, meta))
        step = rows[-1]["step"] if rows else 0
        start = (d / "code.json").stat().st_mtime if (d / "code.json").exists() else None
        n.update(status="running", progress=min(step / N_STEPS, 1.0), step=step, start=start, metric=None)
    else:
        status = "ok" if rec.get("success") else ("timeout" if rec.get("timed_out") else "failed")
        n.update(status=status, metric=rec.get("metric"), duration_s=rec.get("duration_s"),
                 start=rec.get("start"), end=rec.get("end"), progress=1.0)
    p = RUN / "viz" / "pred" / f"{rid}.json"
    n["pred"] = "ready" if p.exists() else ("failed" if rid in _pred_failed else
                                           ("pending" if n["status"] == "ok" else None))
    return n


def agent_state(meta: dict) -> dict:
    """Is the agent's sandbox running (launch.log: last start without an end)?"""
    log = RUN / "launch.log"
    running = False
    if log.exists():
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        running = bool(lines) and " start " in lines[-1]
    # A sandbox killed without a final log line: check the processes too.
    if running:
        running = any(str(RUN / "task") in _cmdline(pid) for pid in _pids())
    return {"running": running, "started": meta.get("session_start") is not None}


def _pids():
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            yield p.name


def _cmdline(pid) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def state() -> dict:
    if not (RUN / "task" / "meta.json").exists():
        return {"exists": False, "run_dir": str(RUN)}
    meta = rj(RUN / "task" / "meta.json")
    host = rj(RUN / "host.json")
    nodes = [node_summary(rid, d, meta) for rid, d in run_dirs()]
    # SOTA: strictly better than every earlier successful node (baseline first), like fml-status.
    better = (lambda a, b: a > b) if meta["metric_direction"] == "higher" else (lambda a, b: a < b)
    best = None
    for n in nodes:
        n["new_sota"] = False
        if n["status"] == "ok" and n["metric"] is not None and (best is None or better(n["metric"], best["metric"])):
            best = n
            n["new_sota"] = True
    sub = rj(RUN / "task" / "runs" / "submission.json")
    idx = index()
    goal = idx.goal_status()
    now = time.time()
    budget = None
    if meta.get("session_start"):
        agent_runs = [n for n in nodes if n["id"] != "baseline"]
        left = meta["deadline"] - now
        last = max([n.get("duration_s") or 0 for n in agent_runs[-3:]] or [0])
        window = max(min(3600, 0.1 * meta["budget_hours"] * 3600), 2 * last)
        reserve = max(1, round(0.05 * meta["max_runs"]))
        n_runs = len(agent_runs)
        phase = ("BUDGET EXHAUSTED" if left <= 0 or n_runs >= meta["max_runs"] else
                 "WRAP-UP" if left < window or n_runs >= meta["max_runs"] - reserve else "SEARCH")
        budget = {"elapsed_s": now - meta["session_start"], "left_s": left, "runs": n_runs, "phase": phase}
    bjob = rj(RUN / "viz" / "jobs" / "baseline.json")
    if bjob and bjob["state"] == "running" and "baseline" not in _jobs and \
            not Path(f"/proc/{bjob.get('pid', 0)}").exists():
        bjob["state"] = "interrupted"          # the process that ran it is gone (e.g. server restart)
    tok = idx.tokens()
    report = RUN / "task" / "scratch" / "FINAL_REPORT.md"
    return {
        "exists": True, "run_dir": str(RUN), "run_id": host["run_id"], "created": host.get("created"),
        "task": meta["task"], "metric": meta["metric"], "direction": meta["metric_direction"],
        "model": host.get("model"), "effort": host.get("reasoning_effort"), "gpu": host.get("gpu"),
        "max_runs": meta["max_runs"], "budget_hours": meta["budget_hours"],
        "baseline_val": meta.get("baseline_val"), "baseline_reference": meta.get("baseline_reference"),
        "nodes": nodes, "sota": best["id"] if best else None, "submission": sub,
        "baseline_job": bjob, "agent": agent_state(meta), "budget": budget,
        "goal": goal, "tokens": tok, "final_report": report.exists(),
        "launch_cmd": (f"tmux new-window -n codex \"bash -c '. {common.REPO}/scripts/codex/env_l4.sh; "
                       f"{common.REPO}/scripts/codex/launch.sh {RUN}; bash'\""),
        "goal_txt": (RUN / "task" / "goal.txt").read_text().strip() if (RUN / "task" / "goal.txt").exists() else None,
    }


# ----------------------------------------------------------------------------- background work

def predictor_loop():
    """Predictions for every finished node (one at a time; jobs.predict holds a file lock)."""
    while True:
        try:
            if (RUN / "viz" / "dataset.json").exists() and "baseline" not in _jobs:
                for rid, d in run_dirs():
                    rec = rj(d / "run.json")
                    if (rec and rec.get("success") and (d / "model.pkl").exists() and rid not in _pred_failed
                            and not (RUN / "viz" / "pred" / f"{rid}.json").exists()):
                        res = jobs.predict(RUN, rid)
                        if not res.get("ok"):
                            _pred_failed.add(rid)
                        break
        except Exception as e:  # keep the loop alive
            print(f"[viz] predictor: {e}", flush=True)
        time.sleep(3)


@app.on_event("startup")
def _startup():
    threading.Thread(target=predictor_loop, daemon=True).start()


def start_job(name: str, fn):
    with _lock:
        if name in _jobs and _jobs[name].is_alive():
            raise HTTPException(409, f"{name} is already running")

        def wrap():
            try:
                fn()
            finally:
                _jobs.pop(name, None)
        t = threading.Thread(target=wrap, daemon=True)
        _jobs[name] = t
        t.start()


# ----------------------------------------------------------------------------- API

@app.get("/api/state")
def api_state():
    return JSONResponse(state())


@app.get("/api/intro")
def api_intro():
    d = rj(RUN / "viz" / "intro.json")
    if d is None:
        raise HTTPException(404, "intro data not prepared")
    return d


@app.get("/api/samples")
def api_samples():
    d = rj(RUN / "viz" / "samples.json")
    if d is None:
        raise HTTPException(404, "samples not prepared")
    return d


@app.get("/api/pred/{rid}")
def api_pred(rid: str):
    d = rj(RUN / "viz" / "pred" / f"{rid}.json")
    if d is None:
        raise HTTPException(404, "no predictions yet")
    return d


def _rdir(rid: str) -> Path:
    if rid != "baseline" and not (rid.isdigit() and len(rid) == 4):
        raise HTTPException(400, "bad run id")
    d = RUN / "task" / "runs" / rid
    if not d.is_dir():
        raise HTTPException(404, "no such run")
    return d


def _code(rid: str) -> dict:
    return rj(_rdir(rid) / "code.json") or {}


@app.get("/api/node/{rid}")
def api_node(rid: str):
    d = _rdir(rid)
    meta = rj(RUN / "task" / "meta.json")
    s = node_summary(rid, d, meta)
    rec = rj(d / "run.json") or {}
    code = _code(rid)
    base = rj(RUN / "task" / "runs" / "baseline" / "code.json") or {}
    changed = {f: {"add": 0, "del": 0} for f in code}
    for f, txt in code.items():
        for l in difflib.unified_diff(base.get(f, "").splitlines(), txt.splitlines(), lineterm="", n=0):
            if l.startswith("+") and not l.startswith("+++"):
                changed[f]["add"] += 1
            elif l.startswith("-") and not l.startswith("---"):
                changed[f]["del"] += 1
    return {**s, "card": rj(d / "card.json"), "notes": rj(d / "note.json") or [], "record": rec,
            "files": list(code), "changes": changed}


@app.get("/api/node/{rid}/diff")
def api_diff(rid: str, against: str = "baseline", file: str | None = None):
    new = _code(rid)
    old = _code(against) if against else {}
    files = [file] if file else list(new)
    out = {}
    for f in files:
        a, b = old.get(f, ""), new.get(f, "")
        out[f] = {"diff": list(difflib.unified_diff(a.splitlines(), b.splitlines(), f"{against}/{f}", f"{rid}/{f}",
                                                    lineterm="", n=3)),
                  "content": b}
    return {"run": rid, "against": against, "files": out}


@app.get("/api/node/{rid}/train")
def api_train(rid: str):
    d = _rdir(rid)
    meta = rj(RUN / "task" / "meta.json")
    rows = train_rows(live_results(rid, d, meta))
    log = d / "output.log"
    tail = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
    return {"rows": rows, "n_steps": N_STEPS, "log_tail": tail,
            "status": node_summary(rid, d, meta)["status"]}


@app.get("/api/node/{rid}/process")
def api_process(rid: str):
    """Agent activity attributed to this node: after the previous fml-val started, up to this one."""
    nodes = [n for n in state()["nodes"] if n["id"] != "baseline"]
    starts = {n["id"]: n.get("start") for n in nodes}
    idx = index()
    if rid == "next":        # what the agent is doing after the latest run started
        t0 = max([s for s in starts.values() if s] or [0]) or None
        t1 = None
    else:
        if rid not in starts:
            return {"events": [], "tokens": None}
        order = [n["id"] for n in nodes]
        i = order.index(rid)
        t1 = (starts[rid] or 0) + 2              # include the fml-val call itself
        t0 = (starts[order[i - 1]] + 2) if i > 0 and starts[order[i - 1]] else None
    ev = idx.window(t0, t1)
    return {"events": ev, "tokens": idx.tokens(t0, t1), "t0": t0, "t1": t1}


@app.get("/api/report")
def api_report():
    p = RUN / "task" / "scratch" / "FINAL_REPORT.md"
    return {"text": p.read_text() if p.exists() else None}


@app.post("/api/baseline")
def api_baseline():
    st = state()
    if not st.get("exists"):
        raise HTTPException(400, "no run")
    if (RUN / "task" / "runs" / "baseline" / "run.json").exists() and st.get("baseline_job", {}) and \
            st["baseline_job"].get("state") == "ok":
        raise HTTPException(409, "baseline already done")
    start_job("baseline", lambda: jobs.baseline(RUN))
    return {"started": True}


@app.post("/api/reset")
def api_reset():
    """Delete the run directory and create a fresh visualization run (same parameters)."""
    st = state()
    if st.get("exists") and st["agent"]["running"]:
        raise HTTPException(409, "the agent's sandbox is still running; exit it first")
    if any(t.is_alive() for t in _jobs.values()):
        raise HTTPException(409, "a job is running")
    if RUN.parent != (common.RUNS_ROOT / "codex" / TASK).resolve() or not RUN.name:
        raise HTTPException(500, f"refusing to delete {RUN}")
    host = rj(RUN / "host.json") or {}
    meta = rj(RUN / "task" / "meta.json") or {}
    if RUN.exists():
        shutil.rmtree(RUN)
    cx = jobs.load_codex_new_run()
    cx.create_run(TASK, gpu=host.get("gpu", RUN_DEFAULTS["gpu"]), cpu_slot=host.get("cpu_slot", RUN_DEFAULTS["cpu_slot"]),
                  budget_hours=meta.get("budget_hours", RUN_DEFAULTS["budget_hours"]),
                  max_runs=meta.get("max_runs", RUN_DEFAULTS["max_runs"]), baseline_dir=REF_BASELINE,
                  model=host.get("model", cx.MODEL), effort=host.get("reasoning_effort", cx.EFFORT),
                  run_id=RUN.name, viz=True)
    _pred_failed.clear()
    _index.update(obj=None)
    jobs.prepare(RUN)
    return {"created": str(RUN)}


@app.get("/api/events")
def api_events():
    """Server-sent events: a 'state' event whenever the state's fingerprint changes."""
    def gen():
        last = None
        while True:
            try:
                st = json.loads(json.dumps(state(), default=str))
                if st.get("budget"):          # elapsed/left change every second: not part of the fingerprint
                    st["budget"].pop("elapsed_s", None)
                    st["budget"].pop("left_s", None)
                fp = hashlib.md5(json.dumps(st, sort_keys=True).encode()).hexdigest()
                if fp != last:
                    last = fp
                    yield f"event: state\ndata: {fp}\n\n"
                else:
                    yield ": ping\n\n"
            except Exception as e:  # noqa: BLE001
                yield f": error {e}\n\n"
            time.sleep(2)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def root():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
