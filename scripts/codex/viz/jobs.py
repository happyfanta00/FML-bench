#!/usr/bin/env python3
"""Host-side jobs of the visualization (docs/codex_viz_design.md, sections 5 and 6).

    python3 scripts/codex/viz/jobs.py prepare  <run_dir>        # sample metadata and images (viz/)
    python3 scripts/codex/viz/jobs.py baseline <run_dir>        # env checks + baseline run + predict
    python3 scripts/codex/viz/jobs.py predict  <run_dir> <rid>  # rid: 0003 or baseline

Everything that executes task or agent code (baseline training, predictions) runs in the agent's
sandbox configuration (masked data, the task's uv env, the run's GPU and CPU slot); the budget clock
is not started. Job state is written to <run_dir>/viz/jobs/<name>.json for the page. One job at a
time (file lock), so GPU jobs never overlap each other.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "claude_code"))
import common  # noqa: E402

TASK = "Generalization_domainbed"
ML = common.REPO / "ml_tasks"


# ----------------------------------------------------------------------------- helpers

def viz_dir(run: Path) -> Path:
    d = run / "viz"
    for sub in ("jobs", "pred", "sandbox_out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(p)


class Job:
    """State file for one job: {name, state, steps: [{key, label, state, detail}], started, ended}."""

    def __init__(self, run: Path, name: str, steps: list[tuple[str, str]]):
        self.p = viz_dir(run) / "jobs" / f"{name}.json"
        self.s = {"name": name, "state": "running", "started": time.time(), "ended": None, "error": None,
                  "pid": os.getpid(),
                  "steps": [{"key": k, "label": l, "state": "pending", "detail": ""} for k, l in steps]}
        self.save()

    def save(self):
        write_json(self.p, self.s)

    def step(self, key, state, detail=""):
        for st in self.s["steps"]:
            if st["key"] == key:
                st.update(state=state, detail=detail)
        self.save()

    def end(self, state, error=None):
        self.s.update(state=state, ended=time.time(), error=error)
        self.save()


def sandbox(run: Path, cmd: list[str], log: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run cmd in the agent's sandbox configuration (no Codex, clock not started)."""
    host = json.loads((run / "host.json").read_text())
    args = common.bwrap_args(task=host["task"], task_dir=run / "task", view="agent", gpu=host["gpu"],
                             cache_dir=run / "cache", claude_config=None)
    args += ["--ro-bind", str(HERE), "/opt/fmlviz", "--bind", str(viz_dir(run) / "sandbox_out"), "/vizout"]
    argv = ["taskset", "-c", host["cpus"], shutil.which("bwrap") or "bwrap", *args, *cmd]
    if log is None:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    with open(log, "w") as f:
        return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def load_codex_new_run():
    """scripts/codex/new_run.py under its own name (scripts/claude_code/new_run.py is `new_run`)."""
    import importlib.util
    if "cx_new_run" not in sys.modules:
        spec = importlib.util.spec_from_file_location("cx_new_run", HERE.parent / "new_run.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cx_new_run"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["cx_new_run"]


def lock(run: Path):
    f = open(viz_dir(run) / "jobs" / ".lock", "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


# ----------------------------------------------------------------------------- prepare

def prepare(run: Path) -> None:
    """Sample metadata and images from the real data on the host (template code, nothing from the agent)."""
    out = viz_dir(run)
    if (out / "samples.json").exists() and (out / "intro.json").exists():
        return
    cfg = common.load_config(TASK)
    tmp = Path(tempfile.mkdtemp(prefix="viz_prepare_", dir=str(out)))
    try:
        common.copy_template(TASK, tmp / TASK)
        repo = tmp / TASK / common.repo_subdir(TASK)
        for seg in cfg["val_command"].split("&&"):
            seg = seg.strip()
            if seg.startswith("cp "):
                subprocess.run(seg.replace("../../../ml_tasks/", f"{ML}/"), shell=True, cwd=repo, check=True)
        py = common.UV_ENVS / cfg["conda_env"] / "bin" / "python"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([str(py), str(HERE / "coloredmnist.py"), "prepare", str(repo), str(out)],
                       env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- predict

def summarize(run: Path, rid: str, raw: dict) -> dict:
    """Aggregates for the right-hand panel from the per-sample predictions."""
    vd = viz_dir(run)
    ds = json.loads((vd / "dataset.json").read_text())
    rows = ds["rows"]                               # label, digit, color, flipped, group
    out = {"run": rid, "ok": raw.get("ok"), "error": raw.get("error"), "seconds": raw.get("seconds"),
           "created": time.time()}
    rdir = run / "task" / "runs" / rid
    rec = json.loads((rdir / "run.json").read_text()) if (rdir / "run.json").exists() else {}
    if not raw.get("ok"):
        return out
    pred, prob = raw["pred"], raw["prob1"]
    n = len(rows)
    assert n == len(pred), (n, len(pred))
    groups = {}
    for g in "ABCD":
        idx = [i for i, r in enumerate(rows) if r[4] == g]
        groups[g] = {"n": len(idx), "acc": sum(pred[i] == rows[i][0] for i in idx) / max(len(idx), 1)}
    conf = [[0, 0], [0, 0]]
    for i, r in enumerate(rows):
        conf[r[0]][pred[i]] += 1
    shape = [int(r[1] < 5) for r in rows]
    out.update(
        n=n, acc=raw["acc"], acc_pass2=raw["acc_pass2"], logged_val=rec.get("metric"),
        # One sample is 1/n (1.4e-4 here); the tolerance only absorbs float rounding.
        matches_logged=(rec.get("metric") is not None and abs(raw["acc"] - rec["metric"]) < 1e-6),
        batch_dependent=raw["batch_dependent"], n_differ_between_passes=raw["n_differ_between_passes"],
        groups=groups, confusion=conf,
        agree_color=sum(pred[i] == rows[i][2] for i in range(n)) / n,
        agree_shape=sum(pred[i] == shape[i] for i in range(n)) / n,
        device=raw.get("device"))
    samples = json.loads((vd / "samples.json").read_text())["samples"]
    out["samples"] = {str(s["i"]): {"pred": pred[s["i"]], "prob1": prob[s["i"]]} for s in samples}
    # Per-environment accuracies at the end of training (DomainBed's own evaluation).
    rj = rdir / "results.jsonl"
    if rj.exists():
        lines = [l for l in rj.read_text().splitlines() if l.strip()]
        if lines:
            last = json.loads(lines[-1])
            out["env_acc"] = {k[:-4]: v for k, v in last.items() if k.endswith("_acc") and k.startswith("env")}
    return out


def predict(run: Path, rid: str) -> dict:
    vd = viz_dir(run)
    prepare(run)
    raw_p = vd / "sandbox_out" / f"{rid}.json"
    raw_p.unlink(missing_ok=True)
    log = vd / "jobs" / f"predict_{rid}.log"
    with lock(run):
        p = sandbox(run, ["python", "/opt/fmlviz/predict_inner.py", rid, f"/vizout/{rid}.json"], log=log,
                    timeout=1800)
    raw = json.loads(raw_p.read_text()) if raw_p.exists() else {"ok": False,
                                                                 "error": f"no output (sandbox rc {p.returncode})"}
    res = summarize(run, rid, raw)
    write_json(vd / "pred" / f"{rid}.json", res)
    raw_p.unlink(missing_ok=True)
    return res


# ----------------------------------------------------------------------------- baseline

STEPS = [("sandbox", "沙盒"), ("gpu", "GPU"), ("data", "掩码数据"), ("train", "训练"),
         ("predict", "预测"), ("compare", "与参考值比较"), ("prompts", "生成提示词")]


def baseline(run: Path) -> int:
    job = Job(run, "baseline", STEPS)
    try:
        prepare(run)
        meta_p = run / "task" / "meta.json"
        meta = json.loads(meta_p.read_text())
        # 1-3: the sandbox starts, sees exactly one GPU, and the masked data with its md5 aliases.
        job.step("sandbox", "running")
        r = sandbox(run, ["bash", "-c", "echo sandbox-ok; nproc"], timeout=120)
        if r.returncode != 0 or "sandbox-ok" not in r.stdout:
            raise RuntimeError(f"sandbox did not start: {r.stderr[-800:]}")
        job.step("sandbox", "ok", f"{r.stdout.split()[-1]} CPU")
        job.step("gpu", "running")
        r = sandbox(run, ["python", "-c", "import torch; assert torch.cuda.is_available(); "
                          "print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"], timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"no GPU in the sandbox: {r.stderr[-800:]}")
        job.step("gpu", "ok", r.stdout.strip())
        job.step("data", "running")
        mnist = common.DATA_CACHE / "mnist" / "MNIST" / "raw"
        r = sandbox(run, ["python", "-c", f"import json, os; a = json.load(open(os.environ['FMLBENCH_MASKED_MD5'])); "
                          f"print(len(a), len(os.listdir({str(mnist)!r})))"], timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"masked data not visible: {r.stderr[-800:]}")
        job.step("data", "ok", f"{r.stdout.split()[0]} md5 aliases, {r.stdout.split()[1]} raw files")
        # 4: the unmodified code through fml-val (runs/baseline/, not counted, live log).
        job.step("train", "running")
        bdir = run / "task" / "runs" / "baseline"
        if not (bdir / "run.json").exists():
            sandbox(run, ["/opt/fml/bin/fml-val", "--baseline"], log=viz_dir(run) / "jobs" / "baseline_fml.log")
        rec = json.loads((bdir / "run.json").read_text()) if (bdir / "run.json").exists() else {}
        if not rec.get("success"):
            raise RuntimeError("baseline training failed: see runs/baseline/output.log")
        job.step("train", "ok", f"{rec['metric']:.4f} in {rec['duration_s']:.0f} s")
        # 5: predictions of the baseline model (first SOTA on the page).
        job.step("predict", "running")
        res = predict(run, "baseline")
        if not res.get("ok"):
            raise RuntimeError(f"prediction failed: {res.get('error')}")
        job.step("predict", "ok", f"acc {res['acc']:.4f} ({'= logged val' if res['matches_logged'] else 'differs'})")
        # 6: compare with the reference measured on this machine (rebaseline_results/..._l4x1).
        job.step("compare", "running")
        ref_dir = Path(meta.get("baseline_source", ""))
        ref = (ref_dir / "val_info.json") if ref_dir.name else None
        if ref and ref.exists():
            same = ref.read_bytes() == (bdir / "val_info.json").read_bytes()
            ref_v = meta.get("baseline_reference")
            job.step("compare", "ok" if same else "warn",
                     "byte-identical to the reference" if same else
                     f"{rec['metric']:.4f} vs reference {ref_v:.4f} ({rec['metric'] - ref_v:+.4f})")
        else:
            job.step("compare", "warn", "no reference to compare with")
        # 7: the measured baseline becomes the run's baseline; now the prompts can be written.
        job.step("prompts", "running")
        meta["baseline_val"] = rec["metric"]
        meta_p.write_text(json.dumps(meta, indent=1))
        load_codex_new_run().write_prompts(run)
        job.step("prompts", "ok", "AGENTS.md, goal.txt")
        job.end("ok")
        return 0
    except Exception as e:
        for st in job.s["steps"]:
            if st["state"] == "running":
                st["state"] = "fail"
        job.end("fail", f"{e}\n{traceback.format_exc()[-1500:]}")
        return 1


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cmd, run = sys.argv[1], Path(sys.argv[2]).resolve()
    if cmd == "prepare":
        prepare(run)
    elif cmd == "baseline":
        sys.exit(baseline(run))
    elif cmd == "predict":
        res = predict(run, sys.argv[3])
        print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
