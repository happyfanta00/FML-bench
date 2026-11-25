"""Predictions of one run's trained model on the val split (runs INSIDE the task sandbox).

    python /opt/fmlviz/predict_inner.py <run dir name: 0003 | baseline> <out json>

Builds a fresh copy of the baseline repo with the run's target files (like fml-val), loads the run's
model.pkl, rebuilds ColoredMNIST exactly as DomainBed's train.py does, and predicts the env2
out-split (= the val split) in random batches of 64, the way DomainBed's evaluation loader batches
it. Two passes with different batch orders: if they disagree, the model's predictions depend on the
batch composition (e.g. batch-level test-time adaptation). It executes the agent's code, hence the
sandbox.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, "/opt/fmlviz")
import coloredmnist  # noqa: E402

TASK = Path("/task")
BATCH = 64        # DomainBed's evaluation batch size (train.py: eval_loaders batch_size=64)


def main():
    rid, out_path = sys.argv[1], Path(sys.argv[2])
    t0 = time.time()
    res = {"run": rid}
    try:
        meta = json.loads((TASK / "meta.json").read_text())
        rdir = TASK / "runs" / rid
        code = json.loads((rdir / "code.json").read_text())
        work = Path("/tmp/viz_exec")
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(TASK / "baseline" / meta["task"], work, symlinks=True,
                        ignore=shutil.ignore_patterns(".git"))
        repo = work / meta["repo_subdir"]
        for tf, content in code.items():
            (repo / tf).write_text(content)
        # The val command first copies the benchmark's train.py/datasets.py/...; do the same.
        for seg in meta["val_command"].split("&&"):
            seg = seg.strip()
            if seg.startswith("cp "):
                subprocess.run(seg.replace("../../../ml_tasks/", "/task/ml_tasks/"), shell=True, cwd=repo,
                               check=True)
        import torch
        rec = coloredmnist.capture(str(repo))
        e2 = rec["envs"][coloredmnist.TEST_ENV]
        keys = torch.tensor(e2["out_keys"])
        x_all, y_all = e2["x"][keys], e2["y"][keys]
        from domainbed import algorithms
        saved = torch.load(rdir / "model.pkl", map_location="cpu", weights_only=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls = algorithms.get_algorithm_class(saved["args"]["algorithm"])
        alg = cls(saved["model_input_shape"], saved["model_num_classes"], saved["model_num_domains"],
                  saved["model_hparams"])
        alg.load_state_dict(saved["model_dict"])
        alg.to(device)
        alg.eval()
        n = len(y_all)
        passes = []
        with torch.no_grad():
            for seed in (0, 1):
                order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
                pred = torch.empty(n, dtype=torch.long)
                prob = torch.empty(n)
                for s in range(0, n, BATCH):
                    idx = order[s:s + BATCH]
                    p = alg.predict(x_all[idx].to(device)).float().cpu()
                    if p.size(1) == 1:      # same rule as misc.accuracy
                        pred[idx] = p[:, 0].gt(0).long()
                        prob[idx] = torch.sigmoid(p[:, 0])
                    else:
                        pred[idx] = p.argmax(1)
                        prob[idx] = p.softmax(1)[:, 1]
                passes.append({"pred": pred, "prob1": prob,
                               "acc": int(pred.eq(y_all).sum()) / n})
        diff = int((passes[0]["pred"] != passes[1]["pred"]).sum())
        res.update(ok=True, n=n, device=device, batch=BATCH,
                   pred=passes[0]["pred"].tolist(), prob1=[round(float(v), 4) for v in passes[0]["prob1"]],
                   acc=passes[0]["acc"], acc_pass2=passes[1]["acc"], n_differ_between_passes=diff,
                   batch_dependent=diff > 0)
    except Exception:
        res.update(ok=False, error=traceback.format_exc()[-4000:])
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res))
    print(f"[predict] run {rid}: ok={res.get('ok')} acc={res.get('acc')} diff={res.get('n_differ_between_passes')} "
          f"in {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
