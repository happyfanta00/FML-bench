"""ColoredMNIST as DomainBed builds it for the task (stdlib + torch; runs in the domainbed env).

capture(repo) runs the task's DomainBed train.py (with the benchmark's train.py/datasets.py already
copied in, as the val command does) up to the data loaders and returns, for every environment, the
image/label tensors, the out-split keys (DomainBed's split_dataset order) and the original MNIST
digit of every sample. Nothing is trained. Used on the host for the sample metadata (prepare) and in
the sandbox by predict_inner.py (same splits, the agent's code).

    python coloredmnist.py prepare <repo> <out_dir>    # host: viz/dataset.json, samples.json, intro.json
"""
from __future__ import annotations

import base64
import io
import json
import os
import runpy
import sys

# The task's train_eval.py arguments (ml_tasks/Generalization_domainbed/train_eval.py).
TRAIN_ARGS = ["--data_dir=../data/", "--algorithm", "ERM", "--dataset", "ColoredMNIST", "--test_env", "2",
              "--holdout_fraction", "0.3", "--output_dir", "./results_tmp"]
TEST_ENV = 2
GROUPS = {   # shape OK = label not flipped by the 25% noise; colour OK = colour agrees with the label
    "A": "形状✓ 颜色✓", "B": "形状✓ 颜色✗", "C": "形状✗ 颜色✓", "D": "形状✗ 颜色✗"}


class _Stop(Exception):
    pass


def capture(repo: str) -> dict:
    import numpy as np
    import torch
    old_cwd, old_argv = os.getcwd(), sys.argv
    os.chdir(repo)
    sys.path.insert(0, repo)
    from domainbed.lib import fast_data_loader, misc
    rec = {"shuffle": None, "envs": []}
    _randperm, _split = torch.randperm, misc.split_dataset

    def randperm(*a, **k):
        r = _randperm(*a, **k)
        if rec["shuffle"] is None and r.numel() >= 10000:
            rec["shuffle"] = r.clone()
        return r

    def split_dataset(dataset, n, seed=0):
        if type(dataset).__name__ == "TensorDataset":   # one call per environment (uda splits are skipped)
            keys = list(range(len(dataset)))
            np.random.RandomState(seed).shuffle(keys)
            x, y = dataset.tensors
            rec["envs"].append({"x": x, "y": y, "out_keys": keys[:n], "in_keys": keys[n:]})
        return _split(dataset, n, seed)

    class NoLoader:
        def __init__(self, *a, **k):
            raise _Stop

    torch.randperm, misc.split_dataset = randperm, split_dataset
    saved_loader = fast_data_loader.InfiniteDataLoader
    fast_data_loader.InfiniteDataLoader = NoLoader
    sys.argv = ["train"] + TRAIN_ARGS
    try:
        runpy.run_module("domainbed.scripts.train", run_name="__main__")
    except _Stop:
        pass
    finally:
        torch.randperm, misc.split_dataset = _randperm, _split
        fast_data_loader.InfiniteDataLoader = saved_loader
        sys.argv = old_argv
        os.chdir(old_cwd)
    # Original digits: images = cat(MNIST train, test)[shuffle]; environment i = [i::3].
    from torchvision.datasets import MNIST
    data_dir = os.path.join(repo, "..", "data")
    labels = torch.cat((MNIST(data_dir, train=True, download=False).targets,
                        MNIST(data_dir, train=False, download=False).targets))
    digits = labels[rec["shuffle"]]
    n_env = len(rec["envs"])
    for i, e in enumerate(rec["envs"]):
        e["digit"] = digits[i::n_env]
    return rec


def sample_meta(e: dict, k: int) -> dict:
    x, y, d = e["x"][k], int(e["y"][k]), int(e["digit"][k])
    color = int(x[1].sum() > x[0].sum())          # kept channel: 0 = red, 1 = green
    shape_label = int(d < 5)                       # DomainBed: label = (digit < 5)
    flipped = y != shape_label
    group = {(False, True): "A", (False, False): "B", (True, True): "C", (True, False): "D"}[(flipped, color == y)]
    return {"label": y, "digit": d, "color": color, "flipped": flipped, "group": group}


def png(x) -> str:
    """2×28×28 image -> 4× nearest-neighbour RGB PNG (channel 0 red, channel 1 green), data URL."""
    import numpy as np
    from PIL import Image
    a = (x.numpy() * 255).clip(0, 255).astype(np.uint8)
    rgb = np.zeros((28, 28, 3), np.uint8)
    rgb[..., 0], rgb[..., 1] = a[0], a[1]
    im = Image.fromarray(rgb).resize((84, 84), Image.NEAREST)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def prepare(repo: str, out_dir: str, n_per_group: int = 12, intro_pool: int = 24) -> None:
    import numpy as np
    rec = capture(repo)
    os.makedirs(out_dir, exist_ok=True)
    e2 = rec["envs"][TEST_ENV]
    val = [sample_meta(e2, k) for k in e2["out_keys"]]      # val = env2 out-split, in split order
    json.dump({"n": len(val), "fields": ["label", "digit", "color", "flipped", "group"],
               "rows": [[m["label"], m["digit"], m["color"], int(m["flipped"]), m["group"]] for m in val]},
              open(os.path.join(out_dir, "dataset.json"), "w"))
    # Fixed display samples: n_per_group per group from the val split (same for every node).
    rng = np.random.RandomState(20261124)
    samples = []
    for g in GROUPS:
        idx = [j for j, m in enumerate(val) if m["group"] == g]
        for j in sorted(rng.choice(idx, size=min(n_per_group, len(idx)), replace=False).tolist()):
            samples.append({"i": j, **val[j], "img": png(e2["x"][e2["out_keys"][j]])})
    json.dump({"groups": GROUPS, "samples": samples}, open(os.path.join(out_dir, "samples.json"), "w"))
    # Intro page: a pool of samples per environment, from the out-splits (env2: the val split only).
    intro = {}
    for i, e in enumerate(rec["envs"]):
        keys = rng.choice(e["out_keys"], size=intro_pool, replace=False).tolist()
        intro[f"env{i}"] = [{**sample_meta(e, k), "img": png(e["x"][k])} for k in keys]
    stats = {}
    for i, e in enumerate(rec["envs"]):
        ms = [sample_meta(e, k) for k in range(len(e["y"]))]
        stats[f"env{i}"] = {"n": len(ms), "n_out": len(e["out_keys"]),
                            "color_agrees": round(float(np.mean([m["color"] == m["label"] for m in ms])), 4),
                            "label_flipped": round(float(np.mean([m["flipped"] for m in ms])), 4)}
    json.dump({"envs": intro, "stats": stats}, open(os.path.join(out_dir, "intro.json"), "w"))
    print(f"[coloredmnist] val samples {len(val)}; display samples {len(samples)}; envs {list(stats)}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "prepare":
        prepare(os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
