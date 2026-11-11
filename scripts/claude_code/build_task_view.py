#!/usr/bin/env python3
"""Build the agent's data view of a task: a copy of its data with the hidden test samples blanked.

    python3 scripts/claude_code/build_task_view.py <TASK> [--force]

Re-executes itself in the task's uv env (needs numpy/torchvision/scipy/PIL). Writes
$FMLBENCH_AGENT_VIEWS/<TASK>/ (default /mnt/nvme/data_agent/<TASK>/):
  <entry path>         masked copy of each masked data entry (see common.TASK_DATA)
  md5_aliases.json     {real file md5: masked file md5}; the sandbox's sitecustomize lets
                       torchvision's integrity check accept the masked files
  view.json            what was masked (counts, indices hash) and how

Only images are blanked; labels and file layout stay identical, so every split computed from
labels or sizes (stratified splits, permutations) is unchanged. Hidden indices are computed with
the task's own code (eval-script split functions, PyCIL's DataManager, DomainBed's train.py).
Check the result with verify_task_view.py.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ML = common.REPO / "ml_tasks"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_function(script: Path, name: str):
    """Load one top-level function from a script without running the script."""
    import numpy as np
    tree = ast.parse(script.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), "exec"), ns)
    return ns[name]


def idx_digest(idx) -> str:
    return hashlib.sha256(",".join(map(str, sorted(int(i) for i in idx))).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------- file writers

def mask_cifar_batches(src: Path, dst: Path, masks: dict[str, "np.ndarray"]) -> dict:
    """Copy a CIFAR python-version dir; zero the image rows given per batch file."""
    shutil.copytree(src, dst)
    changed = {}
    for fname, idx in masks.items():
        with open(src / fname, "rb") as f:
            entry = pickle.load(f, encoding="latin1")      # same as torchvision
        data = entry["data"]
        data[idx] = 0
        entry["data"] = data
        os.chmod(dst / fname, 0o644)
        with open(dst / fname, "wb") as f:
            pickle.dump(entry, f, protocol=2)
        changed[fname] = len(idx)
    return changed


def blanked_indices_stats(n_total: int, idx) -> dict:
    return {"total": int(n_total), "masked": int(len(idx)), "indices_sha256_16": idx_digest(idx)}


# ---------------------------------------------------------------------------- masks per task
# Each returns {"files": {relpath: n_masked}, "stats": {...}} after writing the masked entry.

def cifar10_train_opacus(src, dst):
    # ml_tasks/Privacy_opacus/train_eval_baseline.py: hidden test = first 21% of
    # RandomState(42).permutation(50000) over the CIFAR-10 training set (batches 1..5 in order).
    import numpy as np
    n = 50000
    test_idx = np.random.RandomState(42).permutation(n)[: int(0.21 * n)]
    masks = {}
    for b in range(5):
        lo = b * 10000
        sel = test_idx[(test_idx >= lo) & (test_idx < lo + 10000)] - lo
        masks[f"data_batch_{b + 1}"] = np.sort(sel)
    files = mask_cifar_batches(src, dst, masks)
    return {"files": files, "stats": blanked_indices_stats(n, test_idx)}


def cifar10_test_openood(src, dst):
    # ml_tasks/Robustness_openood/train_eval_baseline.py: hidden ID test = perm[3000:] of
    # RandomState(42).permutation(10000) over the CIFAR-10 test set.
    import numpy as np
    perm = np.random.RandomState(42).permutation(10000)
    hidden = np.sort(perm[int(0.3 * 10000):])
    files = mask_cifar_batches(src, dst, {"test_batch": hidden})
    return {"files": files, "stats": blanked_indices_stats(10000, hidden)}


def svhn_all(src, dst):
    # SVHN test is only the OOD set of the hidden test (val uses CIFAR-100): blank all of it.
    import numpy as np
    import scipy.io as sio
    m = sio.loadmat(src)
    m = {k: v for k, v in m.items() if not k.startswith("__")}
    m["X"] = np.zeros_like(m["X"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(dst, m, do_compression=False)
    n = m["X"].shape[-1]
    return {"files": {"test_32x32.mat": n}, "stats": {"total": n, "masked": n}}


def _cifar100_test(src, dst, hidden):
    files = mask_cifar_batches(src, dst, {"test": hidden})
    return {"files": files, "stats": blanked_indices_stats(10000, hidden)}


def cifar100_test_usb(src, dst):
    # ml_tasks/Data_Efficiency_usb/train_eval_baseline.py: test_idx = perm[3000:] of
    # RandomState(42).permutation(10000) over the CIFAR-100 test set.
    import numpy as np
    perm = np.random.RandomState(42).permutation(10000)
    return _cifar100_test(src, dst, np.sort(perm[int(0.3 * 10000):]))


def cifar100_test_pycil(src, dst):
    # ml_tasks/Continual_Learning_pycil/train_eval_baseline.py: split_cifar100_test() on the
    # DataManager's class-order-remapped test targets (shuffle=True, seed=1993); indices refer to
    # the unchanged order of the CIFAR-100 test set.
    import numpy as np
    repo = common.template_dir("Continual_Learning_pycil") / common.repo_subdir("Continual_Learning_pycil")
    cfg_src = (repo / "algorithm.py").read_text()
    assert '"shuffle": True' in cfg_src and '"seed": [1993]' in cfg_src, "unexpected PyCIL config"
    cwd = os.getcwd()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    try:
        from utils.data_manager import DataManager
        dm = DataManager.__new__(DataManager)
        dm._setup_data("cifar100", True, 1993)
        test_targets = np.array(dm._test_targets)
    finally:
        os.chdir(cwd)
        sys.path.remove(str(repo))
    split = extract_function(ML / "Continual_Learning_pycil" / "train_eval_baseline.py",
                             "split_cifar100_test")
    _, test_idx = split(test_targets)
    return _cifar100_test(src, dst, np.sort(test_idx))


def art_mnist_test(src, dst):
    # ml_tasks/Robustness_and_Reliability_art/eval.py: split_test_stratified(y_test, 0.70, seed=42).
    import numpy as np
    d = dict(np.load(src))
    y = d["y_test"]
    labels = np.argmax(y, axis=1) if y.ndim > 1 else y
    split = extract_function(ML / "Robustness_and_Reliability_art" / "eval.py", "split_test_stratified")
    _, hidden = split(labels, test_fraction=0.70, seed=42)
    d["x_test"] = d["x_test"].copy()
    d["x_test"][hidden] = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        np.savez(f, **d)
    return {"files": {"mnist.npz": len(hidden)},
            "stats": blanked_indices_stats(len(labels), hidden)}


def _domainbed_splits(task: str) -> tuple[dict, list[str]]:
    """Run the task's DomainBed train.py up to the data loaders and record the env splits."""
    cfg = common.load_config(task)
    tmp = Path(tempfile.mkdtemp(prefix=f"view_{task}_", dir="/mnt/nvme"))
    try:
        common.copy_template(task, tmp / task)
        repo = tmp / task / common.repo_subdir(task)
        # The val command copies the benchmark's versions of train.py/datasets.py/... first.
        for seg in cfg["val_command"].split("&&"):
            seg = seg.strip()
            if seg.startswith("cp "):
                seg = seg.replace("../../../ml_tasks/", f"{ML}/")
                subprocess.run(seg, shell=True, cwd=repo, check=True)
        tree = ast.parse((ML / task / "train_eval.py").read_text())
        args = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "cmd":
                args = [e.value if isinstance(e, ast.Constant) else "./results_tmp"
                        for e in node.value.elts]
        assert args and args[:3] == ["python", "-m", "domainbed.scripts.train"], args
        out = tmp / "splits.json"
        driver = f"""
import sys, json, runpy, numpy as np, torch
sys.dont_write_bytecode = True
sys.path.insert(0, '.')
from domainbed.lib import misc, fast_data_loader
rec = {{"shuffle": None, "splits": []}}
_rp = torch.randperm
def randperm(*a, **k):
    r = _rp(*a, **k)
    if rec["shuffle"] is None and r.numel() >= 10000:
        rec["shuffle"] = r.tolist()
    return r
torch.randperm = randperm
_split = misc.split_dataset
def split_dataset(dataset, n, seed=0):
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    e = {{"type": type(dataset).__name__, "len": len(dataset), "n": n, "keys": keys}}
    if hasattr(dataset, "samples"):
        e["samples"] = [s[0] for s in dataset.samples]
    rec["splits"].append(e)
    return _split(dataset, n, seed)
misc.split_dataset = split_dataset
class Stop(Exception):
    pass
class NoLoader:
    def __init__(self, *a, **k):
        raise Stop
fast_data_loader.InfiniteDataLoader = NoLoader
sys.argv = ["train"] + {args[3:]!r}
try:
    runpy.run_module("domainbed.scripts.train", run_name="__main__")
except Stop:
    pass
json.dump(rec, open({str(out)!r}, "w"))
"""
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([sys.executable, "-c", driver], cwd=repo, env=env, check=True,
                       stdout=subprocess.DEVNULL)
        rec = json.loads(out.read_text())
        return rec, args
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _test_env_in_keys(rec: dict, args: list[str]) -> tuple[dict, list[int]]:
    test_env = int(args[args.index("--test_env") + 1])
    env_splits = [s for s in rec["splits"] if s["type"] != "_SplitDataset"]
    e = env_splits[test_env]
    return e, e["keys"][e["n"]:]          # split_dataset: keys[:n] = out-split, keys[n:] = in-split


def coloredmnist_env2_in(src, dst):
    # DomainBed ColoredMNIST: images = cat(MNIST train, test)[shuffle]; env i = [i::3];
    # hidden test = in-split of the test env.
    import numpy as np
    rec, args = _domainbed_splits("Generalization_domainbed")
    e, in_keys = _test_env_in_keys(rec, args)
    test_env = int(args[args.index("--test_env") + 1])
    shuffle = np.array(rec["shuffle"])
    n_env = 3
    orig = shuffle[test_env + n_env * np.array(in_keys)]
    hidden_tr = np.sort(orig[orig < 60000])
    hidden_te = np.sort(orig[orig >= 60000] - 60000)
    (dst / "raw").mkdir(parents=True)
    files = {}
    for name in ("train-labels-idx1-ubyte", "t10k-labels-idx1-ubyte"):
        shutil.copy2(src / "raw" / name, dst / "raw" / name)
    for name, idx in (("train-images-idx3-ubyte", hidden_tr), ("t10k-images-idx3-ubyte", hidden_te)):
        buf = bytearray((src / "raw" / name).read_bytes())
        n = int.from_bytes(buf[4:8], "big")
        rows, cols = int.from_bytes(buf[8:12], "big"), int.from_bytes(buf[12:16], "big")
        img = np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(n, rows * cols).copy()
        img[idx] = 0
        (dst / "raw" / name).write_bytes(bytes(buf[:16]) + img.tobytes())
        files[f"raw/{name}"] = len(idx)
    return {"files": files, "stats": {"env_len": e["len"], "masked": int(len(orig)),
                                      "indices_sha256_16": idx_digest(orig)}}


def officehome_env0_in(src, dst):
    # DomainBed OfficeHome: env = ImageFolder(office_home/<env>); hidden test = in-split of the
    # test env. Hidden images are replaced by black images of the same size and format.
    from PIL import Image
    rec, args = _domainbed_splits("Generalization_domainbed_officehome")
    e, in_keys = _test_env_in_keys(rec, args)
    shutil.copytree(src, dst)
    files = {}
    for k in in_keys:
        rel = e["samples"][k].split("office_home/", 1)[1]
        p = dst / rel
        with Image.open(src / rel) as im:
            size, fmt, mode = im.size, im.format or "JPEG", im.mode
        os.chmod(p, 0o644)
        Image.new(mode if mode in ("RGB", "L") else "RGB", size).save(p, format=fmt)
        files[rel] = 1
    return {"files": {"(images)": len(files)},
            "stats": {"env_len": e["len"], "masked": len(files),
                      "indices_sha256_16": idx_digest(in_keys)}}


MASKS = {f.__name__: f for f in (
    cifar10_train_opacus, cifar10_test_openood, svhn_all, cifar100_test_usb, cifar100_test_pycil,
    art_mnist_test, coloredmnist_env2_in, officehome_env0_in)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--force", action="store_true", help="rebuild an existing view")
    a = ap.parse_args()
    task = a.task
    env_dir = common.UV_ENVS / common.load_config(task)["conda_env"]
    if Path(sys.prefix).resolve() != env_dir.resolve():
        os.execv(str(env_dir / "bin/python"), [str(env_dir / "bin/python"), *sys.argv])

    out = common.VIEW_ROOT / task
    if out.exists():
        if not a.force:
            raise SystemExit(f"{out} exists (use --force to rebuild)")
        subprocess.run(["chmod", "-R", "u+w", str(out)], check=True)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    entries, aliases = [], {}
    for e in common.TASK_DATA[task]:
        rec = {"path": e["path"], "mask": e["mask"]}
        if e["mask"]:
            src, dst = common.DATA_CACHE / e["path"], out / e["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            res = MASKS[e["mask"]](src, dst)
            rec.update(res, seconds=round(time.time() - t0, 1))
            # md5 aliases for every file that differs (torchvision verifies CIFAR/SVHN files).
            pairs = [(src, dst)] if src.is_file() else [
                (src / p.relative_to(dst), p) for p in dst.rglob("*") if p.is_file()]
            for s, d in pairs:
                ms, mdst = md5(s), md5(d)
                if ms != mdst:
                    aliases[ms] = mdst
            print(f"[{task}] {e['path']}: {e['mask']} -> {res['stats']}")
        else:
            print(f"[{task}] {e['path']}: used as is (nothing hidden in this entry)")
        entries.append(rec)
    (out / "md5_aliases.json").write_text(json.dumps(aliases, indent=1))
    (out / "view.json").write_text(json.dumps({
        "task": task, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_cache": str(common.DATA_CACHE), "entries": entries}, indent=1))
    subprocess.run(["chmod", "-R", "a-w", str(out)], check=True)
    print(f"[{task}] view written to {out} ({len(aliases)} md5 aliases)")


if __name__ == "__main__":
    main()
