"""Shared definitions for running Claude Code as an FML-bench agent.

See docs/claude_code_agent_design.md. Host-side only (stdlib), except that
build_task_view.py also imports TASK_DATA from here inside the task's env.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
DATA_CACHE = Path(os.environ.get("FMLBENCH_DATA_CACHE", "/mnt/nvme/data"))   # real data
VIEW_ROOT = Path(os.environ.get("FMLBENCH_AGENT_VIEWS", "/mnt/nvme/data_agent"))  # masked data
RUNS_ROOT = Path(os.environ.get("FMLBENCH_CC_RUNS", "/mnt/nvme/cc_runs"))
UV_ENVS = Path(os.environ.get("FMLBENCH_UV_ENVS", HOME / "fml_envs"))
UV_PYTHON = Path(os.environ.get("UV_PYTHON_INSTALL_DIR", HOME / ".local/share/uv/python"))
TOOLS_DIR = Path(__file__).resolve().parent / "sandbox_tools"
CLAUDE_BIN = (HOME / ".local/bin/claude").resolve()

LITE_TASKS = [
    "Privacy_opacus", "Continual_Learning_pycil", "Data_Efficiency_usb",
    "Generalization_domainbed", "Generalization_domainbed_officehome",
    "Privacy_privacymeter", "Robustness_and_Reliability_art", "Robustness_openood",
]
CPU_ONLY_TASKS = {"Robustness_and_Reliability_art"}

# Data each task reads. "path" is relative to DATA_CACHE (the workspace symlinks point there);
# "mask" names the masking routine in build_task_view.py (None = used as is, nothing hidden).
# "home_dst" mounts the entry under $HOME instead (ART reads ~/.art/data/mnist.npz).
TASK_DATA = {
    "Privacy_opacus": [
        {"path": "cifar10/cifar-10-batches-py", "mask": "cifar10_train_opacus"}],
    "Robustness_openood": [
        {"path": "cifar10/cifar-10-batches-py", "mask": "cifar10_test_openood"},
        {"path": "cifar100/cifar-100-python", "mask": None},
        {"path": "svhn/test_32x32.mat", "mask": "svhn_all"}],
    "Data_Efficiency_usb": [
        {"path": "cifar100/cifar-100-python", "mask": "cifar100_test_usb"}],
    "Continual_Learning_pycil": [
        {"path": "cifar100/cifar-100-python", "mask": "cifar100_test_pycil"}],
    "Generalization_domainbed": [
        {"path": "mnist/MNIST", "mask": "coloredmnist_env2_in"}],
    "Generalization_domainbed_officehome": [
        {"path": "officehome/office_home", "mask": "officehome_env0_in"}],
    "Privacy_privacymeter": [
        {"path": "privacymeter/cifar10.pkl", "mask": None},
        {"path": "privacymeter/cifar10_population.pkl", "mask": None}],
    "Robustness_and_Reliability_art": [
        {"path": "art/mnist.npz", "mask": "art_mnist_test", "home_dst": ".art/data/mnist.npz"}],
}

# Pretrained weights the baseline downloads at run time; copied into each run's ~/.cache.
TASK_CACHE_FILES = {
    "Generalization_domainbed_officehome": ["torch/hub/checkpoints/resnet50-0676ba61.pth"],
}

# CPU slots: whole physical cores (logical CPU i and i+48 are siblings on this machine).
CPU_SLOTS = {
    "G0": "0-7,48-55",
    "G1": "8-15,56-63",
    "G2": "16-23,64-71",
    "G3": "24-31,72-79",
    "ART": "24-47,72-95",
}

MODEL = "us.anthropic.claude-opus-5-5"
EVAL_MODEL = "us.anthropic.claude-haiku-5-5"
VAL_TIMEOUT = 2400
MAX_RUNS = 100
BUDGET_HOURS = 24.0


def load_config(task: str) -> dict:
    return json.loads((REPO / "ml_tasks" / task / "config.json").read_text())


def template_dir(task: str) -> Path:
    """workspace/<TASK> (task template: repo + any template-level data symlinks)."""
    return REPO / "workspace" / task


def repo_subdir(task: str) -> str:
    """Repo dir name inside the template, e.g. 'opacus' for workspace/Privacy_opacus/opacus."""
    return load_config(task)["repo_dir"].split("/", 2)[2]


def ml_tasks_refs(task: str, commands: list[str]) -> list[str]:
    """ml_tasks/<TASK>/... paths referenced by the given commands (relative to ml_tasks/<TASK>)."""
    refs = set()
    for cmd in commands:
        for m in re.finditer(rf"\.\./\.\./\.\./ml_tasks/{re.escape(task)}/([^\s;&|]+)", cmd):
            refs.add(m.group(1).rstrip("/"))
    return sorted(refs)


def primary_metric(info: dict | None, metric: str):
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


def view_info(task: str) -> dict:
    p = VIEW_ROOT / task / "view.json"
    if not p.exists():
        raise SystemExit(f"no agent data view for {task}: run build_task_view.py first ({p})")
    return json.loads(p.read_text())


def data_mounts(task: str, view: str) -> list[tuple[str, str]]:
    """(src, dst) bind mounts for the task's data. view: 'agent' (masked) or 'real'."""
    mounts = []
    for e in TASK_DATA[task]:
        real = DATA_CACHE / e["path"]
        src = VIEW_ROOT / task / e["path"] if (view == "agent" and e["mask"]) else real
        dst = HOME / e["home_dst"] if e.get("home_dst") else real
        if not src.exists():
            raise SystemExit(f"missing data {src}")
        mounts.append((str(src), str(dst)))
    return mounts


def gpu_devices(gpu: int | None) -> list[str]:
    if gpu is None:
        return []
    devs = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset",
            f"/dev/nvidia{gpu}"]
    return [d for d in devs if os.path.exists(d)]


def bwrap_args(*, task: str, task_dir: Path, view: str, gpu: int | None, cache_dir: Path,
               claude_config: Path | None, extra_env: dict | None = None,
               ml_tasks_paths: list[str] | None = None, md5_aliases: bool = True) -> list[str]:
    """bubblewrap arguments for an agent sandbox (view='agent') or a final-test sandbox ('real').

    Layout inside: /task (run's task dir: workspace/, scratch/, runs/), /task/baseline/<TASK>
    (read-only task template), /task/ml_tasks/<TASK>/<refs> (read-only), the task's uv env, the
    task's data at its usual path, only the assigned GPU, /opt/fml (tools), /opt/claude (binary).
    """
    cfg = load_config(task)
    env_dir = UV_ENVS / cfg["conda_env"]
    a = [
        "--die-with-parent", "--new-session", "--unshare-all", "--share-net",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind-try", "/run/systemd/resolve", "/run/systemd/resolve",
        "--ro-bind", "/sys", "/sys",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm", "--tmpfs", "/tmp",
        "--tmpfs", str(HOME),
        "--ro-bind", str(env_dir), str(env_dir),
        "--ro-bind", str(UV_PYTHON), str(UV_PYTHON),
        "--bind", str(cache_dir), str(HOME / ".cache"),
        "--bind", str(task_dir), "/task",
        "--ro-bind", str(template_dir(task)), f"/task/baseline/{task}",
        "--ro-bind", str(TOOLS_DIR), "/opt/fml",
        "--ro-bind", str(CLAUDE_BIN), "/opt/claude/bin/claude",
    ]
    # Base interpreter of the venv, if it is outside /usr and the uv store (see sandbox/run_in_bwrap.sh).
    home_line = [l for l in (env_dir / "pyvenv.cfg").read_text().splitlines() if l.startswith("home")]
    if home_line:
        prefix = str(Path(home_line[0].split("=", 1)[1].strip()).parent)
        if not (prefix == "/usr" or prefix.startswith(str(UV_PYTHON))):
            a += ["--ro-bind-try", prefix, prefix]
    for f in ("CLAUDE.md", "meta.json", "goal.txt"):
        if (task_dir / f).exists():
            a += ["--ro-bind", str(task_dir / f), f"/task/{f}"]
    for rel in (ml_tasks_paths if ml_tasks_paths is not None
                else ml_tasks_refs(task, [cfg["val_command"]])):
        src = REPO / "ml_tasks" / task / rel
        a += ["--ro-bind", str(src), f"/task/ml_tasks/{task}/{rel}"]
    for src, dst in data_mounts(task, view):
        a += ["--ro-bind", src, dst]
    for d in gpu_devices(gpu):
        a += ["--dev-bind", d, d]
    if claude_config is not None:
        a += ["--bind", str(claude_config), str(HOME / ".claude")]

    env = {
        "HOME": str(HOME), "USER": os.environ.get("USER", "user"), "LANG": "C.UTF-8",
        "TERM": os.environ.get("TERM", "xterm-256color"), "TMPDIR": "/tmp",
        "PATH": f"/opt/fml/bin:/opt/claude/bin:{env_dir}/bin:/usr/local/bin:/usr/bin",
        "VIRTUAL_ENV": str(env_dir),
        "CUDA_VISIBLE_DEVICES": "0" if gpu is not None else "",
        "FMLBENCH_TASK": task,
    }
    if view == "agent" and md5_aliases:
        # torchvision checks dataset md5s; the masked files are accepted via this alias table.
        env["PYTHONPATH"] = "/opt/fml/site"
        env["FMLBENCH_MASKED_MD5"] = f"/task/md5_aliases.json"
        a += ["--ro-bind", str(VIEW_ROOT / task / "md5_aliases.json"), "/task/md5_aliases.json"]
    if claude_config is not None:
        env.update({
            "CLAUDE_CONFIG_DIR": str(HOME / ".claude"),
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
            "ANTHROPIC_MODEL": MODEL,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": MODEL,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": EVAL_MODEL,
            "DISABLE_AUTOUPDATER": "1",
        })
    env.update(extra_env or {})
    a.append("--clearenv")
    for k, v in env.items():
        a += ["--setenv", k, v]
    a += ["--chdir", "/task"]
    return a


def copy_template(task: str, dst: Path) -> None:
    """Copy workspace/<TASK> keeping symlinks (the data links point into DATA_CACHE)."""
    shutil.copytree(template_dir(task), dst, symlinks=True)
