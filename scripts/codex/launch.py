#!/usr/bin/env python3
"""Enter the sandbox of a Codex run (created by scripts/codex/new_run.py).

    scripts/codex/launch.sh <run_dir>              # interactive shell in the sandbox; then `cx-start`
    scripts/codex/launch.sh <run_dir> -- CMD ...   # run one command in the sandbox

Same behaviour as scripts/claude_code/launch.py (the first launch starts the wall-clock budget, the
sandbox is pinned to the run's CPU slot and killed one hour after the deadline), with Codex mounted
instead of Claude Code: /opt/codex/bin, <run_dir>/codex_home as ~/.codex, ~/.aws/config read-only.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "claude_code"))
import common  # noqa: E402
import launch as cc_launch  # noqa: E402


def sandbox_argv(run: Path, cmd: list[str]) -> list[str]:
    host = json.loads((run / "host.json").read_text())
    meta_p = run / "task" / "meta.json"
    meta = json.loads(meta_p.read_text())
    if meta.get("viz"):
        # Visualization runs: the host-run baseline (viz page / viz/jobs.py) comes first.
        base = run / "task" / "runs" / "baseline" / "run.json"
        if not (base.exists() and json.loads(base.read_text()).get("success")
                and (run / "codex_home" / "AGENTS.md").exists()):
            raise SystemExit("[launch] this is a visualization run: run the baseline first "
                             "(button on the viz page, or: python3 scripts/codex/viz/jobs.py baseline <run_dir>)")
    now = time.time()
    if meta.get("session_start") is None:
        meta["session_start"] = now
        meta["deadline"] = now + meta["budget_hours"] * 3600
        meta["sandbox_close"] = meta["deadline"] + cc_launch.GRACE
        meta_p.write_text(json.dumps(meta, indent=1))
        print(f"[launch] budget started: deadline {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta['deadline']))}")
    left = meta["deadline"] + cc_launch.GRACE - now
    if left <= 0:
        raise SystemExit("[launch] this run's deadline (plus 1 h grace) has passed")
    args = common.bwrap_args(task=host["task"], task_dir=run / "task", view="agent", gpu=host["gpu"],
                             cache_dir=run / "cache", claude_config=None, codex_home=run / "codex_home",
                             extra_env={"FMLBENCH_AGENT": "codex"})
    return ["taskset", "-c", host["cpus"], "timeout", "--foreground", "-s", "TERM", str(int(left)),
            shutil.which("bwrap") or "bwrap", *args, *cmd]


if __name__ == "__main__":
    cc_launch.sandbox_argv = sandbox_argv
    cc_launch.main()
