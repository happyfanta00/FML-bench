#!/usr/bin/env python3
"""Enter the sandbox of a Claude Code run (created by new_run.py).

    scripts/claude_code/launch.sh <run_dir>              # interactive shell in the sandbox
    scripts/claude_code/launch.sh <run_dir> -- CMD ...   # run one command in the sandbox

The first launch starts the wall-clock budget (session_start/deadline in task/meta.json). The
sandbox is pinned to the run's CPU slot and is killed one hour after the deadline (everything in
it, Claude Code included, ends; the reason is printed and logged in <run_dir>/launch.log).
Re-launching (e.g. to resume Claude Code with `cc-start --continue`) keeps the original deadline.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

GRACE = 3600


def sandbox_argv(run: Path, cmd: list[str]) -> list[str]:
    host = json.loads((run / "host.json").read_text())
    meta_p = run / "task" / "meta.json"
    meta = json.loads(meta_p.read_text())
    now = time.time()
    if meta.get("session_start") is None:
        meta["session_start"] = now
        meta["deadline"] = now + meta["budget_hours"] * 3600
        meta["sandbox_close"] = meta["deadline"] + GRACE
        meta_p.write_text(json.dumps(meta, indent=1))
        print(f"[launch] budget started: deadline {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta['deadline']))}")
    left = meta["deadline"] + GRACE - now
    if left <= 0:
        raise SystemExit("[launch] this run's deadline (plus 1 h grace) has passed")
    args = common.bwrap_args(task=host["task"], task_dir=run / "task", view="agent", gpu=host["gpu"],
                             cache_dir=run / "cache", claude_config=run / "claude_config",
                             extra_env={"CLAUDE_CODE_PROMPT_CACHE_TTL": host.get("prompt_cache_ttl", "5m")})
    return ["taskset", "-c", host["cpus"], "timeout", "--foreground", "-s", "TERM", str(int(left)),
            shutil.which("bwrap") or "bwrap", *args, *cmd]


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run = Path(sys.argv[1]).resolve()
    rest = sys.argv[2:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    cmd = rest or ["bash", "--rcfile", "/opt/fml/bashrc", "-i"]
    argv = sandbox_argv(run, cmd)
    log = run / "launch.log"

    def note(msg):
        with open(log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    note(f"start {' '.join(shlex.quote(c) for c in cmd)}")
    # Wait for the sandbox instead of exec'ing it, so that its end can be reported. Ctrl-C is
    # meant for the programs in the sandbox (same terminal), not for this wrapper.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    p = subprocess.Popen(argv, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    rc = p.wait()
    if rc == 124:
        msg = "sandbox closed: the run's deadline + 1 h grace was reached (killed by timeout)"
    elif rc < 0:
        msg = f"sandbox ended by signal {-rc}"
    else:
        msg = f"sandbox exited with code {rc}"
    note(msg)
    print(f"[launch] {msg}", file=sys.stderr)
    sys.exit(rc if rc >= 0 else 128 - rc)


if __name__ == "__main__":
    main()
