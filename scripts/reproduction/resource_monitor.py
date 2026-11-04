#!/usr/bin/env python3
"""Sample per-task GPU memory / host RSS of benchmark runs.

Usage (from repo root):
    python3 scripts/reproduction/resource_monitor.py OUT.csv [INTERVAL_SEC]
    python3 scripts/reproduction/resource_monitor.py --summary OUT.csv

Processes are attributed to a task by their working directory (any workspace copy workspace/<TASK>__...),
so runs inside sandbox/run_in_bwrap.sh are tracked too. Rows:
    ts, kind=app,  task, gpu index, pid, used MiB         (one per GPU process)
    ts, kind=gpu,  -,    gpu index, -,   used MiB, util %  (one per GPU)
    ts, kind=rss,  task, -,         -,   summed PSS MiB    (all processes of a task)
"""
import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict


def task_of(pid):
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None
    m = re.search(r"/workspace/([^/]+?)__", cwd)
    return m.group(1) if m else None


def smi(query, fields):
    out = subprocess.run(["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def sample(writer):
    ts = int(time.time())
    uuid2idx = {u: i for u, i in smi("gpu", "uuid,index")}
    for i, used, util in smi("gpu", "index,memory.used,utilization.gpu"):
        writer.writerow([ts, "gpu", "-", i, "-", used, util])
    for uuid, pid, used in smi("compute-apps", "gpu_uuid,pid,used_memory"):
        writer.writerow([ts, "app", task_of(pid) or "?", uuid2idx.get(uuid, "?"), pid, used, ""])
    rss = defaultdict(int)
    for pid in filter(str.isdigit, os.listdir("/proc")):
        t = task_of(pid)
        if not t:
            continue
        try:
            with open(f"/proc/{pid}/smaps_rollup") as f:  # PSS: shared pages split across sharers
                rss[t] += next(int(l.split()[1]) for l in f if l.startswith("Pss:")) * 1024
        except OSError:
            pass
    for t, b in rss.items():
        writer.writerow([ts, "rss", t, "-", "-", b // 2**20, ""])


def summary(path):
    peak_gpu = defaultdict(int)   # (task, gpu) -> peak MiB summed over that task's processes
    peak_rss = defaultdict(int)
    peak_card = defaultdict(int)
    per_ts = defaultdict(int)
    for ts, kind, task, gpu, _pid, used, _util in csv.reader(open(path)):
        if kind == "app":
            per_ts[(ts, task, gpu)] += int(used)
        elif kind == "rss":
            peak_rss[task] = max(peak_rss[task], int(used))
        elif kind == "gpu":
            peak_card[gpu] = max(peak_card[gpu], int(used))
    for (_ts, task, gpu), v in per_ts.items():
        peak_gpu[(task, gpu)] = max(peak_gpu[(task, gpu)], v)
    print("task, gpu, peak GPU MiB")
    for (t, g), v in sorted(peak_gpu.items()):
        print(f"{t}, {g}, {v}")
    print("\ntask, peak host PSS MiB")
    for t, v in sorted(peak_rss.items()):
        print(f"{t}, {v}")
    print("\ngpu, peak card MiB (all processes)")
    for g, v in sorted(peak_card.items()):
        print(f"{g}, {v}")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2])
        sys.exit()
    out, interval = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        while True:
            sample(w)
            f.flush()
            time.sleep(interval)
