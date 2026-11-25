#!/usr/bin/env python3
"""Final test, audit and harness-format results for a finished Codex run.

    python3 scripts/codex/finalize.py <run_dir> [--gpu N] [--cpus SLOT] [--force] [--skip-best-val]

Same as scripts/claude_code/finalize.py (final test of the submitted run and of the best-val run on
the real data, audit, harness layout), with the Claude Code transcript parser replaced by a parser
of the Codex rollouts in <run_dir>/codex_home/sessions/**/rollout-*.jsonl:
  - shell commands: exec_command calls (duration and exit code from the CommandExecution items);
    text typed into running sessions (write_stdin with non-empty chars) is audited as a command too;
  - other tools: apply_patch, goal tools, sub-agent tools; web_search calls are reported as WebSearch;
  - tokens: per-response usage (token_usage_record), mapped to the Claude Code keys
    (input = uncached input, cache_creation = cache writes, cache_read = cached input);
  - goal history: thread_goal_updated events (status, tokens and time used).
No USD cost: there are no reliable Bedrock prices for the OpenAI models here; tokens are reported.
Results: benchmark_results/codex/<TASK>/<run_id>/ and <run_dir>/final/report.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "claude_code"))
import common  # noqa: E402
import finalize as cc  # noqa: E402

RESULTS = common.REPO / "benchmark_results" / "codex"
RE_BACKGROUND = re.compile(r"(^|[^&])&\s*($|;|\n)|\bnohup\b|\bsetsid\b|\bdisown\b")


def _dur(d):
    return None if not isinstance(d, dict) else round(d.get("secs", 0) + d.get("nanos", 0) / 1e9, 1)


def parse_rollouts(codex_home: Path) -> dict:
    files = sorted(codex_home.glob("sessions/**/*.jsonl"))
    calls, done, bash_extra, other_tools, goals, goal_calls = {}, {}, [], [], [], {}
    tok = {"input": 0, "cache_creation": 0, "cache_creation_1h": 0, "cache_read": 0, "output": 0}
    reasoning, tok_by_model, sessions, first, last = 0, {}, [], None, None
    for f in files:
        sub = False
        model = None
        for line in f.open(errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ts = cc.parse_ts(d.get("timestamp"))
            if ts:
                first = ts if first is None else min(first, ts)
                last = ts if last is None else max(last, ts)
            t, p = d.get("type"), d.get("payload") or {}
            st = p.get("type")
            if t == "session_meta":
                sessions.append(p.get("id") or f.stem)
                sub = p.get("thread_source") not in (None, "user") or p.get("source") not in (None, "cli", "exec")
            elif t == "event_msg" and st == "thread_settings_applied":
                model = (p.get("thread_settings") or {}).get("model") or model
            elif t == "response_item" and st in ("function_call", "custom_tool_call"):
                name = p.get("name")
                raw = p.get("arguments") if st == "function_call" else p.get("input")
                try:
                    args = json.loads(raw) if isinstance(raw, str) and st == "function_call" else (raw or {})
                except ValueError:
                    args = {"raw": raw}
                if name == "exec_command":
                    calls[p.get("call_id")] = {"ts": ts, "command": args.get("cmd", ""),
                                               "background": bool(RE_BACKGROUND.search(args.get("cmd", ""))),
                                               "sidechain": sub, "file": f.name}
                elif name == "write_stdin":
                    chars = args.get("chars") or ""
                    if chars.strip():   # input typed into a running session (e.g. an interactive shell)
                        bash_extra.append({"ts": ts, "command": chars, "background": False,
                                           "sidechain": sub, "file": f.name, "via_stdin": True,
                                           "duration_s": None, "error": None, "result_tail": ""})
                else:
                    if name in ("create_goal", "update_goal"):
                        goal_calls[p.get("call_id")] = name
                    other_tools.append({"ts": ts, "tool": name,
                                        "input": (raw if isinstance(raw, str) else json.dumps(raw))[:300]})
            elif t == "response_item" and st == "function_call_output" and p.get("call_id") in goal_calls:
                # The goal tools return the goal after the change (update_goal(complete) has no event).
                try:
                    g = (json.loads(p.get("output") or "{}") or {}).get("goal") or {}
                except (ValueError, AttributeError):
                    g = {}
                goals.append({"ts": ts, "via": goal_calls[p["call_id"]], "status": g.get("status"),
                              "objective": (g.get("objective") or "")[:120],
                              "tokens_used": g.get("tokensUsed"), "time_used_s": g.get("timeUsedSeconds")})
            elif t == "response_item" and st == "web_search_call":
                other_tools.append({"ts": ts, "tool": "WebSearch", "input": json.dumps(p.get("action"))[:300]})
            elif t == "event_msg" and st == "item_completed":
                it = p.get("item") or {}
                if it.get("type") == "CommandExecution":
                    done[it.get("id")] = {"ts": ts, "duration_s": _dur(it.get("duration")),
                                          "exit_code": it.get("exit_code"), "status": it.get("status"),
                                          "text": (it.get("aggregated_output") or "")[-400:]}
            elif t == "event_msg" and st == "thread_goal_updated":
                g = p.get("goal") or {}
                goals.append({"ts": ts, "via": "event", "status": g.get("status"), "objective": (g.get("objective") or "")[:120],
                              "tokens_used": g.get("tokensUsed"), "time_used_s": g.get("timeUsedSeconds")})
            elif t == "token_usage_record":
                u = p.get("usage") or {}
                cached, cw = u.get("cached_input_tokens") or 0, u.get("cache_write_input_tokens") or 0
                part = {"input": max((u.get("input_tokens") or 0) - cached - cw, 0), "cache_creation": cw,
                        "cache_creation_1h": 0, "cache_read": cached, "output": u.get("output_tokens") or 0}
                reasoning += u.get("reasoning_output_tokens") or 0
                agg = tok_by_model.setdefault(model or "?", dict.fromkeys(part, 0))
                for k, v in part.items():
                    tok[k] += v
                    agg[k] += v
    bash = []
    for cid, c in calls.items():
        r = done.get(cid, {})
        bash.append({**c, "duration_s": r.get("duration_s"),
                     "error": None if r.get("exit_code") is None else r["exit_code"] != 0,
                     "exit_code": r.get("exit_code"), "result_tail": r.get("text", "")})
    bash += bash_extra
    bash.sort(key=lambda x: x["ts"] or 0)
    return {"files": [str(f) for f in files], "bash": bash, "other_tools": other_tools,
            "tokens": tok, "tokens_by_model": tok_by_model, "reasoning_output_tokens": reasoning,
            "cost_usd": None, "unpriced_models": sorted(tok_by_model), "cost_usd_claude_code": None,
            "sessions": sessions, "sessions_without_cost_state": [], "model_usage": {},
            "goals": goals, "first_ts": first, "last_ts": last}


def main():
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    if run is None:
        raise SystemExit(__doc__)
    host = json.loads((run / "host.json").read_text())
    holder = {}

    def parse(_cfg_dir):   # cc.main passes run/claude_config; read the Codex rollouts instead
        holder["T"] = parse_rollouts(run / "codex_home")
        return holder["T"]

    cc.parse_transcripts = parse
    cc.RESULTS = RESULTS
    cc.main()
    T = holder["T"]
    # Codex-specific fields in the harness summary and the report.
    dst = RESULTS / host["task"] / host["run_id"]
    sp = dst / "summary.json"
    s = json.loads(sp.read_text())
    s.update(agent="codex", model=host["model"], provider="AmazonBedrock (Codex amazon-bedrock)",
             cost_usd=None)
    s["agent_params"].pop("claude_code_version", None)
    s["agent_params"].update(codex_version=host.get("codex_version"), reasoning_effort=host.get("reasoning_effort"),
                             baseline_source=host.get("baseline_source"))
    s["token_usage"]["reasoning_output_tokens"] = T["reasoning_output_tokens"]
    s["goal_history"] = T["goals"]
    sp.write_text(json.dumps(s, indent=2))
    rp = run / "final" / "report.json"
    r = json.loads(rp.read_text())
    r["transcripts"].update(goals=T["goals"], reasoning_output_tokens=T["reasoning_output_tokens"])
    rp.write_text(json.dumps(r, indent=1))
    g = T["goals"][-1] if T["goals"] else None
    print(f"  goal: {len(T['goals'])} updates; last status {g and g['status']}, tokens {g and g['tokens_used']}, "
          f"time {g and g['time_used_s']} s  (cost above is n/a: no Bedrock price table for {T['unpriced_models']})")


if __name__ == "__main__":
    main()
