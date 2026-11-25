"""Index of the Codex rollouts of a run: a time-ordered event list for the process view.

Events (dicts with ts, kind, ...):
  user      a user message (e.g. the /goal objective)
  message   an agent message (phase commentary / final_answer)
  reasoning a reasoning summary (model_reasoning_summary = "auto")
  command   an exec_command call: cmd, duration_s, exit_code, output tail
  stdin     text typed into a running command session (write_stdin with non-empty chars)
  patch     apply_patch: files with +/- line counts, the patch text
  tool      any other tool call (goal tools, sub-agents, ...)
  goal      goal status change
Token usage per response is kept separately (ts, input, cached, output, reasoning) so that any time
window can be summed. Parsing is cached on the files' sizes and mtimes.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


def _ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def _patch_files(text: str) -> list[dict]:
    files, cur = [], None
    for line in (text or "").splitlines():
        m = re.match(r"\*\*\* (Update|Add|Delete) File: (.+)", line)
        if m:
            cur = {"op": m.group(1).lower(), "file": m.group(2).strip(), "add": 0, "del": 0}
            files.append(cur)
        elif cur and line.startswith("+") and not line.startswith("+++"):
            cur["add"] += 1
        elif cur and line.startswith("-") and not line.startswith("---"):
            cur["del"] += 1
    return files


class RolloutIndex:
    def __init__(self, codex_home: Path):
        self.root = codex_home / "sessions"
        self._key = None
        self.events: list[dict] = []
        self.usage: list[dict] = []
        self.goals: list[dict] = []

    def refresh(self) -> "RolloutIndex":
        files = sorted(self.root.glob("**/*.jsonl")) if self.root.exists() else []
        key = tuple((str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files)
        if key == self._key:
            return self
        events, usage, goals = [], [], []
        for f in files:
            calls, goal_calls = {}, {}
            for line in f.open(errors="replace"):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                ts = _ts(d.get("timestamp"))
                t, p = d.get("type"), d.get("payload") or {}
                st = p.get("type")
                if t == "token_usage_record":
                    u = p.get("usage") or {}
                    usage.append({"ts": ts, "input": u.get("input_tokens") or 0,
                                  "cached": u.get("cached_input_tokens") or 0,
                                  "output": u.get("output_tokens") or 0,
                                  "reasoning": u.get("reasoning_output_tokens") or 0})
                elif t == "response_item" and st in ("function_call", "custom_tool_call"):
                    name = p.get("name")
                    raw = p.get("arguments") if st == "function_call" else p.get("input")
                    try:
                        args = json.loads(raw) if (st == "function_call" and isinstance(raw, str)) else {}
                    except ValueError:
                        args = {}
                    cid = p.get("call_id")
                    if name == "exec_command":
                        ev = {"ts": ts, "kind": "command", "cmd": args.get("cmd", ""), "duration_s": None,
                              "exit_code": None, "output": ""}
                        calls[cid] = ev
                        events.append(ev)
                    elif name == "write_stdin":
                        if (args.get("chars") or "").strip():
                            events.append({"ts": ts, "kind": "stdin", "text": args["chars"][:2000]})
                    elif name == "apply_patch":
                        text = raw if isinstance(raw, str) and st == "custom_tool_call" else args.get("input", raw or "")
                        events.append({"ts": ts, "kind": "patch", "files": _patch_files(text),
                                       "patch": (text or "")[:20000]})
                    else:
                        if name in ("create_goal", "update_goal"):
                            goal_calls[cid] = name
                        events.append({"ts": ts, "kind": "tool", "name": name,
                                       "args": (raw if isinstance(raw, str) else json.dumps(raw))[:500]})
                elif t == "response_item" and st == "function_call_output" and p.get("call_id") in goal_calls:
                    try:
                        g = (json.loads(p.get("output") or "{}") or {}).get("goal") or {}
                    except (ValueError, AttributeError):
                        g = {}
                    goals.append({"ts": ts, "status": g.get("status"), "tokens_used": g.get("tokensUsed"),
                                  "time_used_s": g.get("timeUsedSeconds")})
                elif t == "event_msg" and st == "thread_goal_updated":
                    g = p.get("goal") or {}
                    goals.append({"ts": ts, "status": g.get("status"), "tokens_used": g.get("tokensUsed"),
                                  "time_used_s": g.get("timeUsedSeconds")})
                    events.append({"ts": ts, "kind": "goal", "status": g.get("status"),
                                   "objective": (g.get("objective") or "")[:300]})
                elif t == "event_msg" and st == "item_completed":
                    it = p.get("item") or {}
                    typ = it.get("type")
                    if typ == "CommandExecution" and it.get("id") in calls:
                        dur = it.get("duration") or {}
                        calls[it["id"]].update(
                            duration_s=round(dur.get("secs", 0) + dur.get("nanos", 0) / 1e9, 1) if dur else None,
                            exit_code=it.get("exit_code"), output=(it.get("aggregated_output") or "")[-1500:])
                    elif typ == "AgentMessage":
                        text = "".join(c.get("text", "") for c in it.get("content") or [] if isinstance(c, dict))
                        if text.strip():
                            events.append({"ts": ts, "kind": "message", "phase": it.get("phase"), "text": text})
                    elif typ == "UserMessage":
                        text = "".join(c.get("text", "") for c in it.get("content") or [] if isinstance(c, dict))
                        if text.strip():
                            events.append({"ts": ts, "kind": "user", "text": text[:2000]})
                    elif typ == "Reasoning":
                        summ = [s for s in it.get("summary_text") or [] if s.strip()]
                        if summ:
                            events.append({"ts": ts, "kind": "reasoning", "text": "\n".join(summ)})
        events.sort(key=lambda e: e["ts"] or 0)
        usage.sort(key=lambda u: u["ts"] or 0)
        goals.sort(key=lambda g: g["ts"] or 0)
        self.events, self.usage, self.goals, self._key = events, usage, goals, key
        return self

    def tokens(self, t0=None, t1=None) -> dict:
        tot = {"input": 0, "cached": 0, "output": 0, "reasoning": 0, "responses": 0}
        for u in self.usage:
            if (t0 is None or (u["ts"] or 0) > t0) and (t1 is None or (u["ts"] or 0) <= t1):
                for k in ("input", "cached", "output", "reasoning"):
                    tot[k] += u[k]
                tot["responses"] += 1
        return tot

    def window(self, t0=None, t1=None) -> list[dict]:
        return [e for e in self.events
                if (t0 is None or (e["ts"] or 0) > t0) and (t1 is None or (e["ts"] or 0) <= t1)]

    def goal_status(self):
        return self.goals[-1] if self.goals else None
