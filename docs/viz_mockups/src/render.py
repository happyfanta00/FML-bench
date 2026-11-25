"""Render the viz design mockups (docs/codex_viz_design.md section 4) to PNG with headless Chromium."""
import json
import re
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = Path(sys.argv[1])
S = json.loads((HERE / "samples.json").read_text())

CSS = """
:root{--bg:#f4f3f0;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e3e2de;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0ca30c;--crit:#d03b3b;--warn:#fab219;--accent:#6250d6;
--seq1:#cde2fb;--seq3:#6da7ec;--seq5:#256abf;--seq7:#0d366b}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 -apple-system,"Helvetica Neue","Droid Sans Fallback",sans-serif}
.win{margin:18px;border-radius:12px;overflow:hidden;box-shadow:0 8px 30px rgba(0,0,0,.12);background:var(--surface);border:1px solid #d8d7d2}
.chrome{height:34px;background:#ecebe7;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 12px;gap:7px}
.dot{width:11px;height:11px;border-radius:50%} .url{margin-left:16px;background:#fff;border-radius:6px;padding:3px 12px;color:var(--ink2);font-size:12px;width:360px}
.top{display:flex;align-items:center;gap:18px;padding:10px 18px;border-bottom:1px solid var(--line)}
.title{font-weight:650;font-size:15px} .pill{border:1px solid var(--line);border-radius:999px;padding:2px 10px;color:var(--ink2);font-size:12px;background:#fff}
.btn{border-radius:8px;padding:7px 14px;font-weight:600;font-size:13px;border:1px solid var(--line);background:#fff;color:var(--ink)}
.btn.primary{background:var(--s1);border-color:var(--s1);color:#fff}
.grid{display:grid;grid-template-columns:46% 54%;min-height:700px} .col{padding:14px 16px} .col+.col{border-left:1px solid var(--line)}
.card{min-width:0;border:1px solid var(--line);border-radius:10px;background:#fff;padding:12px 14px;margin-bottom:12px}
.h{font-weight:650;font-size:13px;margin-bottom:8px;display:flex;align-items:center;gap:8px} .sub{color:var(--ink2);font-size:12px}
.muted{color:var(--muted)} .tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px}
.tile{border:1px solid var(--line);border-radius:10px;padding:9px 11px;background:#fff} .tile .k{font-size:11px;color:var(--ink2)} .tile .v{font-size:21px;font-weight:650;font-variant-numeric:tabular-nums}
.node{display:grid;grid-template-columns:18px 42px 1fr 104px;align-items:center;gap:6px;padding:6px 8px;border-radius:8px;font-variant-numeric:tabular-nums}
.node.sel{background:#eef4fc;outline:1.5px solid var(--s1)} .node .t{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:8px 0 10px} .tab{padding:6px 12px;color:var(--ink2);border-bottom:2px solid transparent}
.tab.on{color:var(--ink);border-color:var(--ink);font-weight:600}
.kv{display:grid;grid-template-columns:62px 1fr;gap:4px 10px} .kv .k{color:var(--ink2)}
.samples{display:grid;grid-template-columns:repeat(8,1fr);gap:6px}
.s{border-radius:7px;background:#111;padding:3px;text-align:center;color:#fff;font-size:10px;position:relative}
.s img{width:100%;image-rendering:pixelated;display:block;border-radius:4px}
.s.ok{box-shadow:0 0 0 2px var(--good)} .s.bad{box-shadow:0 0 0 2px var(--crit)}
.s .lab{display:flex;justify-content:space-between;padding:2px 2px 0;color:#ddd}
.chip{white-space:nowrap;border:1px solid var(--line);border-radius:999px;padding:2px 9px;font-size:11.5px;color:var(--ink2);background:#fff} .chip.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.bar{display:grid;grid-template-columns:120px 1fr 44px 70px;align-items:center;gap:8px;margin:5px 0;font-variant-numeric:tabular-nums}
.track{height:10px;background:#f0efec;border-radius:5px;overflow:hidden} .fill{height:100%;background:var(--s1);border-radius:0 4px 4px 0}
.annin{display:inline-flex;width:19px;height:19px;border-radius:50%;background:var(--accent);color:#fff;font-size:11px;font-weight:700;align-items:center;justify-content:center;vertical-align:middle;margin:0 4px;flex:none}
.ann{position:absolute;width:20px;height:20px;border-radius:50%;background:var(--accent);color:#fff;font-size:11.5px;font-weight:700;display:flex;align-items:center;justify-content:center;box-shadow:0 0 0 2px #fff}
.legend{display:flex;gap:14px;font-size:11.5px;color:var(--ink2)} .sw{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:5px}
.code{font:11.5px/1.5 "SF Mono",Menlo,monospace;background:#fafaf8;border:1px solid var(--line);border-radius:8px;padding:8px 10px;white-space:pre;overflow:hidden}
.add{background:#e6f4e6;display:block} .del{background:#fbe9e9;display:block}
.notes{padding:10px 18px 16px;border-top:1px solid var(--line);display:grid;grid-template-columns:1fr 1fr;gap:4px 28px;font-size:12px;color:var(--ink2)}
.notes b{color:var(--accent)}
.env{flex:1} .env .row{display:grid;grid-template-columns:repeat(6,92px);gap:8px}
.step{display:flex;gap:10px;align-items:flex-start;margin:6px 0} .num{width:22px;height:22px;border-radius:50%;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;flex:none}
.tag{display:inline-block;font-size:10.5px;padding:0 6px;border-radius:4px;background:#f0efec;color:var(--ink2);white-space:nowrap}
.tag.hot{background:#e7f0fb;color:#1c5cab}
.ev{display:grid;grid-template-columns:52px 58px minmax(0,1fr) 58px;gap:6px;padding:4px 0;border-bottom:1px dashed var(--line);font-variant-numeric:tabular-nums}
"""


def page(body, w=1440, h=900):
    body = re.sub(r"(\S)/>", r"\1 />", body)   # unquoted attr values must not swallow the '/'
    return f"<!doctype html><meta charset=utf-8><style>{CSS}</style><body style='width:{w}px'>{body}</body>"


def chrome(url="localhost:8765"):
    return (f"<div class=chrome><span class=dot style='background:#ff5f57'></span><span class=dot style='background:#febc2e'></span>"
            f"<span class=dot style='background:#28c840'></span><span class=url>{url}</span></div>")


def ann(n, x=None, y=None):
    return f"<span class=annin>{n}</span>"


def tile_html(t, show_pred=True):
    cls = ""
    pred = ""
    if show_pred and "pred" in t:
        ok = t["pred"] == t["label"]
        cls = "ok" if ok else "bad"
        pred = f"<span>p{t['pred']} {'✓' if ok else '✗'}</span>"
    star = "✱" if t["flipped"] else ""
    return (f"<div class='s {cls}'><img src='{t['img']}'><div class=lab><span>y{t['label']}{star}</span>{pred}</div></div>")


# --------------------------------------------------------------------------- charts (inline SVG)

def val_chart(w=560, h=150):
    vals = [0.2626, 0.6837, 0.7315, 0.9057, 0.8761, 0.9051, None]   # run 6 failed
    x0, x1, y0, y1 = 38, w - 12, h - 24, 12
    X = lambda i: x0 + i * (x1 - x0) / 7
    Y = lambda v: y0 - (v - 0.2) / 0.8 * (y0 - y1)
    g = []
    for v in (0.2, 0.4, 0.6, 0.8, 1.0):
        g.append(f"<line x1={x0} x2={x1} y1={Y(v):.1f} y2={Y(v):.1f} stroke='#ecebe7'/>"
                 f"<text x={x0-6} y={Y(v)+4:.1f} font-size=10 fill='#8a8984' text-anchor=end>{v:.1f}</text>")
    for i in range(8):
        g.append(f"<text x={X(i):.1f} y={h-8} font-size=10 fill='#8a8984' text-anchor=middle>{i}</text>")
    best, pts = -1, []
    for i, v in enumerate(vals):
        if v is not None and v > best:
            best = v
        pts.append((X(i), Y(best)))
    path = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}" + "".join(f" H{p[0]:.1f} V{p[1]:.1f}" for p in pts[1:]) + f" H{X(7):.1f}"
    g.append(f"<path d='{path}' fill=none stroke='#52514e' stroke-width=1.5 stroke-dasharray='4 3'/>")
    g.append(f"<line x1={x0} x2={x1} y1={Y(0.75):.1f} y2={Y(0.75):.1f} stroke='#d03b3b' stroke-width=1 stroke-dasharray='2 3'/>"
             f"<text x={x1} y={Y(0.75)-4:.1f} font-size=10 fill='#d03b3b' text-anchor=end>只用形状上限 0.75</text>")
    for i, v in enumerate(vals):
        if v is None:   # failed run: red cross on the x axis
            g.append(f"<text x={X(i):.1f} y={y0+4:.1f} font-size=13 font-weight=700 fill='#d03b3b' text-anchor=middle>✗</text>")
            continue
        prev = [u for u in vals[:i] if u is not None]
        sota = not prev or v > max(prev)
        g.append(f"<circle cx={X(i):.1f} cy={Y(v):.1f} r={5.5 if i == 3 else 4.5} fill='{'#2a78d6' if True else ''}' "
                 f"stroke='#fff' stroke-width=2/>")
        if sota and i in (0, 3):
            g.append(f"<text x={X(i)+8:.1f} y={Y(v)+4:.1f} font-size=10.5 fill='#0b0b0b'>{v:.4f}</text>")
    g.append(f"<circle cx={X(7):.1f} cy={Y(0.55):.1f} r=4.5 fill=none stroke='#2a78d6' stroke-width=2 stroke-dasharray='2 2'/>"
             f"<text x={X(7)-6:.1f} y={Y(0.55)+16:.1f} font-size=10 fill='#8a8984' text-anchor=middle>训练中</text>")
    g.append(f"<line x1={X(3):.1f} x2={X(3):.1f} y1={y1} y2={y0} stroke='#2a78d6' stroke-width=1 opacity=.35/>")
    return f"<svg width={w} height={h}>{''.join(g)}</svg>"


def train_chart(w=330, h=150, kind="acc"):
    import math
    x0, x1, y0, y1 = 34, w - 8, h - 22, 10
    steps = [i * 100 for i in range(33)]   # up to 3200 (live)
    X = lambda s: x0 + s / 5000 * (x1 - x0)
    g = []
    if kind == "loss":
        Y = lambda v: y0 - v / 0.8 * (y0 - y1)
        for v in (0, 0.4, 0.8):
            g.append(f"<line x1={x0} x2={x1} y1={Y(v):.1f} y2={Y(v):.1f} stroke='#ecebe7'/><text x={x0-5} y={Y(v)+4:.1f} font-size=10 fill='#8a8984' text-anchor=end>{v}</text>")
        pts = " ".join(f"{X(s):.1f},{Y(0.22 + 0.48 * math.exp(-s / 500) + 0.01 * math.sin(s / 170)):.1f}" for s in steps)
        g.append(f"<polyline points='{pts}' fill=none stroke='#2a78d6' stroke-width=2/>")
    else:
        Y = lambda v: y0 - v * (y0 - y1)
        for v in (0, 0.5, 1.0):
            g.append(f"<line x1={x0} x2={x1} y1={Y(v):.1f} y2={Y(v):.1f} stroke='#ecebe7'/><text x={x0-5} y={Y(v)+4:.1f} font-size=10 fill='#8a8984' text-anchor=end>{v}</text>")
        for col, f in (("#2a78d6", lambda s: 0.89 - 0.35 * math.exp(-s / 300)),
                       ("#eb6834", lambda s: 0.80 - 0.3 * math.exp(-s / 300)),
                       ("#1baf7a", lambda s: 0.905 - 0.7 * math.exp(-s / 700) + 0.012 * math.sin(s / 130))):
            pts = " ".join(f"{X(s):.1f},{Y(f(s)):.1f}" for s in steps)
            g.append(f"<polyline points='{pts}' fill=none stroke='{col}' stroke-width=2/>")
    g.append(f"<line x1={X(3200):.1f} x2={X(3200):.1f} y1={y1} y2={y0} stroke='#8a8984' stroke-dasharray='2 3'/>")
    for s in (0, 2500, 5000):
        g.append(f"<text x={X(s):.1f} y={h-6} font-size=10 fill='#8a8984' text-anchor=middle>{s}</text>")
    return f"<svg width={w} height={h}>{''.join(g)}</svg>"


def confusion():
    cells = [[3180, 320], [340, 3160]]
    out = "<table style='border-collapse:separate;border-spacing:2px;font-variant-numeric:tabular-nums;font-size:12px'>"
    out += "<tr><td></td><td class=sub style='text-align:center'>预测 0</td><td class=sub style='text-align:center'>预测 1</td></tr>"
    for i, r in enumerate(cells):
        out += f"<tr><td class=sub style='padding-right:6px'>标签 {i}</td>"
        for j, v in enumerate(r):
            bg, fg = ("#256abf", "#fff") if i == j else ("#cde2fb", "#0b0b0b")
            out += f"<td style='background:{bg};color:{fg};width:74px;height:34px;text-align:center;border-radius:4px'>{v:,}</td>"
        out += "</tr>"
    return out + "</table>"


# --------------------------------------------------------------------------- pages

def intro():
    envs = []
    for name, e, desc in (("env0", 0.1, "训练 · 颜色与标签一致 90%"), ("env1", 0.2, "训练 · 一致 80%"),
                          ("env2", 0.9, "测试域 · 一致仅 10%（规则相反）")):
        tiles = "".join(
            f"<div><div class=s><img src='{t['img']}'><div class=lab><span>y{t['label']}{'✱' if t['flipped'] else ''}</span>"
            f"<span>{t['digit']}</span></div></div></div>" for t in S["intro"][name])
        envs.append(f"<div class='card env'><div class=h>{ann(1) if name == 'env0' else ''}{name} <span class=pill>e = {e}</span><span class=sub>{desc}</span></div>"
                    f"<div class=row>{tiles}</div></div>")
    body = f"""<div class=win style='position:relative'>{chrome()}
<div class=top><span class=title>FML-bench · ColoredMNIST · Codex goal</span><span class=pill>run trial_viz_01</span>
<span style='flex:1'></span><span class=sub>1 / 2 任务介绍</span></div>
<div style='padding:18px 26px'>
 <div style='font-size:22px;font-weight:650;margin-bottom:4px'>颜色是“捷径”：训练出能泛化到颜色规则相反的新域的分类器</div>
 <div class=sub style='margin-bottom:16px'>DomainBed · ColoredMNIST · 测试域 env2 · 指标 in_acc_mean（越高越好）</div>
 <div style='display:grid;grid-template-columns:360px 1fr;gap:16px'>
  <div class=card><div class=h>{ann(2)}数据怎么生成</div>
   <div class=step><span class=num>1</span><div>MNIST 数字 → 二分类标签：<b>0–4 → 0，5–9 → 1</b></div></div>
   <div class=step><span class=num>2</span><div>以 <b>25%</b> 概率翻转标签（标签噪声）<div class=sub>只看形状最多对 75%</div></div></div>
   <div class=step><span class=num>3</span><div>按标签上色：<span style='color:#c00;font-weight:700'>红=0</span>、<span style='color:#080;font-weight:700'>绿=1</span>，再以概率 <b>e</b> 翻转颜色</div></div>
   <div class=sub style='margin-top:10px'>每张图下方：<b>y</b>=标签（✱ 表示被噪声翻转），右侧数字为原始数字</div>
  </div>
  <div style='display:flex;flex-direction:column;gap:0'>{''.join(envs)}</div>
 </div>
 <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px'>
  <div class=card><div class=h>评测方式</div><div class=kv>
   <span class=k>训练</span><span>env0、env1（各 70% 训练 / 30% 留出）</span>
   <span class=k>val</span><span>env2 out-split（30%）</span>
   <span class=k>隐藏 test</span><span>env2 in-split（70%），agent 看到的是空白图</span></div></div>
  <div class=card><div class=h>参考水平</div><div class=kv>
   <span class=k>只用颜色</span><span>≈ 0.10（env2 上颜色规则相反）</span>
   <span class=k>只用形状</span><span>上限 0.75</span>
   <span class=k>baseline</span><span>ERM ≈ 0.26（本机实测）</span></div></div>
  <div class=card><div class=h>agent 设置</div><div class=kv>
   <span class=k>agent</span><span>Codex goal · GPT-6 Astra（high）</span>
   <span class=k>预算</span><span>1 小时 · 最多 20 次 fml-val · 1 × L4</span>
   <span class=k>可改文件</span><span>algorithms.py、networks.py</span></div></div>
 </div>
 <div style='display:flex;justify-content:flex-end;gap:10px;margin-top:4px'><span class=btn>换一批样本</span>{ann(3)}<span class='btn primary'>我已了解，进入主页面 →</span></div>
</div>

<div class=notes><div><b>①</b> 每个域 6 张真实样本（由后端用 DomainBed 的代码生成），可以换一批</div>
<div><b>②</b> 用 3 步讲清楚数据怎么生成，这是理解“捷径”和 0.75 上限的关键</div>
<div><b>③</b> 确认后进入主页面；看过一次就不再自动显示（顶部可以重新打开）</div><div></div></div>
</div>"""
    return page(body)


def topbar(state="run"):
    if state == "run":
        right = ("<span class=pill>时间 23m / 60m</span><span class=pill>runs 7 / 20</span><span class=pill>phase SEARCH</span>"
                 "<span class=pill><span style='color:var(--good)'>●</span> goal active</span><span class=pill>tokens 2.1M · cache 96%</span>")
    else:
        right = "<span class=pill>agent 未启动</span><span class=pill>预算尚未开始计时</span>"
    return (f"<div class=top>{ann(1) if state == 'run' else ''}<span class=title>ColoredMNIST · Codex goal</span><span class=pill>run trial_viz_01 ▾</span>{right}"
            f"<span style='flex:1'></span><span class=btn>任务介绍</span><span class=btn>复制启动命令</span></div>")


NODES = [("●", "0000", "baseline · ERM 原版", "0.2626", "base"),
         ("●", "0001", "灰度化 ERM（去掉颜色）", "0.6837", ""),
         ("●", "0002", "+ 仿射增强 / L2 / EMA", "0.7315", ""),
         ("★", "0003", "batch 颜色方向校准", "0.9057", "sel"),
         ("●", "0004", "连续可靠度估计", "0.8761", ""),
         ("●", "0005", "原宽度 CNN + 2×2 池化", "0.9051", ""),
         ("✗", "0006", "SAM 优化器（训练报错）", "失败", "fail"),
         ("◐", "0007", "紧凑 CNN + 标签平滑", "63%", "run")]


def node_list():
    rows = []
    for ic, rid, t, v, k in NODES:
        col = {"★": "#2a78d6", "✗": "#d03b3b", "◐": "#8a8984"}.get(ic, "#52514e")
        extra = ""
        if k == "run":
            v = "<span style='display:inline-block;width:44px;height:6px;background:#f0efec;border-radius:3px;overflow:hidden;vertical-align:middle'><span style='display:block;width:63%;height:100%;background:#2a78d6'></span></span> 63%"
        if k == "base":
            extra = " <span class=pill style='font-size:10.5px;padding:0 6px'>基线</span>"
        if rid == "0003":
            extra = " <span class=pill style='font-size:10.5px;padding:0 6px;color:#2a78d6;border-color:#2a78d6'>当前 SOTA</span>"
        rows.append(f"<div class='node {'sel' if k == 'sel' else ''}'><span style='color:{col};font-weight:700'>{ic}</span>"
                    f"<span class=muted>{rid}</span><span class=t>{t}{extra}</span><span style='text-align:right;{'color:#d03b3b' if k == 'fail' else ''}'>{v}</span></div>")
    return "".join(rows)


def right_panel():
    groups = [("形状✓ 颜色✓", 0.97, "520"), ("形状✓ 颜色✗", 0.92, "4,730"), ("形状✗ 颜色✓", 0.21, "180"), ("形状✗ 颜色✗", 0.88, "1,570")]
    bars = "".join(f"<div class=bar><span>{g}</span><div class=track><div class=fill style='width:{v*100:.0f}%'></div></div>"
                   f"<span style='text-align:right'>{v:.2f}</span><span class=sub>n = {n}</span></div>" for g, v, n in groups)
    tiles = "".join(tile_html(t) for t in S["main"])
    return f"""
<div class=h style='font-size:14px'>{ann(5)}当前 SOTA：run 0003 · batch 颜色方向校准 <span class=pill style='color:var(--good);border-color:var(--good)'>✓ 预测复现 = 记录的 val</span>
<span style='flex:1'></span><span class=sub>☐ 跟随左侧选中的节点</span></div>
<div class=tiles>
 <div class=tile><div class=k>val in_acc（env2 out）</div><div class=v>0.9057</div></div>
 <div class=tile><div class=k>相对 baseline</div><div class=v>+0.6431</div></div>
 <div class=tile><div class=k>env0 留出准确率</div><div class=v>0.8961</div></div>
 <div class=tile><div class=k>env1 留出准确率</div><div class=v>0.8057</div></div>
</div>
<div style='display:grid;grid-template-columns:1fr 250px;gap:12px'>
 <div class=card style='margin:0'><div class=h>{ann(6)}捷径诊断 <span class=sub>全部 val 样本（N = 7,000）分 4 组的准确率</span></div>{bars}
  <div class=sub style='margin-top:6px'>形状✓ = 标签未被噪声翻转 · 颜色✓ = 颜色与标签一致　|　预测与<b>颜色</b>一致 12% · 与<b>形状</b>一致 81%</div></div>
 <div class=card style='margin:0'><div class=h>混淆矩阵 <span class=sub>val</span></div>{confusion()}</div>
</div>
<div class=card style='margin-top:12px'><div class=h>{ann(7)}样本预测 <span class=sub>固定 48 个 val 样本，所有节点相同（4 组 × 12，草图里只画了 32 个）</span></div>
 <div style='display:flex;gap:6px;margin-bottom:8px'><span class='chip on'>全部</span><span class=chip>形状✓颜色✓</span><span class=chip>形状✓颜色✗</span><span class=chip>形状✗颜色✓</span><span class=chip>形状✗颜色✗</span>
 </div><div class=sub style='margin-bottom:8px'><span style='color:var(--good)'>■</span> ✓ 预测正确　<span style='color:var(--crit)'>■</span> ✗ 预测错误　y=标签　p=预测　✱=标签被翻转</div>
 <div class=samples>{tiles}</div>
 <div style='position:relative'><div class=card style='position:absolute;left:290px;top:-150px;width:220px;box-shadow:0 6px 18px rgba(0,0,0,.15);font-size:12px'>
 <b>样本 #4127</b>（悬停提示）<br>原始数字 7 · 标签 1（未翻转）<br>颜色 红 · 预测 1 · p = 0.93 ✓</div></div>
</div>"""


def main_page():
    detail = """
<div class=tabs><span class='tab on'>思路</span><span class=tab>代码</span><span class=tab>训练</span><span class=tab>过程</span><span class=tab>结论</span></div>
<div class=kv>
 <span class=k>标题</span><span><b>Source-calibrated color evidence + batch orientation</b></span>
 <span class=k>思路</span><span>形状模型在任何域上都约 75% 正确；用它在每个未标注 batch 里估计颜色和标签相关的方向，再按训练域学到的强度加上颜色证据。</span>
 <span class=k>预期</span><span>val 明显超过 0.75</span>
 <span class=k>结果</span><span><b>0.9057</b>　+0.1742 vs 上一个 SOTA　<span style='color:#2a78d6;font-weight:600'>★ 新 SOTA</span></span>
 <span class=k>运行</span><span>2m53s · 改了 algorithms.py（+118 −3）· 本节点 token 0.31M</span>
 <span class=k>结论</span><span class=sub>“颜色方向校准有效；下一步检查对 batch 组成的依赖”（agent 用 fml-note 写的）</span>
</div>"""
    body = f"""<div class=win style='position:relative'>{chrome()}{topbar()}
<div class=grid><div class=col>
 <div class=card><div class=h>{ann(2)}val 曲线 <span class=sub>每个点是一次 fml-val；虚线阶梯 = SOTA</span></div>{val_chart()}</div>
 <div class=card style='padding:6px'><div class=h style='margin:4px 6px'>{ann(3)}研究节点 <span class=sub>每次 fml-val 一个节点</span></div>{node_list()}</div>
 <div class=card><div class=h>{ann(4)}选中节点：run 0003</div>{detail}</div>
</div><div class=col>{right_panel()}</div></div>

<div class=notes>
<div><b>①</b> 顶栏：预算、次数、阶段、goal 状态、token，实时更新；可以切换查看其他运行</div>
<div><b>⑤</b> 右侧默认显示当前 SOTA；勾选后改为显示左侧选中的节点，方便对比</div>
<div><b>②</b> val 曲线：点击某个点就选中对应节点；红色虚线是只用形状的上限 0.75</div>
<div><b>⑥</b> 捷径诊断：按“标签是否被翻转 × 颜色是否一致”分组，看模型用的是形状还是颜色</div>
<div><b>③</b> 节点：★ 刷新 SOTA · ◐ 训练中（进度）· ✗ 失败 · 「提交」标记 = 已提交</div>
<div><b>⑦</b> 固定的一批 val 样本：图像、标签、预测，绿框 ✓ 正确，红框 ✗ 错误；可按组筛选</div>
<div><b>④</b> 节点详情：思路、代码、训练、过程、结论 5 个标签页（见图 3）</div><div></div>
</div></div>"""
    return page(body)


def detail_page():
    code = ("<span class=muted>@@ class ERM(Algorithm):  __init__</span>\n"
            "<span class=del>-        self.network = nn.Sequential(self.featurizer, self.classifier)</span>"
            "<span class=add>+        self.color_invariant = tuple(input_shape) == (2, 28, 28) and num_classes == 2</span>"
            "<span class=add>+        if self.color_invariant:</span>"
            "<span class=add>+            self.network_ema = copy.deepcopy(self.network).requires_grad_(False)</span>"
            "<span class=add>+            self.register_buffer(\"color_reliability\", torch.tensor(0.0))</span>"
            "<span class=muted>@@ def predict(self, x):</span>\n"
            "<span class=add>+            orientation = ((score - score.mean()) * (color - color.mean())).mean().sign()</span>"
            "<span class=add>+            evidence = torch.log1p(reliability) - torch.log1p(-reliability)</span>")
    events = [("11:25:01", "<span class=tag>发言</span>", "“第 2 次已到 0.73，接近只用形状的上限；试试用 batch 统计量推断颜色方向……”", ""),
              ("11:25:20", "<span class=tag>命令</span>", "python -c \"…形状检查…\"", "0.8s ✓"),
              ("11:25:44", "<span class=tag>补丁</span>", "apply_patch　domainbed/algorithms.py　+96 −2　<span class=muted>[展开]</span>", ""),
              ("11:26:05", "<span class=tag>思考</span>", "<span class=sub>思考摘要：“单样本时协方差为 0，要退回只用形状……”</span>", ""),
              ("11:26:10", "<span class=tag>命令</span>", "git diff --stat", "0.0s ✓"),
              ("11:26:12", "<span class='tag hot'>fml-val</span>", "<b>fml-val --title “Source-calibrated color…”</b> → run 0003", "2m53s ✓")]
    ev = "".join(f"<div class=ev><span class=muted>{a}</span><span>{b}</span><span>{c}</span><span style='text-align:right' class=sub>{d}</span></div>"
                 for a, b, c, d in events)
    legend = ("<div class=legend><span><span class=sw style='background:#2a78d6'></span>env0 留出</span>"
              "<span><span class=sw style='background:#eb6834'></span>env1 留出</span>"
              "<span><span class=sw style='background:#1baf7a'></span>env2 out（val）</span></div>")
    body = f"""<div class=win style='position:relative'>{chrome()}
<div class=top><span class=title>节点详情的另外 3 个标签页（都在左栏下方）</span></div>
<div style='display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr) minmax(0,1fr);gap:14px;padding:14px 16px'>
 <div class=card><div class=tabs>{ann(1)}<span class=tab>思路</span><span class='tab on'>代码</span><span class=tab>训练</span><span class=tab>过程</span></div>
  <div style='display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap'><span class=sub>对比：</span><span class='chip on'>baseline</span><span class=chip>上一个节点</span><span class=chip>任选 0002 ▾</span>
  <span style='flex:1'></span><span class=chip>algorithms.py ▾</span></div>
  <div class=code>{code}</div>
  <div class=sub style='margin-top:6px'>只改了 1 个文件：+118 −3　·　[完整文件]　[复制]</div></div>
 <div class=card><div class=tabs>{ann(2)}<span class=tab>思路</span><span class=tab>代码</span><span class='tab on'>训练</span><span class=tab>过程</span></div>
  <div style='display:flex;justify-content:space-between' class=sub><span>step 3,200 / 5,000　·　预计剩余 1m02s</span><span>GPU 41%</span></div>
  <div class=track style='height:8px;margin:5px 0 10px'><div class=fill style='width:64%'></div></div>
  <div class=sub>loss</div>{train_chart(kind='loss', h=110)}
  <div class=sub style='margin-top:4px'>留出准确率（env2 in 已置空，不显示）</div>{train_chart(kind='acc', h=130)}{legend}
  <div class=code style='margin-top:8px;font-size:10.5px'>step   loss    env0_out  env1_out  env2_out
3100   0.231   0.8942    0.8011    0.9031
3200   0.228   0.8955    0.8027    0.9044</div></div>
 <div class=card><div class=tabs>{ann(3)}<span class=tab>思路</span><span class=tab>代码</span><span class=tab>训练</span><span class='tab on'>过程</span></div>
  <div class=sub style='margin-bottom:6px'>run 0002 开始之后、run 0003 开始之前，agent 做了什么（来自 Codex 会话记录）</div>
  {ev}
  <div class=sub style='margin-top:8px'>本段 token：输入 0.30M（缓存 97%）· 输出 6.1k（其中推理 3.2k）</div></div>
</div>

<div class=notes><div><b>①</b> 代码：可以对比 baseline、上一个节点或任意节点，统一 diff，带语法高亮</div>
<div><b>②</b> 训练：运行中实时更新进度、曲线和日志尾部（每 100 步一行）</div>
<div><b>③</b> 过程：发言、命令（含耗时和返回码）、补丁、思考摘要（如果开启）和 token，按时间排列</div><div></div></div>
</div>"""
    return page(body, h=760)


def baseline_page():
    body = f"""<div class=win style='position:relative'>{chrome()}{topbar(state='pre')}
<div class=grid style='min-height:430px'><div class=col>
 <div class=card style='padding:6px'>
  <div class='node sel'><span style='color:#8a8984;font-weight:700'>◐</span><span class=muted>0000</span><span class=t>baseline · ERM 原版　<span class=sub>训练中</span></span>
  <span style='text-align:right'>41%</span></div></div>
 <div class=card><div class=h>{ann(2)}baseline 训练进度</div>
  <div class=track style='height:8px;margin:4px 0 8px'><div class=fill style='width:41%'></div></div>
  <div class=sub>step 2,050 / 5,000 · 预计剩余 1m30s</div>{train_chart(kind='acc', w=560, h=120)}</div>
 <div class=card style='border-style:dashed;color:var(--ink2)'>agent 尚未启动：baseline 完成后，这里会显示启动命令。在此之前 <code>launch.sh</code> 会拒绝启动 agent。</div>
</div><div class=col>
 <div class=h style='font-size:14px'>当前 SOTA：无</div>
 <div class=card style='text-align:center;padding:28px'>
  <div style='font-size:15px;font-weight:600;margin-bottom:6px'>还没有 SOTA 模型</div>
  <div class=sub style='margin-bottom:16px'>先在和 agent 相同的沙盒里复现 baseline：它会成为第一个 SOTA，同时检查环境是否可用</div>
  {ann(1)}<span class='btn primary' style='opacity:.45'>▶ 运行 baseline（约 3 分钟）</span>
  <div style='display:flex;justify-content:center;gap:14px;margin-top:18px;font-size:12.5px'>
   <span><span style='color:var(--good)'>✓</span> 沙盒</span><span><span style='color:var(--good)'>✓</span> GPU</span><span><span style='color:var(--good)'>✓</span> 掩码数据</span>
   <span><span style='color:var(--s1)'>◐</span> 训练</span><span class=muted>○ 预测</span><span class=muted>○ 与参考值 0.2626 比较</span></div>
 </div>
 <div class=card style='background:#f3f9f3;border-color:#bfe3bf'><div class=h><span style='color:var(--good)'>✓</span> 完成后（S2）的样子</div>
  baseline 0.2626（与参考值逐字节相同）。右侧开始显示 baseline 的预测；顶部出现：<br>
  <div class=code style='margin-top:6px'>tmux new-window -n codex "… scripts/codex/launch.sh ~/fml_agent_runs/…/trial_viz_01"
→ cx-start → 粘贴 /task/goal.txt 里的 /goal …                              [复制]</div></div>
</div></div>

<div class=notes><div><b>①</b> 初始状态（S0）只有这个按钮；点击后按钮变灰，下方逐项显示环境检查结果</div>
<div><b>②</b> baseline 也是节点 0000：训练进度和曲线的显示方式和 agent 的节点完全相同</div></div>
</div>"""
    return page(body, h=640)


PAGES = {"1_intro": (intro, 900), "2_main": (main_page, 900), "3_node_detail": (detail_page, 760), "4_baseline": (baseline_page, 640)}
OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, (fn, h) in PAGES.items():
        html = fn()
        (OUT / f"{name}.html").write_text(html)
        pg = b.new_page(viewport={"width": 1476, "height": h}, device_scale_factor=1.5)
        pg.set_content(html, wait_until="networkidle")
        pg.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        pg.close()
        print("rendered", name)
    b.close()
