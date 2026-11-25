// Codex × ColoredMNIST visualization (docs/codex_viz_design.md). Vanilla JS, no build step.
// Everything that comes from the agent (titles, ideas, commands, patches, messages) is escaped.

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f4 = (v) => (v == null ? "—" : Number(v).toFixed(4));
const pct = (v) => (v == null ? "—" : (100 * v).toFixed(0) + "%");
const signed = (v) => (v == null ? "" : (v >= 0 ? "+" : "") + Number(v).toFixed(4));
// Node completion time in UTC+8 (Asia/Shanghai), whatever the browser's time zone.
const CST = new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
const cstTime = (t) => { if (!t) return ""; const p = Object.fromEntries(CST.formatToParts(new Date(t * 1000)).map((x) => [x.type, x.value])); return `${p.hour}:${p.minute}:${p.second}`; };
const cstFull = (t) => { if (!t) return ""; const p = Object.fromEntries(CST.formatToParts(new Date(t * 1000)).map((x) => [x.type, x.value])); return `${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}（东8区）`; };
const hms = (t) => (t ? new Date(t * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : "");
const dur = (s) => { if (s == null) return ""; s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60; return h ? `${h}h${String(m).padStart(2, "0")}m` : `${m}m${String(r).padStart(2, "0")}s`; };
const ktok = (n) => (n == null ? "—" : n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n));
const api = async (p, opt) => { const r = await fetch(p, opt); if (!r.ok) throw new Error((await r.text()) || r.status); return r.json(); };
const GROUP_LABEL = { A: "形状✓ 颜色✓", B: "形状✓ 颜色✗", C: "形状✗ 颜色✓", D: "形状✗ 颜色✗" };
const COLOR_NAME = ["红", "绿"];

const S = {
  st: null, samples: null, intro: null, introOffset: 0, view: "main",
  sel: null, tab: "idea", follow: false, filter: "all",
  diffAgainst: "baseline", diffFile: null, diffFull: false,
  pred: {}, node: {}, lastFp: null,
};

// ------------------------------------------------------------------------------------------ boot

async function refresh() {
  try { S.st = await api("/api/state"); S.st._t0 = Date.now() / 1000; } catch (e) { $("#app").innerHTML = `<div class="card">服务不可用：${esc(e.message)}</div>`; return; }
  if (!S.st.exists) { renderNoRun(); return; }
  if (!S.samples) { try { S.samples = await api("/api/samples"); } catch { /* prepared on reset */ } }
  const key = `viz-intro-seen:${S.st.run_id}:${S.st.created}`;
  if (S.view === "main" && !localStorage.getItem(key) && !S._introShown) { S.view = "intro"; S._introShown = true; }
  if (S.sel == null || !S.st.nodes.find((n) => n.id === S.sel) && S.sel !== "next") S.sel = S.st.sota || (S.st.nodes[0] || {}).id || null;
  render();
}

function render() {
  renderTop();
  if (S.view === "intro") renderIntro(); else renderMain();
}

function connect() {
  const es = new EventSource("/api/events");
  es.addEventListener("state", () => refresh());
  es.onerror = () => { /* the browser reconnects */ };
  setInterval(() => { if (S.st && S.st.budget) renderTop(); }, 1000);
}

// ------------------------------------------------------------------------------------------ top bar

function renderTop() {
  const st = S.st;
  if (!st || !st.exists) { $("#top").innerHTML = `<span class="title">ColoredMNIST · Codex goal</span>`; return; }
  const b = st.budget;
  let pills = `<span class="pill">run ${esc(st.run_id)}</span>`;
  if (b) {
    const el = (Date.now() / 1000) - (st._t0 || 0);
    pills += `<span class="pill">时间 ${dur(b.elapsed_s + el)} / ${dur(st.budget_hours * 3600)}</span>`
      + `<span class="pill">runs ${b.runs} / ${st.max_runs}</span><span class="pill">phase ${esc(b.phase)}</span>`;
  } else pills += `<span class="pill">${st.agent.started ? "" : "agent 未启动 · 预算尚未开始计时"}</span>`;
  if (st.goal) {
    const g = st.goal.status;
    const c = g === "complete" ? "good" : g === "active" ? "good" : "warnc";
    pills += `<span class="pill"><span class="${c}">●</span> goal ${esc(g)}</span>`;
  }
  if (st.agent.running) pills += `<span class="pill"><span class="good">●</span> agent 沙盒运行中</span>`;
  if (st.tokens && st.tokens.responses) pills += `<span class="pill">tokens ${ktok(st.tokens.input + st.tokens.output)} · cache ${pct(st.tokens.cached / Math.max(st.tokens.input, 1))}</span>`;
  const baseOk = st.baseline_job && st.baseline_job.state === "ok";
  $("#top").innerHTML = `<span class="title">ColoredMNIST · Codex goal</span>${pills}<span class="spacer"></span>
    ${st.final_report ? `<button class="btn" id="b-report">最终报告</button>` : ""}
    <button class="btn" id="b-intro">任务介绍</button>
    <button class="btn" id="b-copy" ${baseOk ? "" : "disabled"} title="复制启动 agent 的命令">复制启动命令</button>
    <button class="btn danger" id="b-reset" title="删除当前运行并新建一次（清空历史）">新建运行</button>`;
  $("#b-intro").onclick = () => { S.view = "intro"; render(); };
  $("#b-copy").onclick = () => copy(st.launch_cmd, "已复制启动命令");
  $("#b-reset").onclick = resetRun;
  if ($("#b-report")) $("#b-report").onclick = showReport;
}

async function copy(text, msg) {
  try { await navigator.clipboard.writeText(text); toast(msg); } catch { prompt("复制下面的内容：", text); }
}
function toast(msg) { const t = $("#tip"); t.textContent = msg; t.hidden = false; t.style.left = "50%"; t.style.top = "60px"; setTimeout(() => (t.hidden = true), 1400); }

async function resetRun() {
  if (!confirm("删除当前运行的全部记录（节点、模型、会话记录、预测），并新建一次运行？\n此操作不可撤销。")) return;
  try { await api("/api/reset", { method: "POST" }); S.samples = null; S.pred = {}; S.node = {}; S.sel = null; S._introShown = false; S.view = "main"; await refresh(); }
  catch (e) { alert("无法新建运行：" + e.message); }
}

async function showReport() {
  const r = await api("/api/report");
  $("#modal-title").textContent = "FINAL_REPORT.md（agent 写的最终报告）";
  $("#modal-body").innerHTML = `<pre>${esc(r.text || "（还没有）")}</pre>`;
  $("#modal").hidden = false;
}
$("#modal-close").onclick = () => ($("#modal").hidden = true);

function renderNoRun() {
  $("#top").innerHTML = `<span class="title">ColoredMNIST · Codex goal</span>`;
  $("#banner").innerHTML = "";
  $("#app").innerHTML = `<div class="card empty"><div class="h" style="justify-content:center">还没有运行</div>
    <div class="sub" style="margin-bottom:14px">运行目录 ${esc(S.st.run_dir)} 不存在。</div>
    <button class="btn primary" id="b-new">新建运行</button></div>`;
  $("#b-new").onclick = async () => { await api("/api/reset", { method: "POST" }); refresh(); };
}

// ------------------------------------------------------------------------------------------ intro

async function renderIntro() {
  $("#banner").innerHTML = "";
  if (!S.intro) { try { S.intro = await api("/api/intro"); } catch { S.intro = null; } }
  const st = S.st, I = S.intro;
  const env = (name, e, desc) => {
    const pool = I ? I.envs[name] : [];
    const six = pool.slice(S.introOffset % Math.max(pool.length, 1)).concat(pool).slice(0, 6);
    const stt = I ? I.stats[name] : null;
    return `<div class="card"><div class="h">${name} <span class="pill">e = ${e}</span><span class="sub">${desc}</span>
      ${stt ? `<span class="sub">· 实际：颜色与标签一致 ${pct(stt.color_agrees)}，标签被翻转 ${pct(stt.label_flipped)}</span>` : ""}</div>
      <div class="envrow">${six.map((t) => `<div class="s" data-tip="${esc(tipText(t))}"><img src="${t.img}"><div class="lab"><span>y${t.label}${t.flipped ? "✱" : ""}</span><span>${t.digit}</span></div></div>`).join("")}</div></div>`;
  };
  $("#app").innerHTML = `<div class="intro">
   <h1>颜色是“捷径”：训练出能泛化到颜色规则相反的新域的分类器</h1>
   <div class="sub">DomainBed · ColoredMNIST · 测试域 env2 · 指标 ${esc(st.metric)}（越高越好）</div>
   <div class="cols">
    <div class="card"><div class="h">数据怎么生成</div>
      <div class="step"><span class="num">1</span><div>MNIST 数字 → 二分类标签：<b>0–4 → 1，5–9 → 0</b></div></div>
      <div class="step"><span class="num">2</span><div>以 <b>25%</b> 概率翻转标签（标签噪声）<div class="sub">所以只看数字形状最多对 75%</div></div></div>
      <div class="step"><span class="num">3</span><div>按（加噪后的）标签上色：<b style="color:#c00">红 = 0</b>、<b style="color:#080">绿 = 1</b>，再以概率 <b>e</b> 翻转颜色<div class="sub">颜色由加噪后的标签决定，所以颜色比形状更“准”——但它的方向在 env2 上是反的</div></div></div>
      <div class="sub" style="margin-top:12px">样本下方：<b>y</b> = 标签（✱ 表示被噪声翻转），右侧是原始数字。图像实际是 2 通道 28×28（红/绿各一个通道）。</div>
    </div>
    <div>${env("env0", 0.1, "训练 · 颜色与标签一致约 90%")}${env("env1", 0.2, "训练 · 一致约 80%")}${env("env2", 0.9, "测试域 · 一致仅约 10%（规则相反），这里只显示 val 部分")}</div>
   </div>
   <div class="three">
    <div class="card"><div class="h">评测方式</div><div class="kv">
      <span class="k">训练</span><span>env0、env1（各 70% 训练 / 30% 留出）</span>
      <span class="k">val</span><span>env2 out-split（30%，${I ? I.stats.env2.n_out : "≈7000"} 张）</span>
      <span class="k">隐藏 test</span><span>env2 in-split（70%），agent 看到的这部分图像是空白的</span></div></div>
    <div class="card"><div class="h">参考水平</div><div class="kv">
      <span class="k">只用颜色</span><span>≈ 0.10（env2 上颜色规则相反）</span>
      <span class="k">只用形状</span><span>上限 0.75</span>
      <span class="k">baseline</span><span>ERM ${st.baseline_val != null ? f4(st.baseline_val) : `≈ ${f4(st.baseline_reference)}（本机参考值）`}</span></div></div>
    <div class="card"><div class="h">agent 设置</div><div class="kv">
      <span class="k">agent</span><span>Codex goal · ${esc(st.model)}（${esc(st.effort)}）</span>
      <span class="k">预算</span><span>${st.budget_hours} 小时 · 最多 ${st.max_runs} 次 fml-val · GPU ${st.gpu}</span>
      <span class="k">可改文件</span><span>domainbed/algorithms.py、networks.py</span></div></div>
   </div>
   <div style="display:flex;justify-content:flex-end;gap:10px"><button class="btn" id="b-more">换一批样本</button><button class="btn primary" id="b-enter">我已了解，进入主页面 →</button></div>
  </div>`;
  $("#b-more").onclick = () => { S.introOffset += 6; renderIntro(); };
  $("#b-enter").onclick = () => { localStorage.setItem(`viz-intro-seen:${st.run_id}:${st.created}`, "1"); S.view = "main"; render(); };
  bindTips();
}

function tipText(t, pred) {
  let s = `原始数字 ${t.digit} · 标签 ${t.label}（${t.flipped ? "被噪声翻转" : "未翻转"}）· 颜色 ${COLOR_NAME[t.color]}`;
  if (pred) s += ` · 预测 ${pred.pred} · p(1) = ${Number(pred.prob1).toFixed(2)} ${pred.pred === t.label ? "✓" : "✗"}`;
  return s;
}
function bindTips() {
  const tip = $("#tip");
  document.querySelectorAll("[data-tip]").forEach((el) => {
    el.onmouseenter = () => { tip.textContent = el.dataset.tip; tip.hidden = false; };
    el.onmousemove = (e) => { tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px"; };
    el.onmouseleave = () => (tip.hidden = true);
  });
}

// ------------------------------------------------------------------------------------------ main

function renderMain() {
  const st = S.st;
  const bj = st.baseline_job;
  const baseOk = bj && bj.state === "ok";
  // banner: launch instructions, shown for as long as the baseline is done (also after the agent has
  // entered the sandbox: the goal is pasted only after cx-start)
  if (baseOk) {
    if (st.agent.started) {
      $("#banner").innerHTML = `<div class="bar info">agent 已进入沙盒。如果还没有输入 goal：在沙盒里运行 <code>cx-start</code>，再粘贴 goal。
        <button class="btn" id="b-c1">复制启动命令</button> <button class="btn" id="b-c2">复制 goal</button>
        <details style="display:inline-block;margin-left:8px"><summary class="sub">查看命令和 goal 全文</summary><pre>${esc(st.launch_cmd)}</pre><pre>${esc(st.goal_txt || "")}</pre></details></div>`;
      $("#b-c1").onclick = () => copy(st.launch_cmd, "已复制启动命令");
      $("#b-c2").onclick = () => copy(st.goal_txt, "已复制 goal");
    } else {
    $("#banner").innerHTML = `<div class="bar"><b class="good">✓</b> baseline ${f4(st.baseline_val)}（${esc((bj.steps.find((s) => s.key === "compare") || {}).detail || "")}）。现在可以启动 agent：
      <pre>${esc(st.launch_cmd)}</pre>
      进入沙盒后运行 <code>cx-start</code>，再粘贴 goal（<code>cat /task/goal.txt</code>，或用下面的按钮复制）。
      <details><summary class="sub">查看 goal 全文</summary><pre>${esc(st.goal_txt || "")}</pre></details>
      <button class="btn" id="b-c1">复制启动命令</button> <button class="btn" id="b-c2">复制 goal</button></div>`;
    $("#b-c1").onclick = () => copy(st.launch_cmd, "已复制启动命令");
    $("#b-c2").onclick = () => copy(st.goal_txt, "已复制 goal");
    }
  } else $("#banner").innerHTML = "";
  if (!$("#left") || !$("#right")) {
    $("#app").innerHTML = `<div class="grid"><div id="left"></div><div id="right"></div></div>`;
    S._detailKey = S._rightKey = null;
  }
  renderLeft();
  const shown = S.follow && S.sel && S.sel !== "next" ? S.sel : st.sota;
  const rn = st.nodes.find((n) => n.id === shown) || {};
  const rk = JSON.stringify([shown, rn.status, rn.metric, rn.pred, st.baseline_job && st.baseline_job.steps, S.filter]);
  if (rk !== S._rightKey) { S._rightKey = rk; renderRight(); }
}

// ---------------------------------------------------------------- left: chart, nodes, detail

function nodeLabel(n) { return n.id === "baseline" ? "0000" : n.id; }

function renderLeft() {
  const st = S.st;
  const nodes = st.nodes;
  const list = nodes.map((n) => {
    let ic = "●", col = "var(--ink2)", v = f4(n.metric);
    if (n.status === "running") { ic = "◐"; col = "var(--muted)"; v = `<span class="mini"><span style="width:${(100 * n.progress).toFixed(0)}%"></span></span>${pct(n.progress)}`; }
    else if (n.status !== "ok") { ic = "✗"; col = "var(--crit)"; v = `<span class="crit">${n.status === "timeout" ? "超时" : "失败"}</span>`; }
    else if (n.new_sota) { ic = "★"; col = "var(--s1)"; }
    const tags = [];
    if (n.id === "baseline") tags.push(`<span class="tag">基线</span>`);
    if (n.id === st.sota) tags.push(`<span class="tag blue">当前 SOTA</span>`);
    if (st.submission && String(st.submission.run).padStart(4, "0") === n.id) tags.push(`<span class="tag purple">已提交</span>`);
    const title = n.id === "baseline" ? "baseline · ERM 原版" : n.title || "（无标题）";
    return `<div class="node ${S.sel === n.id ? "sel" : ""}" data-id="${n.id}"><span style="color:${col};font-weight:700">${ic}</span>
      <span class="muted" title="${n.end ? "完成于 " + cstFull(n.end) : n.status === "running" ? "运行中" : ""}">${nodeLabel(n)}${n.end ? ` <span class="ntime">${cstTime(n.end)}</span>` : ""}</span><span class="t">${esc(title)} ${tags.join(" ")}</span><span class="v">${v}</span></div>`;
  }).join("");
  const next = st.agent.started && st.goal && st.goal.status !== "complete"
    ? `<div class="node ${S.sel === "next" ? "sel" : ""}" data-id="next"><span class="muted">…</span><span class="muted">next</span><span class="t sub">agent 当前在做什么（下一个实验之前的过程）</span><span class="v"></span></div>` : "";
  const noBase = !nodes.length ? `<div class="node"><span class="muted">○</span><span class="muted">0000</span><span class="t">baseline · ERM 原版 <span class="sub">尚未运行</span></span><span></span></div>` : "";
  $("#left").innerHTML = `
    <div class="card"><div class="h">val 曲线 <span class="sub">每个点是一次 fml-val；虚线阶梯 = SOTA；点击选中节点</span></div><div id="valchart"></div></div>
    <div class="card" style="padding:6px"><div class="h" style="margin:4px 6px">研究节点 <span class="sub">每次 fml-val 一个节点（${st.nodes.filter((n) => n.id !== "baseline").length} / ${st.max_runs}）</span></div>${noBase}${list}${next}
      ${!st.agent.started && nodes.length ? `<div class="sub" style="padding:6px 8px">agent 尚未启动${st.baseline_job && st.baseline_job.state === "ok" ? "：按页面顶部的命令启动" : "：要等 baseline 完成"}</div>` : ""}</div>
    <div class="card" id="detail"></div>`;
  const pick = (id) => { S.sel = id; S.diffFile = null; S._detailKey = null; renderLeft(); if (S.follow) { S._rightKey = null; renderMain(); } };
  document.querySelectorAll(".node[data-id]").forEach((el) => (el.onclick = () => pick(el.dataset.id)));
  $("#valchart").innerHTML = valChart(nodes);
  document.querySelectorAll("#valchart [data-id]").forEach((el) => (el.onclick = () => pick(el.dataset.id)));
  // The detail card is re-rendered only when the selected node changed (keeps open <details>, scroll).
  const sn = nodes.find((n) => n.id === S.sel) || {};
  const dk = JSON.stringify([S.sel, S.tab, sn.status, sn.step, sn.metric, sn.has_note, sn.has_card, sn.pred,
    S.sel === "next" || sn.status === "running" ? (S.st.tokens || {}).responses : 0]);
  if (dk !== S._detailKey) { S._detailKey = dk; delete S.node[S.sel]; renderDetail(); }
  else if (S._detailHTML) $("#detail").replaceWith(S._detailHTML);
}

function valChart(nodes) {
  const W = 600, H = 170, x0 = 40, x1 = W - 14, y0 = H - 24, y1 = 12;
  const maxN = Math.max(8, ...nodes.map((n) => n.num + 1));
  const X = (i) => x0 + (i * (x1 - x0)) / maxN, Y = (v) => y0 - ((v - 0) / 1) * (y0 - y1);
  let g = "";
  for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line x1="${x0}" x2="${x1}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${x0 - 6}" y="${Y(v) + 4}" font-size="10" fill="var(--muted)" text-anchor="end">${v}</text>`;
  const step = Math.ceil(maxN / 10);
  for (let i = 0; i <= maxN; i += step) g += `<text x="${X(i)}" y="${H - 7}" font-size="10" fill="var(--muted)" text-anchor="middle">${i}</text>`;
  g += `<line x1="${x0}" x2="${x1}" y1="${Y(0.75)}" y2="${Y(0.75)}" stroke="var(--crit)" stroke-dasharray="2 3"/><text x="${x1}" y="${Y(0.75) - 4}" font-size="10" fill="var(--crit)" text-anchor="end">只用形状上限 0.75</text>`;
  // SOTA staircase
  let best = null, path = "";
  for (const n of nodes) {
    if (n.status === "ok" && n.new_sota) { path += best == null ? `M${X(n.num)},${Y(n.metric)}` : ` H${X(n.num)} V${Y(n.metric)}`; best = n; }
  }
  if (best) { path += ` H${X(Math.max(...nodes.map((n) => n.num)))}`; g += `<path d="${path}" fill="none" stroke="var(--ink2)" stroke-width="1.5" stroke-dasharray="4 3"/>`; }
  for (const n of nodes) {
    const x = X(n.num), sel = S.sel === n.id;
    if (n.status === "running") g += `<g data-id="${n.id}" style="cursor:pointer"><circle cx="${x}" cy="${Y(0.5)}" r="5" fill="none" stroke="var(--s1)" stroke-width="2" stroke-dasharray="2 2"/><text x="${x}" y="${Y(0.5) + 17}" font-size="10" fill="var(--muted)" text-anchor="middle">训练中</text><rect x="${x - 9}" y="${y1}" width="18" height="${y0 - y1}" fill="transparent"/></g>`;
    else if (n.status !== "ok") g += `<g data-id="${n.id}" style="cursor:pointer"><text x="${x}" y="${y0 + 4}" font-size="13" font-weight="700" fill="var(--crit)" text-anchor="middle">✗</text><rect x="${x - 9}" y="${y1}" width="18" height="${y0 - y1}" fill="transparent"/></g>`;
    else g += `<g data-id="${n.id}" style="cursor:pointer"><title>${nodeLabel(n)} · ${f4(n.metric)}</title><rect x="${x - 9}" y="${y1}" width="18" height="${y0 - y1}" fill="transparent"/>${sel ? `<line x1="${x}" x2="${x}" y1="${y1}" y2="${y0}" stroke="var(--s1)" opacity=".35"/>` : ""}<circle cx="${x}" cy="${Y(n.metric)}" r="${sel ? 6 : 4.5}" fill="${n.id === "baseline" ? "var(--ink2)" : "var(--s1)"}" stroke="#fff" stroke-width="2"/>${n.new_sota ? `<text x="${x + 8}" y="${Y(n.metric) - 6}" font-size="10.5" fill="var(--ink)">${f4(n.metric)}</text>` : ""}</g>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">${g}</svg>`;
}

const TABS = [["idea", "思路"], ["code", "代码"], ["train", "训练"], ["process", "过程"], ["note", "结论"]];

async function renderDetail() {
  const el = $("#detail");
  S._detailHTML = el;
  const id = S.sel;
  if (!id) { el.innerHTML = `<div class="sub">还没有节点。</div>`; return; }
  if (id === "next") { el.innerHTML = `<div class="h">agent 当前的过程 <span class="sub">最近一次 fml-val 开始之后</span></div><div id="tabbody"></div>`; return renderProcess("next"); }
  let n = S.node[id];
  if (!n) { try { n = S.node[id] = await api(`/api/node/${id}`); } catch (e) { el.innerHTML = `<div class="sub">${esc(e.message)}</div>`; return; } }
  el.innerHTML = `<div class="h">选中节点：${id === "baseline" ? "baseline（0000）" : "run " + id}</div>
    <div class="tabs">${TABS.map(([k, l]) => `<button class="tab ${S.tab === k ? "on" : ""}" data-tab="${k}">${l}</button>`).join("")}</div><div id="tabbody"></div>`;
  el.querySelectorAll("[data-tab]").forEach((b) => (b.onclick = () => { S.tab = b.dataset.tab; S._detailKey = null; renderLeft(); }));
  ({ idea: renderIdea, code: renderCode, train: renderTrain, process: renderProcess, note: renderNote })[S.tab](id, n);
}

function prevSota(id) {
  // the SOTA before this node (for "vs previous SOTA")
  let best = null;
  for (const n of S.st.nodes) { if (n.id === id) break; if (n.status === "ok" && n.new_sota) best = n; }
  return best;
}

async function renderIdea(id, n) {
  const c = n.card || {};
  const ps = prevSota(id);
  const files = Object.entries(n.changes || {}).filter(([, v]) => v.add || v.del).map(([f, v]) => `${esc(f)}（+${v.add} −${v.del}）`).join("、") || "没有改动目标文件";
  let result = "—";
  if (n.status === "running") result = `训练中 ${pct(n.progress)}`;
  else if (n.status !== "ok") result = `<span class="crit">✗ ${n.status === "timeout" ? "超时" : "失败"}</span>`;
  else result = `<b>${f4(n.metric)}</b>　${ps ? signed(n.metric - ps.metric) + " vs 上一个 SOTA" : ""}　${n.new_sota && id !== "baseline" ? `<span style="color:var(--s1);font-weight:600">★ 新 SOTA</span>` : ""}`;
  const note = (n.notes || []).slice(-1)[0];
  $("#tabbody").innerHTML = `<div class="kv">
    <span class="k">标题</span><span><b>${id === "baseline" ? "baseline · ERM 原版" : esc(c.title || "（agent 没有写）")}</b></span>
    <span class="k">思路</span><span>${id === "baseline" ? "任务的原版代码，在 agent 开始之前由页面运行一次，作为第一个 SOTA。" : esc(c.idea || "—")}</span>
    <span class="k">预期</span><span>${esc(c.expect || "—")}</span>
    <span class="k">结果</span><span>${result}</span>
    <span class="k">运行</span><span>${n.duration_s ? dur(n.duration_s) : "—"} · ${files}<span id="ntok"></span></span>
    <span class="k">结论</span><span class="sub">${note ? "“" + esc(note.text) + "”" : "未写结论"}</span></div>`;
  if (id !== "baseline") {
    try { const p = await api(`/api/node/${id}/process`); if (p.tokens) $("#ntok").textContent = ` · 本节点 token ${ktok(p.tokens.input + p.tokens.output)}`; } catch { }
  }
}

async function renderCode(id, n) {
  const others = S.st.nodes.filter((m) => m.id !== id);
  const prev = others.filter((m) => m.num < n.num).slice(-1)[0];
  if (!S.diffFile || !(n.files || []).includes(S.diffFile)) S.diffFile = (n.files || [])[0];
  let against = S.diffAgainst === "prev" ? (prev ? prev.id : "baseline") : S.diffAgainst;
  if (against === id) against = "baseline";
  const d = await api(`/api/node/${id}/diff?against=${encodeURIComponent(against)}&file=${encodeURIComponent(S.diffFile || "")}`);
  const f = d.files[S.diffFile] || { diff: [], content: "" };
  const lines = S.diffFull ? f.content.split("\n").map((l) => `<div>${esc(l) || " "}</div>`).join("")
    : (f.diff.length ? f.diff.map((l) => `<div class="${l.startsWith("@@") || l.startsWith("+++") || l.startsWith("---") ? "hh" : l[0] === "+" ? "a" : l[0] === "-" ? "d" : ""}">${esc(l) || " "}</div>`).join("") : `<div class="hh">（和 ${esc(against)} 没有差别）</div>`);
  const ch = (n.changes || {})[S.diffFile] || {};
  $("#tabbody").innerHTML = `<div class="chips"><span class="sub">对比：</span>
      <button class="chip ${S.diffAgainst === "baseline" ? "on" : ""}" data-a="baseline">baseline</button>
      <button class="chip ${S.diffAgainst === "prev" ? "on" : ""}" data-a="prev">上一个节点${prev ? "（" + nodeLabel(prev) + "）" : ""}</button>
      <select class="chip" id="sel-against"><option value="">任选节点…</option>${others.map((m) => `<option value="${m.id}" ${S.diffAgainst === m.id ? "selected" : ""}>${nodeLabel(m)} ${esc(m.title || "")}</option>`).join("")}</select>
      <span class="spacer"></span>
      <select class="chip" id="sel-file">${(n.files || []).map((x) => `<option ${x === S.diffFile ? "selected" : ""}>${esc(x)}</option>`).join("")}</select>
      <button class="chip ${S.diffFull ? "on" : ""}" id="b-full">完整文件</button></div>
    <div class="diff">${lines}</div>
    <div class="sub" style="margin-top:6px">相对 baseline：+${ch.add || 0} −${ch.del || 0} 行 · <a href="#" id="b-copycode">复制此文件</a></div>`;
  document.querySelectorAll("[data-a]").forEach((b) => (b.onclick = () => { S.diffAgainst = b.dataset.a; renderCode(id, n); }));
  $("#sel-against").onchange = (e) => { if (e.target.value) { S.diffAgainst = e.target.value; renderCode(id, n); } };
  $("#sel-file").onchange = (e) => { S.diffFile = e.target.value; renderCode(id, n); };
  $("#b-full").onclick = () => { S.diffFull = !S.diffFull; renderCode(id, n); };
  $("#b-copycode").onclick = (e) => { e.preventDefault(); copy(f.content, "已复制"); };
}

function lineChart(rows, series, { W = 290, H = 140, yMin = null, yMax = null, nSteps = 5000 } = {}) {
  const x0 = 36, x1 = W - 18, y0 = H - 22, y1 = 10;
  const vals = rows.flatMap((r) => series.map((s) => r[s.key]).filter((v) => v != null));
  let lo = yMin ?? Math.min(...vals, 0), hi = yMax ?? Math.max(...vals, 1e-9);
  if (hi - lo < 1e-9) hi = lo + 1;
  const X = (s) => x0 + (s / nSteps) * (x1 - x0), Y = (v) => y0 - ((v - lo) / (hi - lo)) * (y0 - y1);
  let g = "";
  for (const t of [lo, (lo + hi) / 2, hi]) g += `<line x1="${x0}" x2="${x1}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${x0 - 5}" y="${Y(t) + 4}" font-size="10" fill="var(--muted)" text-anchor="end">${Math.abs(t) >= 10 ? t.toFixed(0) : t.toFixed(2)}</text>`;
  for (const s of [0, nSteps / 2, nSteps]) g += `<text x="${X(s)}" y="${H - 6}" font-size="10" fill="var(--muted)" text-anchor="middle">${s}</text>`;
  for (const s of series) {
    const pts = rows.filter((r) => r[s.key] != null).map((r) => `${X(r.step).toFixed(1)},${Y(r[s.key]).toFixed(1)}`).join(" ");
    if (pts) g += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`;
}

async function renderTrain(id, n) {
  const d = await api(`/api/node/${id}/train`);
  const rows = d.rows;
  const last = rows[rows.length - 1] || {};
  const step = last.step || 0;
  let eta = "";
  if (d.status === "running" && step > 0 && n.start) { const el = Date.now() / 1000 - n.start; eta = ` · 预计剩余 ${dur((el / step) * (d.n_steps - step))}`; }
  const skip = new Set(["step", "epoch", "mem_gb", "step_time"]);
  const lossKeys = Object.keys(last).filter((k) => !skip.has(k) && !k.endsWith("_acc"));
  const LC = ["var(--s1)", "var(--s2)", "var(--s3)"];
  const acc = [{ key: "env0_out_acc", color: "var(--s1)", label: "env0 留出" }, { key: "env1_out_acc", color: "var(--s2)", label: "env1 留出" }, { key: "env2_out_acc", color: "var(--s3)", label: "env2 out（val）" }];
  const lossS = lossKeys.slice(0, 3).map((k, i) => ({ key: k, color: LC[i], label: k }));
  const legend = (ss) => `<div class="legend">${ss.map((s) => `<span><span class="sw" style="background:${s.color}"></span>${esc(s.label)}</span>`).join("")}</div>`;
  $("#tabbody").innerHTML = rows.length || d.status === "running" ? `
    <div class="sub" style="display:flex;justify-content:space-between"><span>step ${step.toLocaleString()} / ${d.n_steps.toLocaleString()}${eta}</span><span>${d.status === "running" ? "训练中（每 100 步更新）" : d.status === "ok" ? "已完成" : "未成功"}</span></div>
    <div class="progress"><span style="width:${(100 * Math.min(step / d.n_steps, 1)).toFixed(1)}%"></span></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div><div class="sub">训练损失等（agent 的 update() 返回的数值）</div>${rows.length ? lineChart(rows, lossS) : ""}${legend(lossS)}</div>
      <div><div class="sub">留出准确率（env2 in 已置空，不显示）</div>${rows.length ? lineChart(rows, acc, { yMin: 0, yMax: 1 }) : ""}${legend(acc)}</div>
    </div>
    <div class="sub" style="margin-top:8px">日志尾部</div><div class="log">${esc(d.log_tail.join("\n"))}</div>`
    : `<div class="sub">没有训练记录（运行失败或缺少目标文件）。</div><div class="log">${esc(d.log_tail.join("\n"))}</div>`;
  const lg = $("#tabbody .log"); if (lg) lg.scrollTop = lg.scrollHeight;
}

const KIND = { user: ["用户", ""], message: ["发言", ""], reasoning: ["思考", "purple"], command: ["命令", ""], stdin: ["输入", ""], patch: ["补丁", ""], tool: ["工具", ""], goal: ["goal", "blue"] };

async function renderProcess(id) {
  const d = await api(`/api/node/${id}/process`);
  const evs = d.events || [];
  const rows = evs.map((e) => {
    const [label, cls] = KIND[e.kind] || [e.kind, ""];
    let body = "", right = "";
    if (e.kind === "command") {
      const isVal = /\bfml-val\b/.test(e.cmd);
      body = `<details><summary class="mono">${esc(e.cmd.slice(0, 160))}${e.cmd.length > 160 ? "…" : ""}</summary><pre>${esc(e.cmd)}</pre>${e.output ? `<pre>${esc(e.output)}</pre>` : ""}</details>`;
      right = `${e.duration_s != null ? dur(e.duration_s) : ""} ${e.exit_code == null ? "" : e.exit_code === 0 ? '<span class="good">✓</span>' : '<span class="crit">✗</span>'}`;
      return `<div class="ev"><span class="muted">${hms(e.ts)}</span><span><span class="tag ${isVal ? "blue" : ""}">${isVal ? "fml-val" : label}</span></span><div class="body">${body}</div><span class="sub" style="text-align:right">${right}</span></div>`;
    }
    if (e.kind === "patch") body = `<details><summary>${e.files.map((f) => `${esc(f.file.split("/").slice(-2).join("/"))} +${f.add} −${f.del}`).join("，") || "apply_patch"}</summary><pre>${esc(e.patch)}</pre></details>`;
    else if (e.kind === "tool") body = `<span class="mono">${esc(e.name)}</span> <span class="sub">${esc(e.args)}</span>`;
    else if (e.kind === "goal") body = `goal ${esc(e.status)}${e.objective ? `：<span class="sub">${esc(e.objective)}</span>` : ""}`;
    else if (e.kind === "reasoning") body = `<span class="sub">${esc(e.text)}</span>`;
    else body = esc(e.text || "");
    return `<div class="ev"><span class="muted">${hms(e.ts)}</span><span><span class="tag ${cls}">${label}</span></span><div class="body">${body}</div><span></span></div>`;
  }).join("");
  const t = d.tokens;
  const head = id === "next" ? "" : `<div class="sub" style="margin-bottom:6px">${id === "baseline" ? "baseline 由页面运行，没有 agent 过程。" : "上一次 fml-val 开始之后、这一次开始之前，agent 做了什么（来自 Codex 会话记录）"}</div>`;
  $("#tabbody").innerHTML = head + (rows || `<div class="sub">（没有记录）</div>`)
    + (t && t.responses ? `<div class="sub" style="margin-top:8px">本段 token：输入 ${ktok(t.input)}（缓存 ${pct(t.cached / Math.max(t.input, 1))}）· 输出 ${ktok(t.output)}（其中推理 ${ktok(t.reasoning)}）· ${t.responses} 次模型调用</div>` : "");
}

function renderNote(id, n) {
  const notes = n.notes || [];
  $("#tabbody").innerHTML = notes.length ? notes.map((x) => `<div class="ev" style="grid-template-columns:120px 1fr"><span class="muted">${esc(x.time_iso || hms(x.time))}</span><div class="body">${esc(x.text)}</div></div>`).join("")
    : `<div class="sub">${id === "baseline" ? "baseline 没有结论。" : "agent 还没有用 fml-note 写结论。"}</div>`;
}

// ---------------------------------------------------------------- right: SOTA panel

async function renderRight() {
  const st = S.st, el = $("#right");
  const bj = st.baseline_job;
  if (!st.sota) { renderNoSota(el, bj); return; }
  const shown = S.follow && S.sel && S.sel !== "next" ? S.sel : st.sota;
  const node = st.nodes.find((n) => n.id === shown) || {};
  let p = S.pred[shown];
  if (!p && node.pred === "ready") { try { p = S.pred[shown] = await api(`/api/pred/${shown}`); } catch { } }
  const head = `<div class="h" style="font-size:14px">${shown === st.sota ? "当前 SOTA" : "选中节点"}：${shown === "baseline" ? "baseline · ERM 原版" : "run " + shown + " · " + esc(node.title || "")}
    ${p && p.ok ? (p.batch_dependent ? `<span class="badge warnc" title="两次不同的随机分 batch 预测结果不同">⚠ 预测依赖 batch 组成（${p.n_differ_between_passes} 个样本不同）</span>`
      : p.matches_logged ? `<span class="badge good">✓ 预测复现 = 记录的 val</span>` : `<span class="badge warnc">⚠ 预测 ${f4(p.acc)} ≠ 记录 ${f4(p.logged_val)}</span>`) : ""}
    <span class="spacer"></span><label class="sub"><input type="checkbox" id="cb-follow" ${S.follow ? "checked" : ""}> 跟随左侧选中的节点</label></div>`;
  if (!p) {
    el.innerHTML = head + `<div class="card empty sub">${node.status === "running" ? "这个节点还在训练。" : node.pred === "failed" ? "预测失败（见 viz/jobs/predict_" + esc(shown) + ".log）。" : node.status !== "ok" ? "这个节点没有成功，没有模型。" : "正在计算预测（约 10 秒）…"}</div>`;
    $("#cb-follow").onchange = (e) => { S.follow = e.target.checked; S._rightKey = null; renderMain(); };
    return;
  }
  if (!p.ok) { el.innerHTML = head + `<div class="card"><b class="crit">✗ 预测失败</b><pre class="log">${esc(p.error)}</pre></div>`; $("#cb-follow").onchange = (e) => { S.follow = e.target.checked; S._rightKey = null; renderMain(); }; return; }
  const base = st.baseline_val;
  const ea = p.env_acc || {};
  const bars = [..."ABCD"].map((g) => `<div class="row"><span>${GROUP_LABEL[g]}</span><div class="track"><div class="fill" style="width:${(100 * p.groups[g].acc).toFixed(1)}%"></div></div><span style="text-align:right">${p.groups[g].acc.toFixed(2)}</span><span class="sub">n = ${p.groups[g].n.toLocaleString()}</span></div>`).join("");
  const cf = p.confusion;
  const cmax = Math.max(...cf.flat());
  const cell = (v, diag) => `<td style="background:${diag ? "var(--seq-hi)" : "var(--seq-lo)"};color:${diag ? "#fff" : "var(--ink)"}">${v.toLocaleString()}</td>`;
  const allS = S.samples ? S.samples.samples : [];
  const tile = (s) => {
    const pr = p.samples[String(s.i)];
    const ok = pr && pr.pred === s.label;
    return `<div class="s ${pr ? (ok ? "ok" : "bad") : ""}" data-tip="${esc(tipText(s, pr))}">${pr ? `<span class="mark ${ok ? "good" : "crit"}">${ok ? "✓" : "✗"}</span>` : ""}<img src="${s.img}"><div class="lab"><span>y${s.label}${s.flipped ? "✱" : ""}</span><span>${pr ? "p" + pr.pred : ""}</span></div></div>`;
  };
  const tiles = [..."ABCD"].filter((g) => S.filter === "all" || S.filter === g).map((g) => {
    const gs = allS.filter((s) => s.group === g);
    const nOk = gs.filter((s) => (p.samples[String(s.i)] || {}).pred === s.label).length;
    return `<div class="grow"><div class="glab">${GROUP_LABEL[g]}<div class="sub">${nOk} / ${gs.length} 对</div></div><div class="samples">${gs.map(tile).join("")}</div></div>`;
  }).join("");
  el.innerHTML = head + `
    <div class="tiles">
      <div class="tile"><div class="k">val ${esc(st.metric)}（env2 out）</div><div class="v">${f4(node.metric)}</div></div>
      <div class="tile"><div class="k">相对 baseline</div><div class="v">${base != null && node.metric != null ? signed(node.metric - base) : "—"}</div></div>
      <div class="tile"><div class="k">env0 留出准确率</div><div class="v">${f4(ea.env0_out)}</div></div>
      <div class="tile"><div class="k">env1 留出准确率</div><div class="v">${f4(ea.env1_out)}</div></div>
    </div>
    <div style="display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:12px">
      <div class="card" style="margin:0"><div class="h">捷径诊断 <span class="sub">全部 val 样本（N = ${p.n.toLocaleString()}）分 4 组的准确率</span></div><div class="bars">${bars}</div>
        <div class="sub" style="margin-top:6px">形状✓ = 标签未被噪声翻转 · 颜色✓ = 颜色与标签一致<br>预测与<b>颜色</b>一致 ${pct(p.agree_color)} · 与<b>形状</b>一致 ${pct(p.agree_shape)}</div></div>
      <div class="card" style="margin:0"><div class="h">混淆矩阵 <span class="sub">val</span></div>
        <table class="conf" style="border-collapse:separate;border-spacing:2px"><tr><td></td><td class="sub">预测 0</td><td class="sub">预测 1</td></tr>
        <tr><td class="sub">标签 0</td>${cell(cf[0][0], true)}${cell(cf[0][1], false)}</tr><tr><td class="sub">标签 1</td>${cell(cf[1][0], false)}${cell(cf[1][1], true)}</tr></table></div>
    </div>
    <div class="card" style="margin-top:12px"><div class="h">样本预测 <span class="sub">固定 ${S.samples ? S.samples.samples.length : 0} 个 val 样本（4 组 × 12），所有节点相同；悬停看详情</span></div>
      <div class="chips">${[["all", "全部"], ...Object.entries(GROUP_LABEL)].map(([k, l]) => `<button class="chip ${S.filter === k ? "on" : ""}" data-f="${k}">${l}</button>`).join("")}</div>
      <div class="sub" style="margin-bottom:8px"><span class="good">■</span> ✓ 预测正确　<span class="crit">■</span> ✗ 预测错误　y = 标签　p = 预测　✱ = 标签被翻转</div>
      ${tiles}</div>`;
  $("#cb-follow").onchange = (e) => { S.follow = e.target.checked; S._rightKey = null; renderMain(); };
  el.querySelectorAll("[data-f]").forEach((b) => (b.onclick = () => { S.filter = b.dataset.f; S._rightKey = null; renderMain(); }));
  bindTips();
}

function renderNoSota(el, bj) {
  const steps = bj ? bj.steps : [{ key: "sandbox", label: "沙盒" }, { key: "gpu", label: "GPU" }, { key: "data", label: "掩码数据" }, { key: "train", label: "训练" }, { key: "predict", label: "预测" }, { key: "compare", label: "与参考值比较" }, { key: "prompts", label: "生成提示词" }];
  const ic = (s) => ({ ok: '<span class="good">✓</span>', warn: '<span class="warnc">⚠</span>', fail: '<span class="crit">✗</span>', running: '<span style="color:var(--s1)">◐</span>' }[s.state] || '<span class="muted">○</span>');
  const running = bj && bj.state === "running";
  const failed = bj && (bj.state === "fail" || bj.state === "interrupted");
  el.innerHTML = `<div class="h" style="font-size:14px">当前 SOTA：无</div>
    <div class="card empty"><div style="font-size:15px;font-weight:600;margin-bottom:6px">还没有 SOTA 模型</div>
      <div class="sub" style="margin-bottom:16px">先在和 agent 相同的沙盒里（掩码数据、同样的环境和 GPU）复现 baseline：它会成为第一个 SOTA，同时检查环境是否可用。不计入 agent 的次数，也不开始计时。</div>
      <button class="btn primary" id="b-base" ${running ? "disabled" : ""}>${running ? "baseline 运行中…" : failed ? "↻ 重试 baseline" : "▶ 运行 baseline（约 3 分钟）"}</button>
      <div class="checks">${steps.map((s) => `<span title="${esc(s.detail || "")}">${ic(s)} ${esc(s.label)}${s.detail ? `<span class="sub">（${esc(s.detail)}）</span>` : ""}</span>`).join("")}</div>
      ${failed ? `<pre class="log" style="text-align:left;margin-top:12px">${esc(bj.error || bj.state)}</pre>` : ""}
      <div class="sub" style="margin-top:10px">参考值（本机实测）：${f4(S.st.baseline_reference)}</div></div>`;
  $("#b-base").onclick = async () => { try { await api("/api/baseline", { method: "POST" }); refresh(); } catch (e) { alert(e.message); } };
}

// ------------------------------------------------------------------------------------------
refresh().then(connect);
