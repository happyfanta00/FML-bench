# Codex goal × ColoredMNIST：研究过程可视化（设计稿）

目标：在 Mac 浏览器里看 Codex（goal 模式）改进 ColoredMNIST 的整个过程。先看任务介绍；baseline 在网页上点按钮复现，同时检查环境是否可用；然后 agent 在命令行手动启动。主页面左边是研究流程（每次 `fml-val` 一个节点），右边是当前 SOTA 模型的预测可视化和指标。

本文档只是设计，**界面草图（第 4 节）需要你确认后再开发**。第 9 节列出了需要你决定的事项。

基础是 `docs/codex_agent_design.md` 的沙盒方案（分支 `codex-awesome可视化`，单卡 L4 机器）。

---

## 1. 总体结构

```
 Mac 浏览器 ──(ssh -L 8765:localhost:8765)──► EC2 上的 viz 服务（只监听 127.0.0.1:8765）
                                                 │ 只读 run 目录；按钮触发宿主机上的任务
                                                 ▼
  ~/fml_agent_runs/codex/Generalization_domainbed/<run_id>/
    task/meta.json  task/runs/NNNN/{code.json,run.json,output.log,results.jsonl,model.pkl,card.json}
    task/runs/baseline/...        ← 新增：按钮触发的 baseline（不占 agent 的 20 次）
    codex_home/sessions/**/rollout-*.jsonl   ← agent 的完整会话记录
    viz/                          ← 新增：派生数据（预测结果、解析缓存、任务状态）
```

- **viz 服务**：Python 写，跑在 EC2 上，只监听本机地址，Mac 通过 SSH 端口转发访问，不对外开放端口，所以不需要做登录。
- 它主要只读 run 目录，写入只有三处：`viz/`、`task/runs/baseline/`，以及 baseline 完成后更新 `task/meta.json` 里的 baseline 值。
- **页面**：一个单页应用，不需要打包构建，图表库放在本地，从 viz 服务直接加载。
- **实时更新**：服务端每 2 秒扫描一次 run 目录和会话记录，有变化就通过 SSE（Server-Sent Events）推给页面。
- **两个宿主机任务**（都在沙盒里运行，和 agent 用同样的环境和掩码数据）：
  - `baseline`：按钮触发，第 5 节；
  - `predict`：每有一个新节点完成就自动触发，第 6 节。
- agent 仍按现有方式在命令行启动（`launch.sh` → `cx-start` → 粘贴 `/goal`）。页面会给出可复制的命令，并且在 baseline 完成前不允许启动。

## 2. 页面状态机

| 状态 | 判定条件 | 页面表现 |
|---|---|---|
| S0 未运行 baseline | `task/runs/baseline/run.json` 不存在 | 右侧没有 SOTA，显示"运行 baseline"按钮；左侧只有一个灰色的 baseline 节点 |
| S1 baseline 运行中 | baseline 任务在跑 | baseline 节点显示训练进度条、实时曲线和日志尾部 |
| S2 可以启动 agent | baseline 成功、预测已生成、环境检查通过 | baseline 成为第一个 SOTA，右侧开始有内容；顶部显示可复制的启动命令。agent 第一次进入沙盒时开始计时 |
| S3 agent 运行中 | `meta.session_start` 已设置，goal 未完成 | 节点实时增加；正在跑的节点显示进度；顶部显示时间、次数、阶段和 goal 状态 |
| S4 结束 | goal 为 complete，或截止时间已过 | 显示提交的 run 和 FINAL_REPORT；可选按钮"运行最终 test"（即 `finalize.py`，见第 9 节第 5 条） |

baseline 失败时停在 S0，显示错误和日志，可以重试。

## 3. 数据来源和每个节点的内容

每次 `fml-val` 对应一个节点，baseline 是第 0 个节点。

| 内容 | 来源 | 说明 |
|---|---|---|
| 编号、开始时间、耗时、状态、指标 | `runs/NNNN/run.json` | 有 `code.json` 但没有 `run.json` 的节点算"运行中"；失败或超时的单独标出 |
| **标题、改动思路、预期** | **新增 `runs/NNNN/card.json`**（由 `fml-val --title/--idea/--expect` 写入，第 7 节） | agent 自己写的结构化说明，最适合前端展示 |
| **结论** | **新增 `fml-note`** 写入 `runs/NNNN/note.json` | agent 看到结果后写一两句结论 |
| 代码方案 | `runs/NNNN/code.json` | 给出相对 baseline 和相对上一个节点的 diff（并排或统一视图），也能看完整文件 |
| 训练进度 | 运行中读 `runs/NNNN/output.log`；结束后读**新增保存的** `runs/NNNN/results.jsonl` | DomainBed 每 100 步输出一行（共 5000 步）：loss 和各个 env 的准确率。显示进度条、预计剩余时间、曲线（loss，env0/env1 的训练和留出准确率，env2 out 即 val） |
| agent 的工作过程 | `codex_home/sessions/**/rollout-*.jsonl` | 按时间分到节点里：上一次 `fml-val` 开始之后、这一次开始之前的活动都算这个节点的"准备过程"。包括 agent 的说明性发言、执行的命令（含耗时和返回码）、`apply_patch` 的补丁、reasoning summary（如果开启，第 9 节第 4 条），以及这一段的 token 用量 |
| 预测和指标 | `viz/pred/NNNN.json`（第 6 节） | 右侧面板用 |
| 是否 SOTA、是否提交 | 节点之间比较 val；`runs/submission.json` | SOTA 的规则和 `fml-status` 一样：严格优于才算，打平时保留更早的 |

命令和节点的对应关系：在 `fml-val` 的输出里找到 `[fml] run 0003: started`，就能把这次 `exec_command` 和 run 编号对上。

## 4. 界面草图

草图里的数字只是示意（部分取自 trial_1h），不是实际结果。

渲染好的草图在 `docs/viz_mockups/`（PNG 图片，以及可以直接用浏览器打开的同名 HTML）：

| 图 | 内容 |
|---|---|
| `1_intro.png` | 入口页：数据怎么生成、3 个域的真实样本、评测方式和参考水平（4.1） |
| `2_main.png` | 主页面：左侧研究流程（val 曲线、节点列表、节点详情的思路页），右侧 SOTA（指标、捷径诊断、混淆矩阵、样本预测）（4.2） |
| `3_node_detail.png` | 节点详情的代码、训练、过程三个标签页（4.3） |
| `4_baseline.png` | 还没有 baseline 时的主页面和运行中的 baseline（4.4） |

生成方法：`docs/viz_mockups/src/make_samples.py` 用真实的 MNIST 数字按 ColoredMNIST 的规则上色，生成样本图；`render.py` 写出 HTML 草图，再用无头 Chromium（Playwright）截图。图里的预测和指标都是模拟的。下面的 ASCII 草图和图片内容一致。

### 4.1 入口页：任务介绍（确认后进入主页面）

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  FML-bench · Generalization_domainbed · ColoredMNIST            run: trial_viz_01    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  任务：在颜色是"捷径"的数据上，训练一个能泛化到颜色规则相反的新域的分类器            │
│                                                                                      │
│  ┌─ 数据怎么生成 ──────────────────────────────────────────────────────────────────┐ │
│  │ ① MNIST 数字 → 二分类标签：0-4 → 0，5-9 → 1                                      │ │
│  │ ② 以 25% 概率翻转标签（标签噪声：只看形状最多 75%）                               │ │
│  │ ③ 按标签上色（红=0，绿=1），再以概率 e 翻转颜色                                   │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│   env0  e=0.1（训练）         env1  e=0.2（训练）         env2  e=0.9（测试域）       │
│   颜色和标签一致 90%          一致 80%                    一致 10%（颜色规则相反）     │
│   ┌──┬──┬──┬──┬──┬──┐        ┌──┬──┬──┬──┬──┬──┐        ┌──┬──┬──┬──┬──┬──┐           │
│   │🟥│🟩│🟥│🟩│🟩│🟥│        │🟥│🟩│🟩│🟥│🟩│🟥│        │🟩│🟥│🟥│🟩│🟥│🟩│           │
│   └──┴──┴──┴──┴──┴──┘        └──┴──┴──┴──┴──┴──┘        └──┴──┴──┴──┴──┴──┘           │
│   每张图下方：数字 / 标签 / 颜色；标签被翻转的打 ✱  （实际为 28×28 红/绿数字图）     │
│   [换一批样本]                                                                       │
│                                                                                      │
│  ┌─ 评测方式 ──────────────────────────────────────────────────────────────────────┐ │
│  │ 训练：env0、env1（各 70% 训练，30% 留出）  · val：env2 的 30%（out-split）       │ │
│  │ 隐藏 test：env2 的 70%（in-split），agent 看到的这部分图像已置为空白             │ │
│  │ 指标：in_acc_mean（越高越好）· 只能改 domainbed/algorithms.py、networks.py         │ │
│  │ 参考：只用颜色的模型在 env2 上约 0.10；只用形状上限 0.75；baseline ERM ≈ 0.26     │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│  预算：1 小时 · 最多 20 次 fml-val · 模型 GPT-6 Astra（high）· 1 × L4                │
│                                                                                      │
│                                                     [ 我已了解，进入主页面 → ]        │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

- 样本图由后端用 DomainBed 的代码重建：每张图的原始数字、加噪后的标签、颜色，以及标签是否被翻转。在页面上用 canvas 把红、绿两个通道画成彩色。
- 下次再打开时，如果已经看过介绍（记在本地存储里），直接进入主页面。右上角有"任务介绍"链接可以回来看。

### 4.2 主页面（S2/S3：agent 运行中）

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ColoredMNIST · Codex goal · trial_viz_01   ⏱ 23m/60m  runs 7/20  phase SEARCH  goal ● active     │
│ tokens 2.1M（cache 96%）                                        [任务介绍]  [启动命令 ⧉]       │
├───────────────────────────────────────────────┬────────────────────────────────────────────────┤
│ 研究流程                                       │ 当前 SOTA：run 0005 · 0.9051       [跟随选中 ☐] │
│ ┌ val 曲线 ─────────────────────────────────┐ │ ┌──────────┬──────────┬──────────┬──────────┐  │
│ │0.9 ┤            ●──●──●  ●                │ │ │ val      │ vs base  │ env0 out │ env1 out │  │
│ │    ┤        ●                              │ │ │ 0.9051   │ +0.6425  │ 0.8901   │ 0.7993   │  │
│ │0.5 ┤     ●       ━━ SOTA 阶梯线             │ │ └──────────┴──────────┴──────────┴──────────┘  │
│ │0.26┤ ■ baseline                            │ │                                                │
│ │    └──┬──┬──┬──┬──┬──┬──┬──→ run           │ │ 捷径诊断（全部 val 样本，共 N=7,000）            │
│ └───────────────────────────────────────────┘ │  形状✓ 颜色✓ │████████████░│ 0.97  (n= 520)    │
│                                               │  形状✓ 颜色✗ │███████████░░│ 0.92  (n=4,730)   │
│  ● 0000 baseline  ERM 原版         0.2626  ■  │  形状✗ 颜色✓ │███░░░░░░░░░░│ 0.21  (n= 180)    │
│  ● 0001 灰度化 ERM                 0.6837     │  形状✗ 颜色✗ │█████████████│ 0.88  (n=1,570)   │
│  ● 0002 +仿射增强/L2/EMA           0.7315     │  （"形状✓"=标签没有被噪声翻转；"颜色✓"=颜色和标签一致）│
│  ★ 0003 batch 颜色方向校准         0.9057 ◀   │  预测和颜色一致 12% · 和形状一致 81%            │
│  ● 0004 连续可靠度估计             0.8761     │                                                │
│  ◐ 0005 原宽度 CNN + 2×2 池化  训练中 63% ▓▓▓░ │ 样本预测（固定 48 个 val 样本，每组 12 个）       │
│                                               │ [全部] [形状✓颜色✓] [形状✓颜色✗] [形状✗颜色✓] [形状✗颜色✗] │
│ ─── 选中节点：0003 ──────────────────────────  │  ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐   │
│ [思路] [代码] [训练] [过程] [结论]             │  │ 7 ││ 2 ││ 9 ││ 4 ││ 1 ││ 8 ││ 3 ││ 5 │   │
│                                               │  │y1 ││y0 ││y1 ││y1 ││y0 ││y1 ││y0 ││y1 │   │
│ 标题：Source-calibrated color evidence +       │  │p1✓││p0✓││p0✗││p1✓││p0✓││p1✓││p1✗││p1✓│   │
│       batch orientation                       │  └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘   │
│ 思路：形状模型在任何域上都约 75% 正确；用它在   │   …（6 行 × 8 列；绿框=对，红框=错）           │
│   每个未标注 batch 里估计颜色和标签相关的方向，  │  悬停：数字 7 · 标签 1（未翻转）· 颜色 绿        │
│   再按训练域学到的强度加上颜色证据。            │        · 预测 1 · p=0.93                         │
│ 预期：val 明显超过 0.75                         │                                                │
│ 结果：0.9057（+0.1742 vs 上一个 SOTA）  ★ 新 SOTA │ 混淆矩阵（val）    pred 0   pred 1             │
│ 耗时 2m53s · 本节点 token 0.31M                 │          label 0   3,180     320               │
│                                               │          label 1     340   3,160               │
└───────────────────────────────────────────────┴────────────────────────────────────────────────┘
```

- **左侧节点列表**：`●` 已完成，`★` 刷新了 SOTA（同时是当前 SOTA 的会高亮），`◐` 运行中（显示进度条），`✗` 失败或超时，`📌` 是 agent 提交的。点击节点会在下方展开详情。
- **val 曲线**：每个节点一个点，SOTA 用阶梯线画出，baseline 是第一个台阶。点击曲线上的点和点击列表效果一样。
- **右侧默认显示当前 SOTA**。勾选"跟随选中"后改为显示左侧选中的节点，可以对比任意两个历史节点；标题会标明正在显示哪个 run。
- **捷径诊断**是专门为 ColoredMNIST 设计的：把 val 样本按"标签有没有被噪声翻转 × 颜色和标签是否一致"分成 4 组，分别算准确率，并统计预测和颜色一致、和形状一致的比例。这样能直接看出模型是在用形状还是在用颜色，以及上次 0.9 那种"推断颜色方向"的做法（形状✗的组准确率高，说明它在利用颜色）。

### 4.3 节点详情的其他标签页

```
[代码]  对比：(•) baseline  ( ) 上一个节点  ( ) 任选节点 [0002 ▾]      文件：[algorithms.py ▾]
┌───────────────────────────────────────────────────────────────────────────────┐
│ @@ class ERM(Algorithm):                                                       │
│ -        self.network = nn.Sequential(self.featurizer, self.classifier)        │
│ +        self.color_invariant = tuple(input_shape) == (2, 28, 28) ...         │
│ +            self.network_ema = copy.deepcopy(self.network)...                 │
│   ...（语法高亮；只改了 1 个文件：+118 −3）                    [完整文件] [复制] │
└───────────────────────────────────────────────────────────────────────────────┘

[训练]  step 3200 / 5000 ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  64% · 预计剩余 1m02s · GPU 41%
┌ loss ───────────────────┐  ┌ 准确率 ───────────────────────────────────────┐
│╲                        │  │      env0_out ───  env1_out ───                │
│ ╲___                    │  │      env2_out（val）━━━  （env2_in 已置空，不显示）│
│     ‾‾‾───────          │  │                                                │
└─────────────────────────┘  └────────────────────────────────────────────────┘
日志尾部（实时）：  step  loss   env0_out_acc  env1_out_acc  env2_out_acc ...

[过程]  （上一次 fml-val 开始之后，到这一次开始之前，agent 做了什么）
 11:25:01  💬 "第 2 次已经到 0.73，接近只用形状的上限；下一步试试……"
 11:25:20  ▶ python -c "…形状检查…"                       0.8s  ✓
 11:25:44  ✎ apply_patch  domainbed/algorithms.py  +96 −2            [展开]
 11:26:10  ▶ git diff --stat                                 0.0s  ✓
 11:26:12  🧪 fml-val --title "Source-calibrated color evidence…"  → 本节点
           本段 token：输入 0.30M（缓存 97%） · 输出 6.1k（推理 3.2k）

[结论]  （agent 用 fml-note 写的）"颜色方向校准有效：val +0.17；下一步检查对 batch 组成的依赖……"
```

### 4.4 主页面（S0/S1：还没有 baseline）

```
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│ 研究流程                                      │ 当前 SOTA：无                                │
│                                              │                                              │
│  ○ 0000 baseline（ERM 原版）  尚未运行         │   还没有 SOTA 模型。先在同样的沙盒里复现     │
│                                              │   baseline，作为第一个 SOTA，同时检查环境：  │
│  agent 尚未启动（要等 baseline 完成）          │     · 掩码数据、uv 环境、GPU 是否可用         │
│                                              │     · val 是否和本机参考值 0.2626 一致         │
│                                              │                                              │
│                                              │          [ ▶ 运行 baseline（约 3 分钟）]      │
│                                              │                                              │
│  S1 时：◐ 0000 baseline  训练中 41% ▓▓▓░░░    │  S1 时：环境检查  ✓ 沙盒  ✓ GPU  ✓ 数据      │
│         （选中后可看实时曲线和日志）            │         ◐ 训练   ○ 预测   ○ 和参考值比较      │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
 S2：右侧显示 baseline 的预测；顶部出现
   "baseline 0.2626 ✓（和参考值逐字节相同）。现在可以启动 agent：  [复制命令]
    tmux new-window -n codex "… scripts/codex/launch.sh <run_dir>"  → cx-start → 粘贴 /goal"
```

## 5. baseline 任务（按钮）

1. 用 `exec_host.run_exec` 起一个和 agent 同样配置的沙盒（**掩码数据**、同一个 uv 环境、GPU 0、CPU 槽位 L0），放入未修改的目标文件，运行 val 命令。这一步和 `fml-val` 做的事情一样，只是不计入 agent 的次数，也不触发计时。
2. 结果写到 `task/runs/baseline/`，格式和普通 run 相同（`run.json`、`code.json`、`output.log`、`results.jsonl`、`model.pkl`）。`fml_lib.list_runs` 只统计数字命名的目录，所以 baseline 不会占用 agent 的次数。
3. **环境检查**：沙盒能启动、GPU 可见、掩码数据的 md5 别名表生效、val 返回码为 0；val 和本机参考值（`rebaseline_results/Generalization_domainbed__l4x1/val_info.json`）比较，逐字节相同显示 ✓，否则显示差值并提示。
4. 把测得的 baseline val 写进 `task/meta.json` 的 `baseline_val`，`fml-status` 和 AGENTS.md 都用这个值。所以 AGENTS.md 和 `goal.txt` 要在 baseline 完成后再生成，这一步由 `new_run.py` 拆成两段完成。
5. 自动触发 `predict`，完成后进入 S2。
6. `launch.py` 检查 baseline 是否已经完成，没完成就拒绝启动 agent，防止跳过这一步。

## 6. 预测任务（右侧面板的数据）

- **触发**：baseline 和每个成功的节点完成后各运行一次，结果缓存在 `viz/pred/NNNN.json`，不会重复计算。
- **在哪里运行**：在沙盒里运行，因为它会执行 agent 写的代码。数据用 agent 视图（掩码数据）。用 GPU 0，每次约 10～20 秒；它可能和 agent 的下一次 `fml-val` 同时运行，影响很小，而且记录在案。
- **怎么预测**：用这个节点的代码快照新建一份仓库副本，按 DomainBed 的方式创建 `ColoredMNIST` 数据集，加载 `model.pkl`，**用和评测完全相同的 loader 和 batch（64）对整个 val split 做预测**。
  - 必须这样做，因为上一次 Codex 的方案在预测时依赖整个 batch 的组成（第 4.2 节），单个样本单独预测会得到不同的结果。
  - 校验：用这些预测算出的准确率必须等于 `run.json` 里记录的 val，相同就标 ✓，否则右侧显示警告。
- **输出**：
  - 每个 env 的准确率：env0/env1 的训练和留出部分、env2 out（来自 `results.jsonl` 的最后一行）；
  - 4 组捷径诊断、混淆矩阵、预测和颜色或形状一致的比例；
  - 固定 48 个样本的图像、元数据、预测和概率。
- **样本选择**：从 env2 的 val 部分里，按 4 组各抽 12 个，用固定种子，所有节点用同一批样本，方便对比。各组样本数量差别很大，所以每组固定 12 个，而不是按比例抽。
- **样本元数据**：原始数字、标签是否被翻转、颜色，都用 DomainBed 的代码重建（和 `build_task_view.py` 计算掩码索引的做法一样）。
- **为什么不显示 env2 的 in-split（隐藏 test）**：agent 看到的这部分图像是空白的，在上面的预测没有意义。真实 test 只在最终 test（`finalize.py`）之后显示。

## 7. agent 侧的改动（为了让前端能解析）

1. **`fml-val` 增加参数**：`fml-val --title "<不超过 60 个字符>" --idea "<改了什么、为什么>" [--expect "<预期>"]`。
   - 参数写入 `runs/NNNN/card.json`。
   - **缺少 `--title` 或 `--idea` 时直接拒绝运行，不计次数**，并打印正确用法。这样 agent 必须提供说明。
2. **新增 `fml-note <run> "<结论>"`**：写入 `runs/NNNN/note.json`。提示词要求 agent 每次看到结果后写一两句结论。不强制，没写的节点在页面上显示"未写结论"。
3. **`fml-val` 额外保存** `model.pkl`（ColoredMNIST 的小 CNN 约 1.5 MB）和 `results.jsonl`（训练曲线），在删除执行副本之前复制到 `runs/NNNN/`。
4. **提示词**（AGENTS.md）补充：
   - 上面两个命令的用法；
   - "baseline 已经运行过，记为 run 0，结果是 0.2626；不要花一次 `fml-val` 重跑原版代码"；
   - 研究日志 NOTES.md 不变。

   goal 文本不变。
5. 这些改动只在 viz 运行（`new_run.py --viz`）里启用。普通的 Codex 运行和 Claude Code 流程，`fml-val` 的用法都不变：参数是可选的，只有设置了 `meta.require_card` 时才强制要求。
6. 注意：提示词和工具变了，这次的结果不能和上一次 trial_1h 严格对比（agent 会少花一次 baseline 运行，还要写说明）。

## 8. 技术实现

| 部分 | 选择 | 理由 |
|---|---|---|
| 后端 | Python 3.11 + FastAPI + uvicorn，放在新的 uv 环境 `~/fml_envs/viz` | 方便写 SSE 和静态文件服务；依赖少 |
| 预测 | `scripts/codex/viz/predict.py`，在 domainbed 环境的沙盒里运行 | 需要 torch 和 DomainBed 的代码 |
| 前端 | 一个 HTML 文件 + 原生 JS 模块 + uPlot（曲线）+ diff2html（代码 diff）+ highlight.js，都放在本地 | 不需要 Node 或打包构建；在 Mac 的 Safari 和 Chrome 上都能用 |
| 推送 | SSE：`/api/events` 推送节点或状态的变化；数据从 REST 接口按需取（`/api/state`、`/api/node/{id}`、`/api/pred/{id}`、`/api/samples/intro`） | |
| 任务 | 后端用子进程运行 baseline 和 predict，状态写在 `viz/jobs.json`。服务重启后能恢复显示；正在跑的任务会被标成中断，可以重试 | |
| 启动 | `scripts/codex/viz/serve.sh <run_dir>`，然后在 Mac 上 `ssh -L 8765:localhost:8765 ec2-user@<host>`，打开 http://localhost:8765 | |

新增文件：`scripts/codex/viz/{server.py, rollout_index.py, predict.py, coloredmnist_meta.py, static/…}`。会修改的文件：`fml_lib.py`（第 7 节）、`scripts/codex/new_run.py`（`--viz`、分两段生成）、`scripts/codex/launch.py`（检查 baseline）。

## 9. 需要你决定的事项

1. **访问方式**：SSH 端口转发（推荐，不开公网端口）。也可以在安全组里开端口，但这样就需要加登录。
2. **右侧样本**：只显示 env2 的 val 部分，4 组各 12 个、所有节点用同一批（推荐）；还是也显示训练域 env0/env1 的样本？
3. **强制说明**：`fml-val` 缺少 `--title/--idea` 时拒绝运行（推荐，保证每个节点都有思路说明）；还是只提醒，不强制？
4. **reasoning summary**：在 config 里打开 `model_reasoning_summary`，"过程"页就能看到模型的思考摘要。需要先实测 Bedrock 上的 Astra 支不支持；它也会让输出 token 略多一些。
5. **最终 test 按钮**：agent 结束后在页面上一键运行 `finalize.py` 并显示真实 test 和审计结果，要不要做？
6. **一个页面对应一次运行**：服务启动时指定 run 目录，页面顶部可以切换查看以前的运行（比如 trial_1h 只能看，没有 card 和预测）。可以吗？

---

## 10. 实现（2026-11-24）

按第 9 节的决定实现：SSH 端口转发；右侧只显示 env2 的 val 样本；`fml-val` 缺少 `--title/--idea` 时拒绝运行；开启思考摘要；不做最终 test 按钮；只保留一次运行，"新建运行"会清空历史。

### 10.1 使用

```bash
# EC2 上（服务只监听 127.0.0.1:8765）
cd ~/FML-bench && setsid nohup scripts/codex/viz/serve.sh > /tmp/viz_server.log 2>&1 < /dev/null &
# Mac 上
ssh -L 8765:localhost:8765 ec2-user@<host>      # 然后打开 http://localhost:8765
```

1. 入口页看完后进入主页面。没有运行时点"新建运行"。
2. 点"运行 baseline"（约 3 分钟）：7 项检查逐项显示，完成后 baseline 成为第一个 SOTA。
3. 按页面顶部的命令启动 agent：`launch.sh` → `cx-start` → 粘贴 goal（`cat /task/goal.txt`）。
4. 每次 `fml-val` 结束后，页面会自动在沙盒里算这个节点的预测（约 5 秒）。
5. 想重新开始时点"新建运行"：会删除整个运行目录（节点、模型、会话记录、预测），用同样的参数新建一次。agent 沙盒或任务还在运行时会被拒绝。

运行目录固定为 `~/fml_agent_runs/codex/Generalization_domainbed/viz`（`FMLBENCH_VIZ_RUN` 可以改）。参数：GPU 0、CPU 槽位 L0、1 小时、20 次，参考 baseline 为 `rebaseline_results/Generalization_domainbed__l4x1`。

### 10.2 文件

| 文件 | 内容 |
|---|---|
| `scripts/codex/viz/server.py` | FastAPI 服务：状态汇总（节点、SOTA、阶段、goal、token、agent 是否在运行）、节点详情（思路、diff、训练曲线、过程）、预测、baseline 按钮、新建运行、SSE 推送；后台线程为每个新完成的节点运行预测 |
| `scripts/codex/viz/jobs.py` | 宿主机任务：`prepare`（样本元数据和图片）、`baseline`（7 步检查 + 训练 + 预测 + 比较 + 生成提示词）、`predict`；需要 GPU 的任务用文件锁串行执行；也可以在命令行运行 |
| `scripts/codex/viz/coloredmnist.py` | 运行 DomainBed 自己的 `train.py`，在训练开始之前停下，记录每个域的张量、划分和原始数字；宿主机和沙盒共用 |
| `scripts/codex/viz/predict_inner.py` | 在沙盒里加载节点的代码和 `model.pkl`，对 val 做预测（执行 agent 的代码，所以必须在沙盒里） |
| `scripts/codex/viz/rollout_index.py` | 把 Codex 会话记录整理成事件列表（用户消息、发言、思考摘要、命令、输入、补丁、工具、goal）和 token 用量 |
| `scripts/codex/viz/static/` | 前端：`index.html`、`app.js`（原生 JS，agent 产生的文本全部转义）、`style.css` |
| `scripts/codex/viz/serve.sh` | 启动服务（uv 环境 `~/fml_envs/viz`：fastapi 0.118、uvicorn 0.37） |
| `sandbox_tools/fml_lib.py` 等 | `fml-val --title/--idea/--expect`（`meta.require_card` 时必须提供）、`--baseline`（宿主机触发，写到 `runs/baseline/`，不计次数）、`meta.keep_artifacts` 时保存 `model.pkl` 和 `results.jsonl`、新命令 `fml-note`、`fml-status` 显示 baseline。普通运行不受影响 |
| `scripts/codex/new_run.py` | `--viz`：上面三个开关、开启 `model_reasoning_summary = "auto"`；AGENTS.md 和 goal.txt 推迟到 baseline 完成后再生成（`write_prompts`），AGENTS.md 里加上 `--title/--idea`、`fml-note` 的用法和"baseline 已运行"的说明 |
| `scripts/codex/launch.py` | 可视化运行在 baseline 成功之前拒绝启动 agent |

### 10.3 和设计稿不同的地方

- **标签的方向**：DomainBed 的规则是 `label = (digit < 5)`，所以**数字 0–4 的标签是 1，5–9 是 0**，草图里写反了。颜色由加噪后的标签决定（通道 0 红、通道 1 绿）。重建出的数据：颜色和标签一致的比例 env0 0.898、env1 0.799、env2 0.098，标签翻转率都约 0.247，和设计值一致。val 共 6,999 个样本，4 组分别是 489 / 4,722 / 171 / 1,617。
- **预测的 batch**：DomainBed 评测用的是 `RandomSampler`，batch 的组成取决于训练结束时的全局随机数状态，没法精确复现。所以预测按同样的方式（batch 64、随机分 batch）做两遍，用两个不同的种子：
  - 两遍结果相同，而且准确率等于记录的 val，标 ✓；
  - 两遍不同，标"⚠ 预测依赖 batch 组成"，并给出有差别的样本数，比如上次 Codex 试跑的那种 batch 级方法；
  - 只是准确率不同，显示差值。
- **样本区**：改成每组一行，每行 12 张，左边写组名和本组答对的数量，4 组可以直接对比；对错用图片右上角的 ✓/✗ 图标加边框颜色表示。
- **"下一步"节点**：agent 运行中时，节点列表最后有一个 `next`，显示最近一次 `fml-val` 开始之后 agent 正在做的事。

### 10.4 自测（2026-11-24，单独的运行 `viz_selftest`，测完已删除）

- **baseline 按钮**：7 步都通过（8 CPU、1 × L4、掩码数据、训练 158 秒、预测和记录的 val 相同、和参考值逐字节相同、提示词已生成）；训练中的进度、预计剩余时间、曲线和日志都实时更新。
- **agent 侧**：
  - 不带 `--title` 的 `fml-val` 被拒绝，返回码 2，不计次数；
  - 带说明的灰度化 ERM 运行成功，val 0.6837；
  - `fml-note` 和 `fml-status`（显示 baseline 行）正常。
- **预测和诊断**：
  - baseline 的预测和颜色一致 82%、和形状一致 48%；
  - 灰度化 ERM 的预测和颜色一致 35%、和形状一致 87%；
  - 两个节点的预测都和记录的 val 完全相同，没有依赖 batch。
- **过程页**：在沙盒里用 `codex exec` 生成的会话记录能正确显示（用户消息、发言、命令及耗时和返回码、token）。
- **思考摘要**：Bedrock 上的 Astra 支持，会话记录里有 `Reasoning.summary_text`。
- **"新建运行"**：删除后重建，页面回到 S0。
- 前端用无头 Chromium 检查了入口页、S0、S1、S2、主页面的 5 个标签页，没有控制台报错。
