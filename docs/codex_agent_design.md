# 用 Codex（goal 模式）作为 FML-bench agent：方案与实现

在 Claude Code `/goal` 方案（`docs/claude_code_agent_design.md`）的基础上，把 agent 换成 Codex CLI 的 goal 模式。数据掩码、沙盒布局、`fml-val` / `fml-status` / `fml-submit`、最终 test 和审计规则都沿用，下面只写不同的地方。第一次试跑：单卡 L4 机器上的 domainbed（ColoredMNIST），1 小时、最多 20 次 `fml-val`。

- 分支：`codex-awesome可视化`（基于 `repro/bedrock-uv-sandbox`）
- Codex CLI 0.172.0；模型 Bedrock `openai.gpt-6-astra`，reasoning effort `high`
- 机器：1 × NVIDIA L4（24 GB），8 vCPU（AMD EPYC 7R13），30 GB 内存，Amazon Linux 2023

---

## 1. 和 Claude Code `/goal` 的区别

| 项目 | Claude Code | Codex | 做法 |
|---|---|---|---|
| 结束判定 | 单独的判定模型（Haiku）在每轮结束后读对话，判断条件是否满足 | **没有判定模型**：主模型自己调用 `update_goal(status="complete")` 结束；空闲时自动开始下一轮（goal continuation） | goal 文本改为自检形式："以下条件都由本会话中的命令输出证明后，才把 goal 标为完成" |
| 预算 | 交互模式下没有内置上限 | goal 可以设 token 预算，**不能限挂钟时间** | 时间仍由 `fml-val` 在截止后拒绝运行、沙盒在截止 + 1 小时后关闭来保证；不设 token 预算 |
| 任务说明 | `/task/CLAUDE.md`（自动加载工作目录上层的 CLAUDE.md） | 只读取 git 根目录到 cwd 之间的 `AGENTS.md`，`/task/AGENTS.md` 不会被读到；放进仓库又会被 agent 改动并出现在 `git diff` 里 | 只读挂载为全局说明 `$CODEX_HOME/AGENTS.md`（`~/.codex/AGENTS.md`） |
| 长时间命令 | Bash 工具单次最长 10 分钟，要求后台运行 `fml-val` | `exec_command` 在 PTY 会话里运行，超过 yield 时间后返回会话号，用 `write_stdin` 轮询 | 提示词改为"在命令会话里运行，轮询到命令结束再读结果" |
| 自带沙盒 | 无 | 有（会嵌套在 bwrap 里） | `--dangerously-bypass-approvals-and-sandbox`，隔离靠外层 bwrap |
| 网络工具 | 关闭 WebFetch/WebSearch | 原生 `web_search` 工具 | `web_search = "disabled"`；实测工具列表里没有 web_search |
| 后台服务 | 无 | TUI 默认要启动 app-server 守护进程，依赖 `~/.codex/packages` 的完整安装 | `codex --no-daemon`（沙盒里只挂载 release 目录） |
| 凭证 | `CLAUDE_CODE_USE_BEDROCK=1` + 实例角色 | `model_provider = "amazon-bedrock"`，profile `codex-bedrock`（`credential_process` 读默认链，即实例角色） | 只读挂载 `~/.aws/config`（里面没有密钥），沙盒共享网络能访问 IMDS |
| 会话记录 | `CLAUDE_CONFIG_DIR/projects/**/*.jsonl` | `CODEX_HOME/sessions/**/rollout-*.jsonl`；goal 状态在 `goals_1.sqlite` | `scripts/codex/finalize.py` 里的解析器（第 3 节） |
| 费用 | 按 token 和反推的单价计算 | 没有可靠的 Bedrock 单价 | 只报告 token（含 reasoning token） |

Codex 的 goal 规则（来自 0.172.0 内置提示）：只有目标真正完成、或用户要求暂停时才调用 `update_goal`；同一个阻塞条件连续出现 3 轮以上才标 `blocked`；状态有 `active / paused / blocked / usage_limited / budget_limited / complete`。TUI 命令：`/goal <目标>`、`/goal edit`、`/goal pause`、`/goal resume`、`/goal clear`。

## 2. 本机的适配

- **路径**：`scripts/codex/env_l4.sh` 设置 `FMLBENCH_DATA_CACHE=~/fml_data_cache`、`FMLBENCH_AGENT_VIEWS=~/fml_data_agent`、`FMLBENCH_CC_RUNS=~/fml_agent_runs`（原默认值都在 `/mnt/nvme`）。`build_task_view.py` 里写死的临时目录 `/mnt/nvme` 改为 `FMLBENCH_AGENT_VIEWS`。
- **CPU 槽位**：`common.CPU_SLOTS` 新增 `L0 = 0-7`（整台机器）。
- **baseline**：提示词和 `fml-status` 用本机实测的 val **0.2626**（`rebaseline_results/Generalization_domainbed__l4x1/`，确定性，两次逐字节相同），不用仓库采用的 g7e.24xlarge 版 0.2393。agent 第一次 `fml-val` 跑 baseline 时会得到完全相同的数。`new_run.py --baseline-dir` 指定，记录在 `meta.json` 的 `baseline_source`。
- **耗时提示**：沙盒里实测 baseline val 154 秒（`scripts/codex/val_times_l4.json`），提示词写"about 3 minutes"；GPU 写成 `1 × NVIDIA L4 (24 GB)`，8 CPU cores。
- **WRAP-UP 阶段**：原规则是"剩余 5 次"时进入 WRAP-UP（按 100 次设计），20 次时会在第 15 次就进入，而 goal 条件允许此时结束。改为保留 5% 的次数（至少 1 次）：100 次仍是 95，20 次是 19。时间条件不变：剩余时间少于 max(min(1 小时, 预算的 10%), 2 × 最近 3 次中最长的耗时)，1 小时预算时约为最后 6 分钟。
- **验证**（`verify_task_view.py --baseline-dir ... --val-times scripts/codex/val_times_l4.json`，新增的两个参数，默认行为不变）：掩码 16,334 张（env2 in-split，与 g7e.24xlarge 相同）；掩码数据上的 val 与本机 baseline **逐字节相同**；掩码后的 test 0.5037（真实 0.2706），val 154 秒。

## 3. 实现

| 文件 | 内容 |
|---|---|
| `scripts/codex/new_run.py` | 创建运行目录 `$FMLBENCH_CC_RUNS/codex/<TASK>/<run_id>/`：`task/`（同 Claude Code）、`codex_home/`（`config.toml`、`AGENTS.md`、会话记录）、`cache/`、`host.json`。AGENTS.md 由 Claude Code 的 `build_claude_md` 生成，再替换硬件、耗时和等待 `fml-val` 的说明 |
| `scripts/codex/launch.sh`、`launch.py` | 进入沙盒。复用 `scripts/claude_code/launch.py`（计时、CPU 绑定、截止 + 1 小时关闭），只换挂载 |
| `scripts/claude_code/common.py` | `bwrap_args(..., codex_home=...)`：挂载 Codex release 目录到 `/opt/codex/bin`、`codex_home` 到 `~/.codex`、`~/.aws/config` 只读、`AGENTS.md` 只读覆盖；设置 `CODEX_HOME`、`AWS_REGION`。不传 `codex_home` 时和原来完全一样 |
| `scripts/claude_code/sandbox_tools/bin/cx-start` | 沙盒里启动 Codex TUI：`codex --no-daemon --dangerously-bypass-approvals-and-sandbox -C <repo>` |
| `scripts/claude_code/sandbox_tools/bashrc` | `FMLBENCH_AGENT=codex` 时显示 Codex 的用法 |
| `scripts/codex/finalize.py` | 复用 Claude Code 的最终 test、审计和 harness 格式输出，换成 Codex 会话记录的解析器：`exec_command` 调用（耗时和返回码取自 `CommandExecution` 事件）、`write_stdin` 输入的非空内容（也按命令审计）、`apply_patch` 等其他工具、`web_search_call`（记为 WebSearch）、每次响应的 token、goal 的状态变化（`thread_goal_updated` 事件和 `update_goal` 的返回）。结果写到 `benchmark_results/codex/<TASK>/<run_id>/` |

`config.toml`（每次运行生成）：`model_provider = "amazon-bedrock"`、`model`、`model_reasoning_effort`、`approval_policy = "never"`、`sandbox_mode = "danger-full-access"`、`web_search = "disabled"`、`check_for_update_on_startup = false`、`[features] goals = true, memories = false`、Bedrock profile 和区域、工作目录设为 trusted。`codex exec --strict-config` 能通过。

## 4. 冒烟测试（2026-11-24，运行 `smoke`，0.5 小时 / 3 次）

| 检查 | 结果 |
|---|---|
| 沙盒可见性 | 只有 domainbed 环境、一张 L4、8 个 CPU；看不到 FML-bench 仓库；`~/.codex` 里只有 AGENTS.md 和 config.toml |
| `codex exec`（连通性） | Bedrock 调用正常（实例角色），能读到 AGENTS.md（答出任务名和指标），工具列表里没有 web_search。注意：stdin 不是终端时 `codex exec` 会等 stdin 结束，要加 `< /dev/null` |
| TUI + `/goal`（tmux 驱动） | 简单目标 4 秒完成，agent 自己调用 `update_goal(complete)`，状态栏显示 "Goal achieved" |
| `/goal` + 一次 `fml-val` | agent 用 `write_stdin` 每 45 秒轮询，等 `fml-val` 结束（2m34s，0.262609，与 baseline 相同）后才标完成 |
| `finalize.py` | 真实数据上重训 154 秒，val 0.2626、test 0.2706（与本机 baseline 相同）；审计无标记；7 条命令、token 和 goal 历史都解析正确 |

冒烟测试的 harness 结果改名为 `benchmark_results/codex/Generalization_domainbed/_smoke_20261124`（以 `_` 开头，打分脚本会跳过）。

## 5. 试跑步骤

```bash
cd ~/FML-bench && . scripts/codex/env_l4.sh
# 运行目录已创建：~/fml_agent_runs/codex/Generalization_domainbed/trial_1h（1 小时、20 次、baseline 0.2626）
# 新建其他运行：python3 scripts/codex/new_run.py Generalization_domainbed --gpu 0 --cpus L0 \
#     --budget-hours 1 --max-runs 20 --baseline-dir rebaseline_results/Generalization_domainbed__l4x1 --run-id <id>

# 1. 在 tmux 里进入沙盒（第一次进入时开始计时）
tmux new-window -n codex "bash -c '. scripts/codex/env_l4.sh; scripts/codex/launch.sh ~/fml_agent_runs/codex/Generalization_domainbed/trial_1h; bash'"
# 2. 沙盒内
cat /task/goal.txt      # 一行，以 /goal 开头
cx-start                # 启动 Codex TUI，然后粘贴上面那一行
#    中途退出后：cx-start resume --last（截止时间不变）
# 3. 结束后（宿主机）
python3 scripts/codex/finalize.py ~/fml_agent_runs/codex/Generalization_domainbed/trial_1h
```

- 打分：`compute_agent_metrics.py` 读的是 `ml_tasks/.../test_info.json`（g7e.24xlarge 版 0.2501）。本机 baseline test 是 0.2706，NI 要按两种口径分别算：`NI = max(0, test − p_baseline)`（指标范围 0～1）。
- 1 小时 / 20 次的预算下，每次 `fml-val` 约 2.6 分钟，20 次就要 52 分钟，实际大概率先用完时间，跑不满 20 次。
- 在沙盒的 bash 提示符下按 Ctrl-C 会结束整个沙盒（和 Claude Code 方案相同，见其 7.5 节第 3 条）；在 Codex TUI 里按 Ctrl-C 应该只中断 Codex 自己（按 Claude Code 的经验推断，没有实测）。
