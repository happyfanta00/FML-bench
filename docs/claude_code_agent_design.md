# 用 Claude Code（`/goal` 模式）作为 FML-bench agent：方案

目标：每个任务一个独立的沙盒，沙盒里只有这个任务需要的代码、数据、uv 环境和分配给它的 GPU。你手动启动沙盒，在里面用 `/goal` 运行 Claude Code。会话结束后，由脚本在沙盒外完成最终 test 和结果汇总，结果与现有的 ai_scientist_v2 可比。

---

## 1. 已确定的决定

| # | 事项 | 决定 |
|---|---|---|
| 1 | 网络 | **共享网络**（Claude Code 要访问 Bedrock），靠事后审计发现下载数据、安装包等行为 |
| 2 | 运行次数 | 提示词说明最多 **100 次 `fml-val`**（失败的也算），**不强制**，事后审计是否超出 |
| 3 | 最终提交 | agent 用 **`fml-submit`** 显式选择；val 最优的那次运行的 test 作为参考一并报告 |
| 4 | `fml-val` 之外跑训练 | **不允许**。沙盒里能看到分配给它的 GPU；提示词要求只通过 `fml-val` 训练，GPU 只用于快速检查，事后审计 |
| 5 | 执行方式 | agent 在沙盒内**直接运行 `fml-val`**，不经过宿主机。每次运行用一份新的 baseline 仓库副本，**只放入目标文件** |
| 6 | CPU | 每个 GPU 任务 **16 个逻辑 CPU**，ART **48 个**。合计超过 96，所以分两批运行（4.4 节） |
| 7 | privacymeter 的残余泄漏 | 接受，在结果中注明 |
| 8 | 挂钟时间 | **24 小时**（100 次 × 每次约 10 分钟就要 16.7 小时，还要留出 LLM 思考的时间）；截止后 `fml-val` 拒绝开始新的运行 |
| — | 其他 | 数据掩码、单次运行上限 2400 秒、提示词结构，保留"AI PhD student"人设 |

---

## 2. 已验证的事实（2026-11-04，本机 Claude Code 2.1.318）

| 项目 | 结果 |
|---|---|
| Claude Code 在 bwrap 沙盒里运行 | 可以。原生二进制（`~/.local/share/claude/versions/<版本>`）只读挂载即可；`$HOME` 是空的 tmpfs，配置目录用 `CLAUDE_CONFIG_DIR` 指向一个每次运行单独的可写目录 |
| Bedrock | `CLAUDE_CODE_USE_BEDROCK=1`、`AWS_REGION=us-east-1`、`ANTHROPIC_MODEL=us.anthropic.claude-opus-5-5`；凭证通过共享网络走 IMDS 的实例角色，可用 |
| `/goal` | `claude -p "/goal <条件>"` 在沙盒里能跑完（2 轮，$0.09），判定模型用的是 `ANTHROPIC_DEFAULT_HAIKU_MODEL=us.anthropic.claude-haiku-5-5`（Bedrock 上可用） |
| 配置隔离 | 新建的配置目录里只有本次会话的状态，不会读到宿主机的 `~/.claude`（包括 memory、其他项目的会话记录、本仓库的讨论） |
| 会话记录 | `$CLAUDE_CONFIG_DIR/projects/<cwd 编码>/<session>.jsonl`，包含每次工具调用和 token 用量，可以事后审计 |
| 费用 | `-p --output-format json` 的结果里有 `total_cost_usd` 和按模型拆分的用量；交互模式下 `/goal`（不带参数）会显示当前的 token 花费 |
| 预算上限 | `--max-budget-usd` **只在 `-p` 模式下有效**；交互模式下只能靠监控 |

`/goal` 的机制（官方文档 https://code.claude.com/docs/en/goal.md）：

- 条件最多 4000 字符。设置后立即开始一轮，每轮结束时由小模型根据**对话内容**判断条件是否满足（它不读文件、不运行命令），不满足就自动开始下一轮。
- 结束条件：判定满足、判定不可能满足，或者出现必须人工处理的错误（鉴权失败、模型不可用、上下文压缩失败）。
- 有后台任务在跑时会推迟判定，默认每 30 分钟检查一次后台任务（`CLAUDE_CODE_GOAL_CHECKIN_MINUTES`）。
- `/goal` 依赖 hooks 机制，所以**不能用 `--bare`**。交互模式第一次进入工作目录时需要确认信任。

---

## 3. 风险与对策

### 3.1 隐藏 test 集泄漏

Claude Code 有完整的 shell 权限：它可以直接运行 `--split test`，也可以按评测脚本里写死的种子算出 test 索引。所以要**在数据层面把 test 拿掉**，并且做审计。

| 任务 | 隐藏 test 是什么 | 做法 |
|---|---|---|
| opacus | CIFAR-10 训练集 50K 中按种子 42 抽出的 21%（10,500 张） | agent 视图的 batch 文件里，这些位置的图像置零 |
| openood | CIFAR-10 测试集中的 7K（种子 42）+ SVHN 测试集中的 5K（只在 test 里当 OOD 用） | CIFAR-10 对应位置置零；SVHN 整体置零（val 不用，但脚本无条件加载，保持文件形状） |
| usb | CIFAR-100 测试集的 70%（种子 42 随机） | 置零（usb 专用副本） |
| pycil | CIFAR-100 测试集的 70%（按类分层，种子 42） | 置零（pycil 专用副本，掩码和 usb 不同） |
| ART | MNIST 测试集的 70%（按类分层，种子 42） | `mnist.npz` 副本中置零 |
| domainbed | ColoredMNIST 留出域 env2 的 in-split（70%） | 用 DomainBed 自己的代码算出对应的原始 MNIST 索引后置零。val 运行时 DomainBed 会顺带评估 env2_in，置零后这个数没有意义 |
| officehome | Art 域的 in-split（70%） | 对应图片换成同尺寸的空白图 |
| privacymeter | 同一次运行中训练的 70% 目标模型，不是数据 | 无法隔离：val 运行本身就会算出所有目标模型的 AUC。按决定 7 接受，结果中注明 |

- **只改图像，不改标签。** pycil 和 ART 是按类别分层划分的，改了标签划分就变了。
- **每个任务一份掩码数据**（`/mnt/nvme/data_agent/<TASK>/`，只读），总量不到 2 GB。
- **自动验证**：用掩码数据运行官方 val 命令，结果必须和 baseline 一致（opacus、pycil、usb、domainbed、officehome 应逐字节相同；openood、privacymeter 在噪声范围内）；运行 test 命令，结果应接近随机水平。两项都通过才能用。
- 提示词里直接告诉 agent："隐藏 test 对应的样本已被替换为空白占位"。
- **网络（决定 1）**：沙盒能上外网，agent 理论上可以重新下载原始数据，在 GPU 上算出 test 指标。对策是审计会话记录里所有的网络访问（`curl`、`wget`、`pip`、`git clone`、`huggingface`/`torchvision` 的下载等），并关闭 Claude Code 的 WebFetch/WebSearch 工具。

### 3.2 预算与公平性

| 项目 | 规则 | 怎么执行 |
|---|---|---|
| 挂钟时间 | 24 小时 | `fml-val` 在截止后拒绝开始新的运行；沙盒启动脚本在截止时间加 1 小时后结束沙盒 |
| `fml-val` 次数 | 最多 100 次，失败的也算 | 提示词说明，事后审计 |
| 单次运行 | 2400 秒 | `fml-val` 强制执行（和标准协议相同）。提交的配置也必须在这个时限内跑完 |
| 训练 | 只能通过 `fml-val`；GPU 和 CPU 上只允许快速检查（导入、形状检查、在几个样本上跑前向、显存或速度试探） | 提示词说明，事后审计会话记录里的命令 |
| GPU | 沙盒里只挂载分配到的那一张（`/dev/nvidia<N>` 和 ctl/uvm），用不到其他任务的卡 | 沙盒挂载 |
| Token 费用 | 交互模式下由你监控；`-p` 模式下可以加 `--max-budget-usd` | 汇总时从会话记录统计 |

- **agent 自己在 GPU 上做检查时，如果和 `fml-val` 同时运行，会拖慢 `fml-val`**，甚至让它超过 2400 秒。这是 agent 自己造成的，提示词里会提醒。
- **CPU 核数对耗时的影响很小**：在 16 个 CPU（ART 48 个）下重测，各任务 baseline 的 val 耗时和 96 核下基本相同，最长的 officehome 657 秒（7.1 节）。实测耗时已写进各任务的提示词。

### 3.3 最终提交

- `fml-submit [run_id]` 把一次成功的 `fml-val` 运行标记为最终提交（默认是最后一次）。最终 test 用那次运行的目标文件快照。
- 同时报告 val 最优的那次运行的 test，作为和 ai_scientist_v2 可比的参考（ai_scientist_v2 由 harness 自动选 val 最优）。
- **只有 `target_files` 进入评测**，和 harness 一致。`fml-val` 也只把目标文件放进一份新的 baseline 仓库里运行（决定 5），所以 agent 在开发时看到的 val 和最终评测用的是同一份代码。

### 3.4 其他

- **执行记录在沙盒内可写**：`fml-val` 的记录和 Claude Code 的会话记录都在沙盒里，agent 理论上能改。审计以它们为准，同时用最终 test 重新训练得到的 val 和记录里的 val 做一致性检查。
- **最终 test 的环境**：在一个新的、配置相同的沙盒里运行（任务的 uv 环境、共享网络、真实数据）。agent 会话里才有的东西（自己装的包、目标文件之外的文件）那时都不存在，提示词规则第 4 条要求不要依赖它们。
- **长时间运行**：`fml-val` 会一直阻塞到运行结束。Claude Code 的 Bash 工具单次最长 10 分钟，所以提示词要求把 `fml-val` 放到后台运行。`/goal` 在后台任务结束前会推迟判定。
- **上下文压缩**：长会话会自动压缩上下文。任务说明放在只读的 `CLAUDE.md` 里（每次都会重新加载），并要求 agent 维护研究笔记 `NOTES.md`。

---

## 4. 沙盒设计

每个任务一次运行一个沙盒，运行目录为 `/mnt/nvme/cc_runs/<TASK>/<run_id>/`。

### 4.1 挂载

| 沙盒内路径 | 来源 | 权限 | 说明 |
|---|---|---|---|
| `/task/workspace/<TASK>/<repo>/` | 任务模板的新副本，数据软链指向掩码数据 | 可写 | Claude Code 的工作目录 |
| `/task/baseline/<TASK>/<repo>/` | 同一份 baseline 仓库 | 只读 | `fml-val` 每次从这里复制出执行副本 |
| `/task/ml_tasks/<TASK>/` | 只包含 val 命令引用到的文件 | 只读 | 不包含 `baseline_results/test_info.json`、`config.json`（里面有 test 命令）、`prompt.json` |
| `/task/CLAUDE.md` | 由 `prompt.json` 和 baseline 生成（第 6 节） | 只读 | Claude Code 会自动加载工作目录上层的 CLAUDE.md |
| `/task/scratch/` | 空目录 | 可写 | 笔记、草稿、最终报告 |
| `/task/runs/` | `fml-val` 写入的记录 | 可写 | 每次运行的目标文件快照、`val_info.json`、完整日志、耗时、返回码 |
| `/task/exec/` | `fml-val` 的执行副本 | 可写 | 运行结束后删除（日志和结果已保存到 `/task/runs/`） |
| `/mnt/nvme/data_agent/<TASK>/` | 掩码数据 | 只读 | 只有这个任务的数据 |
| `~/fml_envs/<env>/`、uv 的 Python | 宿主机 | 只读 | 只有这个任务的环境 |
| `~/.cache/` | 每次运行一份，预先放好需要的预训练权重（如 officehome 的 ResNet-50） | 可写 | |
| `$CLAUDE_CONFIG_DIR` | 每次运行一份的空目录 | 可写 | 会话记录、Claude Code 状态 |
| `/usr/local/bin/{claude,fml-val,fml-status,fml-submit}` | Claude Code 二进制和 3 个工具 | 只读 | |
| `/dev/nvidia<N>` + ctl/uvm | 分配到的 GPU | | ART 不挂 GPU |

**看不到的东西**：其他 GPU、FML-bench 仓库本身（agents/、benchmark/、docs/、其他任务）、`rebaseline_results/`、真实数据、宿主机的 `~/.claude`、`~/.ssh`、`~/.aws`、其他 uv 环境。

### 4.2 `fml-val` 的执行方式

每次运行：从 `/task/baseline/` 复制一份新的仓库到 `/task/exec/NNNN/` → 放入工作目录里当前的目标文件 → 在任务的 uv 环境里运行 val 命令（超时 2400 秒）→ 把 `val_info.json`、标准输出和错误、耗时、返回码、目标文件快照写到 `/task/runs/NNNN/` → 删掉执行副本。

- 和最终 test 一样，只有目标文件会生效，其他文件的改动不会被带进来。
- 多次调用按顺序排队（用文件锁），每次都计入次数。

### 4.3 工具

- `fml-val`：按 4.2 运行一次，打印 val 指标、日志路径和预算状态。截止后拒绝运行。
- `fml-status`：打印已用和剩余的时间、`fml-val` 次数（`37/100`）、当前阶段、最优 val 和运行历史。
  - 阶段：`SEARCH` → `WRAP-UP`（剩余时间少于 max(min(1 小时, 预算的 10%), 2 × 最近 3 次运行中最长的耗时)，或者已用 95 次）→ `BUDGET EXHAUSTED`（截止时间已过，或者已用 100 次）。
- `fml-submit [run_id]`：把一次成功的运行标记为最终提交，打印 `SUBMITTED run N`。

### 4.4 CPU 分配和批次

机器是 1 个 socket、48 个物理核 / 96 个逻辑 CPU，逻辑 CPU `i` 和 `i+48` 是同一个物理核。每个任务分整数个物理核（用 `taskset` 绑定整个沙盒），避免两个任务共用同一个物理核：

| 槽位 | 逻辑 CPU | 用途 |
|---|---|---|
| G0 | 0-7, 48-55 | GPU 任务（GPU0） |
| G1 | 8-15, 56-63 | GPU 任务（GPU1） |
| G2 | 16-23, 64-71 | GPU 任务（GPU2） |
| G3 | 24-31, 72-79 | GPU 任务（GPU3），只在第 2 批使用 |
| ART | 24-47, 72-95 | ART，只在第 1 批使用 |

- **第 1 批**（24 小时）：3 个 GPU 任务（G0～G2）+ ART，正好 96 个逻辑 CPU。
- **第 2 批**（24 小时）：剩下 4 个 GPU 任务（G0～G3）。
- 总共约 48 小时。建议把最重的 officehome、openood、privacymeter 放在第 2 批，那时 GPU 任务之间的 CPU 竞争较小；具体分批等 16 个 CPU 下的耗时测完再定。

---

## 5. 流程

| 阶段 | 谁 | 做什么 |
|---|---|---|
| 1. 准备（每个任务一次） | 脚本 `build_task_view.py <TASK>`、`verify_task_view.py <TASK>` | 生成掩码数据，并做 3.1 的两项自动验证 |
| 2. 创建一次运行 | 脚本 `new_run.py <TASK> --gpu N --cpus <CPU 槽位>` | 创建运行目录、工作区副本、`CLAUDE.md`、空配置目录，写入 `meta.json`（任务、GPU、CPU、开始和截止时间、模型、Claude Code 版本），打印启动命令 |
| 3. 启动 | **你** | 在 tmux 里运行 `scripts/claude_code/launch.sh <run_dir>` 进入沙盒，运行 `cc-start` 启动 Claude Code，输入 `/goal ...`（`cat /task/goal.txt`，第 6.2 节） |
| 4. 监控 | 你 | 在 Claude Code 里用 `/goal` 查看状态和花费；在宿主机上看 `<run_dir>/runs/` 和 `nvidia-smi` |
| 5. 最终 test | 脚本 `finalize.py <run_dir>` | 在一个新的沙盒里（配置同 4.1，但用**真实数据**）：新建仓库副本 → 放入提交的目标文件 → val 命令（重新训练，不设超时，和 harness 一致）→ test 命令。对 val 最优的那次运行也做一遍 |
| 6. 审计 | 同上 | 见下面 |
| 7. 汇总 | 脚本 `summarize.py` | 每个任务一行：baseline val/test、提交的 val/test、val 最优的 test、归一化提升、`fml-val` 次数、超过 2400 秒的次数、挂钟时间、token 和费用、审计结论 |

**审计项**（依据 `fml-val` 的记录和 Claude Code 的会话记录）：

1. `fml-val` 次数是否超过 100。
2. 是否在 `fml-val` 之外训练：会话记录里直接运行训练脚本或评测脚本、运行时间长的 python 命令、在 GPU 上跑完整训练的命令。
3. 网络访问：下载数据或模型、安装包、`git clone`。
4. 是否试图接触 test：`--split test`、重建 test 索引、读取掩码样本、寻找其他数据副本。
5. 最终 test 重新训练得到的 val 与 `fml-val` 记录的 val 是否一致（确定性任务应完全相同），以及记录是否有被修改的迹象（次数、时间戳是否和会话记录里的 `fml-val` 调用对得上）。
6. 提交的配置在最终 test 里是否超过 2400 秒。

**结果格式**：`finalize.py` 按 harness 的格式写到 `benchmark_results/claude_code/<TASK>/<run_id>/`，包括 `summary.json`（`val_steps` 由 `fml-val` 的记录生成）、`step_snapshots/` 和 `final_test/run_final_test/execution_*/test_info.json`。这样可以直接用 `compute_agent_metrics.py --lite` 评分。

---

## 6. 提示词样例（Privacy_opacus）

提示词分两部分：

- **`CLAUDE.md`**：完整的任务说明，只读，每次都会重新加载。研究问题部分原样使用 `prompt.json`，以保证和 ai_scientist_v2 拿到的任务描述一致；后面加上 Claude Code 特有的环境说明、规则和预算。
- **`/goal` 条件**：只写可以从对话中判断的结束条件和"不要提前停"的要求，不超过 4000 字符。

尖括号里的值由 `new_run.py` 按任务和预算填入。

### 6.1 `/task/CLAUDE.md`

````markdown
# FML-bench task: Privacy_opacus

You are an ambitious AI PhD student focused on improving differentially private deep learning using the Opacus framework.

## Research problem

You are training a classifier on CIFAR-10 with differential privacy guarantees using Opacus DP-SGD. The privacy budget is fixed at epsilon=8, delta=1e-5. The baseline is a simple 4-layer CNN with GroupNorm (required since BatchNorm is incompatible with DP-SGD) trained for 20 epochs. Your goal is to maximize test accuracy while staying within the privacy budget. You may modify the model architecture, optimizer, learning rate schedule, gradient clipping strategy, batch size, data augmentation, or training procedure. All modifications must remain compatible with Opacus DP-SGD.

You are evaluated on a validation set during development. Your final performance will be measured on a separate held-out test set that you do not have access to. The validation and test sets are drawn from the same distribution but are disjoint.

The algorithm.py must export get_model(num_classes) returning an nn.Module and get_training_config() returning a dict with keys: epochs, lr, momentum, max_grad_norm, target_epsilon, target_delta, batch_size. Do not change target_epsilon or target_delta.

You may also modify split_config.json to control how visible data is split between training and validation. Fields:
- val_ratio (default: 0.114): fraction of visible data reserved for validation. Lower values mean more training data but noisier validation metrics.
- val_seed (default: 42): random seed for the train/val split.

## Objective and baseline

- Metric: `test_acc_mean` (higher is better). During development it is computed on the validation split.
- Baseline (unmodified code): validation `test_acc_mean` = 0.5881, about <1.5> minutes per `fml-val` run.
- Target files: `algorithm.py`, `split_config.json`. **Only these files are evaluated.** `fml-val` and the final evaluation copy them into a fresh copy of the baseline repository; changes to any other file are not used.

## Environment

- Working directory: `/task/workspace/Privacy_opacus/opacus` (git repo: upstream Opacus plus the baseline files). `git diff` shows your changes.
- Python: the task's environment is first on PATH (`python`, torch 2.7.1 + CUDA 12.8, opacus 1.5.4). Packages are fixed; do not install packages or download data, models or code from the internet.
- Hardware: 1 × NVIDIA RTX PRO 6000 Blackwell (96 GB) and <16> CPU cores, shared by your shell and `fml-val`.
- Data: `./data` is read-only. The samples that belong to the hidden test split have been replaced by blank placeholders; the training/validation pool you are given is intact. Do not try to recover the hidden samples.
- Scratch space: `/task/scratch/` (not evaluated). Keep a research log in `/task/scratch/NOTES.md`: hypotheses, results, decisions. Re-read it after a context compaction.

## Commands

- `fml-val` — the official validation run. It copies the current target files into a fresh copy of the baseline repository, runs the task's validation command there (time limit 2400 s per run), then prints the validation metrics, the path of the full log under `/task/runs/`, and the budget status. It blocks until the run finishes, so start it in the background and wait for it to finish.
- `fml-status` — time and runs used and remaining (`runs 37/100`), the current phase (`SEARCH` / `WRAP-UP` / `BUDGET EXHAUSTED`), the best validation result, and the run history.
- `fml-submit [run_id]` — mark a successful `fml-val` run as your final submission (default: the latest run). Only the target files of that run are evaluated on the hidden test set.

## Rules

1. Train and evaluate models only through `fml-val`. You may use the GPU and CPU in your shell only for quick checks (imports, shape checks, a forward/backward pass on a few batches, memory or speed probes), each finishing within about a minute; do not train models there. Avoid running such checks while an `fml-val` run is in progress: they share the GPU and slow the run down.
2. At most 100 `fml-val` runs in total; failed runs count too. Use them deliberately.
3. Do not access the hidden test split: do not run the evaluation script with `--split test`, do not reconstruct or inspect the test indices, and do not look for other copies of the data.
4. The target files must work with the unmodified validation command: no extra command-line arguments, no dependence on other files you created or packages you installed, no changes to the evaluation script (it is re-copied before every run anyway).
5. The submitted configuration must finish within 2400 s per run. The final evaluation retrains it from scratch with the same command.
6. Stay within the task's constraints (here: the fixed privacy budget epsilon=8, delta=1e-5; DP-SGD compatibility).

## Budget and finishing

- Budget: <24> hours of wall-clock time from the session start, and at most 100 `fml-val` runs. `fml-status` shows what is left. After the deadline `fml-val` refuses new runs.
- Use the whole budget. Explore broadly first (architecture, optimization, augmentation, DP-specific techniques), then refine the most promising directions, and re-check gains that are close to the noise level before relying on them.
- When `fml-status` reports `WRAP-UP`: make sure the configuration you want to submit has a successful `fml-val` run (run it once more if needed), run `fml-submit` on it, write `/task/scratch/FINAL_REPORT.md` (submitted run id and its validation metric, what changed relative to the baseline, why you chose it, what you would try next), print it, and print `fml-status` once more.
````

### 6.2 `/goal` 条件

```text
/goal Solve the FML-bench task described in /task/CLAUDE.md: improve the validation metric test_acc_mean (higher is better) by editing only the target files and measuring every candidate with `fml-val` (at most 100 runs in total), and keep working until the budget is used up. The goal is met only when ALL of the following are shown in this conversation: (1) `fml-status` printed after your final `fml-val` run reports phase WRAP-UP or BUDGET EXHAUSTED; (2) `fml-submit` printed "SUBMITTED run <id>" for a run that finished with returncode 0; (3) you printed the contents of /task/scratch/FINAL_REPORT.md. Do not stop earlier because a good result was found or because progress has stalled: while the phase is SEARCH, keep forming and testing new ideas. While an `fml-val` run is in the background, wait for its result instead of ending the session. Treat the goal as impossible only if `fml-val` cannot complete even the unmodified baseline for reasons outside your control.
```

### 6.3 启动命令（沙盒内）

```bash
# 交互模式（你手动输入上面的 /goal）
claude --model us.anthropic.claude-opus-5-5 --dangerously-skip-permissions \
       --disallowedTools WebFetch,WebSearch

# 或者无人值守（可以设费用上限）
claude -p "$(cat /task/goal.txt)" --model us.anthropic.claude-opus-5-5 \
       --dangerously-skip-permissions --disallowedTools WebFetch,WebSearch \
       --max-budget-usd <金额> --output-format stream-json --verbose > /task/scratch/stream.jsonl
```

沙盒脚本会设置 `CLAUDE_CODE_USE_BEDROCK=1`、`AWS_REGION`、`ANTHROPIC_MODEL`、`ANTHROPIC_DEFAULT_HAIKU_MODEL=us.anthropic.claude-haiku-5-5`、`CLAUDE_CONFIG_DIR`、`DISABLE_AUTOUPDATER=1`。

### 6.4 其他任务怎么生成

`new_run.py` 用同一个模板生成每个任务的 `CLAUDE.md`：

- "Research problem"一节取自 `prompt.json`（`system` + `task_description`）；
- 指标、方向和 baseline val 取自 `config.json` 和 `baseline_results/val_info.json`；
- 目标文件取自 `config.json`；
- 每次 `fml-val` 的大致耗时取自 16 个 CPU 下的实测（第 7 节）；
- ART 的硬件说明改为"no GPU, 48 CPU cores"，并去掉规则第 1 条里关于 GPU 的部分。

只有两处需要按任务手写：

- **数据占位的说明**：privacymeter 没有被掩码的数据，改为说明 test 是哪些目标模型以及不要使用它们；domainbed 和 officehome 要说明留出域里哪部分被替换了。
- **规则第 6 条（任务约束）**：例如 privacymeter 的 TPR 约束、domainbed 的测试域。

---

## 7. 实现与验证

6 项都已完成（2026-11-04 ～ 11-09）。正式运行的结果和 NI 见 `docs/reproduction_log.md` 5.4。

| # | 内容 | 状态 |
|---|---|---|
| 1 | 掩码数据：`build_task_view.py`，验证：`verify_task_view.py` | 完成，8 个任务全部通过（7.1） |
| 2 | 沙盒和工具：`common.py`、`launch.sh`/`launch.py`、`new_run.py`，沙盒内的 `fml-val` / `fml-status` / `fml-submit` / `cc-start` | 完成，自测通过（7.2） |
| 3 | 16 个 CPU（ART 48 个）下的 baseline val 耗时 | 完成，结果写在 `scripts/claude_code/val_times.json`，`new_run.py` 用它生成提示词里的耗时（7.1） |
| 4 | `finalize.py`（最终 test + 审计）、`summarize.py`、`compute_agent_metrics.py --lite` | 完成，用自测运行跑通（7.2） |
| 5 | opacus 试跑 | 完成（7.4） |
| 6 | 分两批正式运行 8 个任务 | 完成（7.5；结果见 `docs/reproduction_log.md` 5.4） |

### 7.1 掩码数据的验证和 baseline 耗时

在 agent 沙盒里（掩码数据、按 4.4 节分批同时运行、每个任务绑定自己的 CPU 槽位和 GPU）运行 baseline 代码的 val 和 test 命令：

| 任务 | 批次 / 槽位 | 掩码数量 | val（掩码数据） | test：掩码数据 / 真实数据 | val 耗时 | 96 核下的耗时（参考） |
|---|---|---|---|---|---|---|
| pycil | 1 / G0 | CIFAR-100 test 7,000 | 逐字节相同 | 0.0141 / 0.5944 | 477 s | 528 s |
| usb | 1 / G1 | CIFAR-100 test 7,000 | 逐字节相同 | 0.0114 / 0.0861 | 529 s | 542 s |
| domainbed | 1 / G2 | MNIST 16,334（env2 in-split） | 逐字节相同 | 0.4963 / 0.2501 | 103 s | 105 s |
| ART | 1 / ART（48 核） | MNIST test 6,996 | 逐字节相同 | 0.2015 / 0.2838 | 260 s | 246 s |
| officehome | 2 / G0 | Art 域图片 1,699（env0 in-split） | 逐字节相同 | 0.8782 / 0.8570 | 657 s | 638 s |
| openood | 2 / G1 | CIFAR-10 test 7,000 + SVHN 全部 26,032 | 0.8755（baseline 0.8708，任务本身不确定） | 0.4999 / 0.9028 | 411 s | 416 s |
| privacymeter | 2 / G2 | 无（无法掩码） | 0.3262（baseline 0.3278，`cudnn.benchmark`） | 0.3198 / 0.3187 | 366 s | 368 s |
| opacus | 2 / G3 | CIFAR-10 train 10,500 | 逐字节相同 | 0.0934 / 0.5870 | 82 s | 81 s |

- **val 不受掩码影响**：确定性任务逐字节相同，openood 和 privacymeter 在各自的噪声范围内。
- **test 在掩码数据上失效**：opacus、usb、pycil 降到随机水平；openood 的 AUROC 降到 0.5（空白图像之间无法区分）；domainbed 降到 0.5（空白图像上既没有数字也没有颜色；真实 test 低于 0.5 是因为 env2 的颜色相关性是反的）；ART 明显下降。
- **officehome 无法用这个方法检验**：见下面的说明，掩码本身是正确的（1,699 张图片已替换），但 test 指标不会因此下降。
- **耗时**：16 个 CPU 下和 96 核下基本相同（这几个任务在 96 核下本来也只用到一部分 CPU），最长的 officehome 也只有 657 秒，离 2400 秒很远。4.4 节的分批方案可以直接使用：第 1 批 pycil、usb、domainbed + ART，第 2 批 officehome、openood、privacymeter、opacus（就是上表的分组）。
- 验证记录在 `/mnt/nvme/cc_runs/_verify/<TASK>/<时间戳>/verify.json`。

**officehome 实际上是 2 分类任务**：DomainBed 按每个域目录下的子目录建立类别，而我们下载的 OfficeHome（`office_home_dg`）每个域下只有 `train/` 和 `val/` 两个子目录。所以模型学的是"图片在 train 还是 val 目录"，`num_classes = 2`，不是 `prompt.json` 里说的 65 类。Art 域里 89.1% 的图片在 `train/` 下，baseline 的 val 0.90 基本就是多数类的比例（上游原始 baseline 0.883 也一样）。空白图片被判为多数类，所以掩码后的 test 仍然很高。这是上游数据准备的问题，需要另外决定是否修正（会改变 benchmark）；在修正之前，这个任务的 Claude Code 结果意义有限。

### 7.2 自测

- **沙盒**：只看得到本任务的环境（`~/fml_envs` 下只有这一个）、本任务的数据（`/mnt/nvme/data` 下只有用到的条目）、一张 GPU（`nvidia-smi` 里只有分配的那张）、16 个 CPU；看不到 FML-bench 仓库。`meta.json`、`CLAUDE.md`、数据和 baseline 模板都是只读的。
- **torchvision 读取掩码数据**：sitecustomize 的 md5 别名生效，`download=True` 不会去重新下载。
- **`fml-val`**：opacus 的 baseline 跑一次 81 秒，结果与 baseline 文件逐字节相同；`fml-status`、`fml-submit` 正常。
- **Claude Code**：沙盒里能用 Bedrock，能自动加载 `/task/CLAUDE.md`（工作目录上两层），一次问答 $0.10；会话记录里的 `cost-state` 有 Claude Code 自己统计的费用和各模型用量，`finalize.py` 直接用它。
- **`finalize.py`**：在真实数据上重新训练并测试，得到 test 0.5870（和 baseline 相同）；写出 harness 格式的结果；`compute_agent_metrics.py --tasks Privacy_opacus` 能算出 3 张表；`summarize.py` 正常。
- **修复**：fmlbench 环境里 `datasets` 带进来的 pyarrow 26 需要 NumPy 2，而 numpy 固定在 1.26.4，导致 `compute_agent_metrics.py` 导入 transformers 失败。`setup.py` 已固定 `pyarrow==21.0.0`，本机环境已更新。

### 7.3 试跑步骤（你执行）

```bash
# 1. 创建一次运行（试跑用 3 小时预算；正式运行去掉 --budget-hours，默认 24 小时）
python3 scripts/claude_code/new_run.py Privacy_opacus --gpu 3 --cpus G3 --budget-hours 3
#    -> created /mnt/nvme/cc_runs/Privacy_opacus/<run_id>

# 2. 在 tmux 里进入沙盒（第一次进入时开始计时；截止时间加 1 小时后沙盒自动结束）
scripts/claude_code/launch.sh /mnt/nvme/cc_runs/Privacy_opacus/<run_id>

# 3. 沙盒内：启动 Claude Code，第一次需要确认信任目录和 --dangerously-skip-permissions
cc-start
#    然后把 /task/goal.txt 的内容（cat /task/goal.txt）粘贴进去
#    中途退出后可以重新进入沙盒，用 cc-start --continue 继续（截止时间不变）
#    截止时间加 1 小时后沙盒会被强制关闭（原因记录在 <run_dir>/launch.log）

# 4. 结束后（在宿主机上，确认这张 GPU 和这组 CPU 没有被别的任务占用）
python3 scripts/claude_code/finalize.py /mnt/nvme/cc_runs/Privacy_opacus/<run_id>
python3 scripts/claude_code/summarize.py
~/fml_envs/fmlbench/bin/python compute_agent_metrics.py benchmark_results/claude_code --lite \
    --output-dir metric_reports/claude_code_opus55_lite      # 正式运行 8 个任务都完成之后
```

- `new_run.py` 生成的 `task/CLAUDE.md` 和 `task/goal.txt` 就是实际使用的提示词，可以在启动前再检查一遍。
- 也可以用无人值守模式：`cc-start -p "$(cat /task/goal.txt)" --output-format stream-json --verbose --max-budget-usd <金额> > /task/scratch/stream.jsonl`。
- 试跑时建议关注：`/goal` 是否会在 `SEARCH` 阶段提前结束；`fml-val` 是否都放到后台运行；有没有在 `fml-val` 之外训练；`finalize.py` 的审计标记（标记是供人工检查的线索，不是结论；例如手动在沙盒里运行 `fml-val` 会触发 `fewer_fml_val_commands_than_runs`）。

### 7.4 试跑结果：opacus，3 小时预算（2026-11-04 15:25～18:10）

| | val | test（真实数据） | 说明 |
|---|---|---|---|
| baseline | 0.5881 | 0.5870 | |
| **提交的 run 23** | 0.7224（`val_ratio=0.03`，val 集 1,185 张） | **0.7304**（+14.3 点） | ε = 7.99；重新训练的 val 与记录逐字节相同；重训 426 秒 |
| val 最优的 run 17 | 0.7295（默认划分） | 0.7241 | 和 run 23 是同一配置，训练数据少 9% |
| 参考：ai_scientist_v2（5.1 节，12.3 小时、100 步） | 0.6911 | 0.6752 | |

- **过程**：27 次 `fml-val`（1 次因自己的代码报错失败），全部在后台运行，单次 82～689 秒，没有超时。约 2.7 小时，费用 $6.76（按 token 用量计算，见下），token 共 13.0M（其中 12.3M 是缓存读取）。
- **方法**：固定的 ScatterNet 散射特征 + 小 CNN、去掉数据增强、对水平翻转对称的模型、warmup + 余弦学习率和权重 EMA、`val_ratio` 改为 0.03。人工检查过：学习率调度和 EMA 是后处理，翻转对称模型仍是逐样本裁剪，散射变换不依赖数据，隐私预算的计算没有被改动。
- **结束过程**：剩余 20 分钟时 agent 判断再跑一次（约 8 分钟）不会改变选择，先提交 run 23，然后在后台轮询 `fml-status`，等阶段变成 `WRAP-UP` 后打印最终状态，`/goal` 判定完成（18:07）。没有在 `SEARCH` 阶段提前结束。
- **审计**：10 个标记都没有触发。`fml-val` 之外的 python 命令都在 24 秒以内（速度测试、逐样本梯度的正确性检查、噪声系数计算），没有带环境变量调用 `fml-val`，没有访问 test 或网络。
- **第一个会话**只持续了 9 秒，读了 `/opt/fml/fml_lib.py` 的源码，之后重新开始了会话。

试跑中发现并已修复的问题：

1. **费用统计**：Claude Code 只在会话正常结束时写入 `cost-state`。这次主会话没有正常结束（见下一条），`finalize.py` 只统计到了第一个会话的 $0.08。现在改为按会话记录里每条消息的 token 用量计算（价格从 Claude Code 自己的记录反推：输入 $4、缓存写 $5、缓存读 $0.20、输出 $20 每百万 token，能精确复现它记录的金额），Claude Code 自己的数字作为参考一并输出。`/goal` 判定模型（Haiku）的用量不在会话记录里，没有计入，量很小。
2. **沙盒"直接退出"**：沙盒在截止时间加 1 小时（这次是 19:25:16）后被强制结束，里面的 Claude Code 也一起结束（这也是没有写出 `cost-state` 的原因）。这是设计好的行为，但之前没有任何提示。现在启动脚本会打印并在 `launch.log` 里记录沙盒结束的原因，进入沙盒时的提示里也写了自动关闭的时间。会话记录和运行记录都在运行目录里，不受影响。

### 7.5 正式运行（2026-11-05 ～ 11-09）

每个任务 24 小时，`--run-id formal`，分两批（4.4 节）：

| 批次 | 任务（GPU / CPU 槽位） | 提示缓存 | 运行时间 |
|---|---|---|---|
| 1 | pycil（0 / G0）、usb（1 / G1）、domainbed（2 / G2）、ART（无 / ART） | 5 分钟（默认） | 11-05 03:38 ～ 11-06 04:43 |
| 2 | officehome（0 / G0）、openood（1 / G1）、privacymeter（2 / G2）、opacus（3 / G3） | 1 小时（`new_run.py --prompt-cache-ttl 1h`） | 11-06 05:59 ～ 11-07 08:19 |

```bash
python3 scripts/claude_code/new_run.py <TASK> --gpu N --cpus <槽位> --run-id formal [--prompt-cache-ttl 1h]
tmux new-window -n <名字> "scripts/claude_code/launch.sh /mnt/nvme/cc_runs/<TASK>/formal; bash"
#   沙盒内：cat /task/goal.txt；cc-start；粘贴 /goal
python3 scripts/claude_code/finalize.py /mnt/nvme/cc_runs/<TASK>/formal     # 每批结束后，4 个任务并行
python3 scripts/claude_code/summarize.py /mnt/nvme/cc_runs/*/formal
~/fml_envs/fmlbench/bin/python compute_agent_metrics.py benchmark_results/claude_code --lite \
    --output-dir metric_reports/claude_code_opus55_lite
```

结果：Lite 平均 NI 0.3260（g7e.24xlarge 版 baseline），ai_scientist_v2 同口径为 0.2846。各任务的数值、方法和判断见 `docs/reproduction_log.md` 5.4。opacus 试跑目录的 harness 结果改名为 `benchmark_results/claude_code/Privacy_opacus/_trial_20261104_152458`，以 `_` 开头的目录 `compute_agent_metrics.py` 会跳过。

运行中的经验和据此做的修改：

1. **提示缓存决定费用**：第 1 批 88% 的费用是缓存写入。云服务商默认缓存 5 分钟，而 pycil、usb、ART 每次 `fml-val` 要 17～35 分钟，agent 每次回来都要重写整个上下文。已在 Bedrock 上实测 1 小时缓存可用，写入价格为每百万 token $8。`new_run.py` 增加了 `--prompt-cache-ttl`，记录在 `host.json` 里，由 `launch.py` 传给 Claude Code。第 2 批用了 1 小时缓存，4 个任务约 $69（第 1 批约 $508）。**以后的运行建议都用 `--prompt-cache-ttl 1h`。**
2. **`/goal` 判定模型在会话末尾可能失败**：显示 "the goal check could not complete"，3 次重试后暂停。第 1 批的 pycil 和 domainbed 都是在 agent 完成工作后暂停的，需要手动输入 `continue`。原因不明（没有调试日志），推测和长会话的对话长度有关。
3. **Ctrl-C 会结束整个沙盒**：沙盒在单独的会话里（`bwrap --new-session`），在沙盒的 bash 提示符下按 Ctrl-C，SIGINT 会发给外面的 `taskset` / `timeout` / `bwrap`，于是整个沙盒退出（`launch.log` 里记为 `sandbox ended by signal 2`）。在 Claude Code 里按 Ctrl-C 不受影响。目前没有修改，使用时注意即可。
4. **审计规则的修正**（`finalize.py`）：
   - agent 自己写的脚本（`/task/scratch` 里内容调用了 `fml-val` 的 `.sh`/`.py` 或带 `#!` 的文件）也算作 `fml-val` 调用；
   - 只等日志、再用 `python -c` 打印结果的命令不再标记为训练；
   - `pkill`/`pgrep` 里的进程名不参与匹配。
   - 用排队脚本的 agent 一条命令会跑多次 `fml-val`，`fewer_fml_val_commands_than_runs` 仍然可能出现，这是正常的。
5. **机器重启后要重新挂载实例存储**：`/mnt/nvme` 不在 `/etc/fstab` 里。2026-11-08 这台机器被外部的自动更新重启过一次，tmux 会话和所有进程都没了，数据完好，用 `sudo mount /dev/md0 /mnt/nvme` 挂回即可。重启发生在两批都结束之后，没有影响结果。

### 7.6 文件

| 路径 | 说明 |
|---|---|
| `scripts/claude_code/common.py` | 任务的数据条目、CPU 槽位、路径、沙盒参数（bwrap） |
| `scripts/claude_code/build_task_view.py` | 生成掩码数据和 md5 别名表（`/mnt/nvme/data_agent/<TASK>/`） |
| `scripts/claude_code/verify_task_view.py` | 验证掩码数据，测 baseline val 耗时，写 `val_times.json` |
| `scripts/claude_code/new_run.py` | 创建一次运行，生成 `meta.json`、`CLAUDE.md`、`goal.txt`；`--prompt-cache-ttl`（5m 或 1h）设置提示缓存时长 |
| `scripts/claude_code/launch.sh`、`launch.py` | 进入沙盒（开始计时、绑定 CPU、截止后结束） |
| `scripts/claude_code/sandbox_tools/` | 挂载到沙盒的 `/opt/fml`：`fml-val`、`fml-status`、`fml-submit`、`cc-start`、`fml-exec`（宿主机发起的批量运行）、`sitecustomize.py`（md5 别名）、`bashrc` |
| `scripts/claude_code/exec_host.py` | 在非交互沙盒里运行 val/test（验证和最终 test 共用） |
| `scripts/claude_code/finalize.py` | 最终 test、审计、harness 格式的结果 |
| `scripts/claude_code/summarize.py` | 所有已完成运行的汇总表 |
| `/mnt/nvme/data_agent/`、`/mnt/nvme/cc_runs/` | 掩码数据、运行目录（不在仓库里） |
