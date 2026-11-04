# FML-bench 复现实验记录

本文档记录在本机上复现 FML-bench 的过程：环境适配、baseline 重测、harness 改造（Bedrock / uv / 沙盒），以及 agent 测试结果。后续每次复现实验都追加到「实验记录」一节。

- 仓库基线提交：`d336651`（以下改动均未提交，基于该提交的工作区）
- 记录时间：2026-11-02 ～ 2026-11-03（第 8 节为第二台机器 4 × L4 上的复现）

---

## 1. 机器环境

| 项目 | 值 |
|---|---|
| 实例 | AWS EC2 `g7e.12xlarge`（us-west-2） |
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB，sm_120） |
| 驱动 / CUDA | 580.126.16 / CUDA 13.0 |
| CPU / 内存 | Intel Xeon Platinum 8559C，48 核 / 499 GB |
| 系统 | Amazon Linux 2023（kernel 6.1） |
| 环境管理 | uv 0.11.14（不使用 conda） |
| 沙盒 | bubblewrap 0.10.0 |
| LLM | Amazon Bedrock，`us.anthropic.claude-opus-5-5`，凭证来自实例角色 |

**硬件带来的关键约束**：Blackwell（sm_120）需要 torch ≥ 2.7 + CUDA 12.8。仓库原配置的 torch 2.5.1+cu121、1.12.1+cu113 等版本在这块卡上没有可用的 kernel，统一换成 `torch==2.7.1+cu128`（torchvision 0.22.1、torchaudio 2.7.1）。TF 2.10 无法使用这块 GPU，ART 任务只能在 CPU 上跑。

---

## 2. 环境构建（uv）

- 所有环境建在 `~/fml_envs/<env>`（本地 NVMe），可用 `FMLBENCH_UV_ENVS` 改位置；解释器由 uv 管理，位于 `~/.local/share/uv/python`。
- `setup.py` 新增 `uv_create` / `uv_run`，以下环境已从 conda 迁移到 uv：

| 环境 | Python | 相对原配置的改动 |
|---|---|---|
| fmlbench（harness） | 3.11 | 去掉 conda 的 nodejs（harness 用不到）；`anthropic` 改为 `anthropic[bedrock]==0.84.0`（增加 boto3） |
| opacus / pycil / usb / openood | 3.10 / 3.10 / 3.10 / 3.9 | torch 2.5.1+cu121 → 2.7.1+cu128 |
| domainbed | 3.10 | conda pytorch 1.12.1 → pip torch 2.7.1+cu128；去掉 mkl 系列包；`backpack-for-pytorch` 1.3.0 → 1.7.1（1.3.0 要求 torch<2.0，会把 torch 降级）；新增 `huggingface_hub<1.0`（未锁版本时会装到 2.x，timm 0.9.16 用到的接口在 1.x 已移除） |
| privacy_meter | 3.12 | 从 requirements.txt 中滤掉 torch/nvidia 相关行，单独安装 torch 2.7.1+cu128 |
| art | 3.10 | 版本不变；TF 2.10.1 在本机只能用 CPU |

- 其余环境（easyfsl、lightly、continual_learning、causalml、aif360、fairlearn、gcastle、sololearn、pfllib、open_unlearning）仍是 conda 配置，**未迁移、未测试**。
- 所有已迁移环境都通过了 GPU 计算测试（art 除外，按预期走 CPU）。

### 数据

部分官方下载源太慢（多伦多 CIFAR 服务器约 35 KB/s），改用镜像手动下载，并用 torchvision 内置的 md5 校验。下载逻辑**没有**写进 `setup.py`，在新机器上需要手动重复这一步。

| 数据 | 来源 | 校验 |
|---|---|---|
| CIFAR-10 `cifar-10-python.tar.gz` | Kaggle `pankrzysiu/cifar10-python` | md5 `c58f30108f718f92721af3b95e74349a` |
| CIFAR-100 `cifar-100-python/` | Kaggle `pypiahmad/cifar-100` | 与 torchvision 逐文件 md5 一致 |
| SVHN `test_32x32.mat` | Kaggle `khanhvudo72/svhncropped` | md5 `eb5a983be6a315427106f1b164d9cef3` |
| OfficeHome | `uvx gdown 1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa`（135 MB，含 `office_home_dg/{art,clipart,product,real_world}`） | — |
| MNIST | torchvision 默认的 S3 镜像 | — |

缓存在 `~/fml_data_cache/`，放置位置：

- pycil / usb / openood：`<repo>/data/cifar-100-python`
- openood：另有 `data/cifar-10-python.tar.gz`、`data/svhn/test_32x32.mat`
- privacymeter：`data/cifar10/cifar-10-python.tar.gz`（由 setup 生成 pkl）
- officehome：`workspace/Generalization_domainbed_officehome/data/office_home`

注意：预先放好数据会跳过 setup 里的 git 提交，pycil 和 usb 已手动补上同样的提交（`add cifar-100 dataset for agent benchmark`）。executor 每步都会 `git reset --hard` 到 HEAD，所以数据必须在 HEAD 里。

---

## 3. Baseline 重测（FML-bench-Lite，8 个任务）

### 方法

按各任务 `config.json` 的 `val_command` → `test_command` 手动执行（同一份工作区副本，test 会复用 val 保存的模型），环境为对应的 uv 环境：

```bash
scripts/reproduction/rebaseline_task.sh <TASK> <GPU编号，CPU 填 ""> [TAG]
python3 scripts/reproduction/compare_baseline.py <TASK>
```

结果写入 `rebaseline_results/<TASK>/`（val/test 结果、日志、`run_info.txt`、`pip_freeze.txt`），工作区副本保留为 `workspace/<TASK>__rebaseline_<时间戳>`。

### 结果（原始 → 本机）

| 任务 | 指标（方向） | val | test | 耗时（val/test） | 结论 |
|---|---|---|---|---|---|
| Privacy_opacus | test_acc_mean（↑） | 0.5894 → 0.5881 | 0.5877 → 0.5870 | 1m10s / 5s | 一致 |
| Continual_Learning_pycil | avg_incremental_acc_mean（↑） | 0.5911 → 0.5925 | 0.5952 → 0.5944 | 8m54s / 8m09s | 一致 |
| Privacy_privacymeter | AUC_gap_mean（↓） | 0.3339 → 0.3242 | 0.3214 → 0.3182 | 6m15s / 1s | 一致 |
| Robustness_openood | auroc_mean（↑） | 0.8758 → 0.8804 | 0.9302 → 0.9212 | 10m25s / 8s | 接近 |
| Data_Efficiency_usb | test_acc_mean（↑） | 0.0757 → 0.0827 | 0.0790 → 0.0861 | 11m11s / 5s | 接近（原值本身约 8%） |
| Generalization_domainbed_officehome | avg_acc_mean（↑） | 0.8832 → 0.9011 | 0.8629 → 0.8570 | 10m55s / <1s | 接近（±2 点内） |
| Generalization_domainbed（ColoredMNIST） | in_acc_mean（↑） | 0.2792 → 0.2393 | 0.2868 → 0.2501 | 3m32s / <1s | **偏低约 0.04** |
| Robustness_and_Reliability_art（CPU） | defense_score_mean（↑） | 0.4720 → 0.3005 | 0.4774 → 0.2921 | 4m51s / 7s | **明显偏低** |

pycil 的 test 阶段会重新训练；其他任务的 test 加载 val 保存的模型。

### 偏差分析

- **domainbed** 和 **ART** 都在同一环境下重跑了一次（`rebaseline_results/<TASK>__rerun2/`），结果与第一次完全一致，说明差距不是随机波动，而是环境差异造成的。
- **domainbed**：最可能的原因是 torch 版本变化（2.7.1+cu128 对比原来的 1.12.1）。ColoredMNIST 留出的 env2 是对抗性的域，训练过程中精度在 0.18～0.30 之间跳动，对数值差异非常敏感。
- **ART**：DP-InstaHide 训练 5 个 epoch，种子全部固定。差距集中在部分后门上（attack success：random_corner 0.84 对比原来的 0.30，edge 0.90 对比 0.75），clean_acc 反而更高（0.956 对比 0.944）。最可能的原因是原 baseline 在 GPU 上训练，而本机只能用 CPU，但原始文件里没有硬件信息，无法确认。原始文件之间本身也不一致（`final_info.json` 的 num_samples 是 7500，`val_info.json` 是 8404，另有一个旧版 `final_info.old.json` 的 defense 是 0.6966），可能出自不同版本的代码。
- **officehome** 第一次运行在 10 秒内失败（timm 依赖的 `huggingface_hub` 被装成了 2.x），锁定 `<1.0` 后重跑成功。

### 采用

按决定，**以本机重测结果作为新的 baseline**：

- `ml_tasks/<TASK>/baseline_results/{val,test}_info.json` 替换为重测结果；
- 仓库原有的文件（包括 `final_info.json`）移到 `ml_tasks/<TASK>/baseline_results/original/`。

runner 读 `val_info.json` 作为 agent 提示词里的 baseline，`compute_agent_metrics.py` 读 `val_info.json` / `test_info.json`，两者现在都用新值。

---

## 4. Harness 改造

### 4.1 Bedrock 接入 Opus 5.5

- 用法：`--provider AnthropicBedrock --model us.anthropic.claude-opus-5-5`；区域取 `AWS_REGION`（默认 us-west-2），凭证走 AWS 默认链（本机为实例角色）。
- 调用 LLM 的只有两处，都已加上 Claude 原生分支：
  - `agents/llm.py`：新增 `AnthropicBedrock` provider，以及 `claude_messages_create` / `claude_response_text` / `is_anthropic_client`。`get_response_from_llm` 的 Claude 分支改用这些函数；多轮历史保留完整的 assistant 内容（含 thinking 块），原样回传。
  - `agents/code_editor.py`：遇到 Anthropic 客户端时走 Messages API，否则维持原来的 `chat.completions`。
- 请求参数：
  - 流式输出，`max_tokens=64000`。Opus 5.5 的思考无法关闭且计入 max_tokens，原来的 8192 容易截断；上限调大之后 SDK 要求用流式，否则可能超过 HTTP 超时。
  - 不传 temperature（Opus 5.5 不接受）。
  - `output_config.effort` 默认 `high`，可用 `FMLBENCH_CLAUDE_EFFORT` 调整。
  - 客户端 `max_retries=8`。
- 拒答（`stop_reason == "refusal"`）直接抛错，**不**自动换到其他模型，以免评测结果里混入别的模型。
- 修复：原来 Claude 响应的 token 用量统计为 0，现在 `extract_usage_info` 能正确读取 `input_tokens` / `output_tokens`（含缓存部分）。
- SDK 仍为 0.84.0。Bedrock Mantle 端点在本账号下对该模型返回 404，因此走的是 bedrock-runtime（`anthropic.AnthropicBedrock`）。

### 4.2 运行入口适配 uv 环境

- `benchmark/utils.py` 新增 `env_command(env_name, args)`：若 `~/fml_envs/<conda_env>/bin/python` 存在，就设置 `VIRTUAL_ENV`、把该环境放到 PATH 最前面后直接执行；否则退回 `conda run`。
- 调用方：`benchmark/executor.py::_run_command`（val/test 命令）和 `agents/theaiscientist/theaiscientist.py`（setup 命令）。config.json 的字段名 `conda_env` 保持不变。
- `_run_command` 在每条命令结束时输出 `Command finished in X s (returncode N)`，用于事后统计每步的实际耗时。超时时仍输出原有的 `Command timed out after N seconds`。

### 4.3 沙盒：`sandbox/run_in_bwrap.sh`

```bash
CUDA_VISIBLE_DEVICES=0 sandbox/run_in_bwrap.sh <run_agent_benchmark.py 的参数...>
sandbox/run_in_bwrap.sh --shell [-c CMD]     # 在同一沙盒里开 shell 或执行单条命令
```

| 可见性 | 内容 |
|---|---|
| 只读 | 仓库（agents/、benchmark/、ml_tasks/、configs/…）、`~/fml_envs`、uv 的 Python、`/usr`、`/etc`、`/sys` |
| 可写 | `workspace/`、输出目录（默认 `benchmark_results/`，`--output-dir` 会被识别）、`~/.cache` |
| 不可见 | 宿主 home 的其余部分（包括 `~/.ssh`、`~/.aws`，除非设置了 `AWS_PROFILE`）、EFS 上的其他数据、conda |
| 设备 | NVIDIA 设备节点全部透传。`CUDA_VISIBLE_DEVICES` 原样传入：不设置表示所有 GPU，设为空字符串表示不给 GPU（ART 就这样运行） |
| 解释器 | 如果某个环境的基础解释器既不在 `/usr` 也不在 uv 的 Python 目录里（本机的 privacy_meter 用的是 `~/miniconda3` 的 Python 3.12.9），就把它所在的前缀目录只读挂载进来。conda 不在 PATH 上 |
| 隔离 | user/pid/ipc/uts/cgroup 命名空间独立，`--clearenv`，`--new-session`，`--die-with-parent` |

- **网络与宿主共享**。隔离网络会断掉 IMDS，Bedrock 就拿不到实例角色的凭证；代价是 agent 生成的代码也能访问外网。
- `/etc/resolv.conf` 指向 `/run/systemd/resolve/`，因此只读挂载了这一个目录。
- 已验证：ml_tasks 和 agents 不可写、workspace 可写、`~/.ssh` 不可见；在 opacus 环境里能做 GPU 计算；Bedrock 调用、DNS、HTTPS 都正常。

---

## 5. 实验记录

### 5.1 ai_scientist_v2 × Privacy_opacus × Opus 5.5（2026-11-02 16:34 ～ 2026-11-03 04:55）

**命令**（按 README 的标准协议，max_steps=100）：

```bash
CUDA_VISIBLE_DEVICES=0 sandbox/run_in_bwrap.sh \
  --agent-config configs/agents/ai_scientist_v2.yaml \
  --task-config  configs/tasks/privacy_opacus.yaml \
  --model us.anthropic.claude-opus-5-5 --provider AnthropicBedrock \
  --workspace-label opus55 agent.ai_scientist_v2.max_steps=100
```

此前先做了一次 3 步的冒烟测试（`--output-dir benchmark_results_smoke`，`max_steps=3`），test 准确率 0.6313，全流程跑通。

**结果**

| | val 准确率 | test 准确率 | ε |
|---|---|---|---|
| 新 baseline | 0.5881 | 0.5870 | 8.00 |
| ai_scientist_v2 最优（第 88 步） | 0.6911 | **0.6752** | 7.9997 |
| 提升 | +10.3 点 | **+8.8 点** | 预算内 |

- 4 个阶段的最优 val：基础实现 0.6365（10 步）→ 调参 0.6380（20 步）→ 创新探索 0.6876（50 步）→ 消融精炼 0.6911（20 步）。
- 100 步全部成功运行，没有 LLM 调用失败、拒答或训练超时。
- 最终 test 前会用最优代码重新训练一遍，得到的 val 是 0.6911，与搜索中完全一致（训练是确定性的）。
- 耗时 12.3 小时（平均约 7 分钟/步）。token 共 2,164,265（输入 1,201,852，输出含思考 962,413）。按官方价格 $4/$20 每百万 token 估算约 $24，实际以 Bedrock 账单为准。
- 结果目录：
  - 搜索过程（100 步的代码快照 `step_snapshots/`、每步 val、code_editor 日志）和 `summary.json`：`benchmark_results/ai_scientist_v2/Privacy_opacus/20261102_163447_2a504f3e/`
  - 最终 test：`benchmark_results/ai_scientist_v2/Privacy_opacus/20261103_044518/`
  - 运行日志：`docs/reproduction_logs/aisv2_opacus_full.log`（冒烟测试：`docs/reproduction_logs/aisv2_smoke.log`）

**agent 做了哪些改动**（每次刷新最好成绩时的关键改动；运行编号从 0 开始，对应快照 `step_<编号+1>`）

| 运行 | val | 改动 |
|---|---|---|
| 0 | 0.6046 | 30 epoch；`split_config.json` 的 `val_ratio` 0.114 → 0.05 |
| 3 | 0.6238 | batch 1024，lr 0.15 |
| 5 | 0.6365 | Tramer & Boneh 风格的 6 层卷积网络，tanh 激活，去掉大的全连接层，50 epoch，lr 0.2 |
| 34/35 | 0.6390 | 换路线：固定的 ScatterNet 散射特征（自己实现，J=2、L=8、二阶，243 通道）+ 小型 tanh CNN；60 epoch、lr 4.0、梯度裁剪 0.1、batch 4096 |
| 38 | 0.6542 | 权重 EMA（后处理，不额外消耗隐私预算） |
| 42 | 0.6754 | 水平翻转对称化（原图和镜像取 logits 平均） |
| 46 | 0.6785 | 平移对称化（散射低通改为 stride 2，4 个子网格取平均） |
| 58 | 0.6790 | 训练时 ±1 像素随机平移 |
| 62 | 0.6841 | 每个视角独立平移（增强倍数，De et al. 2022） |
| 74 | 0.6876 | 测试时增强，72 个视角取平均 |
| 87 | 0.6911 | 去掉输入 GroupNorm 多余的可学习参数（少了 486 个要加噪声的参数） |

**需要注意的地方**

1. **想法生成没有运行（纯搜索模式）。** `agents/ai_scientist_v2/agent.py::_generate_ideas` 先检查 `ml_tasks/<TASK>/theaiscientist/seed_ideas.json`，没有任何任务提供这个文件，所以直接跳过了想法生成。而 `generate_ideas` 本身并不需要它（代码注释写着 seed ideas 早已移除），这很可能是个 bug。影响：第 1、2 阶段（30% 的步数）的草稿没有绑定研究想法。第 3、4 阶段即使开启想法生成也不受想法引导。`summary.json` 里的 "Total Ideas: 8" 统计的是起草次数，不是想法数。
2. **`val_ratio` 改为 0.05。** `split_config.json` 在 `target_files` 里，prompt.json 也明确允许修改。改完之后训练集从 35,000 张增加到 37,525 张，val 集从 4,503 张缩小到 1,975 张（标准误约 1 个百分点，所以 val 比 test 高 1.6 点在预期之内）。prompt.json 里说 test "完全不受 split_config 影响"，这不准确：test 加载的是 val 阶段训练的模型，训练集大小正是由 `val_ratio` 决定的。所以 +8.8 点里有一小部分来自训练数据增加约 7%。
3. **搜索过程只能看到 val。** test 只在搜索结束后运行一次，结果不回传给 agent。但隐藏 test 集就在工作区的 `./data` 里（它是 CIFAR-10 训练集的一部分，划分种子固定为 42），agent 生成的代码理论上可以读到；沙盒和 code_editor 的危险代码检查都拦不住。本次检查了全部 100 步的快照，`algorithm.py` 只导入了 torch/math/opacus，没有读取任何数据，没有发生泄漏。
4. **推理代价变大。** 测试时增强让每张图要跑 72 次前向，任务没有限制这一点。

### 5.2 冒烟测试：ai_scientist_v2 × 其余 7 个 Lite 任务 × Opus 5.5（2026-11-03 05:16 ～ 06:44）

**做法**：每个任务跑 3 步，7 个任务同时启动，全部在沙盒中运行。

- 输出目录：`benchmark_results_smoke/`，`--workspace-label smoke`。
- 分配：GPU0 跑 pycil、domainbed、privacymeter；GPU1 跑 usb、officehome、openood；ART 设 `CUDA_VISIBLE_DEVICES=""`，只用 CPU。
- 同时用 `scripts/reproduction/resource_monitor.py` 每 5 秒采样一次。
- 日志和资源数据在 `docs/reproduction_logs/smoke_lite7/`。

当时是手动逐个启动的。之后整理出的 `scripts/reproduction/run_smoke_lite.sh` 用的是同样的命令和分配，但这个脚本本身**还没有实际运行过**。

**过程中发现并修复的问题**

1. privacymeter 第一次启动立刻失败，报错 `No such file or directory: 'conda'`。
   - 原因：建 privacy_meter 环境时，uv 选中了宿主机 `~/miniconda3` 里的 Python 3.12.9，而沙盒里看不到 miniconda，`env_command` 找不到解释器，就退回到了 conda。
   - 修复：沙盒按每个环境 `pyvenv.cfg` 里的 `home` 字段，把对应的解释器前缀只读挂载进来（见 4.3）。环境本身没有重建，保证和测 baseline 时一致。
   - 修复后重启 privacymeter（05:19）。失败的日志保留为 `Privacy_privacymeter.failed_conda.log`。
2. 沙盒原来把 `CUDA_VISIBLE_DEVICES=""` 当成没设置，结果给了所有 GPU。已改为空字符串表示不给 GPU。
3. 监控脚本的两个问题：
   - GPU 编号映射写反了，已修正；
   - 原来把各进程的 RSS 直接相加，DataLoader 子进程的共享内存被重复计算，domainbed 显示 57 GB，已改为 PSS。修正前的记录已从 csv 中删除。

**结果**（每个任务都跑完了 3 步和最终 test）

| 任务 | 新 baseline（val / test） | 冒烟测试（val / test） | 每步耗时（分钟，拥挤时） | 显存峰值 | 内存峰值（PSS） |
|---|---|---|---|---|---|
| pycil | 0.593 / 0.594 | 0.613 / 0.623 | 11.4 / 12.5 / 10.9 | 1.1 GB | 2.2 GB |
| usb | 0.083 / 0.086 | 0.070 / 0.079 | 14.0 / 19.3 / 18.1 | 1.6 GB | 1.6 GB |
| domainbed | 0.239 / 0.250 | 0.693 / 0.705 | 6.7 / 1.6 / 6.3 | 0.9 GB | 3.0 GB |
| officehome | 0.901 / 0.857 | 0.909 / 0.874 | 24.2 / 24.4 / 约 20 | 10.6 GB | 7.9 GB |
| privacymeter（越低越好） | 0.324 / 0.318 | 0.037 / 0.039 | 10.6 / 11.5 / 9.4 | 5.4 GB | 3.0 GB |
| ART（CPU） | 0.301 / 0.292 | 0.931 / 0.931 | 5.9 / 6.4 / 5.3 | — | 1.5 GB |
| openood | 0.880 / 0.921 | 0.914 / 0.986 | 13.9 / 18.3 / 18.7 | 2.0 GB | 2.5 GB |

- **整体情况**：
  - 单卡显存峰值：GPU0 7.3 GB，GPU1 14.2 GB。
  - GPU 利用率：每卡 3 个任务时 GPU0 约 90%，GPU1 约 95%；officehome 单独跑也有 84～97%。所以**瓶颈是 GPU 算力，不是显存**。
  - 没有超时，没有 Bedrock 限流。
  - 7 个任务一共约 52 万 token。
- **privacymeter**：第 1 步因为违反 TPR 约束被 `postprocess.py` 判为无效，这是任务规则在起作用。最好的一步用的是随机裁剪、翻转和 mixup，准确率从 0.611 升到 0.709。
- **openood**：test 比 val 高是任务设计如此。val 用 CIFAR-100 当分布外数据，test 用 SVHN。
- **domainbed（需要注意）**：第 1 步只有一次 code_editor 调用，没有想法输入。它写出的方法是：训练时随机交换红绿两个通道，预测时对两种通道顺序取平均。
  - 模型看到的输入只有任务说明、两个目标文件和 baseline 数值，没有 `datasets.py`、数据或 val/test 结果，所以**没有 test 泄漏**。
  - 但任务说明点名了 ColoredMNIST，这个数据集的构造是公开的（IRM 论文，颜色是伪特征，标签噪声 25%，上限约 75%）。agent 在下一步的分析里也直接写出了这些细节。
  - 所以这个提升来自题目先验，不能代表通用的域泛化能力。
- **ART（需要注意）**：
  - 把图像外框 4 像素清空（后门触发器都放在角落和边缘），并剔除低置信度样本；
  - 还改写了分类器的 `predict`，让评测时输入的图像也先清空外框。
  - 没有读取 test 数据，但同样利用了题目先验，而且绕过了评测接口。
- **usb**：3 步都没有超过 baseline。

### 5.3 正式测试：ai_scientist_v2 × 其余 7 个 Lite 任务 × Opus 5.5（2026-11-03 06:47 启动，进行中）

```bash
setsid nohup scripts/reproduction/run_lite_queue.sh docs/reproduction_logs/full_lite7 \
  > docs/reproduction_logs/full_lite7/queue.out 2>&1 < /dev/null &
python3 scripts/reproduction/resource_monitor.py docs/reproduction_logs/full_lite7/resources.csv 10 &
```

- 参数：每个任务跑 100 步，`--workspace-label opus55`，输出到 `benchmark_results/`。
- **超时改为 4800 秒**（`agent.ai_scientist_v2.execute_timeout=4800`，只在命令行覆盖）：
  - 默认的 2400 秒是所有 agent 共用的协议，agent 的提示词里不会提到这个时限。
  - 每卡跑 2 个任务时，同一份代码最坏会慢一倍，所以把超时加倍，相当于单独跑时仍有 40 分钟。
  - 事后根据 `Command finished in X s` 统计有多少步超过 2400 秒，在结果里注明"按标准协议会有 N 步超时"。
- **排队方式**：每卡 2 个位置，同一位置上的任务依次运行。估算工作量用的是 baseline 代码时的 val 耗时乘以 100 步。

  | 设备 | 位置 A | 位置 B | GPU 工作量估算 |
  |---|---|---|---|
  | GPU0 | officehome | privacymeter → pycil | 约 2600 分钟 |
  | GPU1 | usb | openood → domainbed | 约 2500 分钟 |
  | CPU | ART | | |

  每卡 2 个任务，是为了一个任务等 LLM 回复时，另一个正好用 GPU。officehome 最重，先和较轻的 privacymeter 搭配，降低超时风险。
- **预计**：考虑到 opacus 后期每步耗时涨了约 3 倍（3.5 分钟涨到约 10 分钟），总时长约 2.5～4 天，费用约 $150～200。
- **启动检查**：5 个位置都已启动，日志里打印了 `execute_timeout = 4800` 和 `max_steps = 100`。06:54 ART 第 1 步记录到 `Command finished in 284.4 s (returncode 0)`。`config_used.yaml` 要等任务结束才写出，届时再核对一次超时设置。
- 结果待补充。

---

## 6. 在新机器上重跑冒烟测试（5.2）

以下步骤是根据本次的操作过程整理的。2026-11-03 已在第二台机器（4 × L4）上完整走过一遍，按当时发现的问题修订过，详见第 8 节。

### 6.1 前提条件

| 项目 | 要求 |
|---|---|
| 系统 | Linux x86_64，内核允许非特权用户命名空间（bubblewrap 需要） |
| GPU | NVIDIA，驱动需支持 CUDA 12.8（所有环境都是 torch 2.7.1+cu128，带 sm_75～sm_120 的 kernel，L4/sm_89 实测可用）。脚本默认用 2 张卡，冒烟测试每卡峰值约 15 GB；只有 1 张卡时设 `GPU_A=0 GPU_B=0`，卡数更多或更慢时用 `SMOKE_TASKS` 自定义分配（6.7）。**GPU 比 Blackwell 慢很多时要加大超时**，L4 上 officehome 的 baseline 单步就要 50 分钟（第 8 节） |
| 命令 | `git`、`wget`、`unzip`、`curl`、`uv`、`bwrap`、`nvidia-smi`、`python3`。`conda` 和 `pip` 只在要建的环境里有 conda 环境时才需要（`setup.py` 已改为按需检查）；Lite 的 8 个任务都不需要 |
| AWS | 能在 us-west-2（或通过 `AWS_REGION` 指定的区域）用 Bedrock 调用 `us.anthropic.claude-opus-5-5`，需要 InvokeModel / InvokeModelWithResponseStream 权限 |
| AWS 凭证 | EC2 实例角色（沙盒共享网络，能访问 IMDS），或 `AWS_PROFILE`（沙盒会只读挂载 `~/.aws`），或 `AWS_ACCESS_KEY_ID` 等环境变量（沙盒会传入） |
| 费用和时间 | 冒烟测试约 50 万 token，按官方价格估算约 $5，以 Bedrock 账单为准；两张卡上约 1.5 小时 |

### 6.2 获取代码

改动在 `repro/bedrock-uv-sandbox` 分支上，但**截至 2026-11-03 有一部分还没提交**，在新机器上重跑之前必须先提交并推送：

- `sandbox/run_in_bwrap.sh`：`CUDA_VISIBLE_DEVICES=""` 的处理和解释器前缀挂载。缺了后者，privacymeter 会报 `conda` 找不到；
- `benchmark/executor.py`：命令耗时记录；
- `scripts/reproduction/`：`resource_monitor.py`、`run_smoke_lite.sh`、`run_lite_queue.sh`；
- `ml_tasks/*/baseline_results/`：新 baseline 以及 `original/` 目录；
- 本文档。

```bash
git clone <仓库地址> FML-bench && cd FML-bench
git checkout repro/bedrock-uv-sandbox
```

### 6.3 构建环境

```bash
export UV_PYTHON_PREFERENCE=only-managed   # 只用 uv 自己管理的 Python，见下面的说明
# export FMLBENCH_UV_ENVS=/path/to/envs    # 可选，默认 ~/fml_envs
python3 setup.py --skip-data \
  --task Privacy_opacus --task Continual_Learning_pycil --task Data_Efficiency_usb \
  --task Generalization_domainbed --task Generalization_domainbed_officehome \
  --task Privacy_privacymeter --task Robustness_and_Reliability_art --task Robustness_openood
```

- `UV_PYTHON_PREFERENCE=only-managed` 是 uv 自带的环境变量，不需要改 `setup.py`。本机建环境时没有设置它，所以 privacy_meter 用到了 miniconda 的 Python 3.12.9；设置后会用 uv 管理的 3.12.x，补丁版本可能不同。即使没设，沙盒的解释器挂载也能处理这种情况。
- 环境建在本地盘上即可，本机是 NVMe 上的 `~/fml_envs`。

### 6.4 准备数据

两种方式选一种：

- **官方源**：直接运行上面的 `setup.py` 命令，但去掉 `--skip-data`。本机实测多伦多 CIFAR 服务器只有约 35 KB/s。
- **镜像**：按第 2 节「数据」中的来源下载，放到对应位置，然后运行一次不带 `--skip-data` 的 `setup.py`。它会跳过已有的数据，完成后续处理（比如 privacymeter 的 pkl 生成）。之后：
  - pycil 和 usb 需要**手动提交数据**，因为预先放好数据会跳过 setup 里的提交步骤，而 executor 每步都会 `git reset --hard` 加 `git clean -fd`：
    ```bash
    for r in workspace/Continual_Learning_pycil/PyCIL workspace/Data_Efficiency_usb/Semi-supervised-learning; do
      git -C $r add -f data && git -C $r commit -m "add cifar-100 dataset for agent benchmark"
    done
    ```
    `data/` 在这两个仓库的 `.gitignore` 里，必须用 `add -f`（和 `setup.py` 的 `git_commit_data` 一致）。早先这里写的是 `add -A data`，什么都不会加进去，而且下面的检查也发现不了这个问题。可以用 `git -C $r ls-files data | wc -l` 确认数据确实进了仓库。
  - 最后确认 8 个任务的代码库都是干净的（下面的命令不应有任何输出），否则数据可能在第一步就被清掉：
    ```bash
    for c in ml_tasks/*/config.json; do
      r=$(python3 -c "import json;print(json.load(open('$c'))['repo_dir'])")
      [ -d "$r" ] && git -C "$r" status --porcelain
    done
    ```

officehome 第一次运行时会把 torchvision 的预训练权重下载到 `~/.cache`。沙盒可以写这个目录，网络也是共享的，所以不需要额外操作。

### 6.5 Baseline

- **硬件和第 1 节相同**（Blackwell + 本文的环境）：直接用分支里提交的新 baseline。
- **硬件不同**：建议先重测，用结果覆盖 `ml_tasks/<TASK>/baseline_results/{val,test}_info.json`。agent 提示词里的 baseline 和最终比较都读这两个文件。

  ```bash
  scripts/reproduction/rebaseline_task.sh <TASK> <GPU编号，CPU 填 "">
  python3 scripts/reproduction/compare_baseline.py <TASK>
  ```

  ART 要传 `""`，和 baseline 一样只用 CPU。

### 6.6 检查沙盒

```bash
sandbox/run_in_bwrap.sh --shell -c 'for e in opacus pycil usb domainbed privacy_meter openood art; do
  ~/fml_envs/$e/bin/python -c "import sys; print(\"$e\", sys.version.split()[0])"; done'
sandbox/run_in_bwrap.sh --shell -c '~/fml_envs/opacus/bin/python -c "import torch; print(torch.cuda.device_count())"'
```

每个环境都应该打印出 Python 版本，GPU 数量也应该正确。Bedrock 能否调用，可以参照 4.3 里的验证方式：在沙盒里用 `agents.llm.create_client(..., "AnthropicBedrock")` 发一条短消息。

### 6.7 运行冒烟测试

```bash
GPU_A=0 GPU_B=1 scripts/reproduction/run_smoke_lite.sh docs/reproduction_logs/smoke_lite7_<机器名>
```

- 脚本在前台运行，会启动资源监控，同时运行 7 个任务，每个跑 3 步。分配和 5.2 相同：GPU_A 跑 pycil、domainbed、privacymeter，GPU_B 跑 usb、officehome、openood，ART 只用 CPU。
- 全部结束后，脚本会打印每个任务的命令数、失败数、最佳 val 和 test，以及显存和内存峰值。
- 可调的环境变量：`STEPS`（默认 3）、`OUTPUT_DIR`（默认 `benchmark_results_smoke`）、`SMOKE_TASKS`（用空格分隔的 `任务配置|CUDA_VISIBLE_DEVICES` 列表，替换默认的 7 任务 / 2 卡分配，可以加上 `privacy_opacus`）、`EXTRA_OVERRIDES`（传给 run_agent_benchmark.py 的额外覆盖，例如 `agent.ai_scientist_v2.execute_timeout=7200`）。第 8 节的用法：
  ```bash
  SMOKE_TASKS="generalization_officehome|0 robustness_openood|1 generalization|1 privacy_privacymeter|2 privacy_opacus|2 continual_learning_pycil|3 data_efficiency_usb|3 robustness_and_reliability_art|" \
  EXTRA_OVERRIDES="agent.ai_scientist_v2.execute_timeout=7200" \
    scripts/reproduction/run_smoke_lite.sh docs/reproduction_logs/l4/smoke_lite8
  ```
- 跑这么久，建议用 `setsid nohup ... &` 放到后台，或者在 tmux 里运行。

### 6.8 怎么判断是否成功

- 每个任务都跑完 3 步（日志里有 `Best Val Metric` 和 `Test Metric`），并且清理了工作区副本（`Cleaned up workspace copy`）。
- 日志里没有 `No such file or directory: 'conda'`，没有 `Timeout after`，也没有 Bedrock 鉴权错误。
- 指标和 5.2 不需要完全一致，因为 agent 的输出是随机的。但量级应该相近：pycil、officehome 接近或略高于 baseline，domainbed 和 ART 很可能大幅提升，usb 可能不升。
- privacymeter 某一步因为 TPR 约束被判无效是正常的。

### 6.9 本次踩过的坑

| 现象 | 原因与处理 |
|---|---|
| privacymeter 报 `No such file or directory: 'conda'` | 环境的基础解释器在沙盒里看不到。需要沙盒脚本的解释器挂载（6.2），或者建环境时设 `UV_PYTHON_PREFERENCE=only-managed`（6.3） |
| ART 被分到了 GPU | 需要新版沙盒脚本，并且传 `CUDA_VISIBLE_DEVICES=""` |
| officehome 不到 10 秒就失败（timm 报错） | `huggingface_hub` 装成了 2.x。`setup.py` 已锁定 `<1.0` |
| 沙盒里 DNS 解析失败 | `/etc/resolv.conf` 是指向 `/run/systemd/resolve/` 的软链接，沙盒脚本已挂载这个目录 |
| 数据在第一步被删 | 数据没提交到任务代码库，见 6.4（注意要用 `add -f`） |
| `setup.py` 报 `missing required commands: conda, pip` | 旧版启动检查无条件要求 conda/pip。已改为只有要建 conda 环境时才检查 |
| 全新机器上建 privacy_meter 环境失败（找不到 `requirements.txt`） | 环境要读克隆下来的仓库里的 requirements.txt，但建环境在克隆之前。已改为建这个环境前先克隆 |
| GPU 较慢的机器上 officehome / openood / privacymeter 超时 | 默认 2400 秒是按 Blackwell 的速度定的；L4 上要用 `execute_timeout=7200`（第 8 节） |
| 用 `pkill -f <关键字>` 停监控时，执行命令的 shell 也被杀掉 | 关键字匹配到了这条命令自己的命令行。改用具体的 PID 去 `kill` |

---

## 7. 待办 / 未决事项

- [ ] 是否修复 ai_scientist_v2 跳过想法生成的问题，并用完整流程重跑 opacus
- [ ] 用 `val_ratio=0.05` 重测 opacus baseline，把训练数据增加的影响和算法改进分开
- [ ] 堵住隐藏 test 集可被读取的漏洞（把数据移出 val 阶段可访问的位置，或者在检查里拦截数据读取）
- [ ] ART：是否构建能用 Blackwell GPU 的 TF 环境（会改动锁定的版本）
- [ ] 迁移并测试其余 10 个非 Lite 任务的环境
- [x] 提交 harness 代码与本文档初版（分支 `repro/bedrock-uv-sandbox`：`83030f4`、`7ea0e4f`）
- [ ] 提交之后的改动：`sandbox/run_in_bwrap.sh`、`benchmark/executor.py`、`scripts/reproduction/{resource_monitor.py,run_smoke_lite.sh,run_lite_queue.sh}`、`ml_tasks/*/baseline_results/`、本文档。在其他机器上重跑之前必须完成（见 6.2）
- [ ] 正式测试（5.3）结束后：补充结果，统计超过 2400 秒的步数，检查代码快照里有没有读取数据
- [ ] domainbed（ColoredMNIST）和 ART 利用题目先验大幅提升：要不要通过改任务说明来限制（这属于修改 benchmark，需要决定）
- [ ] `setup.py` 建 uv 环境时是否固定只用 uv 管理的 Python（`--python-preference only-managed`），避免用到宿主机的解释器
- [ ] 清理 `workspace/*__rebaseline_*` 副本（目前按要求保留）
- [ ] 第 8 节的 `setup.py`、`compare_baseline.py`、`run_smoke_lite.sh` 改动和本文档的修订尚未提交
- [ ] 超时协议：默认 2400 秒在 L4 上不够用（officehome 连 baseline 都跑不完）。跨硬件比较时，是按 GPU 速度换算超时，还是统一要求某个最低算力，需要决定

---

## 8. 第二台机器复现：4 × L4（2026-11-03 08:20 ～ 12:50）

按第 6 节的流程在另一台机器上从头走了一遍：建环境 → 重测 8 个 baseline → 8 个 Lite 任务的冒烟测试。第一台机器的正式测试（5.3）当时正在 EFS 的仓库目录里运行，为了不和它冲突（`workspace/`、`rebaseline_results/`、`benchmark_results_smoke/` 和 baseline 文件），整个仓库被复制到本机本地盘 `~/FML-bench`（不含 `workspace/` 和结果目录），所有操作都在这个副本里完成。环境在 `~/fml_envs`。结果和日志都复制到了 EFS 仓库的 `docs/reproduction_logs/l4/` 下（见下文各节）。

### 8.1 机器环境

| 项目 | 第一台（第 1 节） | 本机 |
|---|---|---|
| GPU | 2 × RTX PRO 6000 Blackwell（96 GB，sm_120） | **4 × NVIDIA L4（24 GB，sm_89）** |
| 驱动 / CUDA | 580.126.16 / 13.0 | 595.71.05 / 13.2 |
| CPU / 内存 | Intel Xeon 8559C，48 核 / 499 GB | **AMD EPYC 7R13，96 核** / 363 GB |
| 系统 | Amazon Linux 2023（kernel 6.1） | Amazon Linux 2023（kernel 6.12） |
| uv / bwrap | 0.11.14 / 0.10.0 | 0.12.10 / 0.10.0 |
| conda / pip | 有 | **都没有** |
| AWS | 实例角色 | 实例角色 `my-bedrock-use-role`，同一账号 |

### 8.2 环境（第 1 步）

结论：**uv 方案可以直接使用**，但 `setup.py` 有 3 处要改（已改，未提交）：

1. **启动检查无条件要求 `conda` 和 `pip`。** 本机两者都没有，`setup.py` 直接退出。已改为只有要建的环境里有 conda 环境时才检查 conda/pip；uv 是必需的。
2. **privacy_meter 环境的构建顺序。** 这个环境要读 `workspace/Privacy_privacymeter/ml_privacy_meter/requirements.txt`，但 `setup.py` 先建环境（第 1 阶段）、后克隆仓库（第 2 阶段）。在第一台机器上没出问题，是因为当时仓库已经克隆好了；在全新机器上会失败。已改为建这个环境前先克隆（`clone_privacymeter()`，可重复调用）。
3. **officehome 数据下载依赖宿主机的 `pip install gdown`。** 已改为 `uvx gdown <id> -O office_home.zip`。本次的数据是复制来的，这段代码**没有实际运行过**。

用 `UV_PYTHON_PREFERENCE=only-managed` 构建，privacy_meter 用的是 uv 自带的 Python 3.12.14（第一台是 miniconda 的 3.12.9）。其他环境：fmlbench 3.11.6，opacus/pycil/usb/domainbed/art 3.10.21，openood 3.9.25。

与第一台机器 `rebaseline_results/<TASK>/pip_freeze.txt` 的对比：

- opacus、pycil、usb、privacy_meter、openood 的包版本完全一致（只有 pip 自身是 26.3 对 26.2.2）。
- art 只有 `google-auth` 2.62.1 对 2.62.0。
- domainbed 与锁定 `huggingface_hub<1.0` 之后（officehome 那次）的记录一致。差别只是第一台多了几个包（httpx2、httpcore2、tomli、truststore），是先装了 hub 2.x 又降级时留下的。

所有 torch 环境都能在 L4 上做 GPU 计算（torch 2.7.1+cu128 带 sm_86/sm_90 的 kernel，sm_89 可以直接用）。ART 的 TF 2.10.1 看不到 GPU，和第一台一样只用 CPU。沙盒检查（6.6）全部通过：8 个环境都能启动，GPU 数量正确，`CUDA_VISIBLE_DEVICES=""` 下没有 GPU，ml_tasks 只读、workspace 可写，`~/.ssh`、EFS、conda 都不可见，沙盒内能调用 Bedrock。

数据：用 `setup.py --skip-data` 克隆代码库后，从 EFS 模板复制原始数据（md5 与第 2 节一致），再运行一次不带 `--skip-data` 的 `setup.py`。opacus、openood、privacymeter 由 setup 自己解压、生成 pkl 并提交；pycil 和 usb 手动提交。**文档里原来的手动提交命令（`git add -A data`）是错的**：`data/` 被这两个仓库的 `.gitignore` 忽略，必须用 `add -f`（6.4 已改）。最终 8 个代码库都是干净的。与 EFS 上的模板比较 git 文件树：7 个完全相同；privacymeter 的两个 pkl 字节不同，原因是 pickle 里存了数据集对象的绝对路径（`root`），数据和标签逐项相同。

### 8.3 Baseline 重测（第 2 步）

`rebaseline_task.sh` 并行运行。分配：GPU0 officehome；GPU1 usb + domainbed；GPU2 openood + opacus；GPU3 pycil + privacymeter；ART 只用 CPU。`compare_baseline.py` 已扩展，现在同时输出原始值（`original/`）、当前采用的 baseline（第一台的重测结果）和本机结果。

| 任务 | 指标（方向） | val：原始 / 采用 / **L4** | test：原始 / 采用 / **L4** | L4 耗时（val/test） | 第一台耗时 | 结论 |
|---|---|---|---|---|---|---|
| Privacy_opacus | test_acc（↑） | 0.5894 / 0.5881 / **0.5878** | 0.5877 / 0.5870 / **0.5854** | 2m14s / 7s | 1m10s | 一致 |
| Continual_Learning_pycil | avg_incremental_acc（↑） | 0.5911 / 0.5925 / **0.6075** | 0.5952 / 0.5944 / **0.6034** | 21m36s / 21m02s | 8m54s | 接近（+1～1.5 点） |
| Privacy_privacymeter | AUC_gap（↓） | 0.3339 / 0.3242 / **0.3282** | 0.3214 / 0.3182 / **0.3178** | 40m39s / <1s | 6m15s | 一致 |
| Robustness_openood | auroc（↑） | 0.8758 / 0.8804 / **0.8697** | 0.9302 / 0.9212 / **0.9106** | 32m07s / 9s | 10m25s | 接近（−1 点） |
| Data_Efficiency_usb | test_acc（↑） | 0.0757 / 0.0827 / **0.0723** | 0.0790 / 0.0861 / **0.0829** | 14m39s / 6s | 11m11s | 接近（原值本身约 8%） |
| Generalization_domainbed_officehome | avg_acc（↑） | 0.8832 / 0.9011 / **0.8942** | 0.8629 / 0.8570 / **0.8599** | **50m19s** / <1s | 10m55s | 接近 |
| Generalization_domainbed | in_acc（↑） | 0.2792 / 0.2393 / **0.2626** | 0.2868 / 0.2501 / **0.2706** | 4m48s / <1s | 3m32s | 介于原始和采用值之间 |
| Robustness_and_Reliability_art（CPU） | defense_score（↑） | 0.4720 / 0.3005 / **0.2913** | 0.4774 / 0.2921 / **0.2895** | 5m46s / 7s | 4m51s | 与采用值接近，与原始值仍差很多 |

结果目录：`docs/reproduction_logs/l4/rebaseline_results/<TASK>/`（在本机上是 `~/FML-bench/rebaseline_results/<TASK>/`）。setup 日志：`docs/reproduction_logs/l4/setup_*.log`。

**分析**

- 8 个任务都跑通了，与当前采用的 baseline 相比，全部一致或接近。
- **domainbed 补充了第 3 节的结论。** 每台机器上都只有一个种子，结果是确定性的（第一台跑两次完全相同，都是 0.239320）。本机用同样的 torch 2.7.1，得到 0.2626，落在原始值（torch 1.12.1，0.279）和 Blackwell（0.239）之间。所以第 3 节"最可能是 torch 版本"的判断不完整：**同一个 torch 版本只换 GPU，结果也会变化约 0.02**。这个任务对硬件和 kernel 层面的数值差异本身就很敏感。
- **ART 进一步支持第 3 节的判断。** 本机 CPU 的各项分指标和第一台 CPU 几乎一样（clean_acc 0.9555 对 0.9558，poison_acc 0.816 对 0.813），defense_score 的差距约 0.01，可能来自 AMD 和 Intel 的浮点差异。两台都离原始的 0.472 很远，"原 baseline 用的是 GPU 或另一版代码"的可能性更大了。
- **速度**：上表的 L4 耗时是在多个任务共用 GPU 的情况下测的，和第一台机器的并发情况也不一样，所以不能直接比较。为此又做了一次对照实验：每个任务单独占一张 L4 重跑 val（`rebaseline_results/<TASK>__solo/`），同时用 `nvidia-smi dmon` 每 5 秒采样一次（`docs/reproduction_logs/l4/solo/dmon*.txt`）。除 privacymeter 外，重跑的 val 结果与第一次 L4 测量逐字节相同。privacymeter 开了 `cudnn.benchmark`，卷积算法的选择不固定，结果是 0.3287 对 0.3282。

  | 任务 | 第一台（并发情况） | L4 第一次（并发情况） | **L4 单卡** | **单卡 / 第一台** | L4 单卡时：SM 利用率 / 显存带宽 / 功耗 / SM 时钟 | 瓶颈 |
  |---|---|---|---|---|---|---|
  | officehome | 10m55s（单独） | 50m19s（单独） | 50m19s | **4.6 倍** | 没有用 dmon 记录；baseline 运行时抽查一次是 94% | GPU 算力（ResNet-50，224×224，fp32/TF32） |
  | privacymeter | 6m15s（单独） | 40m39s（前 21 分钟和 pycil 共用） | 25m21s | **4.1 倍** | 96% / 78% / 70.6 W（97% 的时间碰到功耗上限）/ 1182 MHz | GPU 算力（4 个 WRN-28-2，每个 epoch 做一次完整评测） |
  | openood | 10m25s（和 usb 共用） | 32m07s（和 opacus 共用 2 分钟） | 30m55s | **3.0 倍**（第一台有并发，真实差距更大） | 97% / 76% / 70.8 W（95%）/ 1325 MHz | GPU 算力（ResNet-18，32×32，第一层不降采样，100 epoch） |
  | opacus | 1m10s（单独） | 2m14s（和 openood 共用） | 2m31s | 2.1 倍 | 41% / 44% / 47 W / 1903 MHz | 混合：每样本梯度加上 CPU 开销（2 个 worker、隐私预算计算） |
  | pycil | 8m54s（和 domainbed 共用 3.5 分钟） | 21m36s（和 privacymeter 共用全程） | 12m19s | 1.4 倍 | 50% / 25% / 57 W / 1819 MHz | CPU、数据加载、Python 开销（ResNet-32 只有 0.46M 参数，ColorJitter 在 CPU 上做） |
  | usb | 11m11s（和 openood 共用） | 14m39s（和 domainbed 共用 5 分钟） | 14m28s | 1.3 倍 | 40% / 27% / 55 W / 1831 MHz | CPU 和数据加载（2 个 worker；有标注数据只有 200 张，每 3 步就重建一次 worker，全程约 5000 次；RandAugment 在 CPU 上做） |
  | domainbed | 3m32s（和 pycil、ART 共用 CPU） | 4m48s（和 usb、ART 共用） | **1m49s** | **0.5 倍（更快）** | 46% / 34% / 63 W / 1934 MHz | CPU 和 Python（MNIST 小 CNN，每 100 步对 7 万张图做一次评测，8 个 loader × 8 个 worker） |
  | ART（CPU） | 4m51s | 5m46s | — | 1.2 倍 | — | CPU（Intel Xeon 8559C 对比 AMD EPYC 7R13） |

  结论：**这种差距是合理的，原因有两层。**
  1. **任务本身的瓶颈不同。** 受 GPU 算力限制的任务（officehome、privacymeter、openood）慢 3～4.6 倍。L4 的 fp32/TF32 算力和显存带宽本来就比 RTX PRO 6000 低好几倍，而且功耗上限只有 72 W。这几个任务 95% 以上的时间都碰到功耗上限（`SW Power Cap` 处于触发状态），SM 时钟从 2040 MHz 降到 1180～1330 MHz。受 CPU、数据加载或 Python 开销限制的任务（pycil、usb、domainbed），L4 的 GPU 只用到 40～50%，换更快的 GPU 也省不了多少时间，所以只慢 1.3～1.4 倍，甚至更快。opacus 介于两者之间。
  2. **并发的影响。** 第一次 L4 测量时，privacymeter（6.5 倍 → 单卡 4.1 倍）和 pycil（2.4 倍 → 单卡 1.4 倍）都和其他任务共用 GPU，被拖慢了很多。domainbed 对 CPU 竞争很敏感：两台机器第一次测量时都和占满全部 CPU 的 ART（TF CPU）同时运行，单独跑时只要 1m49s，比第一台的 3m32s 还快（本机 96 核，第一台 48 核）。
  - 对冒烟测试和正式测试的影响：超时风险集中在 officehome、privacymeter、openood 这三个受 GPU 算力限制的任务上，同一张卡上再加任务会更慢。在 L4 上，这三个任务最好每个单独占一张卡。
- **采用**：`ml_tasks/*/baseline_results/` **没有改动**，冒烟测试的提示词里用的仍是第一台测出的 baseline，以便和 5.2 比较。本机结果与它的差距都在可接受范围内。如果要以 L4 作为正式评测平台，需要另外决定是否换成本机的值。

### 8.4 冒烟测试：ai_scientist_v2 × 8 个 Lite 任务 × Opus 5.5（09:25 ～ 12:49）

```bash
SMOKE_TASKS="generalization_officehome|0 robustness_openood|1 generalization|1 privacy_privacymeter|2 privacy_opacus|2 continual_learning_pycil|3 data_efficiency_usb|3 robustness_and_reliability_art|" \
EXTRA_OVERRIDES="agent.ai_scientist_v2.execute_timeout=7200" \
  setsid nohup scripts/reproduction/run_smoke_lite.sh docs/reproduction_logs/l4/smoke_lite8 > docs/reproduction_logs/l4/smoke_lite8/smoke.out 2>&1 &
```

- 在 5.2 的 7 个任务之外加上了 opacus，每个任务 3 步，全部在沙盒中运行，输出到 `~/FML-bench/benchmark_results_smoke/`（副本在 `docs/reproduction_logs/l4/benchmark_results_smoke/`），`--workspace-label smoke`。
- **超时改为 7200 秒**（只在命令行覆盖，最终 test 用的是同一个值）。原因：officehome 的 baseline 在 L4 上单独跑一步就要 50 分钟，按默认的 2400 秒，第一步就必然超时。7200 秒大致相当于"L4 比 Blackwell 慢 3 倍，再留一些余量"。
- 分配：officehome 单独占 GPU0（最重）；GPU1 openood + domainbed；GPU2 privacymeter + opacus；GPU3 pycil + usb；ART 只用 CPU。依据是 8.3 里的耗时。

**结果**：8 个任务全部完成 3 步和最终 test，共 40 条命令，全部 returncode 0。没有 `conda` 报错，没有超时，没有 Bedrock 鉴权、限流或拒答，工作区副本都已清理（6.8 的判断标准全部满足）。

| 任务 | baseline（val / test） | 5.2 冒烟（val / test） | **L4 冒烟（val / test）** | 每条命令耗时（秒，3 步 + 最终重训练） | 超过 2400 秒 | 显存峰值 | 内存峰值（PSS） | token |
|---|---|---|---|---|---|---|---|---|
| opacus | 0.588 / 0.587 | （5.1 冒烟 test 0.631） | **0.629 / 0.618** | 427 / 638 / 636 / 473 | 0 | 11.9 GB | 2.2 GB | 2.4 万 |
| pycil | 0.593 / 0.594 | 0.613 / 0.623 | **0.610 / 0.611** | 1238 / 1179 / 1140 / 1190 | 0 | 0.7 GB | 1.9 GB | 3.4 万 |
| usb | 0.083 / 0.086 | 0.070 / 0.079 | **0.109 / 0.107** | 1318 / 1327 / 1784 / 1355 | 0 | 1.4 GB | 1.4 GB | 3.6 万 |
| domainbed | 0.239 / 0.250 | 0.693 / 0.705 | **0.728 / 0.740** | 799 / 824 / 624 / 727 | 0 | 0.7 GB | 2.8 GB | 15.5 万 |
| officehome | 0.901 / 0.857 | 0.909 / 0.874 | **0.911 / 0.874** | 3022 / 3011 / 3001 / 2994 | **4** | 9.9 GB | 7.5 GB | 16.2 万 |
| privacymeter（越低越好） | 0.324 / 0.318 | 0.037 / 0.039 | **0.057 / 0.064** | 2607 / 1528 / 1520 / 1522 | **1** | 4.8 GB | 2.9 GB | 4.9 万 |
| ART（CPU） | 0.301 / 0.292 | 0.931 / 0.931 | **0.967 / 0.966** | 373 / 222 / 370 / 224 | 0 | — | 2.9 GB | 5.7 万 |
| openood | 0.880 / 0.921 | 0.914 / 0.986 | **0.915 / 0.986** | 2497 / 2027 / 1839 / 1841 | **1** | 1.5 GB | 2.3 GB | 2.2 万 |

（评测命令只要几秒，表中没有列出。）

- **整体**：
  - 总耗时 3 小时 24 分钟，由 officehome 决定（4 次训练，每次约 50 分钟）。其余任务都在 2 小时内完成。
  - token 共约 54 万，和 5.2 的 7 个任务（约 52 万）相当，按官方价格估算约 $5～6。
  - 单卡显存峰值：GPU0 9.9 GB，GPU1 2.1 GB，GPU2 13.3 GB，GPU3 2.1 GB，都没有超过 L4 的 24 GB。
- **按标准协议（2400 秒）会有 6 条命令超时**：officehome 全部 4 条（包括最终重训练），privacymeter 第 1 步（和 opacus 共用 GPU 时），openood 第 1 步。officehome 的每一步都和 baseline 一样约 50 分钟，所以不是 agent 让代码变慢了，纯粹是 L4 算力不够。
- 指标量级和 5.2 一致：pycil、officehome、openood 接近或略高于 baseline；domainbed、ART、privacymeter 大幅提升。usb 这次从 0.083 提升到 0.109（5.2 时没有超过 baseline）。opacus 和 5.1 的冒烟测试相近。
- **privacymeter**：最终 test 没有违反 TPR 约束（`constraint_violated: False`），test 准确率 0.755。
- **domainbed（需要注意，和 5.2 相同）**：最优的第 2 步仍然是对红绿通道做置换对称（训练和预测时都对两种通道顺序的 logits 取平均），另外加了权重 EMA 和小损失截断（针对 25% 的标签噪声）。还是在利用 ColoredMNIST 的构造先验。
- **ART（和 5.2 不同）**：最优的第 2 步只改了 `dp_instahide_trainer.py`（`model.py` 和 `split_config.json` 只改了空白）。做法是：先用 2 个 epoch 预训练，再在倒数第二层的特征上按类别做谱签名（spectral signatures）和 2 簇激活聚类（activation clustering），并按标签一致性过滤可疑样本，然后把权重重置为初始值，在过滤后的数据上做 DP-InstaHide 训练。**没有清空图像外框，也没有改写 `predict`**，用的是已发表的通用后门防御方法，不依赖触发器的位置。只有 3 步，样本太少，不能据此判断 agent 的倾向。

日志和资源数据：`docs/reproduction_logs/l4/smoke_lite8/`（`*.log` 被 `.gitignore` 忽略，`resources.csv` 和 `smoke.out` 可以提交）。

---

## 附：相关文件

| 路径 | 说明 |
|---|---|
| `scripts/reproduction/rebaseline_task.sh` | 单任务 baseline 重测（val → test，uv 环境） |
| `scripts/reproduction/compare_baseline.py` | 将重测结果与 `baseline_results/original/` 对比 |
| `rebaseline_results/<TASK>[__rerun2]/` | 重测的原始输出 |
| `ml_tasks/<TASK>/baseline_results/original/` | 仓库原有的 baseline 文件 |
| `sandbox/run_in_bwrap.sh` | 沙盒入口 |
| `scripts/reproduction/resource_monitor.py` | 按工作区副本统计各任务的显存和内存（PSS）；`--summary` 输出峰值 |
| `scripts/reproduction/run_smoke_lite.sh` | Lite 任务的冒烟测试（6.7）。默认是 7 个任务 / 2 张卡，可用 `SMOKE_TASKS` / `EXTRA_OVERRIDES` 自定义 |
| `scripts/reproduction/run_lite_queue.sh` | 7 个 Lite 任务的正式测试，按位置排队（5.3） |
| `benchmark_results_smoke/` | 冒烟测试结果 |
| `docs/reproduction_logs/` | agent 运行日志。`*.log` 被 `.gitignore` 忽略，只存在本机；`smoke_lite7/`、`full_lite7/` 下还有资源采样的 csv |
