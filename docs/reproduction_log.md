# FML-bench 复现实验记录

本文档记录在本机上复现 FML-bench 的过程：环境适配、baseline 重测、harness 改造（Bedrock / uv / 沙盒），以及 agent 测试结果。后续每次复现实验都追加到「实验记录」一节。

- 仓库基线提交：`d336651`（以下改动均未提交，基于该提交的工作区）
- 记录时间：2026-11-02 ～ 2026-11-03

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
| 设备 | NVIDIA 设备节点全部透传，`CUDA_VISIBLE_DEVICES` 原样传入 |
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

---

## 6. 待办 / 未决事项

- [ ] 是否修复 ai_scientist_v2 跳过想法生成的问题，并用完整流程重跑 opacus
- [ ] 用 `val_ratio=0.05` 重测 opacus baseline，把训练数据增加的影响和算法改进分开
- [ ] 堵住隐藏 test 集可被读取的漏洞（把数据移出 val 阶段可访问的位置，或者在检查里拦截数据读取）
- [ ] ART：是否构建能用 Blackwell GPU 的 TF 环境（会改动锁定的版本）
- [ ] 迁移并测试其余 10 个非 Lite 任务的环境
- [ ] 提交改动（`setup.py`、`agents/llm.py`、`agents/code_editor.py`、`benchmark/executor.py`、`benchmark/utils.py`、`agents/theaiscientist/theaiscientist.py`、`ml_tasks/*/baseline_results/`、`sandbox/`、`scripts/reproduction/`、本文档）
- [ ] 清理 `workspace/*__rebaseline_*` 副本（目前按要求保留）

---

## 附：相关文件

| 路径 | 说明 |
|---|---|
| `scripts/reproduction/rebaseline_task.sh` | 单任务 baseline 重测（val → test，uv 环境） |
| `scripts/reproduction/compare_baseline.py` | 将重测结果与 `baseline_results/original/` 对比 |
| `rebaseline_results/<TASK>[__rerun2]/` | 重测的原始输出 |
| `ml_tasks/<TASK>/baseline_results/original/` | 仓库原有的 baseline 文件 |
| `sandbox/run_in_bwrap.sh` | 沙盒入口 |
| `docs/reproduction_logs/` | agent 运行日志 |
