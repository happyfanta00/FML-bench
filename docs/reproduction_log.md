# FML-bench 复现记录

本文档记录 FML-bench 在 uv 环境、Amazon Bedrock（Opus 5.5）和 bubblewrap 沙盒下的复现配置与结果：环境、数据、baseline、harness 改造、agent 测试结果，以及在新机器上复现的步骤。

- 上游基线提交：`d336651`；所有改动在分支 `repro/bedrock-uv-sandbox` 上
- Baseline 的测量日期：2026-11-04（g7e.24xlarge，见第 1 节）

---

## 1. 机器环境

| 用途 | 机器 | GPU | 驱动 / CUDA | CPU / 内存 | 系统 |
|---|---|---|---|---|---|
| **Baseline（第 3 节）** | AWS `g7e.24xlarge`（us-east-1） | 4 × NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB，sm_120） | 595.112.07 / 13.0 | Intel Xeon Platinum 8559C，96 核 / 999 GB | Amazon Linux 2023，kernel 6.12 |
| Agent 测试（5.1、5.2 中的 Blackwell 部分） | AWS `g7e.12xlarge`（us-west-2） | 2 × 同型号 GPU | 580.126.16 / 13.0 | Intel Xeon Platinum 8559C，48 核 / 499 GB | Amazon Linux 2023，kernel 6.1 |
| 其他硬件对照（第 8 节） | 4 × L4 | 4 × NVIDIA L4（24 GB，sm_89） | 595.71.05 / 13.2 | AMD EPYC 7R13，96 核 / 363 GB | Amazon Linux 2023，kernel 6.12 |

- 软件：uv 0.13.7（不使用 conda），bubblewrap 0.12.0。LLM 为 Amazon Bedrock 的 `us.anthropic.claude-opus-5-5`，凭证来自实例角色。
- **硬件约束**：Blackwell（sm_120）需要 torch ≥ 2.7 和 CUDA 12.8。仓库原配置的 torch 2.5.1+cu121、1.12.1+cu113 等版本在这块卡上没有可用的 kernel，因此统一换成 `torch==2.7.1+cu128`（torchvision 0.22.1、torchaudio 2.7.1）。TF 2.10 不能使用这块 GPU，所以 ART 任务只在 CPU 上运行。

---

## 2. 环境构建（uv）与数据

### 2.1 环境

- 环境建在 `~/fml_envs/<env>`，可以用 `FMLBENCH_UV_ENVS` 改位置。解释器由 uv 管理（`UV_PYTHON_PREFERENCE=only-managed`），位于 `~/.local/share/uv/python`。
- `setup.py` 用 `uv_create` / `uv_run` 建环境，conda 和 pip 只在要建 conda 环境时才需要。

| 环境 | Python | 相对原配置的改动 |
|---|---|---|
| fmlbench（harness） | 3.11.6 | 去掉 conda 的 nodejs（harness 用不到）；`anthropic` 改为 `anthropic[bedrock]==0.84.0`（多装 boto3） |
| opacus / pycil / usb | 3.10.22 | torch 2.5.1+cu121 → 2.7.1+cu128 |
| openood | 3.9.25 | torch 2.5.1+cu121 → 2.7.1+cu128 |
| domainbed | 3.10.22 | conda 的 pytorch 1.12.1 → pip 的 torch 2.7.1+cu128；去掉 mkl 系列包；`backpack-for-pytorch` 1.3.0 → 1.7.1（1.3.0 要求 torch<2.0）；锁定 `huggingface_hub<1.0`（timm 0.9.16 用到的接口在 hub 1.x 中已移除） |
| privacy_meter | 3.12.16 | 安装 requirements.txt 时滤掉 torch/nvidia 相关行，torch 2.7.1+cu128 单独安装 |
| art | 3.10.22 | 版本不变；TF 2.10.1 只能用 CPU |

- 6 个 torch 环境都能在 sm_120 和 sm_89 上做 GPU 计算。
- 三台机器上的包版本只有 pip 自身和 art 的 `google-auth` 有补丁版本差异（g7e.12xlarge 的 domainbed 环境里还多出几个降级 hub 时留下的包：httpx2、httpcore2、tomli、truststore）。Python 补丁版本随 uv 版本略有不同。
- 其余 10 个非 Lite 任务的环境（easyfsl、lightly、continual_learning、causalml、aif360、fairlearn、gcastle、sololearn、pfllib、open_unlearning）仍是 conda 配置，**没有迁移，也没有测试**。

### 2.2 数据：共享缓存 + 软链

数据在缓存目录 `/mnt/nvme/data/` 下只存一份，用软链接到各任务实际读取的位置：

```bash
python3 setup.py --skip-envs --data-cache /mnt/nvme/data --task ...   # 或设 FMLBENCH_DATA_CACHE
```

| 缓存条目 | 链接到 | 来源 | 校验 |
|---|---|---|---|
| `cifar10/cifar-10-batches-py/`（另有 `cifar-10-python.tar.gz`） | opacus、openood 的 `data/cifar-10-batches-py` | Kaggle `pankrzysiu/cifar10-python` | tar.gz md5 `c58f30108f718f92721af3b95e74349a`；7 个 batch 文件与 torchvision 内置 md5 一致 |
| `cifar100/cifar-100-python/` | pycil、usb、openood 的 `data/cifar-100-python` | Kaggle `pypiahmad/cifar-100` | train/test/meta 与 torchvision 内置 md5 一致 |
| `svhn/test_32x32.mat` | openood 的 `data/svhn/test_32x32.mat` | Kaggle `khanhvudo72/svhncropped`（只用 test） | md5 `eb5a983be6a315427106f1b164d9cef3` |
| `mnist/MNIST/` | `workspace/Generalization_domainbed/data/MNIST` | torchvision 默认的 S3 镜像 | torchvision 下载时自带 md5 校验 |
| `officehome/office_home/{art,clipart,product,real_world}` | `workspace/Generalization_domainbed_officehome/data/office_home` | `uvx gdown 1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa`（135 MB，15,588 个文件） | — |
| `privacymeter/cifar10{,_population}.pkl` | privacymeter 的 `data/cifar10{,_population}.pkl` | setup 用缓存里的 CIFAR-10 生成 | — |
| `art/mnist.npz` | `~/.art/data/mnist.npz`（ART 运行时读取这里） | `https://s3.amazonaws.com/img-datasets/mnist.npz`（和 ART 代码里的 URL 相同） | sha256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1` |

- **下载源**：多伦多的 CIFAR 官方源约 50 KB/s，斯坦福的 SVHN 官方源连不上。Kaggle 镜像能达到 90～145 MB/s，而且不需要账号。
- **缓存条目缺失时**，setup 会自动下载到缓存（用的是 torchvision、gdown 或 curl）。也可以先按上表手动放好，setup 会直接使用。
- **提交到任务仓库**：`setup.py` 无论数据是下载的、手动放的还是软链，都会执行 `git add -f` 并提交。软链以 symlink 的形式存进 git（mode `120000`），所以 executor 每步执行 `git reset --hard && git clean -fd` 之后数据仍然在，`git status` 也是干净的。重复运行 setup 只会打印 `already links` / `already committed`。domainbed 和 officehome 的数据在任务仓库外面（模板目录 `data/`），不需要提交。
- **工作区副本**：runner 复制工作区时保留软链（`shutil.copytree(..., symlinks=True)`），数据不会被复制。
- **沙盒**：会把 `FMLBENCH_DATA_CACHE`，以及任务模板里所有绝对路径软链的目标，**只读**挂载进去，agent 生成的代码改不了共享数据。沙盒里的 `$HOME` 是空的，所以 ART 的 `mnist.npz` 会在每次运行时从 S3 下载（11 MB）。
- **权限**：缓存设为 `chmod -R a-w`。所有 baseline 代码都只读数据（torchvision 的 `download=True` 校验通过后不会写文件）。
- officehome 第一次运行时会把 ResNet-50 预训练权重下载到 `~/.cache`（沙盒里这个目录可写）。
- `/mnt/nvme` 是实例存储，实例停止后数据会丢失。在同样的路径下重新放好数据，或者重新运行一次带 `--data-cache` 的 setup 即可恢复，工作区里的软链和提交不受影响。

---

## 3. Baseline（FML-bench-Lite，8 个任务）

### 方法

- 按各任务 `config.json` 里的 `val_command` → `test_command` 顺序执行，两者用同一份工作区副本（test 复用 val 保存的模型；pycil 的 test 会重新训练），在对应的 uv 环境里运行。
- 用 `scripts/reproduction/rebaseline_queue.sh`：每张 GPU 一个串行队列，**同一张卡上同时只有一个任务**。ART（只用 CPU）等所有 GPU 任务结束后单独运行，因为 TF 会占满所有 CPU。同时用 `nvidia-smi dmon` 每 5 秒采样一次。
- 分配：GPU0 跑 pycil；GPU1 跑 officehome → opacus；GPU2 跑 usb → domainbed；GPU3 跑 openood → privacymeter；最后在 CPU 上跑 ART。全部共 21.5 分钟。

```bash
scripts/reproduction/rebaseline_queue.sh docs/reproduction_logs/g7e24/rebaseline __g7e24
```

### 结果（原始 → 采用）

| 任务 | 指标（方向） | val | test | 耗时（val / test） | GPU 平均 SM / 显存带宽 / 功耗 | 与原始值相比 |
|---|---|---|---|---|---|---|
| Privacy_opacus | test_acc_mean（↑） | 0.5894 → **0.5881** | 0.5877 → **0.5870** | 1m21s / 5s | 11% / 7% / 107 W | 一致 |
| Continual_Learning_pycil | avg_incremental_acc_mean（↑） | 0.5911 → **0.5925** | 0.5952 → **0.5944** | 8m48s / 8m31s | 42% / 6% / 190 W | 一致 |
| Privacy_privacymeter | AUC_gap_mean（↓） | 0.3339 → **0.3278** | 0.3214 → **0.3187** | 6m08s / <1s | 70% / 32% / 325 W | 一致 |
| Robustness_openood | auroc_mean（↑） | 0.8758 → **0.8708** | 0.9302 → **0.9028** | 6m56s / 7s | 86% / 38% / 374 W | 接近（任务本身不确定，见下） |
| Data_Efficiency_usb | test_acc_mean（↑） | 0.0757 → **0.0827** | 0.0790 → **0.0861** | 9m02s / 5s | 33% / 9% / 173 W | 接近（原始值本身只有约 8%） |
| Generalization_domainbed_officehome | avg_acc_mean（↑） | 0.8832 → **0.9011** | 0.8629 → **0.8570** | 10m38s / <1s | 74% / 55% / 318 W | 接近（相差 2 个点以内） |
| Generalization_domainbed（ColoredMNIST） | in_acc_mean（↑） | 0.2792 → **0.2393** | 0.2868 → **0.2501** | 1m45s / <1s | 25% / 2% / 158 W | **偏低约 0.04** |
| Robustness_and_Reliability_art（CPU） | defense_score_mean（↑） | 0.4720 → **0.2893** | 0.4774 → **0.2838** | 4m06s / 5s | — | **明显偏低** |

- **耗时是单独占一张卡时测的**（4 个 GPU 任务仍然共用 96 核的 CPU），包含 officehome 第一次运行时下载预训练权重的几秒。
- 按 GPU 利用率，任务分两类：
  - 受 GPU 算力限制：officehome、openood、privacymeter；
  - 受 CPU、数据加载或 Python 开销限制：pycil、usb、domainbed、opacus。
- domainbed 对 CPU 竞争很敏感，和占满 CPU 的 ART 同时运行时，耗时会翻倍。

### 确定性与偏差分析

- **在同型号 GPU 上可以复现**。在 g7e.12xlarge（驱动 580、kernel 6.1）上测得的结果：
  - pycil、usb、officehome、domainbed：val/test 结果文件逐字节相同；
  - opacus：只有 ε 在第 15 位有效数字上不同；
  - domainbed 和 ART：在同一台机器上重跑结果完全相同。
- **openood 本身不确定**。同一张卡单独重跑一次，得到 val 0.8764、test 0.9099；g7e.12xlarge 上是 0.8804 / 0.9212。训练脚本只设了随机种子，没有设 `cudnn.deterministic`。val 的噪声范围约为 0.871～0.880，test 约为 0.903～0.921。
- **privacymeter** 开了 `cudnn.benchmark`，卷积算法的选择不固定，所以多次运行的结果会相差约 0.004。
- **domainbed** 对 GPU 架构敏感。
  - 在两台 Blackwell 上都是 0.2393，L4 上是 0.2626（同样是 torch 2.7.1），原始值是 0.2792（torch 1.12.1）。
  - ColoredMNIST 留出的 env2 是对抗性的域，训练过程中它的精度在约 0.10～0.30 之间跳动，对数值上的微小差异非常敏感。
- **ART**：DP-InstaHide 训练 5 个 epoch，种子全部固定。
  - 与原始值的差距集中在部分后门上。攻击成功率（poison_acc）：random_corner 0.84 对原始 0.30，edge 0.90 对 0.75。clean_acc 反而更高（0.955 对 0.944）。
  - 最可能的原因是原 baseline 在 GPU 上训练，或者出自另一个版本的代码。原始文件之间本身也不一致：`final_info.json` 的 num_samples 是 7500，`val_info.json` 是 8404，另一个旧版 `final_info.old.json` 的 defense_score 是 0.6966。
  - 不同机器之间：同型号 CPU 的 g7e.12xlarge 上是 0.3005，AMD 的 L4 机器上是 0.2913。把 CPU 限制为 48 个（`taskset -c 0-47`）后，g7e.24xlarge 上的结果逐字节不变，所以这约 0.01 的差距**不是线程数造成的**，原因不明。

### 采用

- `ml_tasks/<TASK>/baseline_results/{val,test}_info.json` = 上表的采用值（`rebaseline_results/<TASK>__g7e24/`）。runner 读 `val_info.json` 作为 agent 提示词里的 baseline，`compute_agent_metrics.py` 读 `val_info.json` 和 `test_info.json`。
- 仓库原有的文件（包括 `final_info.json`）在 `ml_tasks/<TASK>/baseline_results/original/`。

---

## 4. Harness 改造

### 4.1 Bedrock 接入 Opus 5.5

- 用法：`--provider AnthropicBedrock --model us.anthropic.claude-opus-5-5`。区域取 `AWS_REGION`（默认 us-west-2），凭证走 AWS 的默认链。
- 调用 LLM 的只有两处，都加了 Claude 原生分支：
  - `agents/llm.py`：新增 `AnthropicBedrock` provider，以及 `claude_messages_create` / `claude_response_text` / `is_anthropic_client`。`get_response_from_llm` 的 Claude 分支改用这几个函数；多轮对话的历史保留完整的 assistant 内容（含 thinking 块），原样回传。
  - `agents/code_editor.py`：遇到 Anthropic 客户端时走 Messages API，其他情况仍用 `chat.completions`。
- 请求参数：
  - 流式输出，`max_tokens=64000`。Opus 5.5 的思考无法关闭，而且计入 max_tokens；上限调大之后 SDK 要求使用流式，否则可能超过 HTTP 超时。
  - 不传 temperature（Opus 5.5 不接受这个参数）。
  - `output_config.effort` 默认为 `high`，可以用 `FMLBENCH_CLAUDE_EFFORT` 调整。
  - 客户端 `max_retries=8`。
- 遇到拒答（`stop_reason == "refusal"`）直接抛错，**不会**自动换别的模型，以免评测结果里混入其他模型。
- `extract_usage_info` 能正确读取 Claude 响应的 `input_tokens` / `output_tokens`（含缓存部分）。
- SDK 版本为 0.84.0，走 bedrock-runtime（`anthropic.AnthropicBedrock`）。Bedrock Mantle 端点在本账号下对这个模型返回 404。

### 4.2 运行入口适配 uv 环境

- `benchmark/utils.py::env_command(env_name, args)`：如果 `~/fml_envs/<conda_env>/bin/python` 存在，就设置 `VIRTUAL_ENV`，把该环境放到 PATH 最前面后直接执行；否则退回 `conda run`。
- 调用方：`benchmark/executor.py::_run_command`（val/test 命令）和 `agents/theaiscientist/theaiscientist.py`（setup 命令）。config.json 的字段名 `conda_env` 保持不变。
- `_run_command` 在每条命令结束时输出 `Command finished in X s (returncode N)`，超时时输出 `Command timed out after N seconds`。
- `benchmark/runner.py` 复制工作区时保留软链（见 2.2）。

### 4.3 沙盒：`sandbox/run_in_bwrap.sh`

```bash
CUDA_VISIBLE_DEVICES=0 sandbox/run_in_bwrap.sh <run_agent_benchmark.py 的参数...>
sandbox/run_in_bwrap.sh --shell [-c CMD]     # 在同一个沙盒里开 shell 或执行单条命令
```

| 可见性 | 内容 |
|---|---|
| 只读 | 仓库（agents/、benchmark/、ml_tasks/、configs/…）、`~/fml_envs`、uv 的 Python、`/usr`、`/etc`、`/sys`、数据缓存（`FMLBENCH_DATA_CACHE` 和任务模板里绝对路径软链的目标） |
| 可写 | `workspace/`、输出目录（默认 `benchmark_results/`，能识别 `--output-dir`）、`~/.cache` |
| 不可见 | 宿主 home 的其余部分（包括 `~/.ssh`；`~/.aws` 只在设置了 `AWS_PROFILE` 时可见）、其他数据、conda |
| 设备 | 透传所有 NVIDIA 设备节点。`CUDA_VISIBLE_DEVICES` 原样传入：不设置表示所有 GPU，空字符串表示不给 GPU（ART 就这样运行） |
| 解释器 | 某个环境的基础解释器如果既不在 `/usr`，也不在 uv 的 Python 目录里，就把它所在的前缀目录只读挂载进来。conda 不在 PATH 上 |
| 隔离 | user/pid/ipc/uts/cgroup 命名空间独立，`--clearenv`、`--new-session`、`--die-with-parent` |

- **网络与宿主共享**。隔离网络会断掉 IMDS，Bedrock 就拿不到实例角色的凭证；代价是 agent 生成的代码也能访问外网。
- `/etc/resolv.conf` 指向 `/run/systemd/resolve/`，所以只读挂载了这个目录。
- 已验证：
  - ml_tasks、agents 和数据缓存不可写，workspace 可写，`~/.ssh` 不可见；
  - 能做 GPU 计算，`CUDA_VISIBLE_DEVICES=""` 时没有 GPU；
  - Bedrock 调用、DNS、HTTPS 都正常。

---

## 5. Agent 实验结果

### 5.1 ai_scientist_v2 × Privacy_opacus × Opus 5.5（100 步）

```bash
CUDA_VISIBLE_DEVICES=0 sandbox/run_in_bwrap.sh \
  --agent-config configs/agents/ai_scientist_v2.yaml \
  --task-config  configs/tasks/privacy_opacus.yaml \
  --model us.anthropic.claude-opus-5-5 --provider AnthropicBedrock \
  --workspace-label opus55 agent.ai_scientist_v2.max_steps=100
```

在 g7e.12xlarge 上运行，按 README 的标准协议（超时 2400 秒）。

| | val 准确率 | test 准确率 | ε |
|---|---|---|---|
| baseline | 0.5881 | 0.5870 | 8.00 |
| ai_scientist_v2 最优（第 88 步） | 0.6911 | **0.6752** | 7.9997 |
| 提升 | +10.3 点 | **+8.8 点** | 在预算内 |

- 4 个阶段的最优 val：基础实现 0.6365（10 步）→ 调参 0.6380（20 步）→ 创新探索 0.6876（50 步）→ 消融精炼 0.6911（20 步）。
- 100 步全部成功运行，没有 LLM 调用失败、拒答或训练超时。最终 test 之前用最优代码重新训练，val 仍是 0.6911（训练是确定性的）。
- 耗时 12.3 小时（平均约 7 分钟/步）。token 共 2,164,265（输入 1,201,852，输出含思考 962,413），按官方价格（每百万 token $4 / $20）估算约 $24，以 Bedrock 账单为准。
- 结果目录：
  - 搜索过程（100 步的代码快照 `step_snapshots/`、每步的 val、code_editor 日志）和 `summary.json`：`benchmark_results/ai_scientist_v2/Privacy_opacus/20261102_163447_2a504f3e/`
  - 最终 test：`benchmark_results/ai_scientist_v2/Privacy_opacus/20261103_044518/`
  - 运行日志：`docs/reproduction_logs/aisv2_opacus_full.log`

**agent 做了哪些改动**（每次刷新最好成绩时的关键改动；运行编号从 0 开始，对应快照 `step_<编号+1>`）

| 运行 | val | 改动 |
|---|---|---|
| 0 | 0.6046 | 30 epoch；`split_config.json` 的 `val_ratio` 0.114 → 0.05 |
| 3 | 0.6238 | batch 1024，lr 0.15 |
| 5 | 0.6365 | Tramer & Boneh 风格的 6 层卷积网络，tanh 激活，去掉大的全连接层，50 epoch，lr 0.2 |
| 34/35 | 0.6390 | 换路线：固定的 ScatterNet 散射特征（自己实现，J=2、L=8、二阶，243 通道）+ 小型 tanh CNN；60 epoch，lr 4.0，梯度裁剪 0.1，batch 4096 |
| 38 | 0.6542 | 权重 EMA（属于后处理，不额外消耗隐私预算） |
| 42 | 0.6754 | 水平翻转对称化（原图和镜像的 logits 取平均） |
| 46 | 0.6785 | 平移对称化（散射低通改为 stride 2，4 个子网格取平均） |
| 58 | 0.6790 | 训练时 ±1 像素随机平移 |
| 62 | 0.6841 | 每个视角独立平移（增强倍数，De et al. 2022） |
| 74 | 0.6876 | 测试时增强，72 个视角取平均 |
| 87 | 0.6911 | 去掉输入 GroupNorm 多余的可学习参数（少了 486 个要加噪声的参数） |

**需要注意的地方**

1. **没有运行想法生成（纯搜索模式）。** `agents/ai_scientist_v2/agent.py::_generate_ideas` 会先检查 `ml_tasks/<TASK>/theaiscientist/seed_ideas.json`，没有任何任务提供这个文件，所以想法生成被直接跳过。而 `generate_ideas` 本身并不需要这个文件（代码注释写着 seed ideas 早已移除），这很可能是个 bug。
   - 影响：第 1、2 阶段（30% 的步数）的草稿没有绑定研究想法；第 3、4 阶段即使开启想法生成，也不受想法引导。
   - `summary.json` 里的 "Total Ideas: 8" 统计的是起草次数，不是想法数。
2. **`val_ratio` 被改为 0.05。** `split_config.json` 在 `target_files` 里，prompt.json 也明确允许修改它。
   - 改完之后训练集从 35,000 张增加到 37,525 张，val 集从 4,503 张减少到 1,975 张。val 的标准误约为 1 个百分点，所以 val 比 test 高 1.6 个点在预期之内。
   - prompt.json 说 test "完全不受 split_config 影响"，这不准确：test 加载的是 val 阶段训练的模型，而训练集大小正是由 `val_ratio` 决定的。所以 +8.8 点里有一小部分来自多出的约 7% 训练数据。
3. **搜索过程只能看到 val。** test 只在搜索结束后运行一次，结果不会回传给 agent。
   - 但隐藏的 test 集就在工作区的 `./data` 里（它是 CIFAR-10 训练集的一部分，划分种子固定为 42），agent 生成的代码理论上可以读到；沙盒和 code_editor 的危险代码检查都拦不住。
   - 检查了全部 100 步的快照：`algorithm.py` 只导入了 torch、math、opacus，没有读取任何数据，所以没有发生泄漏。
4. **推理代价变大。** 测试时增强让每张图要跑 72 次前向，任务对此没有限制。

### 5.2 冒烟测试：ai_scientist_v2 × Lite 任务 × Opus 5.5（每个任务 3 步）

全部在沙盒中运行，`--workspace-label smoke`，输出到 `benchmark_results_smoke/`。

- **Blackwell**（g7e.12xlarge，7 个任务，超时 2400 秒）：GPU0 跑 pycil、domainbed、privacymeter；GPU1 跑 usb、officehome、openood；ART 只用 CPU。日志和资源数据在 `docs/reproduction_logs/smoke_lite7/`。
- **L4**（8 个任务，超时 7200 秒）：GPU0 单独跑 officehome；GPU1 跑 openood、domainbed；GPU2 跑 privacymeter、opacus；GPU3 跑 pycil、usb；ART 只用 CPU。命令见 6.7，日志和结果在 `docs/reproduction_logs/l4/smoke_lite8/`、`docs/reproduction_logs/l4/benchmark_results_smoke/`。

| 任务 | baseline（val / test） | Blackwell 冒烟（val / test） | L4 冒烟（val / test） | Blackwell 每步耗时（分钟，每卡 3 个任务） | L4 每条命令耗时（秒，3 步 + 最终重训练） | L4 超过 2400 秒的命令数 | 显存峰值（Blackwell / L4） | 内存峰值 PSS（Blackwell / L4） |
|---|---|---|---|---|---|---|---|---|
| opacus | 0.588 / 0.587 | — / 0.631（3 步，单独测试） | 0.629 / 0.618 | — | 427 / 638 / 636 / 473 | 0 | — / 11.9 GB | — / 2.2 GB |
| pycil | 0.593 / 0.594 | 0.613 / 0.623 | 0.610 / 0.611 | 11.4 / 12.5 / 10.9 | 1238 / 1179 / 1140 / 1190 | 0 | 1.1 / 0.7 GB | 2.2 / 1.9 GB |
| usb | 0.083 / 0.086 | 0.070 / 0.079 | 0.109 / 0.107 | 14.0 / 19.3 / 18.1 | 1318 / 1327 / 1784 / 1355 | 0 | 1.6 / 1.4 GB | 1.6 / 1.4 GB |
| domainbed | 0.239 / 0.250 | 0.693 / 0.705 | 0.728 / 0.740 | 6.7 / 1.6 / 6.3 | 799 / 824 / 624 / 727 | 0 | 0.9 / 0.7 GB | 3.0 / 2.8 GB |
| officehome | 0.901 / 0.857 | 0.909 / 0.874 | 0.911 / 0.874 | 24.2 / 24.4 / 约 20 | 3022 / 3011 / 3001 / 2994 | **4** | 10.6 / 9.9 GB | 7.9 / 7.5 GB |
| privacymeter（↓） | 0.328 / 0.319 | 0.037 / 0.039 | 0.057 / 0.064 | 10.6 / 11.5 / 9.4 | 2607 / 1528 / 1520 / 1522 | **1** | 5.4 / 4.8 GB | 3.0 / 2.9 GB |
| ART（CPU） | 0.289 / 0.284 | 0.931 / 0.931 | 0.967 / 0.966 | 5.9 / 6.4 / 5.3 | 373 / 222 / 370 / 224 | 0 | — | 1.5 / 2.9 GB |
| openood | 0.871 / 0.903 | 0.914 / 0.986 | 0.915 / 0.986 | 13.9 / 18.3 / 18.7 | 2497 / 2027 / 1839 / 1841 | **1** | 2.0 / 1.5 GB | 2.5 / 2.3 GB |

（baseline 列是第 3 节的采用值。两次冒烟测试运行时，提示词里的 openood、privacymeter、ART baseline 用的是旧值 0.880、0.324、0.301，与现在的采用值相差不超过 0.012。）

- **整体**：
  - 所有命令的 returncode 都是 0，没有 Bedrock 鉴权错误、限流或拒答。
  - token 消耗：7 个任务约 52 万，8 个任务约 54 万。按官方价格估算约 $5～6。
  - Blackwell 上每卡跑 3 个任务时，GPU 利用率为 90～95%，显存峰值为 7.3 GB 和 14.2 GB。**瓶颈是 GPU 算力，不是显存。**
  - L4 冒烟测试总共用了 3 小时 24 分钟，由 officehome 决定（4 次训练，每次约 50 分钟）；单卡显存峰值不超过 13.3 GB。
  - **按标准协议（2400 秒）在 L4 上会有 6 条命令超时**：officehome 的全部 4 条，以及 privacymeter 和 openood 的第 1 步。officehome 每一步都和 baseline 一样约 50 分钟，所以是 L4 算力不够，不是 agent 让代码变慢了。
- **各任务**：
  - pycil、officehome、openood 接近或略高于 baseline；domainbed、ART、privacymeter 大幅提升；usb 在 Blackwell 上没有超过 baseline，在 L4 上从 0.083 升到 0.109。
  - **privacymeter**：会有某一步因为违反 TPR 约束被 `postprocess.py` 判为无效，这是任务规则在起作用。Blackwell 上最好的一步用的是随机裁剪、翻转和 mixup，准确率从 0.611 升到 0.709。L4 的最终 test 没有违反约束，test 准确率为 0.755。
  - **openood**：test 比 val 高是任务设计如此：val 用 CIFAR-100 当分布外数据，test 用 SVHN。
- **domainbed（需要注意）**：两次的最优方法都是对红绿两个通道做置换对称（训练时随机交换，预测时对两种通道顺序的 logits 取平均）；L4 那次另外加了权重 EMA 和针对 25% 标签噪声的小损失截断。
  - 模型能看到的只有任务说明、两个目标文件和 baseline 数值，看不到 `datasets.py`、数据或 val/test 结果，所以**没有 test 泄漏**。
  - 但任务说明点名了 ColoredMNIST，它的构造是公开的（IRM 论文：颜色是伪特征，标签噪声 25%，准确率上限约 75%），agent 在分析中也直接写出了这些细节。所以这个提升来自题目先验，不能代表通用的域泛化能力。
- **ART（需要注意）**：
  - Blackwell 那次：把图像外框的 4 个像素清空（后门触发器都放在角落和边缘），剔除低置信度样本，还改写了分类器的 `predict`，让评测输入也先清空外框。没有读取 test 数据，但利用了题目先验，而且绕过了评测接口。
  - L4 那次：只改了 `dp_instahide_trainer.py`。先预训练 2 个 epoch，然后在倒数第二层特征上按类别做谱签名（spectral signatures）和 2 簇激活聚类（activation clustering），按标签一致性过滤可疑样本，再把权重重置为初始值，在过滤后的数据上做 DP-InstaHide 训练。这是已发表的通用后门防御方法，不依赖触发器的位置。
  - 每次只有 3 步，样本太少，不能据此判断 agent 的倾向。

---

## 6. 在新机器上复现

### 6.1 前提条件

| 项目 | 要求 |
|---|---|
| 系统 | Linux x86_64，内核允许非特权用户命名空间（bubblewrap 需要） |
| GPU | NVIDIA，驱动支持 CUDA 12.8（torch 2.7.1+cu128 带 sm_75～sm_120 的 kernel，sm_89 和 sm_120 实测可用）。冒烟测试每卡峰值约 15 GB。**GPU 比 Blackwell 慢很多时要加大超时**：L4 上 officehome 的 baseline 单步就要 50 分钟（第 8 节） |
| 命令 | `git`、`wget`、`unzip`、`curl`、`uv`、`bwrap`、`nvidia-smi`、`python3`。Amazon Linux 2023 用 `sudo dnf install bubblewrap` 安装 bwrap。Lite 的 8 个任务不需要 conda 和 pip |
| AWS | 能在 `AWS_REGION` 指定的区域（默认 us-west-2）用 Bedrock 调用 `us.anthropic.claude-opus-5-5`，需要 InvokeModel 和 InvokeModelWithResponseStream 权限 |
| AWS 凭证 | EC2 实例角色（沙盒共享网络，能访问 IMDS），或 `AWS_PROFILE`（沙盒会只读挂载 `~/.aws`），或 `AWS_ACCESS_KEY_ID` 等环境变量（沙盒会传入） |
| 网络 | 数据从 Kaggle 镜像下载（见 2.2），多伦多和斯坦福的官方源太慢或连不上 |
| 费用和时间 | 冒烟测试约 50 万 token，按官方价格估算约 $5，以 Bedrock 账单为准；Blackwell 上约 1.5 小时 |

### 6.2 获取代码

```bash
git clone <仓库地址> FML-bench && cd FML-bench
git checkout repro/bedrock-uv-sandbox
```

### 6.3 构建环境

```bash
export UV_PYTHON_PREFERENCE=only-managed   # 只用 uv 自己管理的 Python
# export FMLBENCH_UV_ENVS=/path/to/envs    # 可选，默认是 ~/fml_envs
python3 setup.py --skip-data \
  --task Privacy_opacus --task Continual_Learning_pycil --task Data_Efficiency_usb \
  --task Generalization_domainbed --task Generalization_domainbed_officehome \
  --task Privacy_privacymeter --task Robustness_and_Reliability_art --task Robustness_openood
```

在 g7e.24xlarge 上约 6 分钟。环境建在本地盘上即可。

### 6.4 准备数据

1. 按 2.2 的来源把数据下载到缓存目录（例如 `/mnt/nvme/data/`）。压缩包要放在单独的下载目录里，解压出缓存布局，然后对照 2.2 校验 md5/sha256：

   ```bash
   C=/mnt/nvme/data; D=$C/_downloads; mkdir -p $D $C/cifar10 $C/cifar100 $C/svhn $C/officehome $C/art
   curl -sL -o $D/cifar10.zip  https://www.kaggle.com/api/v1/datasets/download/pankrzysiu/cifar10-python
   curl -sL -o $D/cifar100.zip https://www.kaggle.com/api/v1/datasets/download/pypiahmad/cifar-100
   curl -sL -o $D/svhn.zip     https://www.kaggle.com/api/v1/datasets/download/khanhvudo72/svhncropped
   (cd $D && uvx gdown 1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa -O office_home.zip)
   curl -sfL -o $C/art/mnist.npz https://s3.amazonaws.com/img-datasets/mnist.npz
   unzip -q $D/cifar10.zip -d $C/cifar10
   unzip -q $D/cifar100.zip -d $C/cifar100
   unzip -q $D/svhn.zip test_32x32.mat -d $C/svhn
   unzip -q $D/office_home.zip -d $C/officehome && mv $C/officehome/office_home_dg $C/officehome/office_home
   ```

2. 让 setup 补全缓存里缺的条目（MNIST、privacymeter 的 pkl），建好软链并提交到任务仓库：

   ```bash
   python3 setup.py --skip-envs --data-cache /mnt/nvme/data --task ...   # 8 个任务，和 6.3 相同
   chmod -R a-w /mnt/nvme/data/{cifar10,cifar100,svhn,mnist,officehome,privacymeter,art}
   ```

3. 确认 8 个任务的代码库都是干净的（下面的命令不应该有任何输出）：

   ```bash
   for c in ml_tasks/*/config.json; do
     r=$(python3 -c "import json;print(json.load(open('$c'))['repo_dir'])")
     [ -d "$r" ] && git -C "$r" status --porcelain
   done
   ```

不设 `--data-cache` 时，setup 会像上游一样把数据直接下载到各任务的工作区（走官方源），也会提交。

### 6.5 Baseline

- **硬件和第 1 节的 Blackwell 相同**：直接用分支里提交的 baseline。
- **硬件不同**：先重测，用结果覆盖 `ml_tasks/<TASK>/baseline_results/{val,test}_info.json`，agent 提示词里的 baseline 和最终比较都读这两个文件。

  ```bash
  scripts/reproduction/rebaseline_queue.sh <日志目录> <TAG>         # 全部任务，每卡一个串行队列，ART 最后单独跑
  scripts/reproduction/rebaseline_task.sh <TASK> <GPU编号，CPU 填 ""> <TAG>   # 单个任务
  python3 scripts/reproduction/compare_baseline.py <TASK> rebaseline_results/<TASK><TAG>
  ```

  `rebaseline_queue.sh` 的分配可以用 `QUEUES`（`"GPU:任务,任务 ..."`）和 `CPU_TASKS` 修改。结束时会打印耗时和指标对比，`nvidia-smi dmon` 的采样记录写到日志目录。结果写到 `rebaseline_results/<TASK><TAG>/`，所以 TAG 不要和已有的目录重名。

### 6.6 检查沙盒

```bash
sandbox/run_in_bwrap.sh --shell -c 'for e in opacus pycil usb domainbed privacy_meter openood art; do
  ~/fml_envs/$e/bin/python -c "import sys; print(\"$e\", sys.version.split()[0])"; done'
sandbox/run_in_bwrap.sh --shell -c '~/fml_envs/opacus/bin/python -c "import torch; print(torch.cuda.device_count())"'
sandbox/run_in_bwrap.sh --shell -c 'ls workspace/Robustness_openood/OpenOOD/data/cifar-100-python'
```

每个环境都应该打印出 Python 版本，GPU 数量要正确，数据要能通过软链读到。要检查 Bedrock，可以在沙盒里用 `agents.llm.create_client("us.anthropic.claude-opus-5-5", "AnthropicBedrock")` 发一条短消息。

### 6.7 运行冒烟测试

```bash
GPU_A=0 GPU_B=1 scripts/reproduction/run_smoke_lite.sh docs/reproduction_logs/smoke_lite7_<机器名>
```

- 脚本在前台运行，会同时启动资源监控，并行运行 7 个任务，每个跑 3 步。默认分配：GPU_A 跑 pycil、domainbed、privacymeter；GPU_B 跑 usb、officehome、openood；ART 只用 CPU。只有 1 张卡时设 `GPU_A=0 GPU_B=0`。
- 全部结束后会打印每个任务的命令数、失败数、最佳 val 和 test，以及显存和内存峰值。
- 可以调的环境变量：
  - `STEPS`：默认 3；
  - `OUTPUT_DIR`：默认 `benchmark_results_smoke`；
  - `SMOKE_TASKS`：用空格分隔的 `任务配置|CUDA_VISIBLE_DEVICES` 列表，替换默认分配，可以加上 `privacy_opacus`；
  - `EXTRA_OVERRIDES`：传给 run_agent_benchmark.py 的额外覆盖项。
- L4 上的用法（5.2）：
  ```bash
  SMOKE_TASKS="generalization_officehome|0 robustness_openood|1 generalization|1 privacy_privacymeter|2 privacy_opacus|2 continual_learning_pycil|3 data_efficiency_usb|3 robustness_and_reliability_art|" \
  EXTRA_OVERRIDES="agent.ai_scientist_v2.execute_timeout=7200" \
    scripts/reproduction/run_smoke_lite.sh docs/reproduction_logs/l4/smoke_lite8
  ```
- 运行时间较长，建议用 `setsid nohup ... &` 放到后台，或者在 tmux 里运行。

### 6.8 怎么判断是否成功

- 每个任务都跑完 3 步（日志里有 `Best Val Metric` 和 `Test Metric`），并且清理了工作区副本（`Cleaned up workspace copy`）。
- 日志里没有 `No such file or directory: 'conda'`，没有 `Timeout after`，也没有 Bedrock 鉴权错误。
- 指标不需要和 5.2 完全一致（agent 的输出是随机的），但量级应该相近：pycil、officehome 接近或略高于 baseline，domainbed 和 ART 很可能大幅提升，usb 不一定提升。
- privacymeter 有某一步因为 TPR 约束被判无效是正常的。

### 6.9 常见问题

| 现象 | 原因与处理 |
|---|---|
| GPU 较慢的机器上 officehome / openood / privacymeter 超时 | 默认的 2400 秒是按 Blackwell 的速度定的，L4 上要用 `execute_timeout=7200`（第 8 节） |
| 沙盒里找不到数据（软链失效） | 缓存不在沙盒可见的路径上。沙盒会只读挂载任务模板里软链的目标；其他情况要设 `FMLBENCH_DATA_CACHE` |
| ART 用到了 GPU | 要传 `CUDA_VISIBLE_DEVICES=""` |
| 环境在沙盒里报 `No such file or directory: 'conda'` | 环境的基础解释器在沙盒里看不到，`env_command` 退回了 conda。建环境时设 `UV_PYTHON_PREFERENCE=only-managed`；沙盒也会挂载 `pyvenv.cfg` 里 `home` 指向的前缀 |
| 用 `pgrep -f` / `pkill -f <关键字>` 时匹配到（或杀掉了）执行命令的 shell | 关键字出现在这条命令自己的命令行里。改用 `pgrep -x python` 再查看 `/proc/<pid>/cmdline`，或者直接用 PID 去 `kill` |

---

## 7. 待办 / 未决事项

- [ ] 正式测试（ai_scientist_v2 × 7 个 Lite 任务 × 100 步，g7e.12xlarge）：
  - 用 `scripts/reproduction/run_lite_queue.sh` 运行，`execute_timeout=4800`（每卡跑 2 个任务，所以把标准协议的超时加倍）；日志在 `docs/reproduction_logs/full_lite7/`。
  - 结束后：补充结果，按 `Command finished in X s` 统计超过 2400 秒的步数，检查代码快照里有没有读取数据。
  - 这次运行的提示词里，openood、privacymeter、ART 的 baseline 用的是旧值（见 5.2 的表注）。
- [ ] 是否修复 ai_scientist_v2 跳过想法生成的问题（5.1），并用完整流程重跑 opacus
- [ ] 用 `val_ratio=0.05` 重测 opacus 的 baseline，把训练数据增加带来的提升和算法改进分开
- [ ] 堵住隐藏 test 集可以被读取的漏洞：把数据移到 val 阶段访问不到的位置，或者在检查里拦截数据读取
- [ ] domainbed（ColoredMNIST）和 ART 靠题目先验大幅提升：是否通过修改任务说明来限制（这属于修改 benchmark，需要决定）
- [ ] openood 的训练不确定：是否在 `train_eval_baseline.py` 里设置 `cudnn.deterministic`（会改动任务代码）
- [ ] ART：是否构建能用 Blackwell GPU 的 TF 环境（会改动锁定的版本）；同型号 CPU 之间约 0.01 的差距原因不明
- [ ] 超时协议：默认的 2400 秒在 L4 上不够用（officehome 连 baseline 都跑不完）。跨硬件比较时，是按 GPU 速度换算超时，还是统一要求某个最低算力，需要决定
- [ ] 迁移并测试其余 10 个非 Lite 任务的环境
- [ ] 在缓存为空的机器上验证 `setup.py --data-cache` 的自动下载（目前只有 MNIST 实际走过这条路径）
- [ ] 清理 `workspace/*__rebaseline_*` 副本（共约 2.2 GB）和 `/mnt/nvme/data/_downloads/`（2.1 GB）

---

## 8. 其他硬件：4 × L4

### 8.1 Baseline

用同一套环境（包版本相同），两个任务共用一张卡并行重测；之后又让每个任务单独占一张 L4 重跑 val，并用 `nvidia-smi dmon` 每 5 秒采样一次。

| 任务 | 指标（方向） | val：原始 / 采用 / **L4** | test：原始 / 采用 / **L4** | L4 单卡耗时（val） | 单卡耗时 / Blackwell | L4 单卡：SM 利用率 / 显存带宽 / 功耗 / SM 时钟 | 瓶颈 |
|---|---|---|---|---|---|---|---|
| Privacy_opacus | test_acc（↑） | 0.5894 / 0.5881 / **0.5878** | 0.5877 / 0.5870 / **0.5854** | 2m31s | 1.9 倍 | 41% / 44% / 47 W / 1903 MHz | 混合：每样本梯度，加上 CPU 开销（2 个 worker、隐私预算计算） |
| Continual_Learning_pycil | avg_incremental_acc（↑） | 0.5911 / 0.5925 / **0.6075** | 0.5952 / 0.5944 / **0.6034** | 12m19s | 1.4 倍 | 50% / 25% / 57 W / 1819 MHz | CPU、数据加载和 Python 开销（ResNet-32 只有 0.46M 参数，ColorJitter 在 CPU 上做） |
| Privacy_privacymeter | AUC_gap（↓） | 0.3339 / 0.3278 / **0.3282** | 0.3214 / 0.3187 / **0.3178** | 25m21s | 4.1 倍 | 96% / 78% / 70.6 W（97% 的时间碰到功耗上限）/ 1182 MHz | GPU 算力（4 个 WRN-28-2，每个 epoch 做一次完整评测） |
| Robustness_openood | auroc（↑） | 0.8758 / 0.8708 / **0.8697** | 0.9302 / 0.9028 / **0.9106** | 30m55s | 4.5 倍 | 97% / 76% / 70.8 W（95%）/ 1325 MHz | GPU 算力（ResNet-18，32×32，第一层不降采样，100 epoch） |
| Data_Efficiency_usb | test_acc（↑） | 0.0757 / 0.0827 / **0.0723** | 0.0790 / 0.0861 / **0.0829** | 14m28s | 1.6 倍 | 40% / 27% / 55 W / 1831 MHz | CPU 和数据加载（2 个 worker；有标注的数据只有 200 张，每 3 步重建一次 worker，全程约 5000 次；RandAugment 在 CPU 上做） |
| Generalization_domainbed_officehome | avg_acc（↑） | 0.8832 / 0.9011 / **0.8942** | 0.8629 / 0.8570 / **0.8599** | 50m19s | 4.7 倍 | 抽查一次为 94% | GPU 算力（ResNet-50，224×224，fp32/TF32） |
| Generalization_domainbed | in_acc（↑） | 0.2792 / 0.2393 / **0.2626** | 0.2868 / 0.2501 / **0.2706** | 1m49s | 1.0 倍 | 46% / 34% / 63 W / 1934 MHz | CPU 和 Python（MNIST 小 CNN，每 100 步对 7 万张图做一次评测，8 个 loader × 8 个 worker） |
| Robustness_and_Reliability_art（CPU） | defense_score（↑） | 0.4720 / 0.2893 / **0.2913** | 0.4774 / 0.2838 / **0.2895** | 5m46s（AMD EPYC 7R13，与 GPU 任务同时运行） | 1.4 倍 | — | CPU |

（"采用"是第 3 节的值；"单卡耗时 / Blackwell"是和第 3 节单卡耗时的比值。）

- 8 个任务与采用值相比都一致或接近，`ml_tasks/*/baseline_results/` 用的是 Blackwell 的值。如果要以 L4 作为正式评测平台，需要另外决定是否换成 L4 的值。
- 同一张 L4 上重跑：除 privacymeter（`cudnn.benchmark`，0.3287 对 0.3282）以外，val 结果都和第一次逐字节相同。
- **速度差距的原因**：
  - 受 GPU 算力限制的任务（officehome、openood、privacymeter）慢 4～4.7 倍。L4 的 fp32/TF32 算力和显存带宽本来就比 RTX PRO 6000 低好几倍，而且功耗上限只有 72 W：这几个任务 95% 以上的时间都碰到功耗上限（`SW Power Cap` 处于触发状态），SM 时钟从 2040 MHz 降到 1180～1330 MHz。
  - 受 CPU、数据加载或 Python 开销限制的任务（pycil、usb、domainbed）在 L4 上 GPU 只用到 40～50%，所以只慢 1.0～1.6 倍。
  - 多个任务共用一张 L4 时会明显变慢：privacymeter 和 pycil 共用时要 40m39s（单独时 25m21s）。
- **对 agent 测试的影响**：超时风险集中在 officehome、privacymeter、openood 上。在 L4 上，这三个任务最好每个单独占一张卡，并使用 `execute_timeout=7200`。

结果目录：`docs/reproduction_logs/l4/rebaseline_results/<TASK>[__solo]/`。dmon 采样记录在 `docs/reproduction_logs/l4/solo/`，setup 日志为 `docs/reproduction_logs/l4/setup_*.log`。

### 8.2 冒烟测试

结果见 5.2 的表格。

---

## 附：相关文件

和上游一样，运行产物不纳入版本库：`rebaseline_results/`、`benchmark_results/`、`benchmark_results_smoke/`、`docs/reproduction_logs/` 都在 `.gitignore` 里，只存在于产生它们的机器上。版本库里只有 `ml_tasks/<TASK>/baseline_results/` 下的 baseline 文件。

| 路径 | 说明 |
|---|---|
| `ml_tasks/<TASK>/baseline_results/{val,test}_info.json` | 采用的 baseline（第 3 节） |
| `ml_tasks/<TASK>/baseline_results/original/` | 仓库原有的 baseline 文件 |
| `rebaseline_results/<TASK>__g7e24/` | 采用值的原始输出（val/test 结果、日志、`run_info.txt`、`pip_freeze.txt`） |
| `rebaseline_results/Robustness_openood__g7e24_rerun2/`、`rebaseline_results/Robustness_and_Reliability_art__g7e24_48cpu/` | openood 同卡重跑、ART 限制为 48 个 CPU 的对照 |
| `rebaseline_results/<TASK>[__rerun2]/` | g7e.12xlarge 上的重测输出（第 3 节确定性对比用） |
| `docs/reproduction_logs/g7e24/` | 采用值那次的 setup 日志，以及 `rebaseline/` 下的队列记录、`summary.txt`、`dmon.txt` |
| `scripts/reproduction/rebaseline_queue.sh` | 每张 GPU 一个串行队列地重测全部 baseline，ART 最后单独跑，并记录 `nvidia-smi dmon` |
| `scripts/reproduction/rebaseline_task.sh` | 单任务 baseline 重测（val → test，uv 环境） |
| `scripts/reproduction/compare_baseline.py` | 输出原始值、采用值和重测值的对比 |
| `sandbox/run_in_bwrap.sh` | 沙盒入口 |
| `scripts/reproduction/resource_monitor.py` | 按工作区副本统计各任务的显存和内存（PSS）；`--summary` 输出峰值 |
| `scripts/reproduction/run_smoke_lite.sh` | Lite 任务的冒烟测试（6.7） |
| `scripts/reproduction/run_lite_queue.sh` | 7 个 Lite 任务的正式测试，按位置排队 |
| `benchmark_results_smoke/` | 冒烟测试结果 |
| `docs/reproduction_logs/` | agent 运行日志；`smoke_lite7/`、`full_lite7/`、`l4/` 下有资源采样数据 |
