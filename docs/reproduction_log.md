# FML-bench 复现实验记录

本文档记录 FML-bench 在 uv 环境、Amazon Bedrock（Opus 5.5）和 bubblewrap 沙盒下的复现：环境与数据、baseline 重测、harness 改造、agent 测试结果与快照检查（第 5 章，含用 Claude Code 作为 agent 的对比），以及从零复现这些结果的完整流程（第 6 章）。

- 分支：`repro/bedrock-uv-sandbox`，基于上游 `d336651`
- 记录时间：2026-11-02 ～ 2026-11-11
- baseline 有两版（第 3 章）：g7e.12xlarge 版是第 5 章 agent 测试用的；g7e.24xlarge 版是仓库当前采用的。第 8 章是 4 × L4 上的对照

---

## 1. 机器环境

| 用途 | 机器 | GPU | 驱动 / CUDA | CPU / 内存 | 系统 | uv / bwrap |
|---|---|---|---|---|---|---|
| **Agent 测试（第 5 章）**、baseline 的 g7e.12xlarge 版 | AWS `g7e.12xlarge`（us-west-2） | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB，sm_120） | 580.126.16 / 13.0 | Intel Xeon Platinum 8559C，48 核 / 499 GB | Amazon Linux 2023，kernel 6.1 | 0.11.14 / 0.10.0 |
| **baseline 的 g7e.24xlarge 版（仓库采用）**、数据缓存（2.2）、Claude Code 测试（5.4） | AWS `g7e.24xlarge`（us-east-1） | 4 × 同型号 GPU | 595.112.07 / 13.0 | Intel Xeon Platinum 8559C，96 核 / 999 GB | Amazon Linux 2023，kernel 6.12 | 0.13.7 / 0.12.0 |
| 其他硬件对照（第 8 章） | 4 × L4 | 4 × NVIDIA L4（24 GB，sm_89） | 595.71.05 / 13.2 | AMD EPYC 7R13，96 核 / 363 GB | Amazon Linux 2023，kernel 6.12 | 0.12.10 / 0.10.0 |

- 不使用 conda。LLM 是 Amazon Bedrock 的 `us.anthropic.claude-opus-5-5`，凭证来自实例角色。
- **硬件约束**：Blackwell（sm_120）需要 torch ≥ 2.7 和 CUDA 12.8。仓库原配置的 torch 2.5.1+cu121、1.12.1+cu113 等版本在这块卡上没有可用的 kernel，因此统一换成 `torch==2.7.1+cu128`（torchvision 0.22.1、torchaudio 2.7.1）。TF 2.10 不能使用这块 GPU，所以 ART 任务只在 CPU 上运行。

---

## 2. 环境构建（uv）与数据

### 2.1 环境

- 环境建在 `~/fml_envs/<env>`，可以用 `FMLBENCH_UV_ENVS` 改位置。解释器由 uv 管理（`UV_PYTHON_PREFERENCE=only-managed`），位于 `~/.local/share/uv/python`。
- `setup.py` 用 `uv_create` / `uv_run` 建环境，conda 和 pip 只在要建 conda 环境时才需要。

| 环境 | Python | 相对原配置的改动 |
|---|---|---|
| fmlbench（harness） | 3.11.6 | 去掉 conda 的 nodejs（harness 用不到）；`anthropic` 改为 `anthropic[bedrock]==0.84.0`（多装 boto3）；锁定 `pyarrow<26`（打分时 scikit-learn 要用，见 6.9） |
| opacus / pycil / usb | 3.10.22 | torch 2.5.1+cu121 → 2.7.1+cu128 |
| openood | 3.9.25 | torch 2.5.1+cu121 → 2.7.1+cu128 |
| domainbed | 3.10.22 | conda 的 pytorch 1.12.1 → pip 的 torch 2.7.1+cu128；去掉 mkl 系列包；`backpack-for-pytorch` 1.3.0 → 1.7.1（1.3.0 要求 torch<2.0）；锁定 `huggingface_hub<1.0`（timm 0.9.16 用到的接口在 hub 1.x 中已移除） |
| privacy_meter | 3.12.16 | 安装 requirements.txt 时滤掉 torch/nvidia 相关行，torch 2.7.1+cu128 单独安装 |
| art | 3.10.22 | 版本不变；TF 2.10.1 只能用 CPU |

- 6 个 torch 环境都能在 sm_120 和 sm_89 上做 GPU 计算。
- 三台机器上的包版本只有 pip 自身和 art 的 `google-auth` 有补丁版本差异。g7e.12xlarge 的 domainbed 环境里还多出几个降级 hub 时留下的包：httpx2、httpcore2、tomli、truststore。
- Python 的补丁版本随 uv 版本略有不同（上表是 g7e.24xlarge 的）。g7e.12xlarge 建环境时没有设 `UV_PYTHON_PREFERENCE=only-managed`，privacy_meter 用的是 miniconda 的 Python 3.12.9（见 4.3）。
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

- **下载源**：多伦多的 CIFAR 官方源只有 35～50 KB/s，斯坦福的 SVHN 官方源连不上。Kaggle 镜像能达到 90～145 MB/s，而且不需要账号。
- **缓存条目缺失时**，setup 会自动下载到缓存（用的是 torchvision、gdown 或 curl）。也可以先按上表手动放好，setup 会直接使用。
- **提交到任务仓库**：`setup.py` 无论数据是下载的、手动放的还是软链，都会执行 `git add -f` 并提交。软链以 symlink 的形式存进 git（mode `120000`），所以 executor 每步执行 `git reset --hard && git clean -fd` 之后数据仍然在，`git status` 也是干净的。重复运行 setup 只会打印 `already links` / `already committed`。domainbed 和 officehome 的数据在任务仓库外面（模板目录 `data/`），不需要提交。
- **工作区副本**：runner 复制工作区时保留软链（`shutil.copytree(..., symlinks=True)`），数据不会被复制。
- **沙盒**：会把 `FMLBENCH_DATA_CACHE`，以及任务模板里所有绝对路径软链的目标，**只读**挂载进去，agent 生成的代码改不了共享数据。沙盒里的 `$HOME` 是空的，所以 ART 的 `mnist.npz` 会在每次运行时从 S3 下载（11 MB）。
- **权限**：缓存设为 `chmod -R a-w`。所有 baseline 代码都只读数据（torchvision 的 `download=True` 校验通过后不会写文件）。
- officehome 第一次运行时会把 ResNet-50 预训练权重下载到 `~/.cache`（沙盒里这个目录可写）。
- `/mnt/nvme` 是实例存储，实例停止后数据会丢失。在同样的路径下重新放好数据，或者重新运行一次带 `--data-cache` 的 setup 即可恢复，工作区里的软链和提交不受影响。
- **g7e.12xlarge（第 5 章）没有用缓存**：数据从上表同样的镜像下载到 `~/fml_data_cache/`，以实体文件复制进各工作区，再提交到任务仓库（pycil 和 usb 是手动用 `git add -f` 提交的）。两种放法对任务代码没有区别。

---

## 3. Baseline（FML-bench-Lite，8 个任务）

baseline 在两台同型号 Blackwell 机器上各测了一次。g7e.12xlarge 版先测，第 5 章的 agent 测试用的就是这一版；之后在 g7e.24xlarge 上让每个任务单独占一张卡重测，仓库现在采用这一版。

### 3.1 方法

- 按各任务 `config.json` 里的 `val_command` → `test_command` 顺序执行，两者用同一份工作区副本（test 复用 val 保存的模型；pycil 的 test 会重新训练），在对应的 uv 环境里运行。
- **g7e.12xlarge 版**：用 `scripts/reproduction/rebaseline_task.sh` 逐个任务手动启动，有的任务同时运行（openood 和 usb 共用一张卡；pycil、domainbed 和 ART 有一段时间同时运行）。结果在该机的 `rebaseline_results/<TASK>/`，工作区副本保留为 `workspace/<TASK>__rebaseline_<时间戳>`。
- **g7e.24xlarge 版**：用 `scripts/reproduction/rebaseline_queue.sh`，每张 GPU 一个串行队列，**同一张卡上同时只有一个任务**。ART（只用 CPU）等所有 GPU 任务结束后单独运行，因为 TF 会占满所有 CPU。同时用 `nvidia-smi dmon` 每 5 秒采样一次。分配：GPU0 跑 pycil；GPU1 跑 officehome → opacus；GPU2 跑 usb → domainbed；GPU3 跑 openood → privacymeter；最后在 CPU 上跑 ART。全部共 21.5 分钟。结果在该机的 `rebaseline_results/<TASK>__g7e24/`。

```bash
scripts/reproduction/rebaseline_task.sh <TASK> <GPU编号，CPU 填 ""> [TAG]        # g7e.12xlarge 版
scripts/reproduction/rebaseline_queue.sh docs/reproduction_logs/g7e24/rebaseline __g7e24   # g7e.24xlarge 版
```

### 3.2 结果

| 任务 | 指标（方向） | val：原始 / 12xlarge / **24xlarge** | test：原始 / 12xlarge / **24xlarge** | 耗时 val / test（12xlarge；24xlarge 单卡） | 24xlarge GPU 平均 SM / 显存带宽 / 功耗 | 与原始值相比 |
|---|---|---|---|---|---|---|
| Privacy_opacus | test_acc_mean（↑） | 0.5894 / 0.5881 / **0.5881** | 0.5877 / 0.5870 / **0.5870** | 1m10s / 5s；1m21s / 5s | 11% / 7% / 107 W | 一致 |
| Continual_Learning_pycil | avg_incremental_acc_mean（↑） | 0.5911 / 0.5925 / **0.5925** | 0.5952 / 0.5944 / **0.5944** | 8m54s / 8m09s；8m48s / 8m31s | 42% / 6% / 190 W | 一致 |
| Privacy_privacymeter | AUC_gap_mean（↓） | 0.3339 / 0.3242 / **0.3278** | 0.3214 / 0.3182 / **0.3187** | 6m15s / 1s；6m08s / <1s | 70% / 32% / 325 W | 一致 |
| Robustness_openood | auroc_mean（↑） | 0.8758 / 0.8804 / **0.8708** | 0.9302 / 0.9212 / **0.9028** | 10m25s / 8s；6m56s / 7s | 86% / 38% / 374 W | 接近（任务本身不确定，见 3.3） |
| Data_Efficiency_usb | test_acc_mean（↑） | 0.0757 / 0.0827 / **0.0827** | 0.0790 / 0.0861 / **0.0861** | 11m11s / 5s；9m02s / 5s | 33% / 9% / 173 W | 接近（原始值本身只有约 8%） |
| Generalization_domainbed_officehome | avg_acc_mean（↑） | 0.8832 / 0.9011 / **0.9011** | 0.8629 / 0.8570 / **0.8570** | 10m55s / <1s；10m38s / <1s | 74% / 55% / 318 W | 接近（相差 2 个点以内） |
| Generalization_domainbed（ColoredMNIST） | in_acc_mean（↑） | 0.2792 / 0.2393 / **0.2393** | 0.2868 / 0.2501 / **0.2501** | 3m32s / <1s；1m45s / <1s | 25% / 2% / 158 W | **偏低约 0.04** |
| Robustness_and_Reliability_art（CPU） | defense_score_mean（↑） | 0.4720 / 0.3005 / **0.2893** | 0.4774 / 0.2921 / **0.2838** | 4m51s / 7s；4m06s / 5s | — | **明显偏低** |

- g7e.24xlarge 的耗时是单独占一张卡时测的（4 个 GPU 任务仍然共用 96 核的 CPU），包含 officehome 第一次运行时下载预训练权重的几秒。g7e.12xlarge 的耗时里有并发的影响。
- 按 GPU 利用率，任务分两类：
  - 受 GPU 算力限制：officehome、openood、privacymeter；
  - 受 CPU、数据加载或 Python 开销限制：pycil、usb、domainbed、opacus。
- domainbed 对 CPU 竞争很敏感，和占满 CPU 的 ART 同时运行时，耗时会翻倍（g7e.12xlarge 的 3m32s 就是这样来的）。

### 3.3 两版的区别

- **条件**：GPU 和 CPU 型号相同，环境的包版本相同（2.1）。不同的是驱动（580 对 595）、kernel（6.1 对 6.12）、CPU 核数（48 对 96），以及测量方式（g7e.12xlarge 上有任务同时运行，g7e.24xlarge 上每卡只有一个任务）。
- **结果**：
  - pycil、usb、officehome、domainbed 的 val/test 结果文件逐字节相同；opacus 只有 ε 在第 15 位有效数字上不同。
  - 只有 3 个任务的指标不同，差距都不超过 0.02，原因都是任务本身的不确定性：

  | 任务 | val：12xlarge → 24xlarge | test：12xlarge → 24xlarge | 原因 |
  |---|---|---|---|
  | openood | 0.8804 → 0.8708 | 0.9212 → 0.9028 | 训练脚本只设了随机种子，没有设 `cudnn.deterministic`。g7e.24xlarge 同一张卡单独重跑一次，得到 val 0.8764、test 0.9099。val 的噪声范围约为 0.871～0.880，test 约为 0.903～0.921，两版都在范围内 |
  | privacymeter | 0.3242 → 0.3278 | 0.3182 → 0.3187 | 开了 `cudnn.benchmark`，卷积算法的选择不固定，多次运行会相差约 0.004 |
  | ART | 0.3005 → 0.2893 | 0.2921 → 0.2838 | 同型号 CPU 上相差约 0.01。在 g7e.24xlarge 上把 CPU 限制为 48 个（`taskset -c 0-47`）后结果逐字节不变，所以**不是线程数造成的**，原因不明。同一台机器上重跑结果完全相同 |

- **对 NI 的影响**：只有这 3 个任务的 NI 会变。ai_scientist_v2 的 Lite 平均 NI 用 g7e.12xlarge 版是 0.2812，用 g7e.24xlarge 版是 0.2846（5.1）。

### 3.4 与原始值的偏差

- **domainbed** 对 GPU 架构敏感。
  - 在两台 Blackwell 上都是 0.2393，L4 上是 0.2626（同样是 torch 2.7.1），原始值是 0.2792（torch 1.12.1）。每台机器上的结果都是确定性的（g7e.12xlarge 上重跑一次完全相同，见 `rebaseline_results/<TASK>__rerun2/`）。
  - ColoredMNIST 留出的 env2 是对抗性的域，训练过程中它的精度在约 0.10～0.30 之间跳动，对数值上的微小差异非常敏感。
- **ART**：DP-InstaHide 训练 5 个 epoch，种子全部固定。
  - 与原始值的差距集中在部分后门上。攻击成功率（poison_acc）：random_corner 0.84 对原始 0.30，edge 0.90 对 0.75。clean_acc 反而更高（0.955 对 0.944）。
  - 最可能的原因是原 baseline 在 GPU 上训练，或者出自另一个版本的代码。原始文件之间本身也不一致：`final_info.json` 的 num_samples 是 7500，`val_info.json` 是 8404，另一个旧版 `final_info.old.json` 的 defense_score 是 0.6966。
  - AMD CPU 的 L4 机器上是 0.2913，和两版 Blackwell 都接近，离原始值都很远。
- **officehome** 第一次在 g7e.12xlarge 上运行时 10 秒内就失败了（timm 依赖的 `huggingface_hub` 被装成了 2.x），锁定 `<1.0` 后重跑成功。

### 3.5 采用

| 文件 | 内容 | 谁在用 |
|---|---|---|
| `ml_tasks/<TASK>/baseline_results/{val,test}_info.json` | **g7e.24xlarge 版** | runner 读 `val_info.json` 作为 agent 提示词里的 baseline，`compute_agent_metrics.py` 读 `val_info.json` 和 `test_info.json`。之后新启动的测试都用这一版 |
| `ml_tasks/<TASK>/baseline_results/g7e12xlarge/` | g7e.12xlarge 版 | 第 5 章两次正式测试都在 g7e.12xlarge 的工作目录里运行，提示词里用的都是这一版。5.1 表中的 baseline 列和 NI 主列也用这一版 |
| `ml_tasks/<TASK>/baseline_results/original/` | 仓库原有的文件（包括 `final_info.json`） | 5.1 表的"原始 baseline"列 |

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

**Bedrock 上的 OpenAI 模型（GPT-6 Astra，5.5）**

- 用法：`--provider OpenAIBedrock --model us.openai.gpt-6-astra`。区域和凭证的取法同上，不需要 API key。
- 2026-11-10 在 us-west-2 和 us-east-1 能查到的 OpenAI 模型：
  - GPT-6.1 Sol/Terra、GPT-6 Sol/Luna/Astra、GPT-5.6 Sol/Terra/Luna、GPT 5.5、GPT 5.4：只能通过 `us.` / `global.` 推理配置文件调用。
  - gpt-oss-120b/20b、gpt-oss-safeguard-120b/20b：可以直接按需调用。
- 实现：通过 boto3 调用 Bedrock 的 Converse API，改动的位置和 Claude 分支相同。
  - `agents/llm.py`：新增 `OpenAIBedrock` provider、`BedrockConverseClient`、`bedrock_converse_create` / `bedrock_response_text` / `is_bedrock_converse_client`。
    - `get_response_from_llm` 遇到这个客户端时，优先走 Converse 分支，不再按模型名分派。
    - 多轮对话的历史只保留文本，推理内容不回传。
    - `get_batch_responses_from_llm` 里按名字匹配 `gpt` 的 `chat.completions` 分支会跳过这个客户端。
  - `agents/code_editor.py`：遇到这个客户端时走 Converse。
  - `extract_usage_info`：读取 Converse 的 `usage`，`outputTokens` 里包含推理 token。
- 请求参数：
  - `maxTokens=64000`，和 Claude 相同（模型上限是 131072）。
  - `additionalModelRequestFields={"reasoning": {"effort": "high"}}`，和 Claude 的 effort 对应，可以用 `FMLBENCH_BEDROCK_EFFORT` 调整。可选值有 none、low、medium、high、xhigh、max，传错值时 Bedrock 会报错，可见这个参数确实生效了。
  - 不传 temperature。
  - boto3 设 `read_timeout=3600`，自适应重试 8 次。
  - **每次调用都新建 botocore 客户端，不复用连接池里的连接**（原因见下面的"空闲连接问题"）。
  - 被内容过滤拦截（`guardrail_intervened` / `content_filtered`）时直接抛错，输出被截断时打印警告。
- 沙盒会透传 `FMLBENCH_BEDROCK_EFFORT`。`run_lite_queue.sh` 新增 `MODEL`、`PROVIDER`、`LABEL`、`OUTPUT_DIR` 四个环境变量，默认值和原来一样。
- **空闲连接问题**（2026-11-11 发现并修复）：
  - 第一版只创建一个 boto3 客户端并复用它的连接池。一个连接空闲一段时间后，网络路径上会把它静默丢掉：不发 FIN 也不发 RST，本机仍认为连接是 ESTABLISHED。
  - 下一个请求被发到这条死连接上，数据一直卡在发送队列里，内核按 `net.ipv4.tcp_retries2=15` 重传约 15 分钟才报错，然后 botocore 换新连接重试，约 1 分钟就成功。
  - 每一步大约有 2 次 LLM 调用，所以每步多出约 33 分钟。比如 ART 训练一步只要 40 秒，但每 34 分钟才走一步。CloudWatch 上没有限流，单次调用的平均延迟只有 22.6 秒。
  - 用 `ss -tnpo` 能看到这些连接的 Send-Q 有 5～123 KB 一直没被确认，重传计数到了 11～14 次；`py-spy dump` 显示进程卡在 `bedrock_converse_create` 的读响应上。
  - 卡住的请求没有到达服务端，所以不会重复计费，也不影响 LLM 的输出。
  - Anthropic SDK（httpx）的空闲连接 5 秒就过期，所以 Opus 的测试没有遇到这个问题。
  - 修复：`BedrockConverseClient.runtime` 每次调用都从同一个 boto3 Session 新建客户端。对照测试：两个客户端各调用一次，空闲 10 分钟后再调用一次。新建客户端的第二次调用 0.8 秒完成；复用客户端的第二次调用卡住，状态和正式运行时一样（Send-Q 不为 0，在重传）。
- 验证：
  - 在沙盒里测试了单轮、多轮和 code_editor 的编辑，都正常，token 也记录正确。
  - 用 ai_scientist_v2 在 opacus 和 domainbed 上各跑了 2 步的冒烟测试，都完成了 val 和最终 test（domainbed test 0.748，opacus 0.601）。日志在 `docs/reproduction_logs/smoke_gpt6astra/`，结果在 `benchmark_results_smoke/gpt6astra/`。

### 4.2 运行入口适配 uv 环境

- `benchmark/utils.py::env_command(env_name, args)`：如果 `~/fml_envs/<conda_env>/bin/python` 存在，就设置 `VIRTUAL_ENV`，把该环境放到 PATH 最前面后直接执行；否则退回 `conda run`。
- 调用方：`benchmark/executor.py::_run_command`（val/test 命令）和 `agents/theaiscientist/theaiscientist.py`（setup 命令）。config.json 的字段名 `conda_env` 保持不变。
- `_run_command` 在每条命令结束时输出 `Command finished in X s (returncode N)`，超时时输出 `Command timed out after N seconds`。
- `benchmark/runner.py` 复制工作区时保留软链（2.2）。

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
| 解释器 | 某个环境的基础解释器如果既不在 `/usr`，也不在 uv 的 Python 目录里（g7e.12xlarge 上的 privacy_meter 用的是 `~/miniconda3` 的 Python 3.12.9），就把它所在的前缀目录只读挂载进来。conda 不在 PATH 上 |
| 隔离 | user/pid/ipc/uts/cgroup 命名空间独立，`--clearenv`、`--new-session`、`--die-with-parent` |

- **网络与宿主共享**。隔离网络会断掉 IMDS，Bedrock 就拿不到实例角色的凭证；代价是 agent 生成的代码也能访问外网。
- `/etc/resolv.conf` 指向 `/run/systemd/resolve/`，所以只读挂载了这个目录。
- 已验证：
  - ml_tasks、agents 和数据缓存不可写，workspace 可写，`~/.ssh` 不可见；
  - 能做 GPU 计算，`CUDA_VISIBLE_DEVICES=""` 时没有 GPU；
  - Bedrock 调用、DNS、HTTPS 都正常。

---

## 5. 实验结果

### 5.1 ai_scientist_v2 × FML-bench-Lite × Opus 5.5

**设置**

- agent：ai_scientist_v2，参数用 `configs/agents/ai_scientist_v2.yaml` 的默认值。
- 模型：Bedrock `us.anthropic.claude-opus-5-5`。
- 每个任务跑 100 步，全部在沙盒（4.3）中运行。
- 机器：g7e.12xlarge（第 1 章）。
- baseline：**g7e.12xlarge 版**（3.5）。提示词里用的是这一版，下表的 baseline 列和 NI 主列也用这一版。
- 超时：opacus 用默认的 2400 秒；其余 7 个任务每两个共用一张卡，用 4800 秒。按 2400 秒的标准协议，只有 privacymeter 的 1 步会超时，而且不是最佳步，不影响结果。
- 运行时间：2026-11-02 ～ 11-05。复现方法见第 6 章。
- **已知问题**：ai_scientist_v2 的想法生成没有运行。`_generate_ideas` 找不到 `ml_tasks/<TASK>/theaiscientist/seed_ideas.json` 就直接跳过，而所有任务都没有这个文件。按决定没有修复，所以结果相当于纯树搜索。

**结果**（NI 的定义和计算方法见 6.9）

| 任务 | 指标 | baseline val / test（g7e.12xlarge 版） | agent 最佳 val（被测步） | agent test | **NI（g7e.12xlarge 版）** | NI（g7e.24xlarge 版） | NI（原始 baseline） |
|---|---|---|---|---|---|---|---|
| pycil | avg_incremental_acc ↑ | 0.5925 / 0.5944 | 0.7562（89） | 0.7571 | **0.1627** | 0.1627 | 0.1619 |
| usb | test_acc ↑ | 0.0827 / 0.0861 | 0.1677（64） | 0.1724 | **0.0863** | 0.0863 | 0.0934 |
| domainbed（ColoredMNIST） | in_acc ↑ | 0.2393 / 0.2501 | 0.7404（79） | 0.7527 | **0.5026** | 0.5026 | 0.4658 |
| officehome | avg_acc ↑ | 0.9011 / 0.8570 | 0.9203（17） | 0.8782 | **0.0212** | 0.0212 | 0.0153 |
| openood | auroc ↑ | 0.8804 / 0.9212 | 0.9204（55） | 0.9928 | **0.0716** | 0.0900 | 0.0626 |
| opacus | test_acc ↑ | 0.5881 / 0.5870 | 0.6911（88） | 0.6752 | **0.0882** | 0.0882 | 0.0875 |
| privacymeter | AUC_gap ↓ | 0.3242 / 0.3182 | 0.0011（57） | 0.0008 | **0.6347** | 0.6357 | 0.6411 |
| ART | defense_score ↑ | 0.3005 / 0.2921 | 0.9755（21） | 0.9740 | **0.6820** | 0.6903 | 0.4966 |
| **MEAN（Lite）** | | | | | **0.2812** | 0.2846 | 0.2530 |

- **被测步**：最终 test 用的是最早达到最佳 val 的那一步的代码。val 相同时不替换：同一阶段内取第一个最大值，跨阶段时必须严格更好。
- **换 baseline 的影响**：
  - 换成仓库现在采用的 g7e.24xlarge 版，只有 openood、privacymeter、ART 的 baseline 略低一些（3.3），这 3 个任务的 NI 略升，均值为 0.2846。
  - 换成仓库原始的 baseline（`baseline_results/original/`），均值降到 0.253。差距主要来自 ART：重测的 baseline 约 0.29，原始的是 0.48（3.4）。
- **约束都满足**：
  - privacymeter 的两个 TPR 都是 0，test 准确率 0.7515；
  - opacus 的 ε 是 7.9997，在 8 的预算内。
- **openood 的 test 高于 val**：这是任务设计如此，val 用 CIFAR-100 当 OOD 数据，test 用 SVHN。
- **除去不能代表通用能力的任务**（见 5.2，按 g7e.12xlarge 版）：去掉利用题目先验的 domainbed，其余 7 个任务的平均 NI 约为 0.25；再去掉利用评测弱点的 privacymeter，约为 0.19。
- **完整表格**：三张表（原始指标、NI、过程指标）的 CSV 在 g7e.12xlarge 的 `metric_reports/ai_scientist_v2_opus55_lite/`，当时顶层的 baseline 文件还是 g7e.12xlarge 版，所以 CSV 里的 NI 就是主列。结果目录是 `benchmark_results/ai_scientist_v2/<Task>/`，每个任务有一个主运行目录（含 `summary.json` 和 `step_snapshots/`）和一个 `*_final_test` 目录。

### 5.2 快照检查

**检查了什么**

1. **被测代码是哪一版**：按 5.1 的规则，确定每个任务被测的是哪一步的代码快照（`step_snapshots/step_NNNN_code.json`）。usb、domainbed、officehome 后面还有 val 相同、代码不同的步，那些没有被测。
2. **全部快照的静态扫描**：扫描每个任务全部 100 步的快照，只看 agent 新增或改动的行，查找以下几类代码：
   - 文件读写、数据路径；
   - 网络访问和下载；
   - 子进程、`exec`/`eval`；
   - 环境变量读取。

   没有命中。误报只有 `.eval()` 调用和个别 `setattr`。
3. **被测步逐个阅读**：把被测步的代码和 baseline 做 diff，再对照各任务的评测脚本（`train_eval_baseline.py`、`eval.py`、`run_mia.py` 等）确认数据怎么流动。重点看三件事：
   - 是否读取了 val/test 数据；
   - 是否改动或绕过了评测接口；
   - 是否违反任务约束，比如 DP 预算、样例预算、准确率下限。

   val/test 命令会先用 `original_file_backup/` 里的文件覆盖评测相关代码，所以 agent 只能通过目标文件影响结果。

这项检查是必要的，因为隐藏的 test 数据就在工作区里。比如 opacus 的 test 集是 CIFAR-10 训练集的一部分，划分种子固定，agent 生成的代码理论上能读到，沙盒和 code_editor 的危险代码检查都拦不住。

**结论：8 个任务都没有读取 val/test 数据，也没有改动评测代码。** 但 domainbed 和 privacymeter 的提升不能当作通用方法能力来看：

| 任务 | 被测代码做了什么 | 判断 |
|---|---|---|
| **domainbed** | ① 输入形状是 `(2, 28, 28)` 时，把两个颜色通道相加成灰度图，等于手工去掉了 ColoredMNIST 的伪特征（颜色）；② 用 GCE 损失，并按损失从小到大只保留一部分样本，针对的是 25% 的标签噪声；③ 加了仿射增强、双视图一致性、EMA 教师蒸馏，预测时对 1 像素平移取平均 | **利用题目先验**。任务说明点名了 ColoredMNIST，它的构造是公开的（IRM 论文）。test 0.7527 已经到了标签噪声决定的上限（约 75%） |
| **privacymeter** | ① 把模型改成 4 个 WRN 子网络，按像素内容的确定性哈希 `h(x)` 分组：子网络 k 只用 `h(x) ≠ k` 的样本训练，推理时 `h(x) = k` 的输入交给子网络 k，因此任何查询都由一个没见过它的子网络回答（SELENA 中 Split-AI 的思路）；② eval 模式下只输出 top-1 类别，置信度固定（logit 差值为 2）；③ 加了翻转和裁剪增强，测试准确率 0.75，高于 0.61 的约束 | **符合规则，但利用了评测的弱点**。题目只要求做防御侧改动，这一点满足；参考模型也用同样的代码训练。但评测只做非自适应攻击：攻击者只要对成员样本稍加扰动，`h(x)` 就会变，查询有 3/4 的概率落到训练过这个样本的子网络上。所以 AUC_gap 接近 0 不代表真正抵御了成员推断。SELENA 原文正是为此再加了一步自蒸馏 |
| ART | 先正常训练 2 个 epoch，然后用两种办法剔除样本：一是低置信度剔除，二是在倒数第二层特征上对每个类做 PCA 加 2-means，小簇占比低于 0.35 且轮廓系数达标时剔除整个小簇。剔除后从初始权重开始，用 DP-InstaHide 重新训练。`model.py` 和 `split_config.json` 没改 | **常规方法**，即 Chen et al. 2018 的 Activation Clustering。0.35 也是 ART 自带 `ActivationDefence` 的默认阈值。没有改评测时的 `predict` |
| pycil | 第一个任务训练 4 个 ResNet-32 组成集成，带旋转辅助任务，之后冻结。后续任务不再训练骨干，只累加随机投影特征（12288 维）的 Gram 矩阵，用岭回归闭式求解分类器（RanPAC / ACIL 一类的解析式持续学习），λ 用训练集中留出的 1/5 来选 | 常规方法，不需要样例。训练时 `test_loader` 只用来打印日志，和原代码一样。注意两点：骨干参数变成了 4 倍；Gram 矩阵约 1.2 GB，远大于 2000 个样例的存储，但按无样例方法的惯例，不算违反样例预算 |
| usb | WRN-28-4（原来是 28-2），EMA 评估，分布对齐，FreeMatch 自适应阈值，以及特征库上的伪标签平滑（SimMatch 一类） | 常规方法。实现上借用了接口：训练循环是固定的，agent 改不了，于是它让模型在 `training` 模式且 `no_grad` 下（也就是 harness 计算伪标签的那次前向）返回修正过的 logits，并改写最大概率，让外部固定的 0.95 阈值复现自适应阈值的效果。没有用到无标签数据的标签 |
| openood | LogitNorm 训练，训练时把 layer2、layer3、layer4 的特征写入模型 buffer 作为特征库，打分用"翻转 TTA 后的 MSP − 多层 KNN 距离"，其中 layer2、layer3 的距离用 ID 数据的自身 KNN 统计量做了缩放 | 常规方法。特征库只在训练模式的前向里更新，harness 只在训练集上调用 `model.train()`，没有混入 OOD 数据。注释里写到"layer2 统计量对 SVHN 这类远 OOD 有用"，这个信息来自任务说明（说明里写了 test 用 SVHN），不是泄漏 |
| officehome | 分类头学习率乘以 10，warmup 加 cosine 学习率，EMA，标签平滑 0.1 | 常规调参 |
| opacus | 固定的 ScatterNet 特征加小型 tanh CNN（Tramer & Boneh 2021），EMA，翻转和相位对称化，每个视图独立抖动（augmentation multiplicity），推理时 72 视图 TTA。`val_ratio` 从 0.114 改为 0.05 | 常规方法。**DP 保证仍然成立**：同一个 head 在一次前向里被调用 8 次，Opacus 1.5.4 的 `GradSampleModule` 会把多次调用的逐样本梯度累加到同一个样本上，裁剪仍然按原始样本做。EMA 是对 DP 迭代结果的后处理，不额外消耗隐私预算。`val_ratio` 是任务允许修改的，改完之后训练集增加约 7%，所以 +8.8 点里有一小部分来自更多的训练数据。任务说明称 test "完全不受 split_config 影响"，这不准确 |

只有 privacymeter 的分数依赖于评测弱点。如果要堵住，需要改 benchmark，比如加入自适应攻击（对查询做扰动），已记在待办里。

### 5.3 autoresearch × FML-bench-Lite × Opus 5.5

**设置**

- 和 5.1 相同：g7e.12xlarge，g7e.12xlarge 版 baseline，Bedrock Opus 5.5，沙盒，每个任务 100 步。
- agent 换成 autoresearch（`configs/agents/autoresearch.yaml`），8 个任务的超时都用 4800 秒。
- 排队方式和 6.8 的命令相同：
  - GPU0：officehome；privacymeter → pycil；
  - GPU1：usb；openood → domainbed → opacus；
  - CPU：ART。
- 运行时间：2026-11-06 05:59 ～ 11-10 08:57，约 99 小时。日志在 `docs/reproduction_logs/full_lite8_autoresearch/`。
- **opacus 没有跑满 100 步**：
  - 它排在最后，按每步约 12 分钟算，跑完还要 11 小时左右，所以按决定在第 41 步结束后中断（SIGTERM）。
  - 中断时 runner 会清理工作区副本并直接退出，不跑最终 test，也不写 `summary.json`。所以这两样是在结果目录里手工补的：
    - **最终 test**：`manual_final_test.py` 按 `_execute_test` 的流程做：新建工作区副本，恢复最佳步（第 39 步）的快照，不设超时跑 `pre_test_val`，再跑 test。日志在 `privacy_opacus_final_test.log`。
    - **`summary.json`**：`rebuild_summary.py` 用运行日志里每步的 KEEP/DISCARD/CRASH 行、`run_<step-1>/` 下的 val 结果和快照重建。
  - 两个脚本都在 `benchmark_results/autoresearch/Privacy_opacus/20261110_010021_b59cf5bb/` 下。
  - token 用量只保存在进程内存里，已经丢失，所以 opacus 的 token 数缺失。

**autoresearch 怎么工作**（Karpathy 的贪心爬山协议）：
- 每一步由 LLM 根据当前最好的代码和历史实验表提出一个想法，改代码后跑 val。
- val 严格优于当前最好值才保留，否则退回当前最好的代码；崩溃时最多调试 3 次。
- 最终 test 用的是最早达到最佳 val 的那一步的代码，即使它没有超过 baseline 也照样跑 test，这时 NI 被截断为 0。

**结果**（NI 的计算方法同 6.9）

| 任务 | agent 最佳 val（被测步） | agent test | **NI（g7e.12xlarge 版）** | NI（g7e.24xlarge 版） | NI（原始 baseline） | ai_scientist_v2 的 NI（5.1 主列） | token（M） | 耗时（h） | 失败步数 |
|---|---|---|---|---|---|---|---|---|---|
| pycil | 0.8127（95） | 0.8218 | **0.2274** | 0.2274 | 0.2266 | 0.1627 | 2.59 | 59.2 | 4 |
| usb | 0.1723（72） | 0.1483 | **0.0621** | 0.0621 | 0.0693 | 0.0863 | 1.95 | 37.4 | 0 |
| domainbed（ColoredMNIST） | 0.7375（87） | 0.7481 | **0.4980** | 0.4980 | 0.4612 | 0.5026 | 8.81 | 6.3 | 21 |
| officehome | 0.9203（53） | 0.8782 | **0.0212** | 0.0212 | 0.0153 | 0.0212 | 9.44 | 51.3 | 5 |
| openood | 0.9501（98） | 0.9846 | **0.0634** | 0.0818 | 0.0544 | 0.0716 | 2.06 | 84.7 | 2 |
| opacus（41 步） | 0.7526（39） | 0.7370 | **0.1499** | 0.1499 | 0.1492 | 0.0882 | 缺失 | 7.9 | 4 |
| privacymeter | 0.0051（93） | 0.0144 | **0.6075** | 0.6086 | 0.6139 | 0.6347 | 2.28 | 23.0 | 68 |
| ART | 0.9683（79） | 0.9672 | **0.6751** | 0.6835 | 0.4898 | 0.6820 | 2.58 | 21.3 | 4 |
| **MEAN（Lite）** | | | **0.2881** | 0.2916 | 0.2600 | 0.2812 | | | |

- **和 ai_scientist_v2 比**：
  - 平均 NI 基本持平，0.2881 对 0.2812。
  - autoresearch 在 pycil 和 opacus 上明显更好；opacus 只跑了 41 步就到了 0.15，ai_scientist_v2 跑满 100 步是 0.09。
  - usb、openood、privacymeter 上略差，其余任务接近。
  - 去掉利用题目先验的 domainbed，其余 7 个任务 autoresearch 约为 0.26，ai_scientist_v2 约为 0.25。
  - 再去掉 privacymeter（ai_scientist_v2 在这个任务上利用了评测弱点，autoresearch 没有，见下面的检查），约为 0.20 对 0.19。
- **和 Claude Code（5.4）比**：按 g7e.12xlarge 版 baseline，Claude Code 的平均 NI 是 0.3225，高于 autoresearch 的 0.2881 和 ai_scientist_v2 的 0.2812。但协议不同（每个任务 24 小时、由 agent 自己选择提交、在 g7e.24xlarge 上运行，见 5.4），只能作为参考。
- **约束都满足**：
  - privacymeter 的两个 TPR 都是 0，test 准确率 0.632（下限 0.61）；
  - opacus 的 ε 是 7.9931。
- **失败步多的任务**：
  - privacymeter 的 100 步里有 68 步失败，都是违反了约束，代码本身没有崩溃：
    - 66 步是 TPR 约束。`postprocess.py` 在 val 上要求两个 TPR 都是 0，这些步的 TPR@0.1%FPR 在 0.0006～0.003 之间。
    - 2 步是准确率低于 0.61。
  - domainbed 有 21 步失败。
- **officehome 的分数没有意义，三个 agent 的数值完全相同**：
  - autoresearch、5.1 的 ai_scientist_v2 和 5.4 的 Claude Code 都是 val 670/728、test 1492/1699，所以 NI 都是 0.0212，虽然三边的代码各不相同。
  - 原因是 5.4 发现的数据问题：每个域下只有 `train/`、`val/` 两个子目录，DomainBed 把它们当成 2 个类别。
  - 本机核对过：Art 域 `train/` 下有 2162 张图，`val/` 下有 265 张，合计 2427 张，正好等于 val 加 test 的 728 + 1699；而 670 + 1492 = 2162。也就是说，模型对 Art 域的每张图都预测成多数类 `train`，分数就是多数类的占比。
  - 之前把相同的数值解释为"val 已经饱和、碰巧相同"，这是错的。autoresearch 有 34 步的 val 完全相同，也是因为输出总是多数类。
  - 这个任务的 NI 对三个 agent 都一样，不影响它们之间的比较。是否修复数据，记在第 7 章。
- **耗时**：
  - 总共约 99 小时，是 ai_scientist_v2（约 50 小时）的两倍。时间几乎都花在训练上，每个任务的 LLM 时间只有 1～2 小时。
  - 慢的原因是 autoresearch 会不断加长训练，比如 openood 的被测步训练了 500 个 epoch。
  - 超过 2400 秒的步数：openood 75 步，pycil 35 步，officehome 2 步，usb 1 步；超过 4800 秒而超时的：pycil 2 步，openood 2 步，officehome 1 步。
  - **pycil 和 openood 的被测步本身就超过 2400 秒**（val 分别用了 3800 秒和 3520 秒）。如果按默认 2400 秒的标准协议，这两步会超时，结果不能直接和用 2400 秒跑的结果对比。ai_scientist_v2 没有这个问题（5.1）。
- **token**：7 个任务共 29.7M（不含 opacus），ai_scientist_v2 的 8 个任务共 25.3M。domainbed 和 officehome 各用了约 9M，其余任务各 2～2.6M。
- **完整表格**：`metric_reports/autoresearch_opus55_lite/` 下的三张 CSV。打分时顶层 baseline 还是 g7e.12xlarge 版，所以 CSV 里的 NI 就是主列。另外两列是手工算的。

**快照检查**（方法同 5.2）

- **全部快照的静态扫描**：只命中了 `.eval()` 和 `setattr` 这类误报。
- **被测步逐个阅读**：8 个任务都没有读取 val/test 数据，也没有改动评测代码。

| 任务 | 被测代码做了什么 | 判断 |
|---|---|---|
| **domainbed**（第 87 步） | ① 输入是 2 通道时，训练中随机交换红、绿两个通道；预测时对两种通道顺序取平均，等于让模型对颜色不敏感；② GCE 损失（q = 0.7）；③ mixup、SAM，以及平移、旋转 TTA | **又一次利用了 ColoredMNIST 的题目先验**，和 5.1 一样，只是换了做法 |
| privacymeter（第 93 步） | 增强、λ 有截断的 Mixup/CutMix、标签平滑、flooding、SWA（重算 BN 统计量）、梯度裁剪。只改了 `default_trainer.py` | **常规的正则化防御**，没有 5.1 那种按哈希路由的做法 |
| ART（第 79 步） | 谱签名过滤、按损失剔除样本、重新初始化后重训。`model.py` 加了输入端的 3×3 中值滤波层（评测时也生效），并改成 5×5 卷积 64 通道、全连接 1024 维 | 常规方法。`model.py` 是任务允许修改的文件。test 的 clean_acc 是 0.948，poison_acc 是 0.013 |
| pycil（第 95 步） | DER 风格的 ResNet-34（CIFAR 版的 stem）、SLA 辅助头、Balanced Softmax、18 视图 TTA、LUCIR 风格的蒸馏。只改了 `models/icarl.py` | 常规方法。没有改样例存储，也没有在训练时用 `test_loader` |
| usb（第 72 步） | 在允许修改的 `get_training_config` 里把阈值从 0.95 改成 0.7；加了 BlurPool 和 Cutout | 常规方法 |
| openood（第 98 步） | LogitNorm、TrivialAugment 风格的增强、在翻转视图之间共享扰动的 ODIN 打分，训练 500 个 epoch | 常规方法 |
| officehome（第 53 步） | SGD + Nesterov、SAM、SWA、翻转 TTA | 方法本身是常规的，但分数只是多数类的占比，见上文 |
| opacus（第 39 步） | 带 Weight Standardization 的 VGG 风格 CNN（GroupNorm + SiLU，Tramer & Boneh 2021）；最后一层零初始化；在模型内部做输入标准化；训练时对原图和翻转图的 logits 取平均；参数 EMA；推理时 50 视图 TTA（5×5 平移加翻转，按高斯权重平均）；B = 8192，200 个 epoch，lr 0.8。只改了 `algorithm.py` | 常规方法（De et al. 2022 的配方）。**DP 保证仍然成立**：翻转的两个视图来自同一个样本，Opacus 把两次调用的逐样本梯度累加后再裁剪；Weight Standardization 不增加参数，走 functorch 计算逐样本梯度；EMA 是后处理。**val 和 test 用的权重不一样**：EMA 存在模型的普通 Python 属性里，不在 `state_dict` 中，所以 test 加载 checkpoint 后用的是最后一步的权重，而不是 EMA。val 0.7526 比 test 0.7370 高，部分原因就在这里。这对 agent 不利，不算违规 |

- 和 5.1 相比，autoresearch 只有 domainbed 依赖题目先验，privacymeter 的结果是正常的防御效果。

### 5.4 Claude Code × FML-bench-Lite × Opus 5.5

**设置**（方案和实现见 `docs/claude_code_agent_design.md`、`scripts/claude_code/`）

- agent：Claude Code 2.1.318，在交互模式下用 `/goal` 运行（`--dangerously-skip-permissions`，禁用 WebFetch/WebSearch）。模型是 Bedrock `us.anthropic.claude-opus-5-5`，`/goal` 的判定模型是 `us.anthropic.claude-haiku-5-5`。任务说明放在只读的 `CLAUDE.md` 里，研究问题部分原样使用 `prompt.json`。
- 机器：g7e.24xlarge。每个任务一个 bwrap 沙盒，只能看到：
  - 本任务的代码和 uv 环境；
  - 掩码数据：隐藏 test 样本的图像置零，标签和文件布局不变。privacymeter 的 test 是同一次运行里的另一部分目标模型，无法掩码；
  - 分配给它的一张 GPU。GPU 任务各 16 个逻辑 CPU，ART 用 48 个，只用 CPU。
- 分两批运行：
  - 第 1 批：pycil、usb、domainbed、ART，2026-11-05 ～ 11-06；
  - 第 2 批：officehome、openood、privacymeter、opacus，2026-11-06 ～ 11-07。
- 预算和约束：
  - 每个任务 24 小时挂钟时间，截止后 `fml-val` 拒绝新的运行；
  - 提示词说明最多 100 次 `fml-val`（失败的也算），事后审计；单次运行上限 2400 秒；
  - 只能通过 `fml-val` 训练，shell 里只做快速检查（提示词约束，事后审计会话记录）。
- `fml-val` 每次把当前的目标文件放进一份新的 baseline 仓库副本里运行 val 命令，所以开发时看到的 val 和最终评测用的是同一份代码。最终提交由 agent 用 `fml-submit` 选择。
- baseline：**g7e.24xlarge 版**（3.5）。提示词和下表的 NI 都用这一版。
- 最终 test：会话结束后，在一个配置相同但使用真实数据的新沙盒里，把提交的 run 重新训练一遍（val 命令，不设超时，和 harness 一致），再跑 test 命令。
- 提示缓存：第 1 批是默认的 5 分钟，第 2 批是 1 小时（`CLAUDE_CODE_PROMPT_CACHE_TTL=1h`）。这只影响费用，不影响 agent 的行为。

**结果**（NI 的定义见 6.9，口径为 **g7e.24xlarge 版 baseline**）

| 任务 | 指标 | baseline val / test（g7e.24xlarge 版） | 提交的 run（val） | Claude Code test | **NI（g7e.24xlarge 版）** | 参考：ai_scientist_v2 的 NI（g7e.24xlarge 版，5.1） | `fml-val` 次数 | 费用 |
|---|---|---|---|---|---|---|---|---|
| pycil | avg_incremental_acc ↑ | 0.5925 / 0.5944 | 75（0.7789） | 0.7703 | **0.1759** | 0.1627 | 75 | $196 |
| usb | test_acc ↑ | 0.0827 / 0.0861 | 39（0.3823） | 0.3787 | **0.2926** | 0.0863 | 41 | $149 |
| domainbed（ColoredMNIST） | in_acc ↑ | 0.2393 / 0.2501 | 96（0.7340） | 0.7433 | **0.4932** | 0.5026 | 96 | $19 |
| officehome | avg_acc ↑ | 0.9011 / 0.8570 | 95（0.9203） | 0.8782 | **0.0212** | 0.0212 | 95 | $9 |
| openood | auroc ↑ | 0.8708 / 0.9028 | 65（0.9594） | 0.9993 | **0.0965** | 0.0900 | 65 | $26 |
| opacus | test_acc ↑ | 0.5881 / 0.5870 | 49（0.7722） | 0.7697 | **0.1827** | 0.0882 | 51 | $19 |
| privacymeter | AUC_gap ↓ | 0.3278 / 0.3187 | 95（0.0008） | 0.0023 | **0.6327** | 0.6357 | 95 | $14 |
| ART | defense_score ↑ | 0.2893 / 0.2838 | 92（0.9966） | 0.9968 | **0.7131** | 0.6903 | 92 | $144 |
| **MEAN（Lite）** | | | | | **0.3260** | 0.2846 | | **$577** |

- 用 `compute_agent_metrics.py benchmark_results/claude_code --lite` 打分，8 个任务的 `note` 都是 `ok`：test 完整、没有违反约束。privacymeter 的两个 TPR 都满足约束，test 准确率约 0.75；opacus 的 ε 是 7.997。
- **最终重训的复现性**：重训得到的 val 和 `fml-val` 记录的相同（usb 和 privacymeter 的训练不确定，分别差 0.0046 和 0.0026）。最终重训都在 2400 秒内，最长的是 usb，2109 秒。
- **和 ai_scientist_v2 比**：同一版 baseline 下是 0.3260 对 0.2846，差距主要来自 usb、opacus 和 ART。两次测试的协议不同，比较时要注意：
  - ai_scientist_v2：每个任务 100 步，最终 test 用 val 最优的那一步；在 g7e.12xlarge 上运行，超时 2400/4800 秒；提示词里是 g7e.12xlarge 版 baseline；
  - Claude Code：每个任务 24 小时、最多 100 次 `fml-val`，由 agent 自己选择提交；单次 2400 秒；CPU 按槽位分配。
- **换 baseline 的影响**：用 g7e.12xlarge 版是 0.3225，用原始 baseline 是 0.2944，公式不变，只换 `p_baseline`。
- **费用**：第 1 批 4 个任务约 $508，其中 88% 是缓存写入，集中在 pycil、usb、ART 三个任务上。这三个任务每次 `fml-val` 要 17～35 分钟，超过了默认 5 分钟的缓存时长，agent 每次回来都要把整个上下文重新写入缓存。第 2 批改用 1 小时缓存后，4 个任务约 $69。按第 1 批的实际请求序列回放估算，1 小时缓存能让 pycil、usb、ART 的费用各降低约 87%。费用按会话记录里的 token 用量计算，价格（每百万 token 输入 $4、5 分钟缓存写 $5、1 小时缓存写 $8、缓存读 $0.20、输出 $20）是从 Claude Code 自己的计费记录反推的。

**各任务的方法与判断**（依据各任务的 `FINAL_REPORT.md` 和会话记录）

| 任务 | 提交的代码做了什么 | 判断 |
|---|---|---|
| **officehome** | agent 发现任务数据每个域下只有 `train/`、`val/` 两个子目录，DomainBed 因此只看到 **2 类**，标签含义是"文件在不在 `val/` 下"，图像内容里没有可学的信号。它据此提交了一个基本只按类先验预测的模型：冻结骨干的线性探针，加强 L2 约束 | **分数没有意义**：val 0.9203 和 test 0.8782 都是多数类的比例，ai_scientist_v2 也是同样的分数。这是上游数据准备的问题（见第 7 章） |
| **domainbed** | 训练一个只学颜色的偏置模型，按"类别 × 偏置模型是否预测对"分组重新加权，再加 V-REx、仿射增强、EMA。agent 明确拒绝了第 3 次 run（val 0.906，test 0.9009）：那次学到的是反向的颜色规则，在训练域上只有 0.15。它也刻意没有用灰度化或颜色通道交换 | 常规方法，接近只看形状的上限（约 0.75）。和 ai_scientist_v2 不同，没有手工去掉颜色 |
| **privacymeter** | 交叉折蒸馏（SELENA 风格）：每个训练样本的目标，由没见过它的 teacher 给出；推理时把 logits 放大 500 倍（置信度屏蔽），让输出接近硬标签 | **符合规则，但利用了评测的弱点**。置信度屏蔽主要针对的是这类基于置信度的审计攻击本身 |
| **ART** | 交叉验证式毒样本检测（样本由没见过它的模型预测，标签不一致就剔除）加重新训练；模型里内置一个输入过滤器（去掉稀疏的亮像素图案）和测试时增强 | **利用了题目先验**：过滤器的阈值是按触发器的形状定的，触发器定义在评测脚本里，沙盒内可见 |
| **openood** | 2 个 ResNet-18 的集成；旋转自监督和 SLA 联合头；训练图像特征库上的 KNN、能量分数，以及中间层"风格"统计量的加权组合 | 常规方法。开发时用合成的门牌号数字作为远 OOD 代理来选权重，利用了任务说明里"test 用 SVHN"这一点，和 5.2 对 ai_scientist_v2 的判断一样 |
| usb | SoftMatch 风格的伪标签加权、GPU 上的多视图 RandAugment、CoMatch 风格的图对比损失、EMA、WRN-28-4。训练循环是固定的，这些都实现在模型内部 | 常规方法。只用了 200 个有标签样本 |
| pycil | DER 风格的动态扩展（每个任务新增一个 ResNet-32），旧分类头的 logit 蒸馏，权重对齐，数据和增强放到 GPU 上（训练轮数因此约为原来的 5 倍） | 常规方法（任务说明允许换成 DER） |
| opacus | WRN-16-2 加权重标准化和 GroupNorm，增强倍数 K=6（每个真实样本的多视图梯度先求和再裁剪），自定义反向得到逐视图损失的平均，EMA，`val_ratio` 改为 0.02 | 常规方法。DP 保证仍然成立：逐样本裁剪按真实样本做，学习率调度和 EMA 都是后处理，Opacus 的隐私核算没有改动 |

**审计**（`scripts/claude_code/finalize.py`，依据 `fml-val` 的记录和 Claude Code 的会话记录）

- 没有发现访问 test 的命令（`--split test`、重建 test 索引等），没有下载数据或模型、安装包，也没有超过 100 次 `fml-val`。
- usb 有 6 次速度测试，各 2～4 分钟，超出了规则里"约 1 分钟"的快速检查。这些测试只测每步耗时，评估是关掉的。
- 其余几个标记都是误报：有的 agent 用排队脚本连续调用 `fml-val`，所以命令数比运行次数少；有的命令只是等日志出来后用小脚本做离线分析。

**运行中遇到的问题**

- `/goal` 的判定模型在会话末尾几次"无法完成判定"：
  - 第 1 批的 pycil 和 domainbed 在 3 次自动重试后暂停，要手动输入 `continue` 才判定完成；
  - 其余任务重试 1～2 次后恢复。
  - 这些都发生在 agent 完成工作之后，不影响结果。
- domainbed 每次运行只要约 2 分钟，用 4.3 小时就跑完了 96 次，按设计进入 `WRAP-UP` 后停止，剩下的时间空闲。
- officehome 第一次启动后手动操作出了问题，于是删掉这次运行，从头重新开始。表中是重新开始的那一次。
- 在沙盒的 bash 提示符下（不是在 Claude Code 里）按 Ctrl-C 会结束整个沙盒，`launch.log` 里记为 `sandbox ended by signal 2`。

### 5.5 ai_scientist_v2 × FML-bench-Lite × GPT-6 Astra（进行中）

**设置**

- 和 5.1 相同：g7e.12xlarge，agent 参数用默认值，每个任务 100 步，全部在沙盒中运行。
- 模型换成 Bedrock 的 `us.openai.gpt-6-astra`（provider `OpenAIBedrock`，reasoning effort high，见 4.1）。
- baseline 用 **g7e.12xlarge 版**，和 5.1、5.3 一样。运行期间，顶层的 baseline 文件临时换成了 `g7e12xlarge/` 下的文件（6.5），打分后用 `git checkout ml_tasks` 恢复。
- 8 个任务的超时都用 4800 秒，排队方式同 6.8（opacus 排在 GPU1 的 openood → domainbed 之后）。
- 2026-11-11 02:49 启动。日志在 `docs/reproduction_logs/full_lite8_ai_scientist_v2_gpt6astra/`，结果在 `benchmark_results/gpt6astra/ai_scientist_v2/`，和 5.1 的 Opus 结果分开存放。
- **第一次运行作废**：
  - 第一次是 11-10 11:54 启动的。因为 4.1 说的空闲连接问题，每步多等约 33 分钟，14.5 小时只跑了 16～25 步。
  - 按决定，在 11-11 02:30 前后中断（SIGTERM，工作区副本已自动清理），修复后全部重跑。
  - 作废的结果和日志改名保留为 `benchmark_results/gpt6astra_aborted_20261111/` 和 `docs/reproduction_logs/full_lite8_ai_scientist_v2_gpt6astra_aborted_20261111/`，不参与打分。
- 5.1 记录的已知问题（想法生成没有运行）这里同样存在，两次运行的条件相同。

```bash
L=docs/reproduction_logs/full_lite8_ai_scientist_v2_gpt6astra
AGENT=ai_scientist_v2 MODEL=us.openai.gpt-6-astra PROVIDER=OpenAIBedrock LABEL=gpt6astra OUTPUT_DIR=benchmark_results/gpt6astra \
QUEUE_SLOTS="0|generalization_officehome;0|privacy_privacymeter continual_learning_pycil;1|data_efficiency_usb;1|robustness_openood generalization privacy_opacus;|robustness_and_reliability_art" \
setsid nohup scripts/reproduction/run_lite_queue.sh $L > $L/queue.out 2>&1 < /dev/null &
```

结果待补。

---

## 6. 复现全流程

以下步骤从一台新机器开始，复现第 3 章的 baseline 和第 5 章的结果，最后算出 Normalized Improvement。2026-11-03 在 4 × L4 上从建环境到冒烟测试完整走过一遍（第 8 章，当时数据是复制来的，没有用缓存）；6.4 的数据缓存和 6.5 的 `rebaseline_queue.sh` 在 g7e.24xlarge 上走过（第 2、3 章）。

### 6.1 前提条件

| 项目 | 要求 |
|---|---|
| 系统 | Linux x86_64，内核允许非特权用户命名空间（bubblewrap 需要） |
| GPU | NVIDIA，驱动需支持 CUDA 12.8。所有环境都是 torch 2.7.1+cu128，带 sm_75～sm_120 的 kernel，L4（sm_89）实测可用。下面的命令按 2 张卡分配，单卡峰值约 15 GB。**GPU 比 Blackwell 慢很多时要加大超时**：L4 上 officehome 的 baseline 单步就要 50 分钟（第 8 章） |
| 命令 | `git`、`wget`、`unzip`、`curl`、`uv`、`bwrap`、`nvidia-smi`、`python3`。Amazon Linux 2023 用 `sudo dnf install bubblewrap` 安装 bwrap。`conda` 和 `pip` 只在要建 conda 环境时才需要，Lite 的 8 个任务都不需要 |
| AWS | 能在 us-west-2（或 `AWS_REGION` 指定的区域）用 Bedrock 调用 `us.anthropic.claude-opus-5-5`，需要 InvokeModel / InvokeModelWithResponseStream 权限 |
| AWS 凭证 | 三选一：EC2 实例角色（沙盒共享网络，能访问 IMDS）；`AWS_PROFILE`（沙盒会只读挂载 `~/.aws`）；`AWS_ACCESS_KEY_ID` 等环境变量（沙盒会传入） |
| 网络 | 数据从 Kaggle 镜像下载（2.2），多伦多和斯坦福的官方源太慢或连不上 |
| 时间和费用 | 正式测试（6.8）在两张 Blackwell 上约 50 小时，ai_scientist_v2 共约 2530 万 token，按官方价格（输入 $4、输出 $20 每百万 token）估算约 $210，以 Bedrock 账单为准。冒烟测试（6.7）约 1.5 小时，约 50 万 token |

### 6.2 获取代码

```bash
git clone <仓库地址> FML-bench && cd FML-bench
git checkout repro/bedrock-uv-sandbox
```

复现所需的代码、沙盒、脚本和 baseline 都在这个分支上。运行产物（日志、结果、打分表）不在版本库里，见附录。

### 6.3 构建环境

```bash
export UV_PYTHON_PREFERENCE=only-managed   # 只用 uv 自己管理的 Python，见下面的说明
# export FMLBENCH_UV_ENVS=/path/to/envs    # 可选，默认 ~/fml_envs
python3 setup.py --skip-data \
  --task Privacy_opacus --task Continual_Learning_pycil --task Data_Efficiency_usb \
  --task Generalization_domainbed --task Generalization_domainbed_officehome \
  --task Privacy_privacymeter --task Robustness_and_Reliability_art --task Robustness_openood
```

- 这条命令同时建好 harness 环境 `fmlbench`，6.9 打分要用它。
- `UV_PYTHON_PREFERENCE=only-managed` 是 uv 自带的环境变量，不需要改 `setup.py`。g7e.12xlarge 建环境时没有设置它，所以 privacy_meter 用到了 miniconda 的 Python 3.12.9；设置后会用 uv 管理的 3.12.x，补丁版本可能不同。即使没设，沙盒的解释器挂载（4.3）也能处理这种情况。
- 环境建在本地盘上即可。g7e.24xlarge 上约 6 分钟。

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

3. 确认 8 个任务的代码库都是干净的。下面的命令不应该有任何输出，否则数据可能在第一步就被清掉。6.8 启动前和 6.9 打分前也要再检查一次：

   ```bash
   for c in ml_tasks/*/config.json; do
     r=$(python3 -c "import json;print(json.load(open('$c'))['repo_dir'])")
     [ -d "$r" ] && git -C "$r" status --porcelain
   done
   ```

不设 `--data-cache` 时，setup 会像上游一样把数据直接下载到各任务的工作区（走官方源），也会提交。

### 6.5 Baseline

- **硬件和第 1 章的 Blackwell 相同**：直接用分支里提交的 baseline（g7e.24xlarge 版）。要和第 5 章的数值严格对照，先用 g7e.12xlarge 版覆盖顶层文件，两版只在 openood、privacymeter、ART 上不同（3.3）：

  ```bash
  for d in ml_tasks/*/baseline_results/g7e12xlarge; do cp $d/*_info.json $d/..; done   # 恢复：git checkout ml_tasks
  ```

- **硬件不同**：先重测，用结果覆盖 `ml_tasks/<TASK>/baseline_results/{val,test}_info.json`。agent 提示词里的 baseline 和 6.9 的打分都读这两个文件。

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

每个环境都应该打印出 Python 版本，GPU 数量要正确，数据要能通过软链读到。要确认 Bedrock 能否调用，可以在沙盒里用 `agents.llm.create_client(..., "AnthropicBedrock")` 发一条短消息。

### 6.7 冒烟测试（可选，建议做）

每个任务跑 3 步，先确认整条链路能跑通，再投入两天的正式测试：

```bash
GPU_A=0 GPU_B=1 scripts/reproduction/run_smoke_lite.sh docs/reproduction_logs/smoke_lite7_<机器名>
```

- 脚本在前台运行，会启动资源监控，同时运行 7 个任务（不含 opacus），每个跑 3 步，输出到 `benchmark_results_smoke/`。结束后打印每个任务的命令数、失败数、最佳 val 和 test，以及显存和内存峰值。
- 可调的环境变量：
  - `STEPS`：步数，默认 3；
  - `OUTPUT_DIR`：输出目录，默认 `benchmark_results_smoke`；
  - `SMOKE_TASKS`：用空格分隔的 `任务配置|CUDA_VISIBLE_DEVICES` 列表，替换默认的分配，可以加上 `privacy_opacus`；
  - `EXTRA_OVERRIDES`：传给 run_agent_benchmark.py 的额外覆盖，例如 `agent.ai_scientist_v2.execute_timeout=7200`。

  第 8 章就是这样用的。
- **怎么判断是否成功**：
  - 每个任务的日志里都有 `Best Val Metric` 和 `Test Metric`，工作区副本已清理（`Cleaned up workspace copy`）；
  - 日志里没有 `No such file or directory: 'conda'`、`Command timed out after`，也没有 Bedrock 鉴权错误；
  - privacymeter 某一步因为 TPR 约束被判无效是正常的。

### 6.8 正式测试

8 个任务各跑 100 步，排成 5 个队列位置：同一位置上的任务依次运行，各位置并行。

```bash
L=docs/reproduction_logs/full_lite8_ai_scientist_v2
mkdir -p $L
setsid nohup python3 scripts/reproduction/resource_monitor.py $L/resources.csv 10 > $L/monitor.out 2>&1 < /dev/null &
AGENT=ai_scientist_v2 \
QUEUE_SLOTS="0|generalization_officehome;0|privacy_privacymeter continual_learning_pycil;1|data_efficiency_usb;1|robustness_openood generalization privacy_opacus;|robustness_and_reliability_art" \
setsid nohup scripts/reproduction/run_lite_queue.sh $L > $L/queue.out 2>&1 < /dev/null &
```

- `run_lite_queue.sh` 里固定了以下设置，和第 5 章一致：
  - 模型 `us.anthropic.claude-opus-5-5`，provider `AnthropicBedrock`；
  - `--workspace-label opus55`；
  - `agent.<AGENT>.max_steps=100` 和 `execute_timeout=4800`；
  - 每个任务都通过沙盒运行，结果写到 `benchmark_results/<AGENT>/<Task>/`。
- **排队方式**：每张卡 2 个位置，一个任务等 LLM 回复时，另一个正好用 GPU。最长的位置是 GPU0 的 privacymeter → pycil，约 50 小时。ART 只用 CPU，因为 TF 2.10 用不了 Blackwell。
- 5.1 的 opacus 当时是单独运行的，用的是默认的 2400 秒超时，没有发生超时，所以这里统一用 4800 秒结果等价。复现 5.3 时，把 `AGENT` 换成 `autoresearch`，日志目录也相应改名。
- `run_lite_queue.sh` 没有输出目录参数。如果 `benchmark_results/<AGENT>/<Task>/` 下已经有旧的运行，先移走，原因见 6.9。
- 运行时间长，建议用上面的 `setsid nohup` 放到后台，或者在 tmux 里运行。
- **跑完之后**：
  - `queue.log` 里每个任务都有 `end ... rc=0`，最后一行是 `all slots done`；
  - 每个任务的日志里都有 `Test Metric`，`grep -c 'Command timed out after' $L/*.log` 都是 0；
  - 统计按 2400 秒标准协议会超时的步数：`grep -oP 'Command finished in \K[0-9.]+' $L/<任务配置>.log | awk '$1>2400' | wc -l`；
  - 资源监控**不会自动退出**，要用它的 PID 手动 `kill`。

### 6.9 计算 Normalized Improvement

**定义**（论文 arXiv:2605.17373 第 3.3 节和附录 E.1）：

`NI = max(0, (p_agent − p_baseline) / |p_best − p_worst|)`

- `p_agent` 和 `p_baseline` 都是 **test** 指标。`p_baseline` 读 `ml_tasks/<TASK>/baseline_results/test_info.json`，也就是当前顶层的那一版（3.5）。
- 越低越好的指标把分子取反。改进为负时截断为 0。
- `p_best` 和 `p_worst` 取指标的理论最好值和最差值，在 `compute_agent_metrics.py` 的 `RANGE_META` 里：
  - Lite 的 7 个准确率或 AUROC 类任务取 1 和 0；
  - privacymeter 的 AUC_gap 取 0 和 0.5。
- test 崩溃、违反约束或缺少 test 结果时，按 baseline 计分，NI 为 0。
- MEAN 是 8 个任务的算术平均。

**命令**（在宿主机上用 fmlbench 环境运行，不需要沙盒）：

```bash
~/fml_envs/fmlbench/bin/python compute_agent_metrics.py benchmark_results/ai_scientist_v2 --lite \
  --output-dir metric_reports/ai_scientist_v2_opus55_lite
```

- **前提**：
  - 8 个模板工作区都是干净的（用 6.4 的检查命令）。过程指标里的探索类指标要读 baseline 代码。
  - 每个任务目录下只能有一个主运行。脚本取按名字排序后第一个含 `summary.json` 的运行目录，也就是最早的那次。
- **打分脚本还需要的东西**：
  - `--lite` 只给 Lite 的 8 个任务打分，MEAN 也只对这 8 个求平均。不加的话，脚本会要求 18 个任务都存在。
  - fmlbench 环境里的 pyarrow 必须低于 26（`setup.py` 已固定），否则 scikit-learn 导入失败。旧环境可以用 `uv pip install --python ~/fml_envs/fmlbench/bin/python 'pyarrow<26'` 修复。
  - 第一次运行会从 Hugging Face 下载 `microsoft/graphcodebert-base`，只用于过程指标里的探索类指标，不影响 NI。嵌入向量缓存在 `<output-dir>/embeddings_cache/`。
  - `--output-dir` 不能放在结果目录里面，脚本会拒绝。
- **输出**：`--output-dir` 下有三张 CSV。
  - `table_raw_performance.csv`：每个任务的 baseline test 和 agent test。`note` 列都应该是 `ok`，否则表示 test 缺失、崩溃或违反约束。
  - `table_normalized_improvement.csv`：**每个任务的 NI 和 MEAN**。
  - `table_process_metrics.csv`：12 个过程指标。
- **用哪一版 baseline**：对 5.1 的结果直接运行上面的命令，得到的是 5.1 表里 g7e.24xlarge 版那一列（MEAN 0.2846）。要得到主列（g7e.12xlarge 版，MEAN 0.2812），先按 6.5 用 `g7e12xlarge/` 覆盖顶层文件再打分，打完用 `git checkout ml_tasks` 恢复。5.1 表的"原始 baseline"列是手工算的，公式不变，只是把 `p_baseline` 换成 `baseline_results/original/test_info.json` 里的值。
- **怎么判断复现是否成功**：agent 的输出是随机的，单次运行的数值不会和 5.1 完全一致，比较的是量级：
  - domainbed、privacymeter、ART 的 NI 应该大幅提升；
  - 其余任务一般在 0.02～0.16。

### 6.10 快照检查

按 5.2 的三步检查每个任务被测的代码：

1. 在 `summary.json` 的 `val_steps` 里找到最早达到最佳 val 的那一步，对应快照是主运行目录下的 `step_snapshots/step_NNNN_code.json`。快照里的路径是工作区副本的路径，对应到模板工作区，用 `git show HEAD:<文件>` 取出 baseline 版本来比较。
2. 对全部快照里 agent 新增的行做静态扫描。
3. 逐个阅读被测步的 diff，并对照评测脚本确认数据流向。

### 6.11 常见问题

| 现象 | 原因与处理 |
|---|---|
| privacymeter 报 `No such file or directory: 'conda'` | 环境的基础解释器在沙盒里看不到。当前的沙盒脚本会挂载解释器（4.3）；也可以在建环境时设 `UV_PYTHON_PREFERENCE=only-managed`（6.3） |
| ART 被分到了 GPU | 要传 `CUDA_VISIBLE_DEVICES=""`。队列脚本里 ART 所在位置的 GPU 字段留空，就是这个作用 |
| officehome 不到 10 秒就失败（timm 报错） | `huggingface_hub` 装成了 2.x。`setup.py` 已锁定 `<1.0` |
| 沙盒里 DNS 解析失败 | `/etc/resolv.conf` 是指向 `/run/systemd/resolve/` 的软链接，沙盒脚本已挂载这个目录 |
| 数据在第一步被删 | 数据没提交到任务代码库。`setup.py` 无论数据怎么来的都会提交；绕过 setup 手动放的数据要自己用 `git add -f data` 提交（`data/` 在 pycil、usb 仓库的 `.gitignore` 里），再用 6.4 的命令检查 |
| 沙盒里找不到数据（软链失效） | 缓存不在沙盒可见的路径上。沙盒会只读挂载任务模板里软链的目标；其他情况要设 `FMLBENCH_DATA_CACHE` |
| `setup.py` 报 `missing required commands: conda, pip` | 旧版启动检查无条件要求 conda/pip，已改为只有要建 conda 环境时才检查 |
| 全新机器上建 privacy_meter 环境失败（找不到 `requirements.txt`） | 这个环境要读克隆下来的仓库里的 requirements.txt，但旧版先建环境、后克隆。已改为建这个环境前先克隆 |
| GPU 较慢的机器上 officehome / openood / privacymeter 超时 | 默认 2400 秒是按 Blackwell 的速度定的。L4 上要用 `execute_timeout=7200`（第 8 章） |
| 打分时 `FileNotFoundError`（找不到某个任务） | 没加 `--lite`，脚本默认要求 18 个任务都存在 |
| 打分时 scikit-learn 导入失败 | pyarrow 26 要求 NumPy ≥ 2，而 numpy 固定在 1.26.4。装 `pyarrow<26`（6.9） |
| 用 `pgrep -f` / `pkill -f <关键字>` 时匹配到（或杀掉了）执行命令的 shell | 关键字出现在这条命令自己的命令行里。改用 `pgrep -x python` 再查看 `/proc/<pid>/cmdline`，或者直接用 PID 去 `kill` |

---

## 7. 待办 / 未决事项

- [ ] 是否修复 ai_scientist_v2 跳过想法生成的问题，并用完整流程重跑 opacus
- [ ] 用 `val_ratio=0.05` 重测 opacus baseline，把训练数据增加的影响和算法改进分开
- [ ] 堵住隐藏 test 集可被读取的漏洞（把数据移出 val 阶段可访问的位置，或者在检查里拦截数据读取）
- [ ] ART：是否构建能用 Blackwell GPU 的 TF 环境（会改动锁定的版本）；同型号 CPU 之间约 0.01 的差距原因不明（3.3）
- [ ] openood 的训练不确定：是否在 `train_eval_baseline.py` 里设置 `cudnn.deterministic`（会改动任务代码）
- [ ] 迁移并测试其余 10 个非 Lite 任务的环境
- [x] 提交 harness 代码与本文档初版（分支 `repro/bedrock-uv-sandbox`：`83030f4`、`7ea0e4f`）
- [x] 提交沙盒、耗时记录、复现脚本、新 baseline 和第 8 章的 `setup.py` 改动（`6ff6b53`、`74ff570`）
- [x] ai_scientist_v2 的 Lite 正式测试与打分（5.1）
- [x] 检查正式测试中各任务最佳步的代码快照（5.2）：没有读取数据，也没有改动评测代码；domainbed 利用了题目先验，privacymeter 利用了评测只做非自适应攻击这一点
- [ ] privacymeter：是否在评测中加入自适应攻击（例如对查询做扰动），防止基于输入哈希的路由式防御拿到接近 0 的 AUC_gap（这属于修改 benchmark，需要决定）
- [x] 提交 `compute_agent_metrics.py` 的 `--lite`/`--tasks`、`setup.py` 的 `pyarrow<26`、`run_lite_queue.sh` 的 `AGENT`/`QUEUE_SLOTS`、README 的说明和本文档
- [x] 合并 g7e.24xlarge 上的改动：数据缓存、按单卡重测的 baseline、停止跟踪运行产物。两版 baseline 都保留（3.5）
- [ ] domainbed（ColoredMNIST）利用题目先验大幅提升（见 5.2）：要不要通过改任务说明来限制（这属于修改 benchmark，需要决定）
- [ ] `setup.py` 建 uv 环境时是否固定只用 uv 管理的 Python（`--python-preference only-managed`），避免用到宿主机的解释器
- [x] autoresearch 的 Lite 正式测试、打分和快照检查（5.3）。opacus 在第 41 步中断，最终 test 和 `summary.json` 是手工补的
- [x] 把 g7e.12xlarge 上的工作目录更新到合并提交 `81a483c`。不再跟踪的文件用 `git restore --source=7765d6a --worktree -- docs/reproduction_logs benchmark_results_smoke metric_reports rebaseline_results` 恢复到了磁盘上
- [ ] autoresearch 的 pycil、openood 被测步超过 2400 秒（5.3）：要不要用 2400 秒的标准超时重跑这两个任务，或者在比较时注明
- [ ] ai_scientist_v2 × GPT-6 Astra（5.5）结束后：统计超时步数；打分（g7e.12xlarge 版为主，同时给出 g7e.24xlarge 版和原始 baseline 的结果）；检查最佳步的代码快照；和 Opus（5.1）对比；最后用 `git checkout ml_tasks` 恢复顶层 baseline
- [ ] 在缓存为空的机器上验证 `setup.py --data-cache` 的自动下载（目前只有 MNIST 实际走过这条路径）
- [ ] 清理 `workspace/*__rebaseline_*` 副本（目前按要求保留）和 g7e.24xlarge 上的 `/mnt/nvme/data/_downloads/`（2.1 GB）
- [ ] 超时协议：默认 2400 秒在 L4 上不够用（officehome 连 baseline 都跑不完）。跨硬件比较时，是按 GPU 速度换算超时，还是统一要求某个最低算力，需要决定
- [x] Claude Code（`/goal` 模式）的 Lite 测试与打分（5.4）
- [ ] officehome 实际只有 2 类（5.4）：每个域下只有 `train/`、`val/` 两个子目录，DomainBed 把它们当成类别，所以模型学的是"文件在不在 `val/` 下"，不是 65 类物体。是否改为按类别目录组织数据（例如合并每个域的 `train/` 和 `val/`），需要决定。这会改变 baseline 和所有已有结果
- [ ] g7e.24xlarge 的 `/mnt/nvme`（实例存储 RAID）不在 `/etc/fstab` 里，重启后要手动 `sudo mount /dev/md0 /mnt/nvme`。2026-11-08 这台机器被外部自动更新加重启过一次

---

## 8. 第二台机器复现：4 × L4（2026-11-03 08:20 ～ 12:50）

按第 6 章的流程在另一台机器上从头走了一遍：建环境 → 重测 8 个 baseline → 8 个 Lite 任务的冒烟测试。第一台机器的正式测试（5.1）当时正在 EFS 的仓库目录里运行，为了不和它冲突（`workspace/`、`rebaseline_results/`、`benchmark_results_smoke/` 和 baseline 文件），整个仓库被复制到本机本地盘 `~/FML-bench`（不含 `workspace/` 和结果目录），所有操作都在这个副本里完成。环境在 `~/fml_envs`。结果和日志都复制到了 EFS 仓库的 `docs/reproduction_logs/l4/` 下（见下文各节）。本章的"第一台"指 g7e.12xlarge，"本机"指这台 L4 机器。

### 8.1 机器环境

| 项目 | 第一台（第 1 章） | 本机 |
|---|---|---|
| GPU | 2 × RTX PRO 6000 Blackwell（96 GB，sm_120） | **4 × NVIDIA L4（24 GB，sm_89）** |
| 驱动 / CUDA | 580.126.16 / 13.0 | 595.71.05 / 13.2 |
| CPU / 内存 | Intel Xeon 8559C，48 核 / 499 GB | **AMD EPYC 7R13，96 核** / 363 GB |
| 系统 | Amazon Linux 2023（kernel 6.1） | Amazon Linux 2023（kernel 6.12） |
| uv / bwrap | 0.11.14 / 0.10.0 | 0.12.10 / 0.10.0 |
| conda / pip | 有 | **都没有** |
| AWS | 实例角色 | 实例角色 `my-bedrock-use-role`，同一账号 |

### 8.2 环境（第 1 步）

结论：**uv 方案可以直接使用**，但 `setup.py` 有 3 处要改（已改并提交）：

1. **启动检查无条件要求 `conda` 和 `pip`。** 本机两者都没有，`setup.py` 直接退出。已改为只有要建的环境里有 conda 环境时才检查 conda/pip；uv 是必需的。
2. **privacy_meter 环境的构建顺序。** 这个环境要读 `workspace/Privacy_privacymeter/ml_privacy_meter/requirements.txt`，但 `setup.py` 先建环境（第 1 阶段）、后克隆仓库（第 2 阶段）。在第一台机器上没出问题，是因为当时仓库已经克隆好了；在全新机器上会失败。已改为建这个环境前先克隆（`clone_privacymeter()`，可重复调用）。
3. **officehome 数据下载依赖宿主机的 `pip install gdown`。** 已改为 `uvx gdown <id> -O office_home.zip`。本次的数据是复制来的，这段代码**没有实际运行过**。

用 `UV_PYTHON_PREFERENCE=only-managed` 构建，privacy_meter 用的是 uv 自带的 Python 3.12.14（第一台是 miniconda 的 3.12.9）。其他环境：fmlbench 3.11.6，opacus/pycil/usb/domainbed/art 3.10.21，openood 3.9.25。

与第一台机器 `rebaseline_results/<TASK>/pip_freeze.txt` 的对比：

- opacus、pycil、usb、privacy_meter、openood 的包版本完全一致（只有 pip 自身是 26.3 对 26.2.2）。
- art 只有 `google-auth` 2.62.1 对 2.62.0。
- domainbed 与锁定 `huggingface_hub<1.0` 之后（officehome 那次）的记录一致。差别只是第一台多了几个包（httpx2、httpcore2、tomli、truststore），是先装了 hub 2.x 又降级时留下的。

所有 torch 环境都能在 L4 上做 GPU 计算（torch 2.7.1+cu128 带 sm_86/sm_90 的 kernel，sm_89 可以直接用）。ART 的 TF 2.10.1 看不到 GPU，和第一台一样只用 CPU。沙盒检查（6.6）全部通过：8 个环境都能启动，GPU 数量正确，`CUDA_VISIBLE_DEVICES=""` 下没有 GPU，ml_tasks 只读、workspace 可写，`~/.ssh`、EFS、conda 都不可见，沙盒内能调用 Bedrock。

数据：用 `setup.py --skip-data` 克隆代码库后，从 EFS 模板复制原始数据（md5 与第 2 章一致），再运行一次不带 `--skip-data` 的 `setup.py`。opacus、openood、privacymeter 由 setup 自己解压、生成 pkl 并提交；pycil 和 usb 手动提交。**文档里原来的手动提交命令（`git add -A data`）是错的**：`data/` 被这两个仓库的 `.gitignore` 忽略，必须用 `add -f`（6.4 已改）。最终 8 个代码库都是干净的。与 EFS 上的模板比较 git 文件树：7 个完全相同；privacymeter 的两个 pkl 字节不同，原因是 pickle 里存了数据集对象的绝对路径（`root`），数据和标签逐项相同。

### 8.3 Baseline 重测（第 2 步）

`rebaseline_task.sh` 并行运行。分配：GPU0 officehome；GPU1 usb + domainbed；GPU2 openood + opacus；GPU3 pycil + privacymeter；ART 只用 CPU。`compare_baseline.py` 已扩展，现在同时输出原始值（`original/`）、当前顶层的 baseline 和本机结果。下表的"12xlarge"是当时顶层的 g7e.12xlarge 版，g7e.24xlarge 版见 3.2。

| 任务 | 指标（方向） | val：原始 / 12xlarge / **L4** | test：原始 / 12xlarge / **L4** | L4 耗时（val/test） | 第一台耗时 | 结论 |
|---|---|---|---|---|---|---|
| Privacy_opacus | test_acc（↑） | 0.5894 / 0.5881 / **0.5878** | 0.5877 / 0.5870 / **0.5854** | 2m14s / 7s | 1m10s | 一致 |
| Continual_Learning_pycil | avg_incremental_acc（↑） | 0.5911 / 0.5925 / **0.6075** | 0.5952 / 0.5944 / **0.6034** | 21m36s / 21m02s | 8m54s | 接近（+1～1.5 点） |
| Privacy_privacymeter | AUC_gap（↓） | 0.3339 / 0.3242 / **0.3282** | 0.3214 / 0.3182 / **0.3178** | 40m39s / <1s | 6m15s | 一致 |
| Robustness_openood | auroc（↑） | 0.8758 / 0.8804 / **0.8697** | 0.9302 / 0.9212 / **0.9106** | 32m07s / 9s | 10m25s | 接近（−1 点） |
| Data_Efficiency_usb | test_acc（↑） | 0.0757 / 0.0827 / **0.0723** | 0.0790 / 0.0861 / **0.0829** | 14m39s / 6s | 11m11s | 接近（原值本身约 8%） |
| Generalization_domainbed_officehome | avg_acc（↑） | 0.8832 / 0.9011 / **0.8942** | 0.8629 / 0.8570 / **0.8599** | **50m19s** / <1s | 10m55s | 接近 |
| Generalization_domainbed | in_acc（↑） | 0.2792 / 0.2393 / **0.2626** | 0.2868 / 0.2501 / **0.2706** | 4m48s / <1s | 3m32s | 介于原始值和 Blackwell 之间 |
| Robustness_and_Reliability_art（CPU） | defense_score（↑） | 0.4720 / 0.3005 / **0.2913** | 0.4774 / 0.2921 / **0.2895** | 5m46s / 7s | 4m51s | 与 Blackwell 接近，与原始值仍差很多 |

结果目录：`docs/reproduction_logs/l4/rebaseline_results/<TASK>/`（在本机上是 `~/FML-bench/rebaseline_results/<TASK>/`）。setup 日志：`docs/reproduction_logs/l4/setup_*.log`。

**分析**

- 8 个任务都跑通了，与两版 Blackwell baseline 相比，全部一致或接近。
- **domainbed 补充了第 3 章的结论。** 每台机器上都只有一个种子，结果是确定性的（第一台跑两次完全相同，都是 0.239320）。本机用同样的 torch 2.7.1，得到 0.2626，落在原始值（torch 1.12.1，0.279）和 Blackwell（0.239）之间。所以"最可能是 torch 版本"的判断不完整：**同一个 torch 版本只换 GPU，结果也会变化约 0.02**。这个任务对硬件和 kernel 层面的数值差异本身就很敏感。
- **ART 进一步支持 3.4 的判断。** 本机 CPU 的各项分指标和第一台 CPU 几乎一样（clean_acc 0.9555 对 0.9558，poison_acc 0.816 对 0.813），defense_score 的差距约 0.01，可能来自 AMD 和 Intel 的浮点差异。两台都离原始的 0.472 很远，"原 baseline 用的是 GPU 或另一版代码"的可能性更大了。
- **速度**：上表的 L4 耗时是在多个任务共用 GPU 的情况下测的，和第一台机器的并发情况也不一样，所以不能直接比较。为此又做了一次对照实验：每个任务单独占一张 L4 重跑 val（`rebaseline_results/<TASK>__solo/`），同时用 `nvidia-smi dmon` 每 5 秒采样一次（`docs/reproduction_logs/l4/solo/dmon*.txt`）。除 privacymeter 外，重跑的 val 结果与第一次 L4 测量逐字节相同。privacymeter 开了 `cudnn.benchmark`，卷积算法的选择不固定，结果是 0.3287 对 0.3282。

  | 任务 | 第一台（并发情况） | L4 第一次（并发情况） | **L4 单卡** | **L4 单卡 / g7e.24xlarge 单卡** | L4 单卡时：SM 利用率 / 显存带宽 / 功耗 / SM 时钟 | 瓶颈 |
  |---|---|---|---|---|---|---|
  | officehome | 10m55s（单独） | 50m19s（单独） | 50m19s | **4.7 倍** | 没有用 dmon 记录；baseline 运行时抽查一次是 94% | GPU 算力（ResNet-50，224×224，fp32/TF32） |
  | privacymeter | 6m15s（单独） | 40m39s（前 21 分钟和 pycil 共用） | 25m21s | **4.1 倍** | 96% / 78% / 70.6 W（97% 的时间碰到功耗上限）/ 1182 MHz | GPU 算力（4 个 WRN-28-2，每个 epoch 做一次完整评测） |
  | openood | 10m25s（和 usb 共用） | 32m07s（和 opacus 共用 2 分钟） | 30m55s | **4.5 倍** | 97% / 76% / 70.8 W（95%）/ 1325 MHz | GPU 算力（ResNet-18，32×32，第一层不降采样，100 epoch） |
  | opacus | 1m10s（单独） | 2m14s（和 openood 共用） | 2m31s | 1.9 倍 | 41% / 44% / 47 W / 1903 MHz | 混合：每样本梯度加上 CPU 开销（2 个 worker、隐私预算计算） |
  | pycil | 8m54s（和 domainbed 共用 3.5 分钟） | 21m36s（和 privacymeter 共用全程） | 12m19s | 1.4 倍 | 50% / 25% / 57 W / 1819 MHz | CPU、数据加载、Python 开销（ResNet-32 只有 0.46M 参数，ColorJitter 在 CPU 上做） |
  | usb | 11m11s（和 openood 共用） | 14m39s（和 domainbed 共用 5 分钟） | 14m28s | 1.6 倍 | 40% / 27% / 55 W / 1831 MHz | CPU 和数据加载（2 个 worker；有标注数据只有 200 张，每 3 步就重建一次 worker，全程约 5000 次；RandAugment 在 CPU 上做） |
  | domainbed | 3m32s（和 pycil、ART 共用 CPU） | 4m48s（和 usb、ART 共用） | **1m49s** | 1.0 倍 | 46% / 34% / 63 W / 1934 MHz | CPU 和 Python（MNIST 小 CNN，每 100 步对 7 万张图做一次评测，8 个 loader × 8 个 worker） |
  | ART（CPU） | 4m51s | 5m46s | — | 1.4 倍 | — | CPU（Intel Xeon 8559C 对比 AMD EPYC 7R13；L4 机器上和 GPU 任务同时运行） |

  比值的分母是 g7e.24xlarge 上单独占一张卡的耗时（3.2）。第一台有并发，不适合做分母。

  结论：**这种差距是合理的，原因有两层。**
  1. **任务本身的瓶颈不同。** 受 GPU 算力限制的任务（officehome、privacymeter、openood）慢 4.1～4.7 倍。L4 的 fp32/TF32 算力和显存带宽本来就比 RTX PRO 6000 低好几倍，而且功耗上限只有 72 W。这几个任务 95% 以上的时间都碰到功耗上限（`SW Power Cap` 处于触发状态），SM 时钟从 2040 MHz 降到 1180～1330 MHz。受 CPU、数据加载或 Python 开销限制的任务（pycil、usb、domainbed），L4 的 GPU 只用到 40～50%，换更快的 GPU 也省不了多少时间，所以只慢 1.0～1.6 倍。opacus 介于两者之间。
  2. **并发的影响。** 第一次 L4 测量时，privacymeter（6.5 倍 → 单卡 4.1 倍）和 pycil（2.4 倍 → 单卡 1.4 倍）都和其他任务共用 GPU，被拖慢了很多。domainbed 对 CPU 竞争很敏感：两台机器第一次测量时都和占满全部 CPU 的 ART（TF CPU）同时运行，单独跑时只要 1m49s，和 g7e.24xlarge 单卡的 1m45s 相当，第一台的 3m32s 是被 CPU 竞争拖慢的。
  - 对冒烟测试和正式测试的影响：超时风险集中在 officehome、privacymeter、openood 这三个受 GPU 算力限制的任务上，同一张卡上再加任务会更慢。在 L4 上，这三个任务最好每个单独占一张卡。
- **采用**：L4 的结果没有写进 `ml_tasks/`。冒烟测试的提示词里用的是 g7e.12xlarge 版 baseline，以便和 Blackwell 上的冒烟测试比较。本机结果与两版 Blackwell baseline 的差距都在可接受范围内。如果要以 L4 作为正式评测平台，需要另外决定是否换成本机的值。

### 8.4 冒烟测试：ai_scientist_v2 × 8 个 Lite 任务 × Opus 5.5（09:25 ～ 12:49）

```bash
SMOKE_TASKS="generalization_officehome|0 robustness_openood|1 generalization|1 privacy_privacymeter|2 privacy_opacus|2 continual_learning_pycil|3 data_efficiency_usb|3 robustness_and_reliability_art|" \
EXTRA_OVERRIDES="agent.ai_scientist_v2.execute_timeout=7200" \
  setsid nohup scripts/reproduction/run_smoke_lite.sh docs/reproduction_logs/l4/smoke_lite8 > docs/reproduction_logs/l4/smoke_lite8/smoke.out 2>&1 &
```

- 在 Blackwell 冒烟测试的 7 个任务之外加上了 opacus，每个任务 3 步，全部在沙盒中运行，输出到 `~/FML-bench/benchmark_results_smoke/`（副本在 `docs/reproduction_logs/l4/benchmark_results_smoke/`），`--workspace-label smoke`。
- **超时改为 7200 秒**（只在命令行覆盖，最终 test 用的是同一个值）。原因：officehome 的 baseline 在 L4 上单独跑一步就要 50 分钟，按默认的 2400 秒，第一步就必然超时。7200 秒大致相当于"L4 比 Blackwell 慢 3 倍，再留一些余量"。
- 分配：officehome 单独占 GPU0（最重）；GPU1 openood + domainbed；GPU2 privacymeter + opacus；GPU3 pycil + usb；ART 只用 CPU。依据是 8.3 里的耗时。

**结果**：8 个任务全部完成 3 步和最终 test，共 40 条命令，全部 returncode 0。没有 `conda` 报错，没有超时，没有 Bedrock 鉴权、限流或拒答，工作区副本都已清理（6.7 的判断标准全部满足）。

| 任务 | baseline（val / test） | Blackwell 冒烟（val / test） | **L4 冒烟（val / test）** | 每条命令耗时（秒，3 步 + 最终重训练） | 超过 2400 秒 | 显存峰值 | 内存峰值（PSS） | token |
|---|---|---|---|---|---|---|---|---|
| opacus | 0.588 / 0.587 | （Blackwell 冒烟 test 0.631） | **0.629 / 0.618** | 427 / 638 / 636 / 473 | 0 | 11.9 GB | 2.2 GB | 2.4 万 |
| pycil | 0.593 / 0.594 | 0.613 / 0.623 | **0.610 / 0.611** | 1238 / 1179 / 1140 / 1190 | 0 | 0.7 GB | 1.9 GB | 3.4 万 |
| usb | 0.083 / 0.086 | 0.070 / 0.079 | **0.109 / 0.107** | 1318 / 1327 / 1784 / 1355 | 0 | 1.4 GB | 1.4 GB | 3.6 万 |
| domainbed | 0.239 / 0.250 | 0.693 / 0.705 | **0.728 / 0.740** | 799 / 824 / 624 / 727 | 0 | 0.7 GB | 2.8 GB | 15.5 万 |
| officehome | 0.901 / 0.857 | 0.909 / 0.874 | **0.911 / 0.874** | 3022 / 3011 / 3001 / 2994 | **4** | 9.9 GB | 7.5 GB | 16.2 万 |
| privacymeter（越低越好） | 0.324 / 0.318 | 0.037 / 0.039 | **0.057 / 0.064** | 2607 / 1528 / 1520 / 1522 | **1** | 4.8 GB | 2.9 GB | 4.9 万 |
| ART（CPU） | 0.301 / 0.292 | 0.931 / 0.931 | **0.967 / 0.966** | 373 / 222 / 370 / 224 | 0 | — | 2.9 GB | 5.7 万 |
| openood | 0.880 / 0.921 | 0.914 / 0.986 | **0.915 / 0.986** | 2497 / 2027 / 1839 / 1841 | **1** | 1.5 GB | 2.3 GB | 2.2 万 |

（baseline 列是 g7e.12xlarge 版，也就是冒烟测试时提示词里的值。评测命令只要几秒，表中没有列出。）

- **整体**：
  - 总耗时 3 小时 24 分钟，由 officehome 决定（4 次训练，每次约 50 分钟）。其余任务都在 2 小时内完成。
  - token 共约 54 万，和 Blackwell 上 7 个任务的冒烟测试（约 52 万）相当，按官方价格估算约 $5～6。
  - 单卡显存峰值：GPU0 9.9 GB，GPU1 2.1 GB，GPU2 13.3 GB，GPU3 2.1 GB，都没有超过 L4 的 24 GB。
- **按标准协议（2400 秒）会有 6 条命令超时**：officehome 全部 4 条（包括最终重训练），privacymeter 第 1 步（和 opacus 共用 GPU 时），openood 第 1 步。officehome 的每一步都和 baseline 一样约 50 分钟，所以不是 agent 让代码变慢了，纯粹是 L4 算力不够。
- 指标量级和 Blackwell 冒烟一致：pycil、officehome、openood 接近或略高于 baseline；domainbed、ART、privacymeter 大幅提升。usb 这次从 0.083 提升到 0.109（Blackwell 冒烟时没有超过 baseline）。opacus 和 Blackwell 上 opacus 的冒烟测试相近。
- **privacymeter**：最终 test 没有违反 TPR 约束（`constraint_violated: False`），test 准确率 0.755。
- **domainbed（需要注意，和 Blackwell 冒烟相同）**：最优的第 2 步仍然是对红绿通道做置换对称（训练和预测时都对两种通道顺序的 logits 取平均），另外加了权重 EMA 和小损失截断（针对 25% 的标签噪声）。还是在利用 ColoredMNIST 的构造先验。
- **ART（和 Blackwell 冒烟不同）**：最优的第 2 步只改了 `dp_instahide_trainer.py`（`model.py` 和 `split_config.json` 只改了空白）。做法是：先用 2 个 epoch 预训练，再在倒数第二层的特征上按类别做谱签名（spectral signatures）和 2 簇激活聚类（activation clustering），并按标签一致性过滤可疑样本，然后把权重重置为初始值，在过滤后的数据上做 DP-InstaHide 训练。**没有清空图像外框，也没有改写 `predict`**，用的是已发表的通用后门防御方法，不依赖触发器的位置。只有 3 步，样本太少，不能据此判断 agent 的倾向。

日志和资源数据：`docs/reproduction_logs/l4/smoke_lite8/`（不在版本库里，见附录）。

---

## 附：相关文件

和上游一样，运行产物不纳入版本库：`rebaseline_results/`、`benchmark_results/`、`benchmark_results_smoke/`、`metric_reports/`、`docs/reproduction_logs/` 都在 `.gitignore` 里，只存在于产生它们的机器上。版本库里只有 `ml_tasks/<TASK>/baseline_results/` 下的 baseline 文件。

| 路径 | 说明 |
|---|---|
| `ml_tasks/<TASK>/baseline_results/{val,test}_info.json` | 采用的 baseline：g7e.24xlarge 版（3.5） |
| `ml_tasks/<TASK>/baseline_results/g7e12xlarge/` | g7e.12xlarge 版 baseline，第 5 章用的（3.5） |
| `ml_tasks/<TASK>/baseline_results/original/` | 仓库原有的 baseline 文件 |
| `scripts/reproduction/rebaseline_queue.sh` | 每张 GPU 一个串行队列地重测全部 baseline，ART 最后单独跑，并记录 `nvidia-smi dmon` |
| `scripts/reproduction/rebaseline_task.sh` | 单任务 baseline 重测（val → test，uv 环境） |
| `scripts/reproduction/compare_baseline.py` | 输出原始值、顶层 baseline 和重测值的对比 |
| `sandbox/run_in_bwrap.sh` | 沙盒入口 |
| `scripts/reproduction/resource_monitor.py` | 按工作区副本统计各任务的显存和内存（PSS）；`--summary` 输出峰值 |
| `scripts/reproduction/run_smoke_lite.sh` | Lite 任务的冒烟测试（6.7）。默认是 7 个任务 / 2 张卡，可用 `SMOKE_TASKS` / `EXTRA_OVERRIDES` 自定义 |
| `scripts/reproduction/run_lite_queue.sh` | Lite 任务的正式测试，按位置排队；`AGENT`、`QUEUE_SLOTS` 可换 agent 和排队方式（6.8） |
| `compute_agent_metrics.py` | 打分，输出原始指标、NI 和过程指标三张表（6.9） |
| `docs/lite_results_by_method.csv` | ai_scientist_v2、autoresearch、Claude Code 三种方法的 Lite 结果汇总。每行一个任务，每个方法两列：NI 和 baseline test→agent test。前两个方法按 g7e.12xlarge 版 baseline，Claude Code 按 g7e.24xlarge 版（5.1、5.3、5.4） |
| `docs/claude_code_agent_design.md`、`scripts/claude_code/` | Claude Code 作为 agent 的方案、沙盒、`fml-val` 等工具、最终 test、审计和汇总（5.4） |

以下运行产物不在版本库里：

| 机器 | 路径 | 说明 |
|---|---|---|
| g7e.24xlarge | `rebaseline_results/<TASK>__g7e24/` | g7e.24xlarge 版 baseline 的原始输出（val/test 结果、日志、`run_info.txt`、`pip_freeze.txt`） |
| g7e.24xlarge | `rebaseline_results/Robustness_openood__g7e24_rerun2/`、`rebaseline_results/Robustness_and_Reliability_art__g7e24_48cpu/` | openood 同卡重跑、ART 限制为 48 个 CPU 的对照（3.3） |
| g7e.24xlarge | `docs/reproduction_logs/g7e24/` | setup 日志，以及 `rebaseline/` 下的队列记录、`summary.txt`、`dmon.txt` |
| g7e.12xlarge | `rebaseline_results/<TASK>[__rerun2]/` | g7e.12xlarge 版 baseline 的原始输出和重跑对照 |
| g7e.12xlarge | `benchmark_results/<agent>/<Task>/` | 正式测试结果：主运行目录（`summary.json`、`step_snapshots/`）和 `*_final_test` 目录 |
| g7e.12xlarge | `benchmark_results/gpt6astra/ai_scientist_v2/<Task>/` | 5.5 的结果（GPT-6 Astra），目录结构同上 |
| g7e.12xlarge | `metric_reports/ai_scientist_v2_opus55_lite/`、`metric_reports/autoresearch_opus55_lite/` | 5.1、5.3 的三张表（CSV）和嵌入缓存 |
| g7e.12xlarge | `benchmark_results_smoke/` | 冒烟测试结果 |
| g7e.24xlarge | `/mnt/nvme/cc_runs/<TASK>/formal/` | 5.4 的运行目录：`task/`（工作区、`runs/` 运行记录、`scratch/` 笔记和 `FINAL_REPORT.md`）、`claude_config/`（会话记录）、`final/`（最终 test 和 `report.json` 审计结果） |
| g7e.24xlarge | `/mnt/nvme/data_agent/<TASK>/` | 5.4 的掩码数据 |
| g7e.24xlarge | `benchmark_results/claude_code/<Task>/formal/`、`metric_reports/claude_code_opus55_lite/` | 5.4 的 harness 格式结果和打分表（`table_ni_baseline_versions.csv` 是三版 baseline 的 NI 对照） |
| g7e.12xlarge | `docs/reproduction_logs/` | agent 运行日志和资源采样：5.1 的 opacus 在 `aisv2_opacus_full.log`，其余 7 个任务在 `full_lite7/`，5.3 在 `full_lite8_autoresearch/`，冒烟测试在 `smoke_lite7/`；L4 的结果和日志复制在 `l4/` 下 |
