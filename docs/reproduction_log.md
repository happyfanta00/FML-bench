# FML-bench 复现实验记录

本文档记录在本机上复现 FML-bench 的结果：环境适配、baseline 重测、harness 改造（Bedrock / uv / 沙盒）、agent 测试结果与快照检查（第 5 章），以及从零复现这些结果的完整流程（第 6 章）。

- 分支：`repro/bedrock-uv-sandbox`，基于 `d336651`
- 记录时间：2026-11-02 ～ 2026-11-09（第 8 章为第二台机器 4 × L4 上的复现）

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

## 5. 实验结果

### 5.1 ai_scientist_v2 × FML-bench-Lite × Opus 5.5

**设置**

- agent：ai_scientist_v2，参数用 `configs/agents/ai_scientist_v2.yaml` 的默认值。
- 模型：Bedrock `us.anthropic.claude-opus-5-5`。
- 每个任务跑 100 步，全部在沙盒（4.3）中运行。
- baseline：本机重测值（第 3 章）。
- 超时：opacus 用默认的 2400 秒；其余 7 个任务每两个共用一张卡，用 4800 秒。按 2400 秒的标准协议，只有 privacymeter 的 1 步会超时，而且不是最佳步，不影响结果。
- 运行时间：2026-11-02 ～ 11-05。复现方法见第 6 章。
- **已知问题**：ai_scientist_v2 的想法生成没有运行。`_generate_ideas` 找不到 `ml_tasks/<TASK>/theaiscientist/seed_ideas.json` 就直接跳过，而所有任务都没有这个文件。按决定没有修复，所以结果相当于纯树搜索。

**结果**（NI 的定义和计算方法见 6.9）

| 任务 | 指标 | baseline val / test | agent 最佳 val（被测步） | agent test | **NI** | NI（原始 baseline） |
|---|---|---|---|---|---|---|
| pycil | avg_incremental_acc ↑ | 0.5925 / 0.5944 | 0.7562（89） | 0.7571 | **0.1627** | 0.1619 |
| usb | test_acc ↑ | 0.0827 / 0.0861 | 0.1677（64） | 0.1724 | **0.0863** | 0.0934 |
| domainbed（ColoredMNIST） | in_acc ↑ | 0.2393 / 0.2501 | 0.7404（79） | 0.7527 | **0.5026** | 0.4658 |
| officehome | avg_acc ↑ | 0.9011 / 0.8570 | 0.9203（17） | 0.8782 | **0.0212** | 0.0153 |
| openood | auroc ↑ | 0.8804 / 0.9212 | 0.9204（55） | 0.9928 | **0.0716** | 0.0626 |
| opacus | test_acc ↑ | 0.5881 / 0.5870 | 0.6911（88） | 0.6752 | **0.0882** | 0.0875 |
| privacymeter | AUC_gap ↓ | 0.3242 / 0.3182 | 0.0011（57） | 0.0008 | **0.6347** | 0.6411 |
| ART | defense_score ↑ | 0.3005 / 0.2921 | 0.9755（21） | 0.9740 | **0.6820** | 0.4966 |
| **MEAN（Lite）** | | | | | **0.2812** | 0.2530 |

- **被测步**：最终 test 用的是最早达到最佳 val 的那一步的代码。val 相同时不替换：同一阶段内取第一个最大值，跨阶段时必须严格更好。
- **原始 baseline 对照**：改用仓库原始的 baseline（`baseline_results/original/`）后，均值降到 0.253。差距主要来自 ART：本机 baseline 是 0.29，原始的是 0.48（第 3 章）。
- **约束都满足**：
  - privacymeter 的两个 TPR 都是 0，test 准确率 0.7515；
  - opacus 的 ε 是 7.9997，在 8 的预算内。
- **openood 的 test 高于 val**：这是任务设计如此，val 用 CIFAR-100 当 OOD 数据，test 用 SVHN。
- **除去不能代表通用能力的任务**（见 5.2）：去掉利用题目先验的 domainbed，其余 7 个任务的平均 NI 约为 0.25；再去掉利用评测弱点的 privacymeter，约为 0.19。
- **完整表格**：三张表（原始指标、NI、过程指标）的 CSV 在 `metric_reports/ai_scientist_v2_opus55_lite/`。结果目录是 `benchmark_results/ai_scientist_v2/<Task>/`，每个任务有一个主运行目录（含 `summary.json` 和 `step_snapshots/`）和一个 `*_final_test` 目录。

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

### 5.3 autoresearch × FML-bench-Lite × Opus 5.5（进行中）

**设置**：和 5.1 相同，只是把 agent 换成 autoresearch（`configs/agents/autoresearch.yaml`），8 个任务的超时都用 4800 秒。2026-11-06 启动，结果待补。

**autoresearch 怎么工作**（Karpathy 的贪心爬山协议）：
- 每一步由 LLM 根据当前最好的代码和历史实验表提出一个想法，改代码后跑 val。
- val 严格优于当前最好值才保留，否则退回当前最好的代码；崩溃时最多调试 3 次。
- 最终 test 用的是最早达到最佳 val 的那一步的代码，即使它没有超过 baseline 也照样跑 test，这时 NI 被截断为 0。

---

## 6. 复现全流程

以下步骤从一台新机器开始，复现第 3 章的 baseline 和第 5 章的结果，最后算出 Normalized Improvement。6.1～6.7 已于 2026-11-03 在第二台机器（4 × L4）上完整走过一遍，见第 8 章。

### 6.1 前提条件

| 项目 | 要求 |
|---|---|
| 系统 | Linux x86_64，内核允许非特权用户命名空间（bubblewrap 需要） |
| GPU | NVIDIA，驱动需支持 CUDA 12.8。所有环境都是 torch 2.7.1+cu128，带 sm_75～sm_120 的 kernel，L4（sm_89）实测可用。下面的命令按 2 张卡分配，单卡峰值约 15 GB。**GPU 比 Blackwell 慢很多时要加大超时**：L4 上 officehome 的 baseline 单步就要 50 分钟（第 8 章） |
| 命令 | `git`、`wget`、`unzip`、`curl`、`uv`、`bwrap`、`nvidia-smi`、`python3`。`conda` 和 `pip` 只在要建 conda 环境时才需要，Lite 的 8 个任务都不需要 |
| AWS | 能在 us-west-2（或 `AWS_REGION` 指定的区域）用 Bedrock 调用 `us.anthropic.claude-opus-5-5`，需要 InvokeModel / InvokeModelWithResponseStream 权限 |
| AWS 凭证 | 三选一：EC2 实例角色（沙盒共享网络，能访问 IMDS）；`AWS_PROFILE`（沙盒会只读挂载 `~/.aws`）；`AWS_ACCESS_KEY_ID` 等环境变量（沙盒会传入） |
| 时间和费用 | 正式测试（6.8）在两张 Blackwell 上约 50 小时，ai_scientist_v2 共约 2530 万 token，按官方价格（输入 $4、输出 $20 每百万 token）估算约 $210，以 Bedrock 账单为准。冒烟测试（6.7）约 1.5 小时，约 50 万 token |

### 6.2 获取代码

```bash
git clone <仓库地址> FML-bench && cd FML-bench
git checkout repro/bedrock-uv-sandbox
```

复现所需的改动都已提交到这个分支：harness、沙盒、复现脚本和新 baseline（`83030f4`、`6ff6b53`、`74ff570`），以及之后的这些改动：

- `compute_agent_metrics.py`：`--lite` 和 `--tasks`，6.9 只给 Lite 打分时要用；
- `setup.py`：`'pyarrow<26'`。缺了它，打分时 scikit-learn 导入失败；
- `scripts/reproduction/run_lite_queue.sh`：`AGENT` 和 `QUEUE_SLOTS`，6.8 要用；
- `README.md` 和本文档。

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
- `UV_PYTHON_PREFERENCE=only-managed` 是 uv 自带的环境变量，不需要改 `setup.py`。本机建环境时没有设置它，所以 privacy_meter 用到了 miniconda 的 Python 3.12.9；设置后会用 uv 管理的 3.12.x，补丁版本可能不同。即使没设，沙盒的解释器挂载（4.3）也能处理这种情况。
- 环境建在本地盘上即可，本机是 NVMe 上的 `~/fml_envs`。

### 6.4 准备数据

两种方式选一种：

- **官方源**：直接运行上面的 `setup.py` 命令，但去掉 `--skip-data`。本机实测多伦多 CIFAR 服务器只有约 35 KB/s。
- **镜像**：按第 2 章「数据」中的来源下载，放到对应位置，然后运行一次不带 `--skip-data` 的 `setup.py`。它会跳过已有的数据，完成后续处理（比如生成 privacymeter 的 pkl）。之后：
  - pycil 和 usb 需要**手动提交数据**。预先放好数据会跳过 setup 里的提交步骤，而 executor 每步都会运行 `git reset --hard` 加 `git clean -fd`：
    ```bash
    for r in workspace/Continual_Learning_pycil/PyCIL workspace/Data_Efficiency_usb/Semi-supervised-learning; do
      git -C $r add -f data && git -C $r commit -m "add cifar-100 dataset for agent benchmark"
    done
    ```
    `data/` 在这两个仓库的 `.gitignore` 里，必须用 `add -f`（和 `setup.py` 的 `git_commit_data` 一致）。可以用 `git -C $r ls-files data | wc -l` 确认数据确实进了仓库。

最后确认 8 个任务的代码库都是干净的。下面的命令不应有任何输出，否则数据可能在第一步就被清掉。6.8 启动前和 6.9 打分前也要再检查一次：

```bash
for c in ml_tasks/*/config.json; do
  r=$(python3 -c "import json;print(json.load(open('$c'))['repo_dir'])")
  [ -d "$r" ] && git -C "$r" status --porcelain
done
```

officehome 第一次运行时会把 torchvision 的预训练权重下载到 `~/.cache`。沙盒可以写这个目录，网络也是共享的，所以不需要额外操作。

### 6.5 Baseline

- **硬件和第 1 章相同**（Blackwell 加本文的环境）：直接用分支里提交的新 baseline。
- **硬件不同**：建议先重测，用结果覆盖 `ml_tasks/<TASK>/baseline_results/{val,test}_info.json`。agent 提示词里的 baseline 和 6.9 的打分都读这两个文件。

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

每个环境都应该打印出 Python 版本，GPU 数量也应该正确。要确认 Bedrock 能否调用，可以在沙盒里用 `agents.llm.create_client(..., "AnthropicBedrock")` 发一条短消息。

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

- `p_agent` 和 `p_baseline` 都是 **test** 指标。`p_baseline` 读 `ml_tasks/<TASK>/baseline_results/test_info.json`。
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
  - `table_normalized_improvement.csv`：**每个任务的 NI 和 MEAN**，即 5.1 表中的 NI 列。
  - `table_process_metrics.csv`：12 个过程指标。
- **原始 baseline 对照列**：5.1 表的最后一列是手工算的，公式不变，只是把 `p_baseline` 换成 `ml_tasks/<TASK>/baseline_results/original/test_info.json` 里的值。
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
| 数据在第一步被删 | 数据没提交到任务代码库，见 6.4（注意要用 `add -f`） |
| `setup.py` 报 `missing required commands: conda, pip` | 旧版启动检查无条件要求 conda/pip，已改为只有要建 conda 环境时才检查 |
| 全新机器上建 privacy_meter 环境失败（找不到 `requirements.txt`） | 这个环境要读克隆下来的仓库里的 requirements.txt，但旧版先建环境、后克隆。已改为建这个环境前先克隆 |
| GPU 较慢的机器上 officehome / openood / privacymeter 超时 | 默认 2400 秒是按 Blackwell 的速度定的。L4 上要用 `execute_timeout=7200`（第 8 章） |
| 打分时 `FileNotFoundError`（找不到某个任务） | 没加 `--lite`，脚本默认要求 18 个任务都存在 |
| 打分时 scikit-learn 导入失败 | pyarrow 26 要求 NumPy ≥ 2，而 numpy 固定在 1.26.4。装 `pyarrow<26`（6.9） |
| 用 `pkill -f <关键字>` 停监控时，执行命令的 shell 也被杀掉 | 关键字匹配到了这条命令自己的命令行。改用具体的 PID 去 `kill` |

---

## 7. 待办 / 未决事项

- [ ] 是否修复 ai_scientist_v2 跳过想法生成的问题，并用完整流程重跑 opacus
- [ ] 用 `val_ratio=0.05` 重测 opacus baseline，把训练数据增加的影响和算法改进分开
- [ ] 堵住隐藏 test 集可被读取的漏洞（把数据移出 val 阶段可访问的位置，或者在检查里拦截数据读取）
- [ ] ART：是否构建能用 Blackwell GPU 的 TF 环境（会改动锁定的版本）
- [ ] 迁移并测试其余 10 个非 Lite 任务的环境
- [x] 提交 harness 代码与本文档初版（分支 `repro/bedrock-uv-sandbox`：`83030f4`、`7ea0e4f`）
- [x] 提交沙盒、耗时记录、复现脚本、新 baseline 和第 8 章的 `setup.py` 改动（`6ff6b53`、`74ff570`）
- [x] ai_scientist_v2 的 Lite 正式测试与打分（5.1）
- [x] 检查正式测试中各任务最佳步的代码快照（5.2）：没有读取数据，也没有改动评测代码；domainbed 利用了题目先验，privacymeter 利用了评测只做非自适应攻击这一点
- [ ] privacymeter：是否在评测中加入自适应攻击（例如对查询做扰动），防止基于输入哈希的路由式防御拿到接近 0 的 AUC_gap（这属于修改 benchmark，需要决定）
- [x] 提交 `compute_agent_metrics.py` 的 `--lite`/`--tasks`、`setup.py` 的 `pyarrow<26`、`run_lite_queue.sh` 的 `AGENT`/`QUEUE_SLOTS`、README 的说明、本文档和 5.1 的打分表
- [ ] domainbed（ColoredMNIST）利用题目先验大幅提升（见 5.2）：要不要通过改任务说明来限制（这属于修改 benchmark，需要决定）
- [ ] `setup.py` 建 uv 环境时是否固定只用 uv 管理的 Python（`--python-preference only-managed`），避免用到宿主机的解释器
- [ ] autoresearch 正式测试（5.3）结束后：统计超时步数，用 `--lite` 打分，检查最佳步代码快照，和 ai_scientist_v2 对比
- [ ] 清理 `workspace/*__rebaseline_*` 副本（目前按要求保留）
- [ ] 超时协议：默认 2400 秒在 L4 上不够用（officehome 连 baseline 都跑不完）。跨硬件比较时，是按 GPU 速度换算超时，还是统一要求某个最低算力，需要决定

---

## 8. 第二台机器复现：4 × L4（2026-11-03 08:20 ～ 12:50）

按第 6 章的流程在另一台机器上从头走了一遍：建环境 → 重测 8 个 baseline → 8 个 Lite 任务的冒烟测试。第一台机器的正式测试（5.1）当时正在 EFS 的仓库目录里运行，为了不和它冲突（`workspace/`、`rebaseline_results/`、`benchmark_results_smoke/` 和 baseline 文件），整个仓库被复制到本机本地盘 `~/FML-bench`（不含 `workspace/` 和结果目录），所有操作都在这个副本里完成。环境在 `~/fml_envs`。结果和日志都复制到了 EFS 仓库的 `docs/reproduction_logs/l4/` 下（见下文各节）。

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
- **domainbed 补充了第 3 章的结论。** 每台机器上都只有一个种子，结果是确定性的（第一台跑两次完全相同，都是 0.239320）。本机用同样的 torch 2.7.1，得到 0.2626，落在原始值（torch 1.12.1，0.279）和 Blackwell（0.239）之间。所以第 3 章"最可能是 torch 版本"的判断不完整：**同一个 torch 版本只换 GPU，结果也会变化约 0.02**。这个任务对硬件和 kernel 层面的数值差异本身就很敏感。
- **ART 进一步支持第 3 章的判断。** 本机 CPU 的各项分指标和第一台 CPU 几乎一样（clean_acc 0.9555 对 0.9558，poison_acc 0.816 对 0.813），defense_score 的差距约 0.01，可能来自 AMD 和 Intel 的浮点差异。两台都离原始的 0.472 很远，"原 baseline 用的是 GPU 或另一版代码"的可能性更大了。
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
- **采用**：`ml_tasks/*/baseline_results/` **没有改动**，冒烟测试的提示词里用的仍是第一台测出的 baseline，以便和 Blackwell 上的冒烟测试比较。本机结果与它的差距都在可接受范围内。如果要以 L4 作为正式评测平台，需要另外决定是否换成本机的值。

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

（评测命令只要几秒，表中没有列出。）

- **整体**：
  - 总耗时 3 小时 24 分钟，由 officehome 决定（4 次训练，每次约 50 分钟）。其余任务都在 2 小时内完成。
  - token 共约 54 万，和 Blackwell 上 7 个任务的冒烟测试（约 52 万）相当，按官方价格估算约 $5～6。
  - 单卡显存峰值：GPU0 9.9 GB，GPU1 2.1 GB，GPU2 13.3 GB，GPU3 2.1 GB，都没有超过 L4 的 24 GB。
- **按标准协议（2400 秒）会有 6 条命令超时**：officehome 全部 4 条（包括最终重训练），privacymeter 第 1 步（和 opacus 共用 GPU 时），openood 第 1 步。officehome 的每一步都和 baseline 一样约 50 分钟，所以不是 agent 让代码变慢了，纯粹是 L4 算力不够。
- 指标量级和 Blackwell 冒烟一致：pycil、officehome、openood 接近或略高于 baseline；domainbed、ART、privacymeter 大幅提升。usb 这次从 0.083 提升到 0.109（Blackwell 冒烟时没有超过 baseline）。opacus 和 Blackwell 上 opacus 的冒烟测试相近。
- **privacymeter**：最终 test 没有违反 TPR 约束（`constraint_violated: False`），test 准确率 0.755。
- **domainbed（需要注意，和 Blackwell 冒烟相同）**：最优的第 2 步仍然是对红绿通道做置换对称（训练和预测时都对两种通道顺序的 logits 取平均），另外加了权重 EMA 和小损失截断（针对 25% 的标签噪声）。还是在利用 ColoredMNIST 的构造先验。
- **ART（和 Blackwell 冒烟不同）**：最优的第 2 步只改了 `dp_instahide_trainer.py`（`model.py` 和 `split_config.json` 只改了空白）。做法是：先用 2 个 epoch 预训练，再在倒数第二层的特征上按类别做谱签名（spectral signatures）和 2 簇激活聚类（activation clustering），并按标签一致性过滤可疑样本，然后把权重重置为初始值，在过滤后的数据上做 DP-InstaHide 训练。**没有清空图像外框，也没有改写 `predict`**，用的是已发表的通用后门防御方法，不依赖触发器的位置。只有 3 步，样本太少，不能据此判断 agent 的倾向。

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
| `scripts/reproduction/run_lite_queue.sh` | Lite 任务的正式测试，按位置排队；`AGENT`、`QUEUE_SLOTS` 可换 agent 和排队方式（6.8） |
| `compute_agent_metrics.py` | 打分，输出原始指标、NI 和过程指标三张表（6.9） |
| `benchmark_results/<agent>/<Task>/` | 正式测试结果：主运行目录（`summary.json`、`step_snapshots/`）和 `*_final_test` 目录 |
| `metric_reports/ai_scientist_v2_opus55_lite/` | 5.1 的三张表（CSV）和嵌入缓存 |
| `benchmark_results_smoke/` | 冒烟测试结果 |
| `docs/reproduction_logs/` | agent 运行日志：5.1 的 opacus 在 `aisv2_opacus_full.log`，其余 7 个任务在 `full_lite7/`，5.3 在 `full_lite8_autoresearch/`。`*.log` 被 `.gitignore` 忽略，已完成的运行日志用 `git add -f` 提交，进行中的 5.3 还没提交；各目录下还有资源采样的 csv |
