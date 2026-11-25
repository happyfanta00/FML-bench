<p align="center">
  <a href="https://github.com/qrzou/FML-bench">
    <img src="docs/figs/main_logo_v3.png" style="height: 10em" alt="FML-bench" />
  </a>
</p>

<p align="center">
    <a href="http://www.apache.org/licenses/LICENSE-2.0">
        <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue">
    </a>
</p>

> **Note:** Paper link [arxiv:2605.17373](https://arxiv.org/abs/2605.17373). The previous version ([arxiv:2510.10472](https://arxiv.org/abs/2510.10472)) lives on the [`legacy`](https://github.com/qrzou/FML-bench/tree/legacy) branch.

A benchmark for automatic ML research agents on fundamental machine learning
problems. Agents are given a baseline codebase, evaluation harness, and task
description, and are asked to iteratively improve the baseline.

<p align="center">
    <img src="docs/figs/benchmark_pipeline_v3.png" style="width: 60em" alt="FML-bench pipeline" />
</p>


## Contents

- [Quick Start](#quick-start)
- [在一台全新的 g6.2xlarge 上运行 Codex goal 可视化（ColoredMNIST）](#在一台全新的-g62xlarge-上运行-codex-goal-可视化coloredmnist)
- [Setup](#setup)
- [Run an agent (example)](#run-an-agent-example)
- [Run with other models](#run-with-other-models)
- [Run on other tasks](#run-on-other-tasks)
- [FML-bench-Lite](#fml-bench-lite)
- [Remote GPU execution (Modal)](#remote-gpu-execution-modal)
- [Score a run](#score-a-run)
- [Available agents](#available-agents)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)


## Quick Start

Install everything (task repos, datasets, conda envs):

```bash
python setup.py
```

Run an agent (e.g. AI Scientist v2 on Causality_causalml with GPT-5.4):

```bash
conda activate fmlbench
export OPENAI_API_KEY="your_openai_api_key"
python run_agent_benchmark.py \
    --agent-config configs/agents/ai_scientist_v2.yaml \
    --task-config  configs/tasks/causality_causalml.yaml \
    --model gpt-5.4 --provider OpenAI \
    --output-dir results \
    agent.ai_scientist_v2.max_steps=100
```

See [Setup](#setup) and [Run an agent (example)](#run-an-agent-example) for full instructions (per-task setup, GPU selection, other providers).


## 在一台全新的 g6.2xlarge 上运行 Codex goal 可视化（ColoredMNIST）

这一节说明如何从一台刚创建的 AWS g6.2xlarge 实例开始，配置好全部环境，然后在 Mac 的浏览器里观看 Codex（goal 模式）改进 ColoredMNIST 任务的全过程。

整个流程分为两部分：第一部分（第 1～6 步）是一次性的准备工作，大约需要 20 分钟；第二部分（第 7～10 步）是每次实验都要做的事情，一次实验大约 1 小时。

g6.2xlarge 的配置是 1 张 NVIDIA L4 显卡（24 GB 显存）、8 个 CPU 核和 32 GB 内存，本节的所有参数都是按这个配置设定的。

### 实验是怎么进行的

- Codex 运行在一个隔离的沙盒（bubblewrap）里。沙盒里只能看到这个任务的代码、Python 环境、处理过的数据和那一张显卡，看不到这台机器上的其他文件。
- 隐藏测试集对应的图像在沙盒里被替换成了空白图，Codex 只能用验证集来判断自己的改进效果。
- Codex 每做一次尝试，都要调用一次 `fml-val` 命令：它会用 Codex 修改后的代码重新训练模型，并给出验证集上的分数。一次 `fml-val` 大约需要 3 分钟。
- 每次实验的预算是 1 小时，最多 20 次 `fml-val`。
- 浏览器页面把每一次 `fml-val` 显示为一个节点，可以看到 Codex 的改动思路、修改后的代码、训练曲线、它在两次尝试之间做了什么，以及当前最好模型在验证集样本上的预测结果。

### 第 1 步：创建实例

在 AWS 控制台创建实例时，按下面的要求选择：

| 项目 | 选择 |
|---|---|
| 实例类型 | g6.2xlarge |
| 区域 | us-west-2（Codex 通过这个区域的 Amazon Bedrock 调用模型） |
| 镜像（AMI） | Deep Learning OSS Nvidia Driver AMI GPU PyTorch（Amazon Linux 2023）。这个镜像已经装好了显卡驱动、AWS 命令行工具、git 和 tmux |
| 磁盘 | 至少 100 GB |
| IAM 角色 | 需要一个能调用 Amazon Bedrock 模型的角色（例如允许 `bedrock:InvokeModel` 和 `bedrock:InvokeModelWithResponseStream`），并且这个账号已经在 us-west-2 开通了 `openai.gpt-6-astra` 模型的访问权限 |
| 安全组 | 只需要开放 SSH（22 端口）。网页不需要对外开放端口，后面会通过 SSH 转发访问 |

实例启动后，用 SSH 登录（用户名是 `ec2-user`），先确认显卡可以使用：

```bash
nvidia-smi
```

输出里应该能看到一张 NVIDIA L4。

### 第 2 步：安装系统工具

```bash
# bubblewrap 是沙盒工具；tmux 用来让 Codex 在后台持续运行
sudo dnf install -y bubblewrap tmux git

# uv 用来创建 Python 环境
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

然后按照 Codex 的官方说明安装 Codex 命令行工具（本仓库测试时用的是 0.172.0 版本的独立安装包）。安装完成后要满足两个条件：

- `codex --version` 能输出版本号；
- `~/.local/bin/codex` 存在。沙盒会把这个文件所在的安装目录整个挂载进去，因为 Codex 除了主程序以外还需要同一目录下的两个辅助程序。如果你的 Codex 装在别的位置，请设置环境变量 `FMLBENCH_CODEX_BIN_DIR`，指向包含 `codex` 主程序的那个目录。

### 第 3 步：配置 Bedrock 的访问方式

Codex 通过 AWS 配置文件里名为 `codex-bedrock` 的配置来调用 Bedrock，它实际使用的是实例角色的临时凭证。先查看 AWS 命令行工具的位置：

```bash
which aws        # 在上面推荐的镜像里是 /usr/local/bin/aws
```

然后创建 `~/.aws/config`，把下面第二段里的 `/usr/local/bin/aws` 换成上一条命令输出的路径：

```ini
[default]
region = us-west-2

[profile codex-bedrock]
region = us-west-2
credential_process = /usr/local/bin/aws configure export-credentials --profile default --format process
```

这个文件里没有任何密钥，沙盒会以只读方式使用它。

### 第 4 步：下载代码

```bash
cd ~
git clone <仓库地址> FML-bench
cd FML-bench
git checkout codex-awesome可视化
```

### 第 5 步：创建 Python 环境并准备数据

下面的命令都在 `~/FML-bench` 目录下运行。

```bash
# 1. 创建这个任务用的 Python 环境（domainbed）和评测工具的环境（fmlbench），大约 3 分钟。
#    环境建在 ~/fml_envs 下面；Python 解释器全部由 uv 下载管理，不使用系统自带的 Python。
UV_PYTHON_PREFERENCE=only-managed python3 setup.py --skip-data --task Generalization_domainbed

# 2. 下载 MNIST 数据到 ~/fml_data_cache，并在任务目录里建立指向它的链接。
UV_PYTHON_PREFERENCE=only-managed python3 setup.py --skip-envs --data-cache ~/fml_data_cache --task Generalization_domainbed

# 3. 创建网页服务用的 Python 环境。
uv venv --python 3.11 ~/fml_envs/viz
VIRTUAL_ENV=~/fml_envs/viz uv pip install "fastapi==0.118.*" "uvicorn==0.37.*"
```

### 第 6 步：测量这台机器上的参考分数，并生成沙盒用的数据

不同型号的显卡会让这个任务的原始代码（baseline）得到略有不同的分数，所以要先在这台机器上测一次，作为后面的参考值。在 L4 上，这个分数应该是 0.2626。

```bash
cd ~/FML-bench

# 1. 用原始代码训练并评测一次，大约 3 分钟。结果保存在 rebaseline_results/Generalization_domainbed__l4x1/，
#    网页上的“与参考值比较”会读取这个目录，所以目录名不要改。
scripts/reproduction/rebaseline_task.sh Generalization_domainbed 0 __l4x1
cat rebaseline_results/Generalization_domainbed__l4x1/val_info.json     # in_acc_mean 应该是 0.2626 左右

# 2. 设置本节所有脚本使用的数据和运行目录（每打开一个新的终端都要先执行这一行）。
. scripts/codex/env_l4.sh

# 3. 生成沙盒里使用的数据：把隐藏测试集对应的图像替换成空白图，大约 10 秒。
python3 scripts/claude_code/build_task_view.py Generalization_domainbed

# 4. 检查替换后的数据：在沙盒里用原始代码跑一次，验证集分数必须和第 1 步完全相同，大约 3 分钟。
python3 scripts/claude_code/verify_task_view.py Generalization_domainbed --gpu 0 --cpus L0 \
    --baseline-dir rebaseline_results/Generalization_domainbed__l4x1 \
    --val-times scripts/codex/val_times_l4.json
```

最后一条命令的输出里应该有 `val OK (byte-identical ...)`，表示沙盒、显卡和数据都没有问题。

到这里，一次性的准备工作就完成了。

### 第 7 步：启动网页服务，并在 Mac 上打开页面

在实例上启动网页服务。它只在实例本机的 8765 端口上提供服务，外部无法直接访问：

```bash
cd ~/FML-bench
setsid nohup scripts/codex/viz/serve.sh > /tmp/viz_server.log 2>&1 < /dev/null &
```

在 Mac 上打开一个终端，建立 SSH 端口转发（这个终端窗口在看页面期间要一直开着）：

```bash
ssh -i <你的密钥文件> -L 8765:localhost:8765 ec2-user@<实例的公网地址>
```

然后在 Mac 的浏览器里打开 http://localhost:8765 。

### 第 8 步：在页面上新建一次运行，并复现 baseline

1. 第一次打开时，页面上还没有运行，点击“新建运行”。
2. 页面会先显示任务介绍：数据是怎么生成的、三个域的样本图片、评测方式和参考分数。看完后点击“我已了解，进入主页面”。
3. 在主页面右侧点击“运行 baseline”，大约需要 3 分钟。页面会依次显示七项检查：沙盒、显卡、数据、训练、预测、与参考值比较、生成提示词。全部通过后，baseline 会成为第一个“当前最好模型”，右侧开始显示它的预测结果。

这一步不计入 Codex 的 20 次 `fml-val`，也不会开始 1 小时的计时。baseline 完成之前，Codex 无法启动。

### 第 9 步：启动 Codex

baseline 完成后，页面顶部会出现启动命令和 goal 的复制按钮。

1. 在实例上（另开一个 SSH 窗口）进入 tmux：

   ```bash
   tmux new -s fml
   ```

2. 在 tmux 里粘贴页面上复制的启动命令。它会新开一个名为 `codex` 的 tmux 窗口，并进入沙盒。从这一刻开始计时 1 小时。
3. 在沙盒里输入 `cx-start`，启动 Codex 的交互界面。界面上方显示 `permissions: YOLO mode` 是正常的：它表示 Codex 执行命令时不会请求确认，安全由外层的沙盒保证。
4. 在 Codex 界面里粘贴页面上复制的 goal（以 `/goal` 开头的一整行），按回车。

之后就不需要再操作了。Codex 会自己反复修改代码、调用 `fml-val`、记录结论，直到用完预算并提交最终结果。可以随时在浏览器里查看进度，页面会自动刷新。

查看进度时需要注意：

- 想离开 tmux 但让 Codex 继续运行时，按 `Ctrl-b` 然后按 `d`。之后用 `tmux attach -t fml` 回来。
- 不要在沙盒的命令提示符下按 `Ctrl-C`，这会关闭整个沙盒。在 Codex 的界面里按 `Ctrl-C` 只会中断 Codex 当前的操作。
- 如果中途退出了 Codex，可以重新运行同一条启动命令进入沙盒，再用 `cx-start resume --last` 继续，截止时间不会改变。

### 第 10 步：结束和下一次实验

- Codex 完成后，页面顶部的 goal 状态会变成 `complete`，并出现“最终报告”按钮，可以查看 Codex 写的总结。
- 在 Codex 界面里输入 `/quit` 退出，再在沙盒里输入 `exit` 离开沙盒。如果不手动退出，沙盒会在截止时间之后 1 小时自动关闭。
- 如果想用隐藏测试集评测 Codex 提交的模型，在实例上运行：

  ```bash
  cd ~/FML-bench && . scripts/codex/env_l4.sh
  python3 scripts/codex/finalize.py ~/fml_agent_runs/codex/Generalization_domainbed/viz
  ```

  它会用真实数据重新训练 Codex 提交的那次运行，并在测试集上评测；如果验证集分数最高的是另一次运行，也会一起评测，作为对照。总共需要 3 到 6 分钟，结果打印在终端里，并保存到 `benchmark_results/codex/Generalization_domainbed/viz/`。
- 想开始下一次实验时，在页面上点击“新建运行”，然后从第 8 步开始。注意：这会删除上一次运行的全部记录（节点、模型、Codex 的会话记录和预测结果）。如果要保留，请先把 `~/fml_agent_runs/codex/Generalization_domainbed/viz` 复制到别的地方，并在复制前先运行上面的评测命令。

### 遇到问题时

| 现象 | 原因和处理方法 |
|---|---|
| 浏览器打不开 http://localhost:8765 | 检查 Mac 上的 SSH 端口转发窗口是否还开着；在实例上用 `cat /tmp/viz_server.log` 查看网页服务有没有报错，必要时按第 7 步重新启动 |
| 启动命令提示 “run the baseline first” | baseline 还没有完成。回到页面，按第 8 步运行 baseline |
| baseline 的“显卡”检查失败 | 用 `nvidia-smi` 确认显卡正常，并确认没有别的程序占满显存 |
| baseline 的“与参考值比较”显示警告 | 说明这次的分数和第 6 步测得的参考值不完全一样。环境本身仍然可用；如果差得很多，请检查是不是在别的型号的实例上做了第 6 步 |
| Codex 报错无法调用模型，或者提示没有权限 | 检查实例的 IAM 角色是否有 Bedrock 调用权限、us-west-2 是否开通了 `openai.gpt-6-astra`，以及第 3 步的 `~/.aws/config` 是否正确 |
| `cx-start` 提示找不到 Codex 的安装包 | 确认 `~/.local/bin/codex` 存在；如果 Codex 装在别的位置，按第 2 步设置 `FMLBENCH_CODEX_BIN_DIR` 后重启网页服务，并点击“新建运行” |
| 沙盒突然关闭 | 在沙盒命令提示符下按了 `Ctrl-C`，或者已经超过截止时间 1 小时。原因记录在 `~/fml_agent_runs/codex/Generalization_domainbed/viz/launch.log` 里 |

更详细的设计说明见 `docs/codex_agent_design.md`（沙盒和 Codex 的配置）和 `docs/codex_viz_design.md`（网页的设计和实现）。


## Setup

Make sure you have Anaconda/Miniconda installed before running setup.

Everything — task repositories, datasets, and the conda environments required
to run agents — is bootstrapped by a single command:

```bash
python setup.py
```

`setup.py` is idempotent: re-running it skips repos, datasets, and envs that
are already present.

If you only want to set up a single task (much faster — only the conda
environments that task needs are created), pass `--task`:

```bash
python setup.py --task Causality_causalml
```

Other options:

```bash
python setup.py --list           # list all available tasks
python setup.py --skip-data      # clone repos and create envs, but skip dataset downloads
python setup.py --skip-envs      # set up workspaces only, skip conda env creation
```

After setup completes, the harness env `fmlbench` is ready, and each task has
its own conda env (e.g. `causalml`, `domainbed`, …) used to execute the
baseline code.


## Run an agent (example)

This example runs **AI Scientist v2** on **Causality_causalml** using
**GPT-5.4**.

```bash
# 1. set up just this task
python setup.py --task Causality_causalml

# 2. activate the harness env
conda activate fmlbench

# 3. provide your API key
export OPENAI_API_KEY="your_openai_api_key"

# 4. pick a GPU
export CUDA_VISIBLE_DEVICES=0

# 5. run the agent
python run_agent_benchmark.py \
    --agent-config configs/agents/ai_scientist_v2.yaml \
    --task-config  configs/tasks/causality_causalml.yaml \
    --model        gpt-5.4 \
    --provider     OpenAI \
    --output-dir   results \
    agent.ai_scientist_v2.max_steps=100
```

Results, the per-step token usage, and a `summary.json` are written under
the chosen `--output-dir` (defaults to `benchmark_results/`); per-agent
step budget is controlled by the `agent.<type>.max_steps=N` override (the
example above sets it to 100).


## Run with other models

The model and provider are command-line flags. Anything supported by the
provider works (e.g. OpenAI GPT family, Google Gemini, Anthropic Claude,
OpenRouter passthrough). Examples:

```bash
# Gemini 2.5 Pro via Google
python run_agent_benchmark.py \
    --agent-config configs/agents/ai_scientist_v2.yaml \
    --task-config  configs/tasks/causality_causalml.yaml \
    --model gemini-2.5-pro --provider Google \
    --output-dir results \
    agent.ai_scientist_v2.max_steps=100

# Claude via OpenRouter
python run_agent_benchmark.py \
    --agent-config configs/agents/ai_scientist_v2.yaml \
    --task-config  configs/tasks/causality_causalml.yaml \
    --model anthropic/claude-3.5-sonnet --provider OpenRouter \
    --output-dir results \
    agent.ai_scientist_v2.max_steps=100
```

Set the corresponding API key in your shell:

| Provider     | Environment variable        |
|--------------|-----------------------------|
| OpenAI       | `OPENAI_API_KEY`            |
| Google       | `GOOGLE_API_KEY`            |
| Anthropic    | `ANTHROPIC_API_KEY`         |
| OpenRouter   | `OPENROUTER_API_KEY`        |

Per-agent hyperparameters can be overridden inline via positional `key=value`
arguments, e.g.:

```bash
python run_agent_benchmark.py \
    --agent-config configs/agents/ai_scientist_v2.yaml \
    --task-config  configs/tasks/causality_causalml.yaml \
    --model gpt-5.4 --provider OpenAI \
    --output-dir results \
    agent.ai_scientist_v2.max_steps=100 \
    agent.ai_scientist_v2.num_ideas=5 \
    agent.ai_scientist_v2.max_debug_depth=2
```


## Run on other tasks

Each task has a YAML in `configs/tasks/`. Set up the workspace for the task,
then point `--task-config` at the corresponding file. For example, to run on
DomainBed (Generalization):

```bash
python setup.py --task Generalization_domainbed
python run_agent_benchmark.py \
    --agent-config configs/agents/ai_scientist_v2.yaml \
    --task-config  configs/tasks/generalization.yaml \
    --model gpt-5.4 --provider OpenAI \
    --output-dir results \
    agent.ai_scientist_v2.max_steps=100
```

Available task configs (one per task):

| Task                                     | Config file                                             |
|------------------------------------------|---------------------------------------------------------|
| Causality (CausalML)                     | `configs/tasks/causality_causalml.yaml`                 |
| Causality (gCastle)                      | `configs/tasks/causality_gcastle.yaml`                  |
| Continual Learning (continual-learning)  | `configs/tasks/continual_learning.yaml`                 |
| Continual Learning (PyCIL)               | `configs/tasks/continual_learning_pycil.yaml`           |
| Data Efficiency (easy-few-shot-learning) | `configs/tasks/data_efficiency.yaml`                    |
| Data Efficiency (USB)                    | `configs/tasks/data_efficiency_usb.yaml`                |
| Fairness (AIF360)                        | `configs/tasks/fairness_and_bias_aif360.yaml`           |
| Fairness (Fairlearn)                     | `configs/tasks/fairness_fairlearn.yaml`                 |
| Federated Learning (PFLlib)              | `configs/tasks/federated_learning_pfllib.yaml`          |
| Generalization (DomainBed, ColoredMNIST) | `configs/tasks/generalization.yaml`                     |
| Generalization (DomainBed, OfficeHome)   | `configs/tasks/generalization_officehome.yaml`          |
| Privacy (Opacus)                         | `configs/tasks/privacy_opacus.yaml`                     |
| Privacy (PrivacyMeter)                   | `configs/tasks/privacy_privacymeter.yaml`               |
| Representation Learning (Lightly)        | `configs/tasks/representation_learning.yaml`            |
| Representation Learning (solo-learn)     | `configs/tasks/representation_learning_solo_learn.yaml` |
| Robustness (ART)                         | `configs/tasks/robustness_and_reliability_art.yaml`     |
| Robustness (OpenOOD)                     | `configs/tasks/robustness_openood.yaml`                 |
| Unlearning (open-unlearning)             | `configs/tasks/unlearning_open_unlearning.yaml`         |

`python setup.py --list` prints the task names accepted by `--task`.


## FML-bench-Lite

**FML-bench-Lite** is a subset of the full benchmark, offered as a cheaper proxy
for the full 18-task suite.

| Task                                     | Config file                                         |
|------------------------------------------|-----------------------------------------------------|
| Continual Learning (PyCIL)               | `configs/tasks/continual_learning_pycil.yaml`       |
| Data Efficiency (USB)                    | `configs/tasks/data_efficiency_usb.yaml`            |
| Generalization (DomainBed, ColoredMNIST) | `configs/tasks/generalization.yaml`                 |
| Generalization (DomainBed, OfficeHome)   | `configs/tasks/generalization_officehome.yaml`      |
| Robustness (OpenOOD)                     | `configs/tasks/robustness_openood.yaml`             |
| Privacy (Opacus)                         | `configs/tasks/privacy_opacus.yaml`                 |
| Privacy (PrivacyMeter)                   | `configs/tasks/privacy_privacymeter.yaml`           |
| Robustness (ART)                         | `configs/tasks/robustness_and_reliability_art.yaml` |

Run an agent on these task configs and [score](#score-a-run) it exactly as you
would the full suite. Continual Learning (continual-learning) and Unlearning
(open-unlearning) tasks are not suitable as subset tasks due to high variance and large metric scale.
On this subset the overall ranking of agents closely tracks
what the full 18-task benchmark shows, making Lite a useful, cheaper proxy when a
full sweep is out of reach.



## Remote GPU execution (Modal)

By default evaluations run as local subprocesses on this machine. FML-bench also
supports an **opt-in [Modal](https://modal.com) backend** that offloads only the
experiment-execution step (each task's validation/test command) to an ephemeral
remote GPU sandbox, while the entire agent loop — search, code edits, metric
parsing — stays local. This lets you run on remote GPUs and fan many tasks out
in parallel, with no change to agent or task behavior.

That backend lives on its own branch and does not affect the local default on
this branch. For setup and usage, see the
[`Modal` branch](https://github.com/qrzou/FML-bench/tree/Modal).


## Score a run

After an agent has run on all 18 tasks under a single `--output-dir`, score it
with `compute_agent_metrics.py`. Point the script at that agent's result
directory — `<output-dir>/<agent_name>`, which holds one subdirectory per task:

```bash
conda activate fmlbench
python compute_agent_metrics.py results/ai_scientist_v2
```

For a [FML-bench-Lite](#fml-bench-lite) run, add `--lite` to score only the 8
Lite tasks (the MEAN rows then average over those 8). `--tasks <TASK> ...`
scores any other subset.

It prints and writes three tables (as CSVs under `metric_reports/<agent_name>/`
by default; override the location with `--output-dir`):

1. **Raw Performance** — the canonical test metric for each task.
2. **Normalized Improvement** — each task's improvement over its baseline,
   normalized to `[0, 1]` and averaged across tasks.
3. **Process-Level metrics** — 12 metrics spanning Exploration, Generalization,
   Reliability, Efficiency, and Cost.

The 4 Exploration metrics embed each step's code snapshot with GraphCodeBERT and
additionally require `torch`, `transformers`, and `scikit-learn` (embeddings are
cached under the output dir). They read each task's baseline code from
`workspace/`, so that workspace must be at its reset (baseline) state when
scoring. The script only reads from the result directory and refuses an
`--output-dir` that is, or is nested inside, the result directory.


## Available agents

Seven agents are registered in this benchmark. Each has a config in
`configs/agents/`:

| Agent                 | Config file                                    |
|-----------------------|------------------------------------------------|
| The AI Scientist v1   | `configs/agents/theaiscientist.yaml`           |
| The AI Scientist v2   | `configs/agents/ai_scientist_v2.yaml`          |
| AIDE                  | `configs/agents/aide.yaml`                     |
| AIRA                  | `configs/agents/aira_mcts.yaml`                |
| Autoresearch          | `configs/agents/autoresearch.yaml`             |
| OpenEvolve            | `configs/agents/openevolve.yaml`               |
| AdaptiveSearch (ours) | `configs/agents/adaptivesearch.yaml`           |

Swap `--agent-config` to switch agents — everything else (task, model,
provider) stays the same.


## Repository layout

```
setup.py                  # one-shot environment + workspace setup
run_agent_benchmark.py    # entry point for running an agent on a task
compute_agent_metrics.py  # score one agent's results (3 tables + CSVs)
agents/                   # agent implementations
benchmark/                # benchmark runner / executor
configs/agents/           # agent YAMLs
configs/tasks/            # task YAMLs
ml_tasks/                 # task definitions: train.py, prompts, configs
workspace/                # populated by setup.py with task codebases
```


## Citation

If you find FML-bench useful in your research, please cite our paper:

```bibtex
@article{zou2026fml,
  title={FML-bench: A Controlled Study of AI Research Agent Strategies from the Perspective of Search Dynamics},
  author={Zou, Qiran and Lam, Hou Hei and Zhao, Wenhao and Chen, Tingting and Tang, Yiming and Yu, Samson and Zhu, Yingtao and Anumasa, Srinivas and Zhang, Zufeng and Zhang, Tianyi and others},
  journal={arXiv preprint arXiv:2605.17373},
  year={2026}
}
```


## Acknowledgements

We thank the maintainers of the following upstream projects:

- **ML Repos:** [DomainBed](https://github.com/facebookresearch/DomainBed), [easy-few-shot-learning](https://github.com/sicara/easy-few-shot-learning), [USB](https://github.com/microsoft/Semi-supervised-learning), [Lightly](https://github.com/lightly-ai/lightly), [solo-learn](https://github.com/vturrisi/solo-learn), [continual-learning](https://github.com/GMvandeVen/continual-learning), [PyCIL](https://github.com/LAMDA-CL/PyCIL), [CausalML](https://github.com/uber/causalml), [gCastle](https://github.com/huawei-noah/trustworthyAI), [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox), [OpenOOD](https://github.com/Jingkang50/OpenOOD), [Opacus](https://github.com/pytorch/opacus), [PrivacyMeter](https://github.com/privacytrustlab/ml_privacy_meter), [AIF360](https://github.com/Trusted-AI/AIF360), [Fairlearn](https://github.com/fairlearn/fairlearn), [PFLlib](https://github.com/TsingZ0/PFLlib), [open-unlearning](https://github.com/locuslab/open-unlearning)
- **ML Research Agents:** [The AI Scientist v1](https://github.com/SakanaAI/AI-Scientist), [The AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2), [AIDE](https://github.com/WecoAI/aideml), [AIRA](https://github.com/facebookresearch/aira-dojo), [Autoresearch](https://github.com/karpathy/autoresearch), [OpenEvolve](https://github.com/codelion/openevolve)
