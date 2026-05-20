# AgentRL Infra 投稿实验 TODO

Last updated: 2026-05-20

目标论文主张：

> Failure-aware, replayable rollout infrastructure improves Agent RL data quality,
> debugging efficiency, and training sample efficiency under realistic long-horizon
> agent failures.

本文件把模型、任务、数据集、baseline、指标和阶段任务固定下来，作为后续实现与实验的执行清单。

## 0. 固定实验栈

### 0.1 代码与依赖

- [x] 使用 `git` 管理项目。
- [x] 使用 `uv` 管理 Python 依赖。
- [x] 创建 package：`agentrl_infra`。
- [x] 初版模块：
  - [x] failure taxonomy: `src/agentrl_infra/failures.py`
  - [x] event log: `src/agentrl_infra/events.py`
  - [x] session runtime: `src/agentrl_infra/session.py`
  - [x] replay report: `src/agentrl_infra/replay.py`
  - [x] scheduler: `src/agentrl_infra/scheduler.py`
- [x] 单元测试：`uv run pytest`
- [x] lint：`uv run ruff check .`

### 0.2 固定模型

主实验优先使用开源权重，保证可复现；闭源模型只作为上界或 debug oracle。

#### 训练/rollout 主模型

- [ ] `Qwen/Qwen3-8B`
  - 用途：主训练模型、主 rollout 模型。
  - 原因：8B 规模适中，可在 1-2 张 80GB GPU 或量化环境中迭代；Hugging Face 已支持 `transformers`。
  - 推理后端：`vLLM` 或 `SGLang`，优先 `SGLang` 以便与 AgentRL 风格保持一致。
  - 训练方式：LoRA/QLoRA smoke test；正式实验用 full fine-tune 或 FSDP/LoRA 视资源决定。

#### 低成本 smoke model

- [ ] `Qwen/Qwen3-4B`
  - 用途：CI/smoke、synthetic failure benchmark、开发时快速回归。
  - 不作为主论文结论模型。

#### 跨模型泛化评估

- [ ] `meta-llama/Llama-3.1-8B-Instruct`
  - 用途：证明 failure-aware runtime 不依赖 Qwen tokenizer/template。
  - 只做 rollout/eval，不做主训练。

#### 强模型上界 / oracle

- [ ] `Qwen/Qwen3-32B`
  - 用途：较强开源上界，可用于 replay 中的 model-substituted replay。
  - 如本地资源不足，改用 API 托管版本。

- [ ] `gpt-5-mini` 或当前可用等价闭源模型
  - 用途：少量 root-cause oracle / debugging user study 辅助，不作为主训练模型。
  - 注意：所有主结果必须能在开源模型上复现。

### 0.3 固定推理/训练后端

- [ ] 推理后端：
  - [ ] 首选：SGLang
  - [ ] 备用：vLLM
- [ ] 训练后端：
  - [ ] Phase 1：本项目轻量 GRPO prototype，只用于 synthetic 和小任务验证。
  - [ ] Phase 2：接入 AgentRL trainer 或 verl trainer，做正式训练曲线。
- [ ] 统一接口：
  - [ ] OpenAI-compatible chat endpoint
  - [ ] token-native trace export：`input_ids`、`logprobs`、`loss_mask`、`model_version`

## 1. 固定任务与数据集

实验分四类：synthetic controllable、tool/web、browser、code/sandbox。

### 1.1 Synthetic FailureBench-AgentRL

这是必须先做的可控 benchmark，用来证明 failure taxonomy、watchdog、replay、scheduler 的有效性。

- [ ] 新建目录：`benchmarks/failurebench/`
- [ ] 固定 8 类 failure scenario：
  - [ ] `agent_invalid_action`
  - [ ] `tool_timeout`
  - [ ] `tool_execution_error`
  - [ ] `environment_crash`
  - [ ] `environment_contamination`
  - [ ] `context_limit`
  - [ ] `repetitive_loop`
  - [ ] `rate_limit`
- [ ] 每类 100 个 deterministic seeds，共 800 episodes。
- [ ] 固定 split：
  - [ ] dev: 每类 20，共 160
  - [ ] test: 每类 80，共 640
- [ ] 每个 episode 输出：
  - [ ] `trace.jsonl`
  - [ ] `oracle_failure_type`
  - [ ] `oracle_attribution`
  - [ ] `oracle_replayability`
  - [ ] `expected_salvageable`
- [ ] 论文用途：
  - [ ] failure detection precision/recall
  - [ ] root-cause attribution accuracy
  - [ ] replayability classification accuracy
  - [ ] overhead measurement

### 1.2 WebShop / AgentBench Web Shopping

真实多轮工具/网页购物任务，用于证明 rollout data quality 和训练收益。

- [ ] 数据源：WebShop / AgentBench WS task。
- [ ] 固定任务：
  - [ ] `webshop-env_train`
  - [ ] `webshop-std`
- [ ] 固定 split：
  - [ ] train: 1,000 episodes
  - [ ] dev: 200 episodes
  - [ ] test: 500 episodes
- [ ] 如果完整 WebShop 环境部署成本过高：
  - [ ] 先使用 AgentBench FC/WebShop function-call 环境。
  - [ ] 保留真实 WebShop full environment 作为扩展实验。
- [ ] 主要 failure：
  - [ ] invalid tool call
  - [ ] repetitive search loop
  - [ ] context overrun
  - [ ] no-progress interaction
  - [ ] delayed sparse reward
- [ ] 论文用途：
  - [ ] GRPO training curve
  - [ ] useful trajectories/hour
  - [ ] invalid action rate
  - [ ] task success rate

### 1.3 MiniWoB++ Browser Subset

浏览器状态、重复动作、页面等待、DOM/action mismatch 的受控真实环境。

- [ ] 数据源：MiniWoB++。
- [ ] 固定 20 个任务：
  - [ ] `click-button`
  - [ ] `click-checkboxes`
  - [ ] `click-checkboxes-large`
  - [ ] `click-collapsible`
  - [ ] `click-dialog`
  - [ ] `click-link`
  - [ ] `click-menu`
  - [ ] `click-option`
  - [ ] `click-pie`
  - [ ] `enter-date`
  - [ ] `enter-password`
  - [ ] `enter-text`
  - [ ] `focus-text`
  - [ ] `login-user`
  - [ ] `multi-layouts`
  - [ ] `navigate-tree`
  - [ ] `search-engine`
  - [ ] `social-media`
  - [ ] `use-autocomplete`
  - [ ] `use-spinner`
- [ ] 固定 seeds：
  - [ ] train: seeds 0-999
  - [ ] dev: seeds 1000-1199
  - [ ] test: seeds 1200-1699
- [ ] 每个 episode 最大 15 turns。
- [ ] 主要 failure：
  - [ ] repeated click loop
  - [ ] stale DOM
  - [ ] page timeout
  - [ ] invalid selector/action
  - [ ] no-progress
- [ ] 论文用途：
  - [ ] browser/runtime failure benchmark
  - [ ] replay determinism levels
  - [ ] environment reset/reuse experiments

### 1.4 SWE-bench Verified Debug Subset

代码 agent 任务不作为第一阶段训练主任务，主要用于 replay/debugging 和环境生命周期实验。

- [ ] 数据源：SWE-bench Verified。
- [ ] 固定 subset：
  - [ ] dev-debug: 50 instances
  - [ ] test-debug: 100 instances
- [ ] subset 选择规则：
  - [ ] 按 repository 分层采样。
  - [ ] 排除需要超长上下文或极高资源的任务。
  - [ ] 固定随机种子：`20260520`。
- [ ] 每个 instance 最大预算：
  - [ ] wall-clock: 20 min
  - [ ] tool calls: 80
  - [ ] shell command timeout: 60 sec
- [ ] 主要 failure：
  - [ ] shell timeout
  - [ ] test command failure
  - [ ] environment setup failure
  - [ ] invalid patch
  - [ ] repeated file inspection loop
- [ ] 论文用途：
  - [ ] replay/debugging benchmark
  - [ ] environment snapshot/rollback cost
  - [ ] failure attribution on realistic coding agents

## 2. 固定 Baseline

所有实验至少跑以下 baseline。

### 2.1 Runtime baseline

- [ ] `B0_raw_agent`
  - 普通 agent loop。
  - 无 typed failure。
  - 无 event-sourced trace。
  - fixed timeout。
  - failed trajectory 直接丢弃。

- [ ] `B1_capacity_scheduler`
  - 按 worker capacity 调度。
  - fixed timeout。
  - message-only trace。

- [ ] `B2_retry_only`
  - capacity scheduler + simple retry。
  - 不区分 failure type。

- [ ] `B3_message_trace`
  - 保存完整 messages 和 stdout/stderr。
  - 无 token-native trace，无 typed event。

### 2.2 Ours variants

- [ ] `O1_failure_runtime`
  - typed failure taxonomy。
  - watchdog。
  - loop/no-progress detection。

- [ ] `O2_failure_runtime_salvage`
  - O1 + partial trajectory salvage。

- [ ] `O3_replayable_trace`
  - O2 + event-sourced trace。
  - exact/model-substituted/environment-substituted replay report。

- [ ] `O4_failure_scheduler`
  - O3 + failure-aware scheduler。

- [ ] `O5_full`
  - O4 + token-native validation。
  - scheduler 使用 failure rate、cost、policy lag、reward variance。

## 3. 固定指标

### 3.1 Failure detection

- [ ] detection precision
- [ ] detection recall
- [ ] macro F1 by failure type
- [ ] false positive rate
- [ ] mean time to detection
- [ ] turns wasted after oracle failure point

### 3.2 Rollout system

- [ ] completed trajectories/hour
- [ ] useful trajectories/hour
- [ ] successful trajectories/hour
- [ ] failed rollout cost
- [ ] environment-hours/success
- [ ] zombie session rate
- [ ] P50/P95/P99 rollout latency
- [ ] GPU idle time caused by rollout bottleneck
- [ ] logging overhead
- [ ] storage overhead per trajectory

### 3.3 Training

- [ ] reward curve
- [ ] success rate
- [ ] sample efficiency measured by environment-hours
- [ ] sample efficiency measured by rollout tokens
- [ ] invalid action rate
- [ ] tool error rate
- [ ] context-limit failure rate
- [ ] average turns to success
- [ ] seed variance across 3 seeds

### 3.4 Replay/debugging

- [ ] replay success rate
- [ ] determinism level distribution
- [ ] root-cause attribution accuracy
- [ ] trajectory diff accuracy
- [ ] percentage of failures with actionable diagnosis
- [ ] time-to-root-cause

### 3.5 Environment lifecycle

- [ ] reset latency
- [ ] contamination rate
- [ ] reuse success rate
- [ ] environment utilization
- [ ] failed rollout due to bad state
- [ ] cost per successful trajectory

## 4. 固定主实验

### Experiment 1: FailureBench controllable failure benchmark

目的：证明 failure taxonomy 和 runtime detector 有效。

- [ ] 实现 `benchmarks/failurebench`。
- [ ] 运行模型：
  - [ ] Qwen3-4B smoke
  - [ ] Qwen3-8B main
  - [ ] Llama-3.1-8B cross-model
- [ ] 对比：
  - [ ] B0_raw_agent
  - [ ] B1_capacity_scheduler
  - [ ] O1_failure_runtime
  - [ ] O3_replayable_trace
- [ ] 报告表：
  - [ ] per-failure precision/recall/F1
  - [ ] mean time to detection
  - [ ] turns wasted after oracle failure
  - [ ] trace overhead
- [ ] 通过标准：
  - [ ] macro F1 >= 0.85 on synthetic test
  - [ ] false positive rate <= 5%

### Experiment 2: Useful rollout throughput

目的：证明我们的系统提高 useful trajectories/hour，而不只是 raw throughput。

- [ ] 任务：
  - [ ] FailureBench test
  - [ ] MiniWoB++ 20-task subset
  - [ ] WebShop dev
- [ ] 模型：
  - [ ] Qwen3-8B
- [ ] 对比：
  - [ ] B1_capacity_scheduler
  - [ ] B2_retry_only
  - [ ] O2_failure_runtime_salvage
  - [ ] O4_failure_scheduler
  - [ ] O5_full
- [ ] 报告：
  - [ ] useful trajectories/hour
  - [ ] successful trajectories/hour
  - [ ] failed rollout cost
  - [ ] environment-hours/success
  - [ ] zombie session rate
  - [ ] P95/P99 latency
- [ ] 通过标准：
  - [ ] useful trajectories/hour 相比 B1 提升 >= 20%
  - [ ] zombie session rate 相比 B1 降低 >= 50%

### Experiment 3: Agent RL training on WebShop/AgentBench WS

目的：证明 infra 改进影响训练效率。

- [ ] 任务：
  - [ ] `webshop-env_train`: 1,000 train episodes
  - [ ] `webshop-std`: 500 test episodes
- [ ] 模型：
  - [ ] Qwen3-8B
- [ ] 算法：
  - [ ] GRPO
- [ ] 训练配置：
  - [ ] rollout group size `n=8`
  - [ ] max turns `20`
  - [ ] max total length `8192`
  - [ ] temperature `0.8`
  - [ ] learning rate `1e-6`
  - [ ] seeds: 3 seeds (`1`, `2`, `3`)
- [ ] 对比：
  - [ ] discard all failures
  - [ ] uniform negative reward for all failures
  - [ ] O2 typed failure + salvage
  - [ ] O5 full
- [ ] 报告：
  - [ ] success rate vs environment-hours
  - [ ] reward vs rollout tokens
  - [ ] invalid action rate
  - [ ] tool error rate
  - [ ] context-limit rate
  - [ ] seed variance
- [ ] 通过标准：
  - [ ] O5 在相同 environment-hours 下达到更高 success rate。
  - [ ] O5 在最终 success rate 上不低于 baseline，且显著减少 invalid/tool failures。

### Experiment 4: Replay/debugging benchmark

目的：证明 replayable event trace 比 raw logs/message trace 更适合定位失败。

- [ ] 数据：
  - [ ] FailureBench test: 640 traces
  - [ ] MiniWoB++ failed traces: at least 300
  - [ ] SWE-bench Verified dev-debug failed traces: at least 50
- [ ] 对比日志格式：
  - [ ] raw stdout/stderr
  - [ ] message-only trace
  - [ ] event-sourced trace
- [ ] 自动评估：
  - [ ] 使用 oracle labels 计算 attribution accuracy。
  - [ ] 使用 deterministic replay 判断 replay success。
- [ ] 人类评估（可选但强烈建议）：
  - [ ] 3-5 名参与者。
  - [ ] 每人 30 个 failure cases。
  - [ ] 测 time-to-root-cause 和 diagnosis accuracy。
- [ ] 报告：
  - [ ] root-cause attribution accuracy
  - [ ] replay success rate
  - [ ] time-to-root-cause
  - [ ] actionable diagnosis rate
- [ ] 通过标准：
  - [ ] event-sourced trace attribution accuracy 高于 message-only >= 20 个百分点。

### Experiment 5: Environment lifecycle and contamination

目的：证明 agent environment 需要 snapshot/reuse/contamination-aware 管理。

- [ ] 任务：
  - [ ] MiniWoB++ browser subset
  - [ ] SWE-bench Verified dev-debug
  - [ ] synthetic mutable DB/file task
- [ ] 对比：
  - [ ] recreate every episode
  - [ ] blind reuse
  - [ ] fixed TTL reuse
  - [ ] health-check reuse
  - [ ] contamination-aware reuse
- [ ] 报告：
  - [ ] reset latency
  - [ ] contamination rate
  - [ ] environment utilization
  - [ ] failed rollout due to bad state
  - [ ] cost per successful trajectory
- [ ] 通过标准：
  - [ ] 相比 recreate every episode 降低环境成本。
  - [ ] 相比 blind reuse 显著降低 contamination-induced failure。

## 5. 必做消融

- [ ] 去掉 taxonomy：所有 failure 都映射到 `unknown_runtime_error`。
- [ ] 去掉 salvage：所有 failed trajectories 丢弃。
- [ ] 去掉 scheduler：使用 capacity-only。
- [ ] 去掉 token-native validation：只存 message trace，训练时重新 tokenize。
- [ ] 去掉 replay metadata：只保留 event text，不保存 model/tool/env provenance。

每个消融至少在：

- [ ] FailureBench test
- [ ] WebShop dev
- [ ] MiniWoB++ subset

上跑。

## 6. 实现路线

### Phase A: 本项目 primitives 到 runnable demo

- [ ] 新增 `benchmarks/failurebench/`。
- [ ] 新增 `agentrl_infra.runner`：
  - [ ] episode runner
  - [ ] tool runtime wrapper
  - [ ] timeout wrapper
  - [ ] event sink
- [ ] 新增 CLI：
  - [ ] `agentrl-infra run-failurebench`
  - [ ] `agentrl-infra inspect-trace`
  - [ ] `agentrl-infra replay-trace`
  - [ ] `agentrl-infra summarize-runs`
- [ ] 新增输出规范：
  - [ ] `runs/{run_id}/config.yaml`
  - [ ] `runs/{run_id}/traces/*.jsonl`
  - [ ] `runs/{run_id}/metrics.json`
  - [ ] `runs/{run_id}/summary.md`

### Phase B: MiniWoB++ 接入

- [ ] 创建 `integrations/miniwob/`。
- [ ] 封装 browser action schema。
- [ ] 封装 observation extractor。
- [ ] 实现 loop/no-progress detector。
- [ ] 实现 environment reset health check。
- [ ] 跑 20-task subset baseline。

### Phase C: WebShop / AgentBench WS 接入

- [ ] 创建 `integrations/webshop/` 或 `integrations/agentbench_ws/`。
- [ ] 支持 OpenAI-compatible agent loop。
- [ ] 支持 tool-call validation。
- [ ] 导出 token-native trajectory。
- [ ] 跑 WebShop dev baseline。
- [ ] 接 GRPO 训练。

### Phase D: SWE-bench Debug 接入

- [ ] 创建 `integrations/swebench_debug/`。
- [ ] 固定 50 dev-debug subset。
- [ ] 封装 shell tool：
  - [ ] command timeout
  - [ ] stdout/stderr truncation
  - [ ] file diff capture
  - [ ] test command capture
- [ ] 实现 replay/debug report。

### Phase E: 论文实验自动化

- [ ] 新增 `experiments/configs/`。
- [ ] 每个实验一个 YAML：
  - [ ] `exp1_failurebench_detection.yaml`
  - [ ] `exp2_useful_throughput.yaml`
  - [ ] `exp3_webshop_grpo.yaml`
  - [ ] `exp4_replay_debugging.yaml`
  - [ ] `exp5_env_lifecycle.yaml`
- [ ] 新增 `scripts/run_experiment.py`。
- [ ] 新增 `scripts/make_tables.py`。
- [ ] 新增 `scripts/make_figures.py`。
- [ ] 所有表格输出 LaTeX：
  - [ ] `paper/tables/*.tex`
- [ ] 所有图输出 PDF/PNG：
  - [ ] `paper/figures/*`

## 7. 论文图表固定计划

### Main paper figures

- [ ] Figure 1: 系统架构图。
- [ ] Figure 2: event-sourced trajectory schema。
- [ ] Figure 3: useful trajectories/hour 对比。
- [ ] Figure 4: WebShop GRPO training curve。
- [ ] Figure 5: replay root-cause accuracy / time-to-debug。

### Main paper tables

- [ ] Table 1: failure taxonomy and semantics。
- [ ] Table 2: FailureBench detection results。
- [ ] Table 3: rollout throughput and cost。
- [ ] Table 4: ablation study。
- [ ] Table 5: environment lifecycle results。

### Appendix

- [ ] 全部任务和 seeds。
- [ ] 模型超参。
- [ ] scheduler 公式。
- [ ] failure detector 阈值。
- [ ] replay determinism levels。
- [ ] 更多 per-task MiniWoB++ 结果。
- [ ] SWE-bench debug case studies。

## 8. 当前最近三周执行计划

### Week 1

- [ ] 实现 FailureBench。
- [ ] 实现 runner/event sink。
- [ ] 实现 summary metrics。
- [ ] 跑 Qwen3-4B smoke 或 mock policy。
- [ ] 生成第一批 traces。

### Week 2

- [ ] 接 Qwen3-8B OpenAI-compatible inference。
- [ ] 跑 FailureBench dev/test。
- [ ] 做 detection precision/recall 表。
- [ ] 实现 replay CLI。
- [ ] 写 2 个 case study。

### Week 3

- [ ] 接 MiniWoB++ 5-task pilot。
- [ ] 实现 browser loop/no-progress detector。
- [ ] 跑 capacity baseline vs O1/O3。
- [ ] 输出 useful trajectories/hour 初版图。
- [ ] 根据结果调整 failure detector 阈值。

## 9. 外部资料固定引用

- AgentRL: https://arxiv.org/abs/2510.04206
- AgentBench: https://arxiv.org/abs/2308.03688
- SWE-bench datasets: https://www.swebench.com/SWE-bench/guides/datasets/
- Qwen3-8B model card: https://huggingface.co/Qwen/Qwen3-8B
- Qwen2.5-7B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct model card: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- MiniWoB++ release: https://github.com/Farama-Foundation/miniwob-plusplus
- WebShop: https://github.com/princeton-nlp/WebShop

