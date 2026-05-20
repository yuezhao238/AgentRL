# RolloutOS / AgentRL Infra TODO

Last updated: 2026-05-20

目标：把本仓库推进成 MLSys 系统论文，而不是 Agent RL 算法论文。

核心主张：

> Failure-aware, replayable rollout infrastructure improves Agent RL data quality,
> debugging efficiency, and environment/GPU efficiency under realistic long-horizon
> agent failures.

当前论文名暂定为 **RolloutOS**：a failure-aware, replayable runtime for long-horizon
agent RL rollouts.

## 0. 当前基线状态

### 已完成并进入主线

- [x] Python package：`agentrl_infra`
- [x] 依赖管理：`uv`
- [x] 模型缓存准备：
  - [x] `Qwen/Qwen3-4B`
  - [x] `Qwen/Qwen3-8B`
  - [x] `Qwen/Qwen3-32B`
  - [x] `meta-llama/Llama-3.1-8B-Instruct`
- [x] Runtime primitives：
  - [x] typed failure taxonomy：`src/agentrl_infra/failures.py`
  - [x] event-sourced trace：`src/agentrl_infra/events.py`
  - [x] session watchdog：`src/agentrl_infra/session.py`
  - [x] episode runner：`src/agentrl_infra/runner.py`
  - [x] environment/tool protocols：`src/agentrl_infra/resources.py`
  - [x] failure-aware scheduler：`src/agentrl_infra/scheduler.py`
  - [x] executable replay：`src/agentrl_infra/replay.py`
- [x] Synthetic systems workloads：
  - [x] FailureBench：8 类 deterministic rollout failure
  - [x] mutable environment lifecycle benchmark
- [x] Baselines：
  - [x] raw timeout
  - [x] message trace
  - [x] retry-only
  - [x] FIFO/capacity scheduler baseline
- [x] Automation：
  - [x] run validation
  - [x] batch replay validation
  - [x] run comparison
  - [x] one-shot local experiment suite
  - [x] paper CSV/LaTeX table generation
- [x] Paper：
  - [x] MLSys template
  - [x] initial `paper/main.tex`
  - [x] FailureBench table
  - [x] lifecycle table

### 当前关键结果

- [x] FailureBench synthetic：
  - [x] `rolloutos` macro F1 = 1.000
  - [x] `rolloutos` useful/cost = 0.480
  - [x] baselines range from 0.000 to 0.333 macro F1
- [x] Lifecycle synthetic：
  - [x] recreate：50/50 success, cost/success 1.100
  - [x] blind reuse：1/50 success, 49 contamination failures
  - [x] contamination-aware：50/50 success, cost/success 0.316

### 新增推进状态

- [x] MiniWoB++ browser contract harness：
  - [x] fixed 20-task subset schema
  - [x] browser action schema：click/type/wait
  - [x] DOM observation schema with stable DOM hash
  - [x] stale-DOM detection
  - [x] invalid action detection
  - [x] no-progress detection through runtime watchdog
  - [x] environment health check
  - [x] contamination check
  - [x] deterministic trace emission
  - [x] CLI：`agentrl-infra run-miniwob-contract`
- [x] MiniWoB++ contract quantitative suite：
  - [x] 20 tasks x 5 dev seeds = 100 episodes per policy
  - [x] oracle：100/100 success
  - [x] stale DOM：0/100 success, 100 invalid-action failures
  - [x] invalid selector：0/100 success, 100 invalid-action failures
  - [x] wait loop：0/100 success, 100 no-progress failures
  - [x] repeated wrong text：60/100 success, 40 no-progress failures

## 1. MLSys 主线

### 主线 A：Rollout Runtime

目标：证明 typed failure + replay + scheduler 是一个系统 runtime，而不是算法 trick。

- [x] Event log captures session/model/action/tool/reward/failure events.
- [x] Failure semantics define retryability, salvageability, reset requirement, and training signal.
- [x] Runtime watchdog catches turn budget, repeated action, and no-progress failures.
- [x] Replay validates event-log structure and deterministic executable replay.
- [x] Scheduler uses failure-aware priority and cancellation semantics.
- [ ] Add resource accounting for wall-clock, CPU, GPU, browser, and storage overhead.
- [ ] Add token-native trajectory fields for real model rollouts:
  - [ ] `input_ids`
  - [ ] `logprobs`
  - [ ] `loss_mask`
  - [ ] `model_version`
  - [ ] `tokenizer_hash`

### 主线 B：Realistic Environment Evidence

优先级：MiniWoB++ > WebShop/AgentBench WS > SWE-bench Debug。

原因：MiniWoB++ 最直接暴露系统问题：browser reset、DOM stale state、重复点击、
页面等待、环境污染和 replay mismatch。

- [x] MiniWoB++ deterministic contract harness
- [x] MiniWoB++ contract result table in `paper/tables/miniwob_contract_summary.tex`
- [ ] Live MiniWoB++ browser binding:
  - [ ] install/lock browser dependency
  - [ ] launch task by name/seed
  - [ ] extract DOM/actionable elements
  - [ ] execute click/type/wait actions
  - [ ] capture reward/done
  - [ ] expose health/contamination checks
  - [ ] save screenshots or DOM snapshots for replay debugging
- [ ] MiniWoB++ lifecycle experiment:
  - [ ] recreate
  - [ ] blind reuse
  - [ ] fixed TTL
  - [ ] health check
  - [ ] contamination-aware reuse
- [ ] MiniWoB++ replay experiment:
  - [ ] exact replay
  - [ ] environment-substituted replay
  - [ ] stale DOM mismatch report
  - [ ] reward/done divergence report

### 主线 C：Throughput and Scheduling

目标：把“检测更准”推进到“单位资源产出更多可用轨迹”。

- [x] synthetic useful/cost metric
- [x] FIFO vs failure-aware scheduled FailureBench run
- [ ] worker-pool throughput simulator:
  - [ ] environment lease queue
  - [ ] per-task service-time distribution
  - [ ] zombie session modeling
  - [ ] P50/P95/P99 rollout latency
  - [ ] useful trajectories/hour
  - [ ] environment-hours/success
- [ ] real MiniWoB++ throughput run on multiple browser workers
- [ ] GPU/LLM server backpressure model for SGLang/vLLM rollout workers

### 主线 D：Training Impact

目标：证明 RolloutOS 改善训练数据质量和 sample efficiency。

- [ ] OpenAI-compatible policy client
- [ ] SGLang rollout client
- [ ] vLLM rollout client
- [ ] token-native trace export
- [ ] lightweight GRPO prototype for smoke validation
- [ ] connect to AgentRL/verl trainer for formal curves
- [ ] WebShop/AgentBench WS training run:
  - [ ] discard-failures baseline
  - [ ] uniform negative failure baseline
  - [ ] typed failure + salvage
  - [ ] full RolloutOS

## 2. 固定 Workloads

### FailureBench

- [x] 8 failure types:
  - [x] `agent_invalid_action`
  - [x] `tool_timeout`
  - [x] `tool_execution_error`
  - [x] `environment_crash`
  - [x] `environment_contamination`
  - [x] `context_limit`
  - [x] `repetitive_loop`
  - [x] `rate_limit`
- [x] deterministic dev/test seed generation
- [x] trace output
- [x] oracle failure labels
- [x] replayability labels
- [x] baseline comparison
- [ ] increase default paper run to 800 episodes when finalizing numbers
- [ ] report storage/logging overhead

### MiniWoB++ 20-task Subset

- [x] Fixed task names:
  - [x] `click-button`
  - [x] `click-checkboxes`
  - [x] `click-checkboxes-large`
  - [x] `click-collapsible`
  - [x] `click-dialog`
  - [x] `click-link`
  - [x] `click-menu`
  - [x] `click-option`
  - [x] `click-pie`
  - [x] `enter-date`
  - [x] `enter-password`
  - [x] `enter-text`
  - [x] `focus-text`
  - [x] `login-user`
  - [x] `multi-layouts`
  - [x] `navigate-tree`
  - [x] `search-engine`
  - [x] `social-media`
  - [x] `use-autocomplete`
  - [x] `use-spinner`
- [x] deterministic contract harness
- [ ] live browser execution
- [ ] fixed splits:
  - [ ] train seeds 0-999
  - [ ] dev seeds 1000-1199
  - [ ] test seeds 1200-1699
- [ ] max 15 turns per episode
- [ ] benchmark failures:
  - [x] stale DOM
  - [x] invalid selector/action
  - [x] no-progress
  - [ ] page timeout
  - [ ] repeated click loop on live pages

### WebShop / AgentBench WS

- [ ] choose exact environment source and pin version
- [ ] function-call adapter
- [ ] OpenAI-compatible agent loop
- [ ] tool-call validation
- [ ] trace export
- [ ] dev baseline
- [ ] GRPO integration

### SWE-bench Debug

- [ ] fixed 50-instance dev-debug subset
- [ ] shell tool wrapper
- [ ] command timeout
- [ ] stdout/stderr truncation
- [ ] diff capture
- [ ] replay/debug report

## 3. 固定 Metrics

### Failure Detection

- [x] detection accuracy
- [x] macro F1
- [x] per-failure precision/recall/F1
- [x] turns wasted after oracle failure
- [ ] false positive rate
- [ ] mean time to detection

### Rollout System

- [x] useful trajectories/cost unit
- [x] failed rollout cost units
- [x] replayable failure rate
- [ ] useful trajectories/hour with real workers
- [ ] successful trajectories/hour
- [ ] environment-hours/success
- [ ] zombie session rate
- [ ] P50/P95/P99 rollout latency
- [ ] GPU idle/backpressure time
- [ ] storage overhead per trajectory

### Replay / Debugging

- [x] event-log validation
- [x] deterministic replay match/mismatch
- [x] batch replay report
- [ ] replay determinism distribution on MiniWoB++
- [ ] root-cause attribution accuracy on real failures
- [ ] time-to-root-cause human or oracle-assisted study

### Environment Lifecycle

- [x] reset count
- [x] restore count
- [x] reuse count
- [x] contamination-induced failures
- [x] cost per successful trajectory
- [ ] real browser reset latency
- [ ] real browser contamination rate
- [ ] environment utilization

## 4. 近期执行计划

### P0：把 MiniWoB++ 从 contract 推到 live browser

- [ ] Decide and pin live environment package.
- [ ] Add optional dependency group for browser workload.
- [ ] Add live `MiniWoBEnvironmentAdapter`.
- [ ] Keep the current deterministic contract as CI and replay oracle.
- [ ] Add live smoke test gated by environment availability.
- [ ] Run 20-task dev subset with oracle or scripted policy.

### P1：补齐系统指标

- [ ] Add latency percentile helpers.
- [ ] Add trace byte-size and event-count overhead.
- [ ] Add environment lease utilization metrics.
- [ ] Add zombie session definition and measurement.

### P2：实验自动化和论文闭环

- [ ] Add `experiments/configs/exp1_failurebench_detection.json`.
- [ ] Add `experiments/configs/exp2_minwob_lifecycle.json`.
- [ ] Add `experiments/configs/exp3_throughput.json`.
- [ ] Add paper table generation for MiniWoB++ contract/live runs.
- [ ] Replace synthetic-only claims in `paper/main.tex` with real workload methodology.

## 5. 投稿差距

当前距离 MLSys Oral 的主要差距不是代码 skeleton，而是证据强度：

- [ ] 需要至少一个真实环境的大规模系统实验，MiniWoB++ 是第一优先级。
- [ ] 需要 throughput/latency/utilization 结果，而不只是 failure classification。
- [ ] 需要训练曲线或至少 rollout-data-quality 对训练输入的量化影响。
- [ ] 需要更强的 related work 和 architecture figure。
- [ ] 需要把所有实验配置、依赖和 artifact validation 做到一键复现。
