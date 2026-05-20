# RolloutOS / AgentRL Infra TODO

Last updated: 2026-05-21

目标：把本仓库推进成 MLSys 系统论文，而不是 Agent RL 算法论文。

核心主张：

> Failure-aware, replayable rollout infrastructure improves Agent RL data quality,
> debugging efficiency, and environment/GPU efficiency under realistic long-horizon
> agent failures.

当前论文名暂定为 **RolloutOS**：a failure-aware, replayable runtime for long-horizon
agent RL rollouts.

## 0. 当前判断

RolloutOS 的投稿主线必须是：真实 rollout failure 是系统事件；typed failure handling、
replay 和 scheduling 把原本浪费的 rollout 资源转化成 useful/replayable training signal。

最终证据链必须闭环：

1. 真实模型在 action channel 上产生 parse failure、invalid action、no-progress、stale
   observation、environment contamination。
2. runtime 把这些失败记录成 replayable typed events，而不是普通 exception/string log。
3. scheduler/resource manager 使用这些 typed events 改善 useful trajectories per resource-hour。

当前最硬的证据：

- FailureBench 已证明 failure typing 和 replay metadata 可以提高 failure F1/useful-cost。
- MiniWoB contract 已证明 stale DOM、invalid selector、wait loop、repeated action 等浏览器系统失败可复现、可分类。
- Qwen3-4B action-channel ablation 已证明 default thinking、no-thinking、guided JSON 在 parse/success/token/latency 上有显著差异。
- Worker-pool runtime 已证明 failure-aware scheduling 能降低 failed cost 和 zombie sessions。
- 2026-05-21 新增主线：throughput workload 必须从 `model_action_summary.json` 自动派生，避免 scheduler 实验和真实模型失败分布脱节。

当前最大缺口：

- Live browser evidence 仍不足：MiniWoB 现在是 deterministic contract，不是完整 live browser benchmark。
- Serving backend evidence 仍不足：guided JSON 当前是 Transformers + LM Format Enforcer，还没有 SGLang/vLLM server-side structured output 对比。
- Experiment scale 仍不足：Qwen3-4B action table 只有 3 tasks x 3 seeds，需要扩到 20 tasks x 5 seeds，再扩模型。
- Training impact 仍未闭环：目前还没有 GRPO/verl/WebShop 训练曲线。

下一次验收标准：

- [ ] 用 20 MiniWoB contract tasks x 5 seeds 重新跑 Qwen3-4B action-channel table。
- [ ] 从该 `model_action_summary.json` 自动生成 empirical throughput table。
- [ ] 至少报告 default/no-thinking/guided-json 在 useful/hr、failed cost、zombie rate、P95 latency 上的系统后果。
- [ ] 如果 SGLang structured output 能稳定启动，加入 server-side guided JSON 对比。

## 1. 当前基线状态

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

- [x] External systems alignment：
  - [x] documented alignment with OpenAI Structured Outputs, vLLM/SGLang guided decoding, Qwen thinking controls, OpenAI Agents tracing, LangGraph persistence, and rollout-as-a-service systems in `docs/systems_alignment.md`
  - [x] conclusion: RolloutOS must compare against structured decoding/tool-call runtimes, not only prompt-only JSON parsing
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
- [x] Worker-pool throughput simulation：
  - [x] FIFO / retry-only / failure-aware policies
  - [x] metrics: useful trajectories/hour, success/hour, failed cost, zombie rate, P95/P99 latency, utilization
  - [x] CLI：`agentrl-infra run-throughput-bench`
  - [x] 240 requests, 8 workers:
    - [x] FIFO：1750.5 useful/hr, 1289.0 failed cost, 7.1% zombie rate, P95 latency 36.0
    - [x] retry-only：1581.8 useful/hr, 1561.0 failed cost, 7.1% zombie rate, P95 latency 24.0
    - [x] failure-aware：8037.2 useful/hr, 307.0 failed cost, 0% zombie rate, P95 latency 6.0
- [x] Empirical throughput workload from real action-channel summaries：
  - [x] `build_model_action_throughput_workload(...)` reads `model_action_summary.json`
  - [x] derives per-protocol failure probability, replayable failure probability, service time, detection time, and zombie probability
  - [x] `run-throughput-bench --model-action-summary ...` runs FIFO/retry/failure-aware on real model-action failure distributions
  - [x] experiment suite emits `throughput_model_action/model_action_{policy}` when model-action runs are enabled
- [x] Real model tokenizer provenance audit：
  - [x] Qwen/Qwen3-4B：mean 20.0 prompt tokens, max 23, drift 3/3
  - [x] Qwen/Qwen3-8B：mean 20.0 prompt tokens, max 23, drift 3/3
  - [x] meta-llama/Llama-3.1-8B-Instruct：mean 46.7 prompt tokens, max 50, drift 3/3
  - [x] token-native JSONL traces include `model_version`, `tokenizer_hash`, `token_ids`, `loss_mask`
- [x] Real local model generation smoke：
  - [x] Qwen/Qwen3-4B loaded with `AutoModelForCausalLM`
  - [x] 3 prompts, 60 input tokens, 96 generated tokens
  - [x] 21.54 generated tokens/sec after model load
  - [x] mean generated-token logprob -0.103, min logprob -1.331
  - [x] generated-token event traces include `logprobs`
- [x] Real model-to-action MiniWoB contract token-budget / constrained-decoding sweep：
  - [x] Qwen3-4B action protocol ablation: 3 tasks x 3 seeds over 5 protocol-budget cells = 45 episodes
  - [x] 96-token default thinking protocol: 0/9 parsed actions, 0/9 success, 9 invalid-action failures
  - [x] 96-token no-thinking protocol: 9/9 parsed actions, 9/9 success, 727 generated tokens
  - [x] 96-token guided JSON protocol: 9/9 parsed actions, 9/9 success, 727 generated tokens, 16.62 tok/s
  - [x] 384-token default thinking protocol: 7/9 parsed actions, 7/9 success, 2 invalid-action failures, 3096 generated tokens
  - [x] 384-token no-thinking protocol: 9/9 parsed actions, 9/9 success, 727 generated tokens
  - [x] traces include model request/response, proposed action, tool execution, reward, and typed failure events
  - [x] conclusion: structured action channels require runtime-level chat-template, constrained decoding, and token-budget control

## 2. MLSys 主线

### 主线 A：Rollout Runtime

目标：证明 typed failure + replay + scheduler 是一个系统 runtime，而不是算法 trick。

- [x] Event log captures session/model/action/tool/reward/failure events.
- [x] Failure semantics define retryability, salvageability, reset requirement, and training signal.
- [x] Runtime watchdog catches turn budget, repeated action, and no-progress failures.
- [x] Replay validates event-log structure and deterministic executable replay.
- [x] Scheduler uses failure-aware priority and cancellation semantics.
- [ ] Add resource accounting for wall-clock, CPU, GPU, browser, and storage overhead.
- [x] Add token-native trajectory fields for real tokenizer/model provenance:
  - [x] `token_ids`
  - [x] `loss_mask`
  - [x] `model_version`
  - [x] `tokenizer_hash`
- [x] Add generation-time `logprobs` from local Transformers smoke.
- [ ] Add generation-time `logprobs` from SGLang/vLLM rollout workers.
- [ ] Add OpenAI/LangGraph-style trace coverage matrix:
  - [ ] model call
  - [ ] tool call
  - [ ] guardrail/failure event
  - [ ] checkpoint/replay boundary
  - [ ] determinism level

### 主线 A2：Structured Action Channel

目标：和 OpenAI Structured Outputs、vLLM guided decoding、SGLang structured outputs、
Qwen reasoning/tool parser 对齐，证明 action channel 是 runtime/serving 问题。

- [x] Prompt-only JSON parser baseline
- [x] Qwen3 `enable_thinking=False` action protocol
- [x] Token-budget sweep: 96 and 384 max generated tokens
- [x] Local guided JSON backend via LM Format Enforcer / Transformers `prefix_allowed_tokens_fn`
- [ ] Add 1024-token budget point to show whether default thinking fully recovers or remains wasteful
- [ ] Add guided JSON backend:
  - [x] local Transformers guided JSON baseline
  - [ ] vLLM guided JSON if local vLLM works with Qwen3-4B
  - [ ] SGLang constrained JSON if SGLang backend is easier to stabilize
  - [ ] record backend name, schema hash, and decoding mode in traces
- [ ] Add tool-call parser backend:
  - [ ] Qwen parser/tool-call format
  - [ ] invalid tool-call failure type separation from invalid browser action
- [ ] Table target:
  - [ ] prompt-default
  - [ ] prompt-no-thinking
  - [ ] guided-json
  - [ ] tool-call-parser
  - [ ] repair/retry
- [ ] Metrics target:
  - [ ] parse rate
  - [ ] task success
  - [ ] invalid-action rate
  - [ ] generated tokens/action
  - [ ] latency/action
  - [ ] retry count
  - [ ] replayability

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
- [x] worker-pool throughput simulator:
  - [x] per-task service-time distribution
  - [x] zombie session modeling
  - [x] P50/P95/P99 rollout latency
  - [x] useful trajectories/hour
  - [x] worker utilization
- [x] empirical workload adapter from real model-action summaries:
  - [x] action protocol cells become workload classes
  - [x] success/parse/no-progress failures become scheduler-visible failure probabilities
  - [x] measured model latency becomes service time
  - [x] no-progress rate contributes to zombie probability
- [ ] paper table: empirical model-action throughput consequence
  - [ ] FIFO
  - [ ] retry-only
  - [ ] failure-aware/RolloutOS
- [ ] environment-hours/success on live workers
- [ ] real MiniWoB++ throughput run on multiple browser workers
- [ ] GPU/LLM server backpressure model for SGLang/vLLM rollout workers

### 主线 D：Training Impact

目标：证明 RolloutOS 改善训练数据质量和 sample efficiency。

- [ ] OpenAI-compatible policy client
- [ ] SGLang rollout client
- [ ] vLLM rollout client
- [x] tokenizer-level token-native trace export
- [x] generation-level token-native trace export with logprobs for local model smoke
- [x] model-generated browser action trace export on MiniWoB contract smoke
- [ ] generation-level token-native trace export with logprobs for served SGLang/vLLM workers
- [ ] lightweight GRPO prototype for smoke validation
- [ ] connect to AgentRL/verl trainer for formal curves
- [ ] WebShop/AgentBench WS training run:
  - [ ] discard-failures baseline
  - [ ] uniform negative failure baseline
  - [ ] typed failure + salvage
  - [ ] full RolloutOS

## 3. 固定 Workloads

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

## 4. 固定 Metrics

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
- [x] empirical useful trajectories/hour from model-action summary distributions
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

## 5. 近期执行计划

### P0：把 MiniWoB++ 从 contract 推到 live browser

- [ ] Decide and pin live environment package.
- [ ] Add optional dependency group for browser workload.
- [ ] Add live `MiniWoBEnvironmentAdapter`.
- [ ] Keep the current deterministic contract as CI and replay oracle.
- [ ] Add live smoke test gated by environment availability.
- [ ] Run 20-task dev subset with oracle or scripted policy.

### P1：补齐系统指标

- [x] Add empirical throughput workload adapter from model-action summaries.
- [ ] Add latency percentile helpers for live/browser workers.
- [ ] Add trace byte-size and event-count overhead.
- [ ] Add environment lease utilization metrics.
- [ ] Add zombie session definition and measurement.

### P2：实验自动化和论文闭环

- [ ] Add `experiments/configs/exp1_failurebench_detection.json`.
- [ ] Add `experiments/configs/exp2_minwob_lifecycle.json`.
- [ ] Add `experiments/configs/exp3_throughput.json`.
- [ ] Add paper table generation for MiniWoB++ contract/live runs.
- [ ] Replace synthetic-only claims in `paper/main.tex` with real workload methodology.

## 6. 投稿差距

当前距离 MLSys Oral 的主要差距不是代码 skeleton，而是证据强度：

- [ ] 需要至少一个真实环境的大规模系统实验，MiniWoB++ 是第一优先级。
- [ ] 需要 throughput/latency/utilization 结果，而不只是 failure classification。
- [ ] 需要训练曲线或至少 rollout-data-quality 对训练输入的量化影响。
- [ ] 需要更强的 related work 和 architecture figure。
- [ ] 需要把所有实验配置、依赖和 artifact validation 做到一键复现。
