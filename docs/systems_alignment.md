# RolloutOS Systems Alignment

Last updated: 2026-05-20

This note records where RolloutOS must align with existing LLM agent and serving
systems. The purpose is to keep the project from overfitting to its own
benchmarks.

## External System Patterns

### Structured actions are a decoding/runtime problem

OpenAI Structured Outputs, vLLM guided decoding, and SGLang structured outputs
all treat JSON/tool output reliability as a runtime constraint rather than a
prompting convention. The shared pattern is:

- The user-facing tool/action schema is explicit.
- The serving layer constrains or parses the model output against that schema.
- Invalid structure is measured as a runtime failure, not hidden inside task
  success.

Implication for RolloutOS: `parse_browser_action()` is only the compatibility
baseline. The main action-channel experiment must compare prompt-only output
against guided JSON/tool-call backends.

### Reasoning and action channels are separate

Reasoning models such as Qwen3 can spend output budget on a hidden or visible
thinking channel. Qwen and vLLM expose explicit controls for thinking/reasoning
parsing. Tool/action generation therefore needs a policy:

- when reasoning is allowed,
- when reasoning is disabled,
- how many output tokens each channel may consume,
- how structured actions are extracted or constrained.

Implication for RolloutOS: token budget is a first-class runtime resource. The
paper should report parse rate, success rate, generated tokens, latency, and
invalid-action failures under multiple budgets, not a single short smoke test.

### Agent runtimes persist traces and checkpoints

OpenAI Agents tracing and LangGraph persistence make model calls, tool calls,
handoffs, guardrails, graph state, and checkpoints observable. Mature agent
runtimes are not simple chat loops; they expose execution state for debugging,
resumption, and fault tolerance.

Implication for RolloutOS: event-sourced JSONL traces should be positioned as a
rollout-training trace format with failure attribution, loss masks, token IDs,
logprobs, environment provenance, and replay determinism levels.

### Rollout systems separate execution from training

Agent RL infrastructure and rollout-as-a-service systems decouple environment
execution from RL training. Their core system question is how to generate,
observe, schedule, and reuse multi-turn agent rollouts at scale.

Implication for RolloutOS: the main claim should not be a new agent algorithm.
It should be a runtime layer that improves useful trajectories per unit cost by
combining typed failures, replay boundaries, lifecycle control, and scheduling.

## Alignment Matrix

| External pattern | RolloutOS status | Gap to close |
| --- | --- | --- |
| JSON/tool schema is explicit | BrowserAction schema exists | Add guided JSON/tool-call backend |
| Structured output is constrained | Prompt/parser baseline only | Add vLLM or SGLang guided decoding experiment |
| Reasoning channel is controlled | Qwen3 `no_thinking` ablation exists | Add budget sweep and parser/constrained variants |
| Trace/checkpoint is first-class | JSONL event stream exists | Add determinism-level and replay-boundary table |
| Runtime failures drive policy | Failure taxonomy and scheduler exist | Connect real action failures to scheduler simulation |
| Rollout execution is decoupled | CLI suite exists | Add service-style worker API or long-running runner |

## Required Experiments Before Submission

1. Action-channel systems ablation:
   - protocols: prompt default, no-thinking prompt, guided JSON, tool-call parser
   - budgets: at least 96, 384, 1024
   - metrics: parse rate, success rate, invalid-action rate, generated tokens,
     latency, retry count, replayability

2. Real environment validation:
   - keep MiniWoB contract for deterministic action-channel isolation
   - add live MiniWoB/WebShop/ALFWorld evidence for end-to-end rollout behavior

3. Runtime policy evaluation:
   - feed observed invalid-action/no-progress/tool-timeout distributions into the
     worker-pool scheduler
   - report useful trajectories/hour, failed cost, zombie worker rate, and P95
     latency under FIFO, retry-only, and failure-aware scheduling

## References to Track

- OpenAI Structured Outputs: https://openai.com/index/introducing-structured-outputs-in-the-api/
- OpenAI Agents tracing: https://openai.github.io/openai-agents-python/tracing/
- vLLM structured outputs and reasoning parser documentation: https://docs.vllm.ai/
- SGLang structured outputs documentation: https://docs.sglang.ai/
- Qwen function calling documentation: https://qwen.readthedocs.io/en/stable/framework/function_call.html
- LangGraph persistence documentation: https://langchain-ai.github.io/langgraph/concepts/persistence/
