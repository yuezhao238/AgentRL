# AgentRL Infra

Failure-aware and replayable rollout infrastructure primitives for Agent RL research.

This repository starts from the research proposal in `proposal_failure_aware_agentrl.tex`
and implements a minimal Python substrate for:

- typed rollout failure taxonomy;
- event-sourced trajectory logs;
- session watchdog and loop detection;
- replay metadata/reporting;
- failure-aware rollout scheduling.

The goal is not to replace AgentRL, verl, or other trainers. The package is a research
prototype for the runtime layer beneath long-horizon agent RL rollouts.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

For the full experiment/runtime environment on the current A100 machine:

```bash
uv sync --extra dev --extra experiments --extra sglang
```

Switch to the backup vLLM backend with:

```bash
uv sync --extra dev --extra experiments --extra vllm
```

## Quick Example

```python
from agentrl_infra import EventLog, EventType, FailureRecord, FailureType

log = EventLog.new(session_id="s1", task_id="webshop", sample_id="42")
log.append(EventType.SESSION_STARTED)
log.append_failure(FailureRecord(type=FailureType.REPETITIVE_LOOP))
```

## FailureBench

Run the deterministic synthetic failure benchmark:

```bash
uv run agentrl-infra run-failurebench --output-dir runs/failurebench
```

Scheduler policies are explicit:

```bash
uv run agentrl-infra run-failurebench --scheduler-policy failure_aware
uv run agentrl-infra run-failurebench --scheduler-policy fifo
uv run agentrl-infra run-failurebench --scheduler-policy none
```

This writes:

- `config.json`
- `schedule.json` when a scheduler policy is used
- `traces/*.jsonl`
- `metrics.json`
- `summary.json`
- `summary.md`

Summarize an existing run:

```bash
uv run agentrl-infra summarize-runs runs/failurebench/<run_id>/metrics.json
```

Inspect a trace:

```bash
uv run agentrl-infra inspect-trace runs/failurebench/<run_id>/traces/<sample>.jsonl
```

Execute deterministic replay for FailureBench:

```bash
uv run agentrl-infra replay-trace runs/failurebench/<run_id>/traces/<sample>.jsonl --execute
```

Validate and replay a full run:

```bash
uv run agentrl-infra validate-run runs/failurebench/<run_id>
uv run agentrl-infra replay-run runs/failurebench/<run_id>
```

Compare two runs:

```bash
uv run agentrl-infra compare-runs runs/failurebench/a/metrics.json runs/failurebench/b/metrics.json
```

Run the synthetic environment lifecycle benchmark:

```bash
uv run agentrl-infra run-lifecycle-bench --policy recreate
uv run agentrl-infra run-lifecycle-bench --policy blind_reuse
uv run agentrl-infra run-lifecycle-bench --policy contamination_aware
```

Run the deterministic MiniWoB browser contract harness:

```bash
uv run agentrl-infra run-miniwob-contract --tasks click-button,enter-text --seeds 1000,1001
uv run agentrl-infra run-miniwob-contract --tasks all --policy repeated_action
```

The contract harness fixes the browser action schema, DOM hash observation format,
stale-DOM checks, no-progress watchdog behavior, and contamination/health checks before
binding the same interface to a live MiniWoB++ browser environment.

Run the deterministic worker-pool throughput benchmark:

```bash
uv run agentrl-infra run-throughput-bench --policy fifo --workers 8
uv run agentrl-infra run-throughput-bench --policy retry_only --workers 8
uv run agentrl-infra run-throughput-bench --policy failure_aware --workers 8
```

Run the local experiment suite and generate paper-ready CSV/LaTeX tables:

```bash
uv run agentrl-infra run-experiment-suite --suite-id local-suite
uv run agentrl-infra run-experiment-suite --config-path experiments/configs/local_suite.json
```

This runs FailureBench baselines, the RolloutOS configuration, lifecycle policies,
MiniWoB browser contract policies, artifact validation, batch replay, and writes tables under
`runs/experiments/<suite_id>/tables/`.

## Project Status

This is an early research codebase. APIs are intentionally small and explicit so that
the core abstractions can be evaluated before integrating with a full trainer/runtime.
