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

## Quick Example

```python
from agentrl_infra import EventLog, EventType, FailureRecord, FailureType

log = EventLog.new(session_id="s1", task_id="webshop", sample_id="42")
log.append(EventType.SESSION_STARTED)
log.append_failure(FailureRecord(type=FailureType.REPETITIVE_LOOP))
```

## Project Status

This is an early research codebase. APIs are intentionally small and explicit so that
the core abstractions can be evaluated before integrating with a full trainer/runtime.

