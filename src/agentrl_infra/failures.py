from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FailureType(StrEnum):
    AGENT_INVALID_ACTION = "agent_invalid_action"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    ENVIRONMENT_CRASH = "environment_crash"
    ENVIRONMENT_CONTAMINATION = "environment_contamination"
    CONTEXT_LIMIT = "context_limit"
    REPETITIVE_LOOP = "repetitive_loop"
    NO_PROGRESS = "no_progress"
    RATE_LIMIT = "rate_limit"
    SCHEDULER_CANCELLED = "scheduler_cancelled"
    UNKNOWN_RUNTIME_ERROR = "unknown_runtime_error"


class FailureSemantics(BaseModel):
    """Operational semantics attached to a rollout failure."""

    attributable_to_policy: bool
    retryable: bool
    salvageable: bool
    reset_environment: bool
    emit_training_signal: bool
    default_reward_delta: float = 0.0


DEFAULT_FAILURE_SEMANTICS: dict[FailureType, FailureSemantics] = {
    FailureType.AGENT_INVALID_ACTION: FailureSemantics(
        attributable_to_policy=True,
        retryable=False,
        salvageable=True,
        reset_environment=False,
        emit_training_signal=True,
        default_reward_delta=-0.2,
    ),
    FailureType.TOOL_TIMEOUT: FailureSemantics(
        attributable_to_policy=False,
        retryable=True,
        salvageable=True,
        reset_environment=False,
        emit_training_signal=False,
    ),
    FailureType.TOOL_EXECUTION_ERROR: FailureSemantics(
        attributable_to_policy=False,
        retryable=True,
        salvageable=True,
        reset_environment=False,
        emit_training_signal=False,
    ),
    FailureType.ENVIRONMENT_CRASH: FailureSemantics(
        attributable_to_policy=False,
        retryable=True,
        salvageable=False,
        reset_environment=True,
        emit_training_signal=False,
    ),
    FailureType.ENVIRONMENT_CONTAMINATION: FailureSemantics(
        attributable_to_policy=False,
        retryable=True,
        salvageable=False,
        reset_environment=True,
        emit_training_signal=False,
    ),
    FailureType.CONTEXT_LIMIT: FailureSemantics(
        attributable_to_policy=True,
        retryable=False,
        salvageable=True,
        reset_environment=False,
        emit_training_signal=True,
        default_reward_delta=-0.1,
    ),
    FailureType.REPETITIVE_LOOP: FailureSemantics(
        attributable_to_policy=True,
        retryable=False,
        salvageable=True,
        reset_environment=False,
        emit_training_signal=True,
        default_reward_delta=-0.2,
    ),
    FailureType.NO_PROGRESS: FailureSemantics(
        attributable_to_policy=True,
        retryable=False,
        salvageable=True,
        reset_environment=False,
        emit_training_signal=True,
        default_reward_delta=-0.1,
    ),
    FailureType.RATE_LIMIT: FailureSemantics(
        attributable_to_policy=False,
        retryable=True,
        salvageable=True,
        reset_environment=False,
        emit_training_signal=False,
    ),
    FailureType.SCHEDULER_CANCELLED: FailureSemantics(
        attributable_to_policy=False,
        retryable=True,
        salvageable=True,
        reset_environment=False,
        emit_training_signal=False,
    ),
    FailureType.UNKNOWN_RUNTIME_ERROR: FailureSemantics(
        attributable_to_policy=False,
        retryable=True,
        salvageable=False,
        reset_environment=True,
        emit_training_signal=False,
    ),
}


class FailureRecord(BaseModel):
    type: FailureType
    message: str | None = None
    source: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    semantics: FailureSemantics | None = None

    def resolved_semantics(self) -> FailureSemantics:
        return self.semantics or DEFAULT_FAILURE_SEMANTICS[self.type]

