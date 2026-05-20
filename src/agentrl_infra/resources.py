from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from .failures import FailureRecord
from .runner import StepResult


class ResourceState(StrEnum):
    CREATED = "created"
    READY = "ready"
    LEASED = "leased"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass(frozen=True)
class HealthCheckResult:
    healthy: bool
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentLease:
    environment_id: str
    worker_id: str
    state: ResourceState
    snapshot_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    observation: dict[str, Any]
    reward: float = 0.0
    done: bool = False
    failure: FailureRecord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_step_result(self) -> StepResult:
        return StepResult(
            observation=self.observation,
            reward=self.reward,
            done=self.done,
            failure=self.failure,
            info=self.metadata,
        )


@runtime_checkable
class EnvironmentAdapter(Protocol):
    environment_id: str

    def reset(self) -> dict[str, Any]:
        """Reset the environment and return an initial observation."""

    def step(self, action: dict[str, Any]) -> StepResult:
        """Apply an action and return a typed transition."""

    def snapshot(self) -> str:
        """Create a replay/restoration snapshot and return its id."""

    def restore(self, snapshot_id: str) -> None:
        """Restore a previously created snapshot."""

    def health_check(self) -> HealthCheckResult:
        """Return whether the environment is safe for use."""

    def contamination_check(self) -> HealthCheckResult:
        """Return whether state is clean enough for reuse."""

    def close(self) -> None:
        """Release local or remote resources."""


class ToolAdapter(Protocol):
    name: str

    def validate(self, action: dict[str, Any]) -> FailureRecord | None:
        """Return a typed failure if the action is invalid."""

    def execute(self, action: dict[str, Any]) -> ToolResult:
        """Execute a tool action and return structured output."""


class InMemoryEnvironmentPool:
    def __init__(self) -> None:
        self._leases: dict[str, EnvironmentLease] = {}

    def register(self, environment_id: str, worker_id: str = "local") -> EnvironmentLease:
        lease = EnvironmentLease(
            environment_id=environment_id,
            worker_id=worker_id,
            state=ResourceState.READY,
        )
        self._leases[environment_id] = lease
        return lease

    def lease(self, environment_id: str, worker_id: str) -> EnvironmentLease:
        current = self._leases.get(environment_id)
        if current is None:
            current = self.register(environment_id, worker_id)
        if current.state == ResourceState.LEASED:
            raise RuntimeError(f"environment already leased: {environment_id}")
        lease = EnvironmentLease(
            environment_id=environment_id,
            worker_id=worker_id,
            state=ResourceState.LEASED,
            snapshot_id=current.snapshot_id,
            metadata=current.metadata,
        )
        self._leases[environment_id] = lease
        return lease

    def release(
        self,
        environment_id: str,
        *,
        healthy: bool = True,
        snapshot_id: str | None = None,
    ) -> EnvironmentLease:
        current = self._leases[environment_id]
        lease = EnvironmentLease(
            environment_id=environment_id,
            worker_id=current.worker_id,
            state=ResourceState.READY if healthy else ResourceState.UNHEALTHY,
            snapshot_id=snapshot_id or current.snapshot_id,
            metadata=current.metadata,
        )
        self._leases[environment_id] = lease
        return lease

    def get(self, environment_id: str) -> EnvironmentLease | None:
        return self._leases.get(environment_id)
