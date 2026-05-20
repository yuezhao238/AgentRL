from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .events import EventLog, EventType
from .failures import FailureRecord, FailureType
from .session import SessionRuntime, SessionRuntimeConfig, SessionState


class Policy(Protocol):
    def next_action(self, observation: dict[str, Any], log: EventLog) -> dict[str, Any]:
        """Return the next environment/tool action for a rollout."""


class Environment(Protocol):
    def reset(self) -> dict[str, Any]:
        """Reset the environment and return the initial observation."""

    def step(self, action: dict[str, Any]) -> StepResult:
        """Apply an action and return the resulting transition."""


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    reward: float = 0.0
    done: bool = False
    failure: FailureRecord | None = None
    info: dict[str, Any] = field(default_factory=dict)


class EpisodeResult(BaseModel):
    session_id: str
    task_id: str
    sample_id: str | None = None
    state: SessionState
    event_log: EventLog
    total_reward: float = 0.0
    turn_count: int = 0
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def failure(self) -> FailureRecord | None:
        failures = self.event_log.failures()
        return failures[-1] if failures else None


class EpisodeRunner:
    def __init__(self, config: SessionRuntimeConfig | None = None) -> None:
        self.config = config or SessionRuntimeConfig()

    def run(
        self,
        *,
        session_id: str,
        task_id: str,
        policy: Policy,
        environment: Environment,
        sample_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeResult:
        started = perf_counter()
        runtime = SessionRuntime(
            session_id=session_id,
            task_id=task_id,
            sample_id=sample_id,
            config=self.config,
        )
        runtime.start(metadata=metadata or {})
        total_reward = 0.0

        try:
            observation = environment.reset()
            failure = runtime.record_observation(observation)
            if failure:
                return self._result(runtime, total_reward, started, metadata)

            while runtime.state == SessionState.RUNNING:
                runtime.event_log.append(
                    EventType.MODEL_REQUESTED,
                    payload={"observation": observation, "turn": runtime.turn_count},
                )
                action = policy.next_action(observation, runtime.event_log)
                runtime.event_log.append(
                    EventType.MODEL_RESPONDED,
                    payload={"action": action, "turn": runtime.turn_count},
                )

                failure = runtime.record_action(action)
                if failure:
                    break

                step = environment.step(action)
                runtime.event_log.append(
                    EventType.TOOL_CALL_EXECUTED,
                    payload={"action": action, "info": step.info},
                )
                total_reward += step.reward
                runtime.event_log.append(
                    EventType.REWARD_EMITTED,
                    payload={"reward": step.reward, "total_reward": total_reward},
                )

                if step.failure:
                    runtime.fail(step.failure)
                    break

                observation = step.observation
                failure = runtime.record_observation(observation)
                if failure:
                    break

                if step.done:
                    runtime.complete(total_reward=total_reward, info=step.info)
                    break

            if runtime.state == SessionState.RUNNING:
                runtime.fail(
                    FailureRecord(
                        type=FailureType.UNKNOWN_RUNTIME_ERROR,
                        message="episode loop exited without terminal state",
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            runtime.fail(
                FailureRecord(
                    type=FailureType.UNKNOWN_RUNTIME_ERROR,
                    message=str(exc),
                    source=exc.__class__.__name__,
                )
            )

        return self._result(runtime, total_reward, started, metadata)

    @staticmethod
    def _result(
        runtime: SessionRuntime,
        total_reward: float,
        started: float,
        metadata: dict[str, Any] | None,
    ) -> EpisodeResult:
        return EpisodeResult(
            session_id=runtime.session_id,
            task_id=runtime.task_id,
            sample_id=runtime.sample_id,
            state=runtime.state,
            event_log=runtime.event_log,
            total_reward=total_reward,
            turn_count=runtime.turn_count,
            latency_seconds=perf_counter() - started,
            metadata=metadata or {},
        )
