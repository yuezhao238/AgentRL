from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from ..failures import FailureRecord, FailureType
from ..resources import HealthCheckResult, InMemoryEnvironmentPool, ResourceState
from ..runner import StepResult


class ReusePolicy(StrEnum):
    RECREATE = "recreate"
    BLIND_REUSE = "blind_reuse"
    FIXED_TTL = "fixed_ttl"
    HEALTH_CHECK = "health_check"
    CONTAMINATION_AWARE = "contamination_aware"


class LifecycleEpisodeMetrics(BaseModel):
    policy: ReusePolicy
    episode_index: int
    environment_id: str
    success: bool
    reset_count: int
    restore_count: int
    reuse_count: int
    contamination_detected: bool
    contamination_induced_failure: bool
    reset_latency_units: float
    total_cost_units: float


class LifecycleSummary(BaseModel):
    policy: ReusePolicy
    episode_count: int
    success_count: int
    reset_count: int
    restore_count: int
    reuse_count: int
    contamination_count: int
    contamination_induced_failures: int
    total_cost_units: float
    cost_per_success: float


@dataclass
class MutableSyntheticEnvironment:
    environment_id: str
    reset_latency_units: float = 1.0
    restore_latency_units: float = 0.2
    step_cost_units: float = 0.1
    dirty: bool = False
    reset_count: int = 0

    def reset(self) -> dict[str, object]:
        self.dirty = False
        self.reset_count += 1
        return {"environment_id": self.environment_id, "dirty": self.dirty}

    def step(self, action: dict[str, object]) -> StepResult:
        if self.dirty:
            return StepResult(
                observation={"dirty": True},
                failure=FailureRecord(
                    type=FailureType.ENVIRONMENT_CONTAMINATION,
                    message="dirty mutable state leaked into next episode",
                    source="lifecycle_benchmark",
                ),
                info={"contamination_induced_failure": True},
            )
        self.dirty = bool(action.get("mutates_state", True))
        return StepResult(
            observation={"dirty": self.dirty},
            reward=1.0,
            done=True,
            info={"contamination_induced_failure": False},
        )

    def snapshot(self) -> str:
        return f"{self.environment_id}-reset-{self.reset_count}"

    def restore(self, snapshot_id: str) -> None:
        self.dirty = False

    def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=not self.dirty, details={"dirty": self.dirty})

    def contamination_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=not self.dirty, details={"dirty": self.dirty})

    def close(self) -> None:
        self.dirty = False


def run_lifecycle_benchmark(
    *,
    policy: ReusePolicy,
    episodes: int = 100,
    ttl: int = 5,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> tuple[list[LifecycleEpisodeMetrics], LifecycleSummary]:
    pool = InMemoryEnvironmentPool()
    env = MutableSyntheticEnvironment(environment_id="mutable-env-0")
    pool.register(env.environment_id)
    metrics: list[LifecycleEpisodeMetrics] = []
    reuse_age = 0
    reuse_count = 0
    reset_count = 0
    restore_count = 0

    for index in range(episodes):
        should_reset = _should_reset(policy, env, reuse_age, ttl)
        should_restore = False
        contamination_detected = False
        if policy == ReusePolicy.CONTAMINATION_AWARE:
            contamination_detected = not env.contamination_check().healthy
            should_restore = contamination_detected and env.reset_count > 0
            should_reset = should_reset or (contamination_detected and env.reset_count == 0)
        if policy == ReusePolicy.HEALTH_CHECK:
            should_reset = should_reset or not env.health_check().healthy

        if should_reset:
            env.reset()
            reset_count += 1
            reuse_age = 0
            setup_cost = env.reset_latency_units
        elif should_restore:
            env.restore(env.snapshot())
            restore_count += 1
            reuse_age = 0
            setup_cost = env.restore_latency_units
        else:
            reuse_count += 1
            setup_cost = 0.0

        lease = pool.lease(env.environment_id, "lifecycle-worker")
        step = env.step({"mutates_state": True})
        success = step.failure is None
        contamination_failure = bool(step.info.get("contamination_induced_failure", False))
        healthy = success or policy == ReusePolicy.BLIND_REUSE
        pool.release(
            lease.environment_id,
            healthy=healthy,
            snapshot_id=env.snapshot() if lease.state == ResourceState.LEASED else None,
        )
        reuse_age += 1
        metrics.append(
            LifecycleEpisodeMetrics(
                policy=policy,
                episode_index=index,
                environment_id=env.environment_id,
                success=success,
                reset_count=reset_count,
                restore_count=restore_count,
                reuse_count=reuse_count,
                contamination_detected=contamination_detected,
                contamination_induced_failure=contamination_failure,
                reset_latency_units=setup_cost,
                total_cost_units=setup_cost + env.step_cost_units,
            )
        )

    summary = summarize_lifecycle_metrics(policy, metrics)
    if output_dir and run_id:
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "lifecycle_metrics.json").write_text(
            json.dumps([metric.model_dump(mode="json") for metric in metrics], indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "lifecycle_summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    return metrics, summary


def summarize_lifecycle_metrics(
    policy: ReusePolicy,
    metrics: list[LifecycleEpisodeMetrics],
) -> LifecycleSummary:
    success_count = sum(1 for metric in metrics if metric.success)
    total_cost = sum(metric.total_cost_units for metric in metrics)
    return LifecycleSummary(
        policy=policy,
        episode_count=len(metrics),
        success_count=success_count,
        reset_count=max((metric.reset_count for metric in metrics), default=0),
        restore_count=max((metric.restore_count for metric in metrics), default=0),
        reuse_count=max((metric.reuse_count for metric in metrics), default=0),
        contamination_count=sum(1 for metric in metrics if metric.contamination_detected),
        contamination_induced_failures=sum(
            1 for metric in metrics if metric.contamination_induced_failure
        ),
        total_cost_units=total_cost,
        cost_per_success=total_cost / success_count if success_count else float("inf"),
    )


def _should_reset(
    policy: ReusePolicy,
    env: MutableSyntheticEnvironment,
    reuse_age: int,
    ttl: int,
) -> bool:
    if policy == ReusePolicy.RECREATE:
        return True
    if policy == ReusePolicy.BLIND_REUSE:
        return env.reset_count == 0
    if policy == ReusePolicy.FIXED_TTL:
        return env.reset_count == 0 or reuse_age >= ttl
    if policy in {ReusePolicy.HEALTH_CHECK, ReusePolicy.CONTAMINATION_AWARE}:
        return env.reset_count == 0
    raise ValueError(f"unknown reuse policy: {policy}")
