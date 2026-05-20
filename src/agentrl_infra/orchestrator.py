from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .metrics import EpisodeMetrics
from .scheduler import FailureAwareScheduler, RolloutRequest, SchedulerDecision, TaskStats

EpisodeCallable = Callable[[RolloutRequest], EpisodeMetrics]


class ScheduledEpisodeMetrics(BaseModel):
    order: int
    scheduler_score: float
    scheduler_reason: str
    metrics: EpisodeMetrics


class SchedulerRunSummary(BaseModel):
    policy: str
    episode_count: int
    decision_count: int
    skipped_count: int
    decisions: list[ScheduledEpisodeMetrics] = Field(default_factory=list)


@dataclass
class BatchOrchestrator:
    scheduler: FailureAwareScheduler

    def run(
        self,
        requests: list[RolloutRequest],
        stats_by_task: dict[str, TaskStats],
        execute: EpisodeCallable,
    ) -> SchedulerRunSummary:
        pending = list(requests)
        decisions: list[ScheduledEpisodeMetrics] = []
        order = 0

        while pending:
            decision = self.scheduler.choose(pending, stats_by_task)
            if decision is None:
                break

            pending.remove(decision.request)
            stats = stats_by_task.setdefault(decision.request.task_id, TaskStats())
            stats.active_sessions += 1
            try:
                metrics = execute(decision.request)
            finally:
                stats.active_sessions = max(0, stats.active_sessions - 1)

            if metrics.success:
                stats.success_count += 1
            else:
                stats.failure_count += 1
            stats.mean_cost = _moving_average(stats.mean_cost, max(metrics.turn_count, 1))

            decisions.append(
                ScheduledEpisodeMetrics(
                    order=order,
                    scheduler_score=decision.score,
                    scheduler_reason=decision.reason,
                    metrics=metrics,
                )
            )
            order += 1

        return SchedulerRunSummary(
            policy="failure_aware",
            episode_count=len(decisions),
            decision_count=len(decisions),
            skipped_count=len(pending),
            decisions=decisions,
        )


def fifo_decisions(
    requests: list[RolloutRequest],
    execute: EpisodeCallable,
) -> SchedulerRunSummary:
    decisions: list[ScheduledEpisodeMetrics] = []
    for order, request in enumerate(requests):
        metrics = execute(request)
        decision = SchedulerDecision(
            request=request,
            score=request.priority,
            reason="fifo",
        )
        decisions.append(
            ScheduledEpisodeMetrics(
                order=order,
                scheduler_score=decision.score,
                scheduler_reason=decision.reason,
                metrics=metrics,
            )
        )
    return SchedulerRunSummary(
        policy="fifo",
        episode_count=len(decisions),
        decision_count=len(decisions),
        skipped_count=0,
        decisions=decisions,
    )


def _moving_average(previous: float, observed: float, alpha: float = 0.1) -> float:
    return (1 - alpha) * previous + alpha * observed
