from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutRequest:
    task_id: str
    sample_id: str
    priority: float = 1.0
    estimated_cost: float = 1.0


@dataclass
class TaskStats:
    active_sessions: int = 0
    capacity: int = 1
    success_count: int = 0
    failure_count: int = 0
    reward_variance: float = 0.0
    mean_cost: float = 1.0
    policy_lag: float = 0.0

    @property
    def available_capacity(self) -> int:
        return max(0, self.capacity - self.active_sessions)

    @property
    def failure_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.failure_count / total if total else 0.0


@dataclass(frozen=True)
class SchedulerDecision:
    request: RolloutRequest
    score: float
    reason: str


class FailureAwareScheduler:
    def __init__(
        self,
        failure_penalty: float = 2.0,
        cost_penalty: float = 0.5,
        policy_lag_penalty: float = 0.25,
        reward_variance_bonus: float = 0.2,
    ) -> None:
        self.failure_penalty = failure_penalty
        self.cost_penalty = cost_penalty
        self.policy_lag_penalty = policy_lag_penalty
        self.reward_variance_bonus = reward_variance_bonus

    def score(self, request: RolloutRequest, stats: TaskStats) -> float:
        if stats.available_capacity <= 0:
            return float("-inf")
        return (
            request.priority
            + self.reward_variance_bonus * stats.reward_variance
            - self.failure_penalty * stats.failure_rate
            - self.cost_penalty * max(request.estimated_cost, stats.mean_cost)
            - self.policy_lag_penalty * stats.policy_lag
        )

    def choose(
        self,
        requests: list[RolloutRequest],
        stats_by_task: dict[str, TaskStats],
    ) -> SchedulerDecision | None:
        best: SchedulerDecision | None = None
        for request in requests:
            stats = stats_by_task.get(request.task_id, TaskStats())
            score = self.score(request, stats)
            decision = SchedulerDecision(
                request=request,
                score=score,
                reason=(
                    f"capacity={stats.available_capacity}, "
                    f"failure_rate={stats.failure_rate:.3f}, "
                    f"mean_cost={stats.mean_cost:.3f}, "
                    f"policy_lag={stats.policy_lag:.3f}"
                ),
            )
            if best is None or decision.score > best.score:
                best = decision
        if best is None or best.score == float("-inf"):
            return None
        return best

