from agentrl_infra import RolloutRequest, TaskStats
from agentrl_infra.scheduler import FailureAwareScheduler


def test_scheduler_prefers_available_low_failure_task() -> None:
    scheduler = FailureAwareScheduler()
    requests = [
        RolloutRequest(task_id="unstable", sample_id="a", estimated_cost=1),
        RolloutRequest(task_id="stable", sample_id="b", estimated_cost=1),
    ]
    stats = {
        "unstable": TaskStats(capacity=4, failure_count=9, success_count=1),
        "stable": TaskStats(capacity=4, failure_count=1, success_count=9),
    }

    decision = scheduler.choose(requests, stats)

    assert decision is not None
    assert decision.request.task_id == "stable"


def test_scheduler_returns_none_when_all_tasks_are_full() -> None:
    scheduler = FailureAwareScheduler()
    requests = [RolloutRequest(task_id="full", sample_id="a")]
    stats = {"full": TaskStats(active_sessions=1, capacity=1)}

    assert scheduler.choose(requests, stats) is None

