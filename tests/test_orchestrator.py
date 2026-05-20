from __future__ import annotations

from agentrl_infra.benchmarks.failurebench import run_failurebench_scheduled
from agentrl_infra.scheduler import RolloutRequest, TaskStats


def test_failurebench_scheduled_run_records_decisions(tmp_path) -> None:
    summary = run_failurebench_scheduled(
        output_dir=tmp_path,
        run_id="scheduled",
        split="dev",
        dev_seeds_per_type=1,
        test_seeds_per_type=0,
        scheduler_policy="failure_aware",
    )

    assert summary.policy == "failure_aware"
    assert summary.episode_count == 8
    assert summary.skipped_count == 0
    assert all(decision.metrics.correct_detection for decision in summary.decisions)


def test_scheduler_full_capacity_skips_pending() -> None:
    from agentrl_infra.orchestrator import BatchOrchestrator
    from agentrl_infra.scheduler import FailureAwareScheduler

    request = RolloutRequest(task_id="full", sample_id="a")
    stats = {"full": TaskStats(active_sessions=1, capacity=1)}

    summary = BatchOrchestrator(FailureAwareScheduler()).run(
        [request],
        stats,
        lambda _: (_ for _ in ()).throw(AssertionError("should not execute")),
    )

    assert summary.skipped_count == 1
    assert summary.episode_count == 0
