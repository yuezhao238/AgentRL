from __future__ import annotations

from agentrl_infra.benchmarks.throughput import (
    ThroughputPolicy,
    run_throughput_benchmark,
)


def test_throughput_failure_aware_improves_useful_yield(tmp_path) -> None:
    fifo = run_throughput_benchmark(
        policy=ThroughputPolicy.FIFO,
        worker_count=8,
        output_dir=tmp_path,
        run_id="fifo",
    )
    failure_aware = run_throughput_benchmark(
        policy=ThroughputPolicy.FAILURE_AWARE,
        worker_count=8,
        output_dir=tmp_path,
        run_id="failure_aware",
    )

    assert failure_aware.summary.useful_trajectories_per_hour > (
        fifo.summary.useful_trajectories_per_hour
    )
    assert failure_aware.summary.zombie_session_rate < fifo.summary.zombie_session_rate
    assert (tmp_path / "failure_aware" / "throughput_summary.json").exists()


def test_throughput_retry_only_adds_attempts() -> None:
    fifo = run_throughput_benchmark(policy=ThroughputPolicy.FIFO, worker_count=8)
    retry = run_throughput_benchmark(policy=ThroughputPolicy.RETRY_ONLY, worker_count=8)

    assert retry.summary.attempt_count > fifo.summary.attempt_count
    assert retry.summary.failed_cost_units > 0
