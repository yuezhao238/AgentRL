from __future__ import annotations

from agentrl_infra.benchmarks.lifecycle import ReusePolicy, run_lifecycle_benchmark


def test_lifecycle_recreate_avoids_contamination_with_higher_cost() -> None:
    _, recreate = run_lifecycle_benchmark(policy=ReusePolicy.RECREATE, episodes=5)
    _, blind = run_lifecycle_benchmark(policy=ReusePolicy.BLIND_REUSE, episodes=5)

    assert recreate.contamination_induced_failures == 0
    assert blind.contamination_induced_failures > 0
    assert recreate.total_cost_units > blind.total_cost_units


def test_lifecycle_contamination_aware_resets_dirty_state() -> None:
    _, recreate = run_lifecycle_benchmark(policy=ReusePolicy.RECREATE, episodes=5)
    _, summary = run_lifecycle_benchmark(policy=ReusePolicy.CONTAMINATION_AWARE, episodes=5)

    assert summary.contamination_induced_failures == 0
    assert summary.contamination_count > 0
    assert summary.success_count == 5
    assert summary.restore_count > 0
    assert summary.total_cost_units < recreate.total_cost_units
