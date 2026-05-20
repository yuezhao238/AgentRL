from __future__ import annotations

from agentrl_infra.benchmarks.failurebench import run_failurebench
from agentrl_infra.compare import compare_metric_files
from agentrl_infra.metrics import save_metrics


def test_compare_metric_files(tmp_path) -> None:
    metrics_a = run_failurebench(
        output_dir=tmp_path,
        run_id="a",
        split="dev",
        dev_seeds_per_type=1,
        test_seeds_per_type=0,
    )
    metrics_b = run_failurebench(
        output_dir=tmp_path,
        run_id="b",
        split="dev",
        dev_seeds_per_type=1,
        test_seeds_per_type=0,
    )
    save_metrics(tmp_path / "a", metrics_a)
    save_metrics(tmp_path / "b", metrics_b)

    comparison = compare_metric_files(
        tmp_path / "a" / "metrics.json",
        tmp_path / "b" / "metrics.json",
    )

    assert comparison.baseline_summary.episode_count == 8
    assert comparison.candidate_summary.episode_count == 8
    assert comparison.macro_f1_delta == 0.0
