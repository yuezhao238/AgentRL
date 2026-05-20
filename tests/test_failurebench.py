from __future__ import annotations

from agentrl_infra.benchmarks.failurebench import (
    FailureBenchScenario,
    build_failurebench_cases,
    run_failurebench,
    run_failurebench_case,
)
from agentrl_infra.metrics import save_metrics, summarize_metrics


def test_failurebench_builds_expected_case_count() -> None:
    cases = build_failurebench_cases(split="all", dev_seeds_per_type=2, test_seeds_per_type=3)

    assert len(cases) == len(FailureBenchScenario) * 5
    assert {case.split for case in cases} == {"dev", "test"}


def test_failurebench_case_detects_oracle_failure() -> None:
    cases = build_failurebench_cases(split="dev", dev_seeds_per_type=1, test_seeds_per_type=0)

    for case in cases:
        result = run_failurebench_case(case)
        assert result.failure is not None
        assert result.failure.type == case.oracle_failure_type


def test_failurebench_run_writes_metrics(tmp_path) -> None:
    metrics = run_failurebench(
        output_dir=tmp_path,
        run_id="r1",
        split="dev",
        dev_seeds_per_type=1,
        test_seeds_per_type=0,
    )
    summary = save_metrics(tmp_path / "r1", metrics)

    assert summary.episode_count == len(FailureBenchScenario)
    assert summary.detection_accuracy == 1.0
    assert (tmp_path / "r1" / "metrics.json").exists()
    assert (tmp_path / "r1" / "summary.md").exists()
    assert len(list((tmp_path / "r1" / "traces").glob("*.jsonl"))) == len(FailureBenchScenario)


def test_failurebench_summary_has_macro_f1(tmp_path) -> None:
    metrics = run_failurebench(
        output_dir=tmp_path,
        run_id="r-unused",
        split="dev",
        dev_seeds_per_type=1,
        test_seeds_per_type=0,
    )

    summary = summarize_metrics(metrics)

    assert summary.macro_f1 == 1.0
