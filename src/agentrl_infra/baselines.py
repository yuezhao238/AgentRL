from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from .benchmarks.failurebench import FailureBenchCase, build_failurebench_cases
from .failures import FailureType
from .metrics import EpisodeMetrics, RunSummary, save_metrics, summarize_metrics


class RuntimeBaseline(StrEnum):
    RAW_TIMEOUT = "raw_timeout"
    MESSAGE_TRACE = "message_trace"
    RETRY_ONLY = "retry_only"


def evaluate_failurebench_baseline(
    *,
    baseline: RuntimeBaseline,
    split: str = "all",
    dev_seeds_per_type: int = 20,
    test_seeds_per_type: int = 80,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> tuple[list[EpisodeMetrics], RunSummary]:
    cases = build_failurebench_cases(
        split=split,
        dev_seeds_per_type=dev_seeds_per_type,
        test_seeds_per_type=test_seeds_per_type,
    )
    metrics = [_baseline_metric(case, baseline) for case in cases]
    if output_dir and run_id:
        run_dir = output_dir / run_id
        summary = save_metrics(run_dir, metrics)
    else:
        summary = summarize_metrics(metrics)
    return metrics, summary


def _baseline_metric(case: FailureBenchCase, baseline: RuntimeBaseline) -> EpisodeMetrics:
    detected = _detected_failure(case, baseline)
    turn_count = _turn_count(case, baseline)
    replayable = _replayable(case, baseline)
    return EpisodeMetrics(
        session_id=f"{baseline.value}-{case.sample_id}",
        task_id=case.task_id,
        sample_id=case.sample_id,
        scenario=case.scenario.value,
        split=case.split,
        seed=case.seed,
        success=False,
        detected_failure_type=detected,
        oracle_failure_type=case.oracle_failure_type,
        correct_detection=detected == case.oracle_failure_type,
        turn_count=turn_count,
        oracle_failure_turn=case.oracle_failure_turn,
        turns_wasted_after_oracle=max(0, turn_count - case.oracle_failure_turn),
        event_count=_event_count(baseline, turn_count),
        failure_count=1 if detected else 0,
        total_reward=0.0,
        latency_seconds=0.001 * max(1, turn_count),
        resource_cost_units=_resource_cost(case, baseline, turn_count),
        replayable=replayable,
        trace_path="",
    )


def _detected_failure(case: FailureBenchCase, baseline: RuntimeBaseline) -> FailureType | None:
    if baseline == RuntimeBaseline.RAW_TIMEOUT:
        if case.oracle_failure_type in {
            FailureType.CONTEXT_LIMIT,
            FailureType.REPETITIVE_LOOP,
            FailureType.TOOL_TIMEOUT,
        }:
            return FailureType.SCHEDULER_CANCELLED
        return None
    if baseline == RuntimeBaseline.MESSAGE_TRACE:
        if case.oracle_failure_type in {
            FailureType.AGENT_INVALID_ACTION,
            FailureType.TOOL_EXECUTION_ERROR,
        }:
            return case.oracle_failure_type
        return FailureType.UNKNOWN_RUNTIME_ERROR
    if baseline == RuntimeBaseline.RETRY_ONLY:
        if case.oracle_failure_type in {
            FailureType.TOOL_TIMEOUT,
            FailureType.TOOL_EXECUTION_ERROR,
            FailureType.RATE_LIMIT,
        }:
            return case.oracle_failure_type
        return FailureType.UNKNOWN_RUNTIME_ERROR
    raise ValueError(f"unknown baseline: {baseline}")


def _turn_count(case: FailureBenchCase, baseline: RuntimeBaseline) -> int:
    if baseline == RuntimeBaseline.RAW_TIMEOUT:
        return max(8, case.oracle_failure_turn)
    if baseline == RuntimeBaseline.MESSAGE_TRACE:
        return max(case.oracle_failure_turn, 2)
    if baseline == RuntimeBaseline.RETRY_ONLY:
        if case.oracle_failure_type in {
            FailureType.TOOL_TIMEOUT,
            FailureType.TOOL_EXECUTION_ERROR,
            FailureType.RATE_LIMIT,
        }:
            return case.oracle_failure_turn + 1
        return max(case.oracle_failure_turn, 3)
    raise ValueError(f"unknown baseline: {baseline}")


def _event_count(baseline: RuntimeBaseline, turn_count: int) -> int:
    if baseline == RuntimeBaseline.RAW_TIMEOUT:
        return turn_count
    if baseline == RuntimeBaseline.MESSAGE_TRACE:
        return 2 * turn_count
    if baseline == RuntimeBaseline.RETRY_ONLY:
        return 3 * turn_count
    raise ValueError(f"unknown baseline: {baseline}")


def _resource_cost(case: FailureBenchCase, baseline: RuntimeBaseline, turn_count: int) -> float:
    base = float(turn_count)
    if baseline == RuntimeBaseline.RAW_TIMEOUT:
        return base * 1.0
    if baseline == RuntimeBaseline.MESSAGE_TRACE:
        return base * 1.05
    if baseline == RuntimeBaseline.RETRY_ONLY:
        retries = 1 if case.oracle_failure_type in {
            FailureType.TOOL_TIMEOUT,
            FailureType.TOOL_EXECUTION_ERROR,
            FailureType.RATE_LIMIT,
        } else 0
        return base * 1.15 + retries * 0.5
    raise ValueError(f"unknown baseline: {baseline}")


def _replayable(case: FailureBenchCase, baseline: RuntimeBaseline) -> bool:
    if baseline == RuntimeBaseline.RAW_TIMEOUT:
        return False
    if baseline == RuntimeBaseline.MESSAGE_TRACE:
        return case.oracle_failure_type in {
            FailureType.AGENT_INVALID_ACTION,
            FailureType.TOOL_EXECUTION_ERROR,
        }
    if baseline == RuntimeBaseline.RETRY_ONLY:
        return case.oracle_failure_type in {
            FailureType.TOOL_TIMEOUT,
            FailureType.TOOL_EXECUTION_ERROR,
            FailureType.RATE_LIMIT,
        }
    raise ValueError(f"unknown baseline: {baseline}")
