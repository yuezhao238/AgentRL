from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .metrics import RunSummary, load_metrics, summarize_metrics


class RunComparison(BaseModel):
    baseline: str
    candidate: str
    baseline_summary: RunSummary
    candidate_summary: RunSummary
    useful_trajectories_per_hour_delta: float
    useful_trajectories_per_cost_unit_delta: float
    macro_f1_delta: float
    failed_rollout_cost_turns_delta: int
    failed_rollout_cost_units_delta: float
    replayable_failure_rate_delta: float


def compare_metric_files(baseline_metrics: Path, candidate_metrics: Path) -> RunComparison:
    baseline = summarize_metrics(load_metrics(baseline_metrics))
    candidate = summarize_metrics(load_metrics(candidate_metrics))
    return RunComparison(
        baseline=str(baseline_metrics),
        candidate=str(candidate_metrics),
        baseline_summary=baseline,
        candidate_summary=candidate,
        useful_trajectories_per_hour_delta=(
            candidate.useful_trajectories_per_hour - baseline.useful_trajectories_per_hour
        ),
        useful_trajectories_per_cost_unit_delta=(
            candidate.useful_trajectories_per_cost_unit
            - baseline.useful_trajectories_per_cost_unit
        ),
        macro_f1_delta=candidate.macro_f1 - baseline.macro_f1,
        failed_rollout_cost_turns_delta=(
            candidate.failed_rollout_cost_turns - baseline.failed_rollout_cost_turns
        ),
        failed_rollout_cost_units_delta=(
            candidate.failed_rollout_cost_units - baseline.failed_rollout_cost_units
        ),
        replayable_failure_rate_delta=(
            candidate.replayable_failure_rate - baseline.replayable_failure_rate
        ),
    )
