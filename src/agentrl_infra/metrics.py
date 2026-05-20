from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from .failures import FailureType
from .runner import EpisodeResult
from .session import SessionState


class EpisodeMetrics(BaseModel):
    session_id: str
    task_id: str
    sample_id: str | None
    scenario: str
    split: str
    seed: int
    success: bool
    detected_failure_type: FailureType | None
    oracle_failure_type: FailureType
    correct_detection: bool
    turn_count: int
    oracle_failure_turn: int
    turns_wasted_after_oracle: int
    event_count: int
    failure_count: int
    total_reward: float
    latency_seconds: float
    resource_cost_units: float = 0.0
    replayable: bool
    trace_path: str


class FailureTypeMetrics(BaseModel):
    failure_type: FailureType
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class RunSummary(BaseModel):
    episode_count: int
    success_count: int
    failure_count: int
    correct_detection_count: int
    detection_accuracy: float
    macro_f1: float
    useful_trajectories_per_hour: float
    useful_trajectories_per_cost_unit: float
    failed_rollout_cost_turns: int
    failed_rollout_cost_units: float
    mean_turns_wasted_after_oracle: float
    mean_latency_seconds: float
    replayable_failure_count: int
    replayable_failure_rate: float
    by_failure_type: list[FailureTypeMetrics] = Field(default_factory=list)


def metrics_from_result(
    result: EpisodeResult,
    *,
    oracle_failure_type: FailureType,
    oracle_failure_turn: int,
    scenario: str,
    split: str,
    seed: int,
    trace_path: Path,
    replayable: bool,
    resource_cost_units: float | None = None,
) -> EpisodeMetrics:
    detected = result.failure.type if result.failure else None
    success = result.state == SessionState.COMPLETED
    return EpisodeMetrics(
        session_id=result.session_id,
        task_id=result.task_id,
        sample_id=result.sample_id,
        scenario=scenario,
        split=split,
        seed=seed,
        success=success,
        detected_failure_type=detected,
        oracle_failure_type=oracle_failure_type,
        correct_detection=detected == oracle_failure_type,
        turn_count=result.turn_count,
        oracle_failure_turn=oracle_failure_turn,
        turns_wasted_after_oracle=max(0, result.turn_count - oracle_failure_turn),
        event_count=len(result.event_log.events),
        failure_count=len(result.event_log.failures()),
        total_reward=result.total_reward,
        latency_seconds=result.latency_seconds,
        resource_cost_units=resource_cost_units
        if resource_cost_units is not None
        else float(max(result.turn_count, 1)),
        replayable=replayable,
        trace_path=str(trace_path),
    )


def summarize_metrics(metrics: list[EpisodeMetrics]) -> RunSummary:
    if not metrics:
        return RunSummary(
            episode_count=0,
            success_count=0,
            failure_count=0,
            correct_detection_count=0,
            detection_accuracy=0.0,
            macro_f1=0.0,
            useful_trajectories_per_hour=0.0,
            useful_trajectories_per_cost_unit=0.0,
            failed_rollout_cost_turns=0,
            failed_rollout_cost_units=0.0,
            mean_turns_wasted_after_oracle=0.0,
            mean_latency_seconds=0.0,
            replayable_failure_count=0,
            replayable_failure_rate=0.0,
        )

    by_type = _failure_type_metrics(metrics)
    total_latency = sum(metric.latency_seconds for metric in metrics)
    total_cost = sum(metric.resource_cost_units for metric in metrics)
    useful = sum(1 for metric in metrics if metric.replayable or metric.success)
    failures = [metric for metric in metrics if not metric.success]
    replayable_failures = sum(1 for metric in failures if metric.replayable)
    correct = sum(1 for metric in metrics if metric.correct_detection)

    return RunSummary(
        episode_count=len(metrics),
        success_count=sum(1 for metric in metrics if metric.success),
        failure_count=len(failures),
        correct_detection_count=correct,
        detection_accuracy=correct / len(metrics),
        macro_f1=mean([item.f1 for item in by_type]) if by_type else 0.0,
        useful_trajectories_per_hour=useful / total_latency * 3600 if total_latency > 0 else 0.0,
        useful_trajectories_per_cost_unit=useful / total_cost if total_cost > 0 else 0.0,
        failed_rollout_cost_turns=sum(metric.turn_count for metric in failures),
        failed_rollout_cost_units=sum(metric.resource_cost_units for metric in failures),
        mean_turns_wasted_after_oracle=mean(
            [metric.turns_wasted_after_oracle for metric in metrics]
        ),
        mean_latency_seconds=mean([metric.latency_seconds for metric in metrics]),
        replayable_failure_count=replayable_failures,
        replayable_failure_rate=replayable_failures / len(failures) if failures else 0.0,
        by_failure_type=by_type,
    )


def save_metrics(run_dir: Path, metrics: list[EpisodeMetrics]) -> RunSummary:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_metrics(metrics)
    (run_dir / "metrics.json").write_text(
        json.dumps([metric.model_dump(mode="json") for metric in metrics], indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    return summary


def load_metrics(path: Path) -> list[EpisodeMetrics]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EpisodeMetrics.model_validate(item) for item in data]


def render_summary_markdown(summary: RunSummary) -> str:
    lines = [
        "# Run Summary",
        "",
        f"- episodes: {summary.episode_count}",
        f"- successes: {summary.success_count}",
        f"- failures: {summary.failure_count}",
        f"- detection accuracy: {summary.detection_accuracy:.4f}",
        f"- macro F1: {summary.macro_f1:.4f}",
        f"- useful trajectories/hour: {summary.useful_trajectories_per_hour:.2f}",
        f"- useful trajectories/cost unit: {summary.useful_trajectories_per_cost_unit:.4f}",
        f"- failed rollout cost (turns): {summary.failed_rollout_cost_turns}",
        f"- failed rollout cost (units): {summary.failed_rollout_cost_units:.2f}",
        f"- mean wasted turns after oracle: {summary.mean_turns_wasted_after_oracle:.2f}",
        f"- mean latency seconds: {summary.mean_latency_seconds:.6f}",
        f"- replayable failure rate: {summary.replayable_failure_rate:.4f}",
        "",
        "## By Failure Type",
        "",
        "| failure_type | precision | recall | f1 | tp | fp | fn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary.by_failure_type:
        lines.append(
            "| "
            f"{item.failure_type.value} | {item.precision:.4f} | {item.recall:.4f} | "
            f"{item.f1:.4f} | {item.true_positive} | {item.false_positive} | "
            f"{item.false_negative} |"
        )
    return "\n".join(lines) + "\n"


def _failure_type_metrics(metrics: list[EpisodeMetrics]) -> list[FailureTypeMetrics]:
    counts: dict[FailureType, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for metric in metrics:
        oracle = metric.oracle_failure_type
        detected = metric.detected_failure_type
        if detected == oracle:
            counts[oracle]["tp"] += 1
        else:
            counts[oracle]["fn"] += 1
            if detected is not None:
                counts[detected]["fp"] += 1

    result: list[FailureTypeMetrics] = []
    for failure_type in sorted(counts.keys(), key=lambda item: item.value):
        tp = counts[failure_type]["tp"]
        fp = counts[failure_type]["fp"]
        fn = counts[failure_type]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.append(
            FailureTypeMetrics(
                failure_type=failure_type,
                true_positive=tp,
                false_positive=fp,
                false_negative=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return result
