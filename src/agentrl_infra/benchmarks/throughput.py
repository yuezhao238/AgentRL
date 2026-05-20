from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from .model_action import ModelActionSummary


class ThroughputPolicy(StrEnum):
    FIFO = "fifo"
    RETRY_ONLY = "retry_only"
    FAILURE_AWARE = "failure_aware"


@dataclass(frozen=True)
class WorkloadClass:
    name: str
    count: int
    service_time_units: float
    failure_probability: float
    replayable_failure_probability: float
    zombie_probability: float
    detection_time_units: float
    retryable: bool
    priority: float = 1.0

    @property
    def expected_success_probability(self) -> float:
        return 1.0 - self.failure_probability

    @property
    def expected_useful_probability(self) -> float:
        return self.expected_success_probability + (
            self.failure_probability * self.replayable_failure_probability
        )


class ThroughputEpisodeMetrics(BaseModel):
    policy: ThroughputPolicy
    sample_id: str
    workload_class: str
    worker_id: int
    attempt: int
    start_time_units: float
    end_time_units: float
    latency_units: float
    service_time_units: float
    success: bool
    useful: bool
    retry_scheduled: bool
    zombie: bool
    failure_detected_early: bool


class ThroughputSummary(BaseModel):
    policy: ThroughputPolicy
    worker_count: int
    episode_count: int
    attempt_count: int
    success_count: int
    useful_count: int
    failure_count: int
    zombie_count: int
    makespan_units: float
    total_busy_units: float
    failed_cost_units: float
    worker_utilization: float
    useful_trajectories_per_hour: float
    successful_trajectories_per_hour: float
    zombie_session_rate: float
    p50_latency_units: float
    p95_latency_units: float
    p99_latency_units: float


class ThroughputRun(BaseModel):
    policy: ThroughputPolicy
    metrics: list[ThroughputEpisodeMetrics] = Field(default_factory=list)
    summary: ThroughputSummary


def build_model_action_throughput_workload(
    summary_path: Path,
    *,
    episodes_per_cell: int = 40,
    min_service_time_units: float = 0.05,
) -> list[WorkloadClass]:
    """Build scheduler workload classes from real model-to-action benchmark summaries."""
    summaries = [
        ModelActionSummary.model_validate(item)
        for item in json.loads(summary_path.read_text(encoding="utf-8"))
    ]
    workload: list[WorkloadClass] = []
    for summary in summaries:
        episodes = summary.episode_count
        if episodes == 0:
            continue
        failures = episodes - summary.success_count
        failure_probability = failures / episodes
        replayable_failures = summary.invalid_action_count + summary.no_progress_count
        replayable_failure_probability = replayable_failures / failures if failures else 0.0
        failed_records = [record for record in summary.records if not record.success]
        failure_latency = (
            mean(record.latency_seconds for record in failed_records)
            if failed_records
            else summary.total_latency_seconds / episodes
        )
        service_time = max(summary.total_latency_seconds / episodes, min_service_time_units)
        detection_time = max(min(failure_latency, service_time), min_service_time_units)
        zombie_probability = min(0.50, 0.05 + (summary.no_progress_count / episodes) * 0.75)
        workload.append(
            WorkloadClass(
                name=(
                    f"model_action_{_short_model_id(summary.model_id)}_"
                    f"{summary.prompt_protocol}_{summary.max_new_tokens}"
                ),
                count=episodes_per_cell,
                service_time_units=service_time,
                failure_probability=failure_probability,
                replayable_failure_probability=replayable_failure_probability,
                zombie_probability=zombie_probability if failures else 0.0,
                detection_time_units=detection_time,
                retryable=failures > 0,
                priority=1.0,
            )
        )
    return workload


def default_throughput_workload() -> list[WorkloadClass]:
    return [
        WorkloadClass(
            name="short_browser_success",
            count=80,
            service_time_units=2.0,
            failure_probability=0.10,
            replayable_failure_probability=0.50,
            zombie_probability=0.00,
            detection_time_units=1.0,
            retryable=False,
            priority=1.0,
        ),
        WorkloadClass(
            name="stale_dom_invalid_action",
            count=50,
            service_time_units=3.0,
            failure_probability=0.85,
            replayable_failure_probability=1.00,
            zombie_probability=0.02,
            detection_time_units=1.0,
            retryable=False,
            priority=1.2,
        ),
        WorkloadClass(
            name="wait_loop_timeout",
            count=40,
            service_time_units=12.0,
            failure_probability=0.95,
            replayable_failure_probability=1.00,
            zombie_probability=0.25,
            detection_time_units=4.0,
            retryable=False,
            priority=0.9,
        ),
        WorkloadClass(
            name="tool_timeout_retryable",
            count=40,
            service_time_units=6.0,
            failure_probability=0.55,
            replayable_failure_probability=0.30,
            zombie_probability=0.08,
            detection_time_units=3.0,
            retryable=True,
            priority=0.8,
        ),
        WorkloadClass(
            name="contaminated_environment",
            count=30,
            service_time_units=8.0,
            failure_probability=0.80,
            replayable_failure_probability=0.00,
            zombie_probability=0.15,
            detection_time_units=2.0,
            retryable=True,
            priority=0.6,
        ),
    ]


def run_throughput_benchmark(
    *,
    policy: ThroughputPolicy,
    worker_count: int = 8,
    output_dir: Path | None = None,
    run_id: str | None = None,
    workload: list[WorkloadClass] | None = None,
) -> ThroughputRun:
    workload = workload or default_throughput_workload()
    pending = _build_pending_samples(workload)
    if policy == ThroughputPolicy.FAILURE_AWARE:
        pending.sort(key=lambda item: _failure_aware_priority(item[0]), reverse=True)

    worker_available_at = [0.0 for _ in range(worker_count)]
    metrics: list[ThroughputEpisodeMetrics] = []
    completed_sample_ids: set[str] = set()

    while pending:
        workload_class, sample_id, attempt = pending.pop(0)
        worker_id = min(range(worker_count), key=lambda item: worker_available_at[item])
        start = worker_available_at[worker_id]
        success = not _draw_bool(sample_id, attempt, "failure", workload_class.failure_probability)
        zombie = (
            (not success)
            and _draw_bool(sample_id, attempt, "zombie", workload_class.zombie_probability)
        )
        replayable_failure = (
            (not success)
            and _draw_bool(
                sample_id,
                attempt,
                "replayable",
                workload_class.replayable_failure_probability,
            )
        )
        failure_detected_early = policy == ThroughputPolicy.FAILURE_AWARE and not success
        service_time = _service_time(policy, workload_class, success=success, zombie=zombie)
        end = start + service_time
        worker_available_at[worker_id] = end

        retry_scheduled = (
            policy == ThroughputPolicy.RETRY_ONLY
            and (not success)
            and workload_class.retryable
            and attempt == 0
        )
        if retry_scheduled:
            pending.append((workload_class, sample_id, attempt + 1))

        useful = success or (policy == ThroughputPolicy.FAILURE_AWARE and replayable_failure)
        if success:
            completed_sample_ids.add(sample_id)
        metrics.append(
            ThroughputEpisodeMetrics(
                policy=policy,
                sample_id=sample_id,
                workload_class=workload_class.name,
                worker_id=worker_id,
                attempt=attempt,
                start_time_units=start,
                end_time_units=end,
                latency_units=service_time,
                service_time_units=service_time,
                success=success,
                useful=useful,
                retry_scheduled=retry_scheduled,
                zombie=zombie and policy != ThroughputPolicy.FAILURE_AWARE,
                failure_detected_early=failure_detected_early,
            )
        )

    summary = summarize_throughput_metrics(
        policy=policy,
        worker_count=worker_count,
        original_episode_count=sum(item.count for item in workload),
        metrics=metrics,
    )
    run = ThroughputRun(policy=policy, metrics=metrics, summary=summary)
    if output_dir and run_id:
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "throughput_metrics.json").write_text(
            json.dumps([metric.model_dump(mode="json") for metric in metrics], indent=2)
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "throughput_summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    return run


def summarize_throughput_metrics(
    *,
    policy: ThroughputPolicy,
    worker_count: int,
    original_episode_count: int,
    metrics: list[ThroughputEpisodeMetrics],
) -> ThroughputSummary:
    makespan = max((metric.end_time_units for metric in metrics), default=0.0)
    total_busy = sum(metric.service_time_units for metric in metrics)
    failures = [metric for metric in metrics if not metric.success]
    latencies = sorted(metric.latency_units for metric in metrics)
    useful_count = _unique_count(metric.sample_id for metric in metrics if metric.useful)
    success_count = _unique_count(metric.sample_id for metric in metrics if metric.success)
    zombie_count = sum(1 for metric in metrics if metric.zombie)
    return ThroughputSummary(
        policy=policy,
        worker_count=worker_count,
        episode_count=original_episode_count,
        attempt_count=len(metrics),
        success_count=success_count,
        useful_count=useful_count,
        failure_count=len(failures),
        zombie_count=zombie_count,
        makespan_units=makespan,
        total_busy_units=total_busy,
        failed_cost_units=sum(metric.service_time_units for metric in failures),
        worker_utilization=total_busy / (makespan * worker_count) if makespan else 0.0,
        useful_trajectories_per_hour=useful_count / makespan * 3600 if makespan else 0.0,
        successful_trajectories_per_hour=success_count / makespan * 3600 if makespan else 0.0,
        zombie_session_rate=zombie_count / len(metrics) if metrics else 0.0,
        p50_latency_units=_percentile(latencies, 0.50),
        p95_latency_units=_percentile(latencies, 0.95),
        p99_latency_units=_percentile(latencies, 0.99),
    )


def _build_pending_samples(
    workload: list[WorkloadClass],
) -> list[tuple[WorkloadClass, str, int]]:
    samples: list[tuple[WorkloadClass, str, int]] = []
    for workload_class in workload:
        for index in range(workload_class.count):
            samples.append((workload_class, f"{workload_class.name}-{index:04d}", 0))
    return samples


def _failure_aware_priority(workload_class: WorkloadClass) -> float:
    expected_time = (
        workload_class.expected_success_probability * workload_class.service_time_units
        + workload_class.failure_probability * workload_class.detection_time_units
    )
    expected_time = max(expected_time, 1e-6)
    return workload_class.priority * workload_class.expected_useful_probability / expected_time


def _service_time(
    policy: ThroughputPolicy,
    workload_class: WorkloadClass,
    *,
    success: bool,
    zombie: bool,
) -> float:
    if success:
        return workload_class.service_time_units
    if policy == ThroughputPolicy.FAILURE_AWARE:
        return workload_class.detection_time_units
    if zombie:
        return workload_class.service_time_units * 3.0
    return workload_class.service_time_units


def _draw_bool(sample_id: str, attempt: int, salt: str, probability: float) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    digest = sha256(f"{sample_id}:{attempt}:{salt}".encode()).hexdigest()
    value = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    return value < probability


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * quantile))))
    return sorted_values[index]


def _unique_count(values: Iterable[str]) -> int:
    return len(set(values))


def _short_model_id(value: str) -> str:
    return value.split("/")[-1].replace("-", "_").replace(".", "_")
