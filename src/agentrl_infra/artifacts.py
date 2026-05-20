from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .events import EventLog
from .metrics import load_metrics
from .replay import ReplayEngine, ReplayExecutionReport, ReplayMode, validate_event_log


class ArtifactValidationReport(BaseModel):
    run_dir: str
    valid: bool
    trace_count: int = 0
    metric_count: int = 0
    missing_files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BatchReplayReport(BaseModel):
    run_dir: str
    attempted: int
    replayable: int
    matched: int
    mismatched: int
    skipped: int
    reports: list[ReplayExecutionReport] = Field(default_factory=list)


def validate_run_artifacts(run_dir: Path) -> ArtifactValidationReport:
    required = ["config.json", "metrics.json", "summary.json", "summary.md"]
    missing = [name for name in required if not (run_dir / name).exists()]
    errors: list[str] = []

    trace_dir = run_dir / "traces"
    if not trace_dir.exists():
        missing.append("traces/")
        trace_paths: list[Path] = []
    else:
        trace_paths = sorted(trace_dir.glob("*.jsonl"))

    metrics = []
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = load_metrics(metrics_path)
        except Exception as exc:
            errors.append(f"failed to load metrics.json: {exc}")

    for trace_path in trace_paths:
        try:
            log = EventLog.load_jsonl(trace_path)
        except Exception as exc:
            errors.append(f"{trace_path.name}: failed to load trace: {exc}")
            continue
        for error in validate_event_log(log):
            errors.append(f"{trace_path.name}: {error}")

    metric_trace_paths = {Path(metric.trace_path).name for metric in metrics}
    actual_trace_paths = {path.name for path in trace_paths}
    missing_traces = sorted(metric_trace_paths - actual_trace_paths)
    if missing_traces:
        errors.append(f"metrics reference missing traces: {missing_traces[:10]}")

    return ArtifactValidationReport(
        run_dir=str(run_dir),
        valid=not missing and not errors,
        trace_count=len(trace_paths),
        metric_count=len(metrics),
        missing_files=missing,
        errors=errors,
    )


def replay_run_artifacts(
    run_dir: Path,
    *,
    benchmark: str = "failurebench",
    mode: ReplayMode = ReplayMode.EXACT,
) -> BatchReplayReport:
    if benchmark != "failurebench":
        raise ValueError(f"unsupported replay benchmark: {benchmark}")

    from .benchmarks.failurebench import rerun_failurebench_log

    engine = ReplayEngine(rerun_failurebench_log)
    trace_paths = sorted((run_dir / "traces").glob("*.jsonl"))
    reports: list[ReplayExecutionReport] = []
    skipped = 0
    for trace_path in trace_paths:
        log = EventLog.load_jsonl(trace_path)
        if validate_event_log(log):
            skipped += 1
            continue
        reports.append(engine.execute(log, mode))

    return BatchReplayReport(
        run_dir=str(run_dir),
        attempted=len(reports),
        replayable=sum(1 for report in reports if report.replayable),
        matched=sum(1 for report in reports if report.matched),
        mismatched=sum(1 for report in reports if report.replayable and not report.matched),
        skipped=skipped,
        reports=reports,
    )


def save_artifact_report(run_dir: Path, report: ArtifactValidationReport) -> None:
    (run_dir / "artifact_validation.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def save_batch_replay_report(run_dir: Path, report: BatchReplayReport) -> None:
    (run_dir / "replay_report.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "replay_report.md").write_text(
        render_batch_replay_markdown(report),
        encoding="utf-8",
    )


def render_batch_replay_markdown(report: BatchReplayReport) -> str:
    payload = {
        "attempted": report.attempted,
        "replayable": report.replayable,
        "matched": report.matched,
        "mismatched": report.mismatched,
        "skipped": report.skipped,
    }
    return "# Replay Report\n\n```json\n" + json.dumps(payload, indent=2) + "\n```\n"
