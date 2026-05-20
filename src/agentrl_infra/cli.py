from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from .benchmarks.failurebench import (
    rerun_failurebench_log,
    run_failurebench,
    run_failurebench_scheduled,
)
from .events import EventLog
from .metrics import load_metrics, save_metrics, summarize_metrics
from .replay import ReplayEngine, ReplayMode, ReplayReport, validate_event_log

app = typer.Typer(help="AgentRL Infra research utilities.")

OUTPUT_DIR_OPTION = typer.Option(Path("runs/failurebench"), help="Directory for run outputs.")
RUN_ID_OPTION = typer.Option(None, help="Run identifier. Defaults to UTC timestamp.")
SPLIT_OPTION = typer.Option("all", help="Split to run: dev, test, or all.")
DEV_SEEDS_OPTION = typer.Option(20, help="Number of dev seeds per failure type.")
TEST_SEEDS_OPTION = typer.Option(80, help="Number of test seeds per failure type.")
SCHEDULER_POLICY_OPTION = typer.Option(
    "failure_aware",
    help="Scheduler policy: failure_aware, fifo, or none.",
)


@app.command()
def inspect_trace(path: Path, replay_mode: ReplayMode = ReplayMode.EXACT) -> None:
    """Inspect a JSONL event trace and print a replayability report."""
    log = EventLog.load_jsonl(path)
    report = ReplayReport.from_log(log, replay_mode)
    typer.echo(report.model_dump_json(indent=2))


@app.command("replay-trace")
def replay_trace(
    path: Path,
    replay_mode: ReplayMode = ReplayMode.EXACT,
    execute: bool = typer.Option(False, help="Execute deterministic replay when supported."),
    benchmark: str = typer.Option("failurebench", help="Benchmark replay adapter."),
) -> None:
    """Produce a replayability report, optionally with deterministic re-execution."""
    log = EventLog.load_jsonl(path)
    validation_errors = validate_event_log(log)
    if validation_errors:
        typer.echo(json.dumps({"valid": False, "errors": validation_errors}, indent=2))
        raise typer.Exit(code=1)
    if not execute:
        report = ReplayReport.from_log(log, replay_mode)
        typer.echo(report.model_dump_json(indent=2))
        return
    if benchmark != "failurebench":
        raise typer.BadParameter(f"unsupported replay benchmark: {benchmark}")
    report = ReplayEngine(rerun_failurebench_log).execute(log, replay_mode)
    typer.echo(report.model_dump_json(indent=2))


@app.command("run-failurebench")
def run_failurebench_cmd(
    output_dir: Path = OUTPUT_DIR_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    split: str = SPLIT_OPTION,
    dev_seeds_per_type: int = DEV_SEEDS_OPTION,
    test_seeds_per_type: int = TEST_SEEDS_OPTION,
    scheduler_policy: str = SCHEDULER_POLICY_OPTION,
) -> None:
    """Run the deterministic FailureBench workload and write traces plus metrics."""
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark": "failurebench",
        "run_id": run_id,
        "split": split,
        "dev_seeds_per_type": dev_seeds_per_type,
        "test_seeds_per_type": test_seeds_per_type,
        "scheduler_policy": scheduler_policy,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if scheduler_policy == "none":
        metrics = run_failurebench(
            output_dir=output_dir,
            run_id=run_id,
            split=split,
            dev_seeds_per_type=dev_seeds_per_type,
            test_seeds_per_type=test_seeds_per_type,
        )
    else:
        scheduler_summary = run_failurebench_scheduled(
            output_dir=output_dir,
            run_id=run_id,
            split=split,
            dev_seeds_per_type=dev_seeds_per_type,
            test_seeds_per_type=test_seeds_per_type,
            scheduler_policy=scheduler_policy,
        )
        (run_dir / "schedule.json").write_text(
            scheduler_summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        metrics = [decision.metrics for decision in scheduler_summary.decisions]
    summary = save_metrics(run_dir, metrics)
    typer.echo(summary.model_dump_json(indent=2))


@app.command("summarize-runs")
def summarize_runs(metrics_path: Path) -> None:
    """Summarize a metrics.json file emitted by run-failurebench."""
    metrics = load_metrics(metrics_path)
    summary = summarize_metrics(metrics)
    typer.echo(summary.model_dump_json(indent=2))
