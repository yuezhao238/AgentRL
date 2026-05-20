from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from .artifacts import (
    replay_run_artifacts,
    save_artifact_report,
    save_batch_replay_report,
    validate_run_artifacts,
)
from .benchmarks.failurebench import (
    rerun_failurebench_log,
    run_failurebench,
    run_failurebench_scheduled,
)
from .benchmarks.lifecycle import ReusePolicy, run_lifecycle_benchmark
from .compare import compare_metric_files
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
LIFECYCLE_OUTPUT_DIR_OPTION = typer.Option(
    Path("runs/lifecycle"),
    help="Directory for run outputs.",
)
LIFECYCLE_POLICY_OPTION = typer.Option(ReusePolicy.CONTAMINATION_AWARE, help="Reuse policy.")
LIFECYCLE_EPISODES_OPTION = typer.Option(100, help="Number of synthetic episodes.")
LIFECYCLE_TTL_OPTION = typer.Option(5, help="TTL for fixed_ttl reuse.")


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


@app.command("validate-run")
def validate_run(run_dir: Path) -> None:
    """Validate run artifact completeness and trace structure."""
    report = validate_run_artifacts(run_dir)
    save_artifact_report(run_dir, report)
    typer.echo(report.model_dump_json(indent=2))
    if not report.valid:
        raise typer.Exit(code=1)


@app.command("replay-run")
def replay_run(
    run_dir: Path,
    benchmark: str = typer.Option("failurebench", help="Replay adapter."),
    replay_mode: ReplayMode = ReplayMode.EXACT,
    verbose: bool = typer.Option(False, help="Print per-trace replay details."),
) -> None:
    """Execute deterministic replay over all supported traces in a run directory."""
    report = replay_run_artifacts(run_dir, benchmark=benchmark, mode=replay_mode)
    save_batch_replay_report(run_dir, report)
    if verbose:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(
            json.dumps(
                {
                    "run_dir": report.run_dir,
                    "attempted": report.attempted,
                    "replayable": report.replayable,
                    "matched": report.matched,
                    "mismatched": report.mismatched,
                    "skipped": report.skipped,
                },
                indent=2,
            )
        )
    if report.mismatched:
        raise typer.Exit(code=1)


@app.command("run-lifecycle-bench")
def run_lifecycle_bench(
    output_dir: Path = LIFECYCLE_OUTPUT_DIR_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    policy: ReusePolicy = LIFECYCLE_POLICY_OPTION,
    episodes: int = LIFECYCLE_EPISODES_OPTION,
    ttl: int = LIFECYCLE_TTL_OPTION,
) -> None:
    """Run synthetic environment lifecycle/reuse benchmark."""
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    _, summary = run_lifecycle_benchmark(
        policy=policy,
        episodes=episodes,
        ttl=ttl,
        output_dir=output_dir,
        run_id=run_id,
    )
    typer.echo(summary.model_dump_json(indent=2))


@app.command("compare-runs")
def compare_runs(baseline_metrics: Path, candidate_metrics: Path) -> None:
    """Compare two metrics.json files."""
    comparison = compare_metric_files(baseline_metrics, candidate_metrics)
    typer.echo(comparison.model_dump_json(indent=2))
