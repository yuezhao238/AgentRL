from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from .benchmarks.failurebench import run_failurebench
from .events import EventLog
from .metrics import load_metrics, save_metrics, summarize_metrics
from .replay import ReplayMode, ReplayReport

app = typer.Typer(help="AgentRL Infra research utilities.")

OUTPUT_DIR_OPTION = typer.Option(Path("runs/failurebench"), help="Directory for run outputs.")
RUN_ID_OPTION = typer.Option(None, help="Run identifier. Defaults to UTC timestamp.")
SPLIT_OPTION = typer.Option("all", help="Split to run: dev, test, or all.")
DEV_SEEDS_OPTION = typer.Option(20, help="Number of dev seeds per failure type.")
TEST_SEEDS_OPTION = typer.Option(80, help="Number of test seeds per failure type.")


@app.command()
def inspect_trace(path: Path, replay_mode: ReplayMode = ReplayMode.EXACT) -> None:
    """Inspect a JSONL event trace and print a replayability report."""
    log = EventLog.load_jsonl(path)
    report = ReplayReport.from_log(log, replay_mode)
    typer.echo(report.model_dump_json(indent=2))


@app.command("replay-trace")
def replay_trace(path: Path, replay_mode: ReplayMode = ReplayMode.EXACT) -> None:
    """Produce a replayability report for a JSONL event trace."""
    inspect_trace(path, replay_mode)


@app.command("run-failurebench")
def run_failurebench_cmd(
    output_dir: Path = OUTPUT_DIR_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    split: str = SPLIT_OPTION,
    dev_seeds_per_type: int = DEV_SEEDS_OPTION,
    test_seeds_per_type: int = TEST_SEEDS_OPTION,
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
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    metrics = run_failurebench(
        output_dir=output_dir,
        run_id=run_id,
        split=split,
        dev_seeds_per_type=dev_seeds_per_type,
        test_seeds_per_type=test_seeds_per_type,
    )
    summary = save_metrics(run_dir, metrics)
    typer.echo(summary.model_dump_json(indent=2))


@app.command("summarize-runs")
def summarize_runs(metrics_path: Path) -> None:
    """Summarize a metrics.json file emitted by run-failurebench."""
    metrics = load_metrics(metrics_path)
    summary = summarize_metrics(metrics)
    typer.echo(summary.model_dump_json(indent=2))
