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
from .benchmarks.model_provenance import DEFAULT_MODEL_IDS, run_model_provenance_audit
from .benchmarks.throughput import ThroughputPolicy, run_throughput_benchmark
from .compare import compare_metric_files
from .events import EventLog
from .experiments import ExperimentSuiteConfig, load_experiment_suite_config, run_experiment_suite
from .integrations.miniwob import MINIWOB_20_TASKS, run_miniwob_contract_subset
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
THROUGHPUT_OUTPUT_DIR_OPTION = typer.Option(
    Path("runs/throughput"),
    help="Directory for throughput benchmark outputs.",
)
THROUGHPUT_WORKERS_OPTION = typer.Option(8, help="Number of simulated rollout workers.")
THROUGHPUT_POLICY_OPTION = typer.Option(ThroughputPolicy.FAILURE_AWARE, help="Policy.")
MODEL_OUTPUT_DIR_OPTION = typer.Option(
    Path("runs/model_provenance"),
    help="Directory for model provenance audit outputs.",
)
EXPERIMENT_OUTPUT_DIR_OPTION = typer.Option(
    Path("runs/experiments"),
    help="Suite output root.",
)
SUITE_ID_OPTION = typer.Option("local-suite", help="Suite identifier.")
SUITE_CONFIG_PATH_OPTION = typer.Option(None, help="Optional JSON suite config.")
MINIWOB_OUTPUT_DIR_OPTION = typer.Option(
    Path("runs/miniwob"),
    help="Directory for MiniWoB contract run outputs.",
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


@app.command("run-throughput-bench")
def run_throughput_bench(
    output_dir: Path = THROUGHPUT_OUTPUT_DIR_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    policy: ThroughputPolicy = THROUGHPUT_POLICY_OPTION,
    workers: int = THROUGHPUT_WORKERS_OPTION,
) -> None:
    """Run deterministic worker-pool throughput simulation."""
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = run_throughput_benchmark(
        policy=policy,
        worker_count=workers,
        output_dir=output_dir,
        run_id=run_id,
    )
    typer.echo(run.summary.model_dump_json(indent=2))


@app.command("run-model-provenance")
def run_model_provenance(
    output_dir: Path = MODEL_OUTPUT_DIR_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    models: str = typer.Option(
        ",".join(DEFAULT_MODEL_IDS),
        help="Comma-separated Hugging Face model ids.",
    ),
) -> None:
    """Audit local tokenizer provenance and token-native trace metadata."""
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_ids = [item for item in models.split(",") if item]
    summaries = run_model_provenance_audit(
        output_dir=output_dir,
        run_id=run_id,
        model_ids=model_ids,
    )
    typer.echo(json.dumps([item.model_dump(mode="json") for item in summaries], indent=2))


@app.command("compare-runs")
def compare_runs(baseline_metrics: Path, candidate_metrics: Path) -> None:
    """Compare two metrics.json files."""
    comparison = compare_metric_files(baseline_metrics, candidate_metrics)
    typer.echo(comparison.model_dump_json(indent=2))


@app.command("run-experiment-suite")
def run_experiment_suite_cmd(
    config_path: Path | None = SUITE_CONFIG_PATH_OPTION,
    output_dir: Path = EXPERIMENT_OUTPUT_DIR_OPTION,
    suite_id: str = SUITE_ID_OPTION,
    split: str = SPLIT_OPTION,
    dev_seeds_per_type: int = typer.Option(5, help="Dev seeds per FailureBench type."),
    test_seeds_per_type: int = typer.Option(0, help="Test seeds per FailureBench type."),
    lifecycle_episodes: int = typer.Option(50, help="Lifecycle benchmark episode count."),
) -> None:
    """Run reproducible local experiments and generate paper-ready tables."""
    if config_path:
        config = load_experiment_suite_config(config_path)
    else:
        config = ExperimentSuiteConfig(
            suite_id=suite_id,
            output_dir=output_dir,
            split=split,
            dev_seeds_per_type=dev_seeds_per_type,
            test_seeds_per_type=test_seeds_per_type,
            lifecycle_episodes=lifecycle_episodes,
        )
    report = run_experiment_suite(config)
    typer.echo(report.model_dump_json(indent=2))


@app.command("run-miniwob-contract")
def run_miniwob_contract_cmd(
    output_dir: Path = MINIWOB_OUTPUT_DIR_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    tasks: str = typer.Option(
        "click-button,enter-text,login-user",
        help="Comma-separated MiniWoB tasks, or all.",
    ),
    seeds: str = typer.Option("1000", help="Comma-separated integer seeds."),
    policy: str = typer.Option("oracle", help="Policy: oracle or repeated_action."),
) -> None:
    """Run the deterministic MiniWoB browser contract harness."""
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    task_names = (
        list(MINIWOB_20_TASKS) if tasks == "all" else [item for item in tasks.split(",") if item]
    )
    seed_values = [int(item) for item in seeds.split(",") if item]
    _, summary = run_miniwob_contract_subset(
        output_dir=output_dir,
        run_id=run_id,
        task_names=task_names,
        seeds=seed_values,
        policy_name=policy,
    )
    typer.echo(summary.model_dump_json(indent=2))
