from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .artifacts import (
    replay_run_artifacts,
    save_artifact_report,
    save_batch_replay_report,
    validate_run_artifacts,
)
from .baselines import RuntimeBaseline, evaluate_failurebench_baseline
from .benchmarks.failurebench import run_failurebench_scheduled
from .benchmarks.lifecycle import ReusePolicy, run_lifecycle_benchmark
from .metrics import save_metrics
from .tables import (
    build_lifecycle_rows,
    build_summary_rows,
    write_lifecycle_csv,
    write_lifecycle_latex,
    write_summary_csv,
    write_summary_latex,
)


class ExperimentSuiteConfig(BaseModel):
    suite_id: str = "local-suite"
    output_dir: Path = Path("runs/experiments")
    split: str = "dev"
    dev_seeds_per_type: int = 5
    test_seeds_per_type: int = 0
    lifecycle_episodes: int = 50


class ExperimentSuiteReport(BaseModel):
    suite_dir: str
    failurebench_runs: dict[str, str] = Field(default_factory=dict)
    lifecycle_runs: dict[str, str] = Field(default_factory=dict)
    tables: dict[str, str] = Field(default_factory=dict)


def load_experiment_suite_config(path: Path) -> ExperimentSuiteConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentSuiteConfig.model_validate(data)


def run_experiment_suite(config: ExperimentSuiteConfig) -> ExperimentSuiteReport:
    suite_dir = config.output_dir / config.suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "suite_config.json").write_text(
        config.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    failurebench_dir = suite_dir / "failurebench"
    failurebench_runs: dict[str, str] = {}
    for baseline in RuntimeBaseline:
        run_id = baseline.value
        metrics, _ = evaluate_failurebench_baseline(
            baseline=baseline,
            split=config.split,
            dev_seeds_per_type=config.dev_seeds_per_type,
            test_seeds_per_type=config.test_seeds_per_type,
            output_dir=failurebench_dir,
            run_id=run_id,
        )
        (failurebench_dir / run_id / "config.json").write_text(
            json.dumps({"baseline": baseline.value}, indent=2) + "\n",
            encoding="utf-8",
        )
        failurebench_runs[baseline.value] = str(failurebench_dir / run_id / "metrics.json")

    ours_run_id = "rolloutos"
    scheduler_summary = run_failurebench_scheduled(
        output_dir=failurebench_dir,
        run_id=ours_run_id,
        split=config.split,
        dev_seeds_per_type=config.dev_seeds_per_type,
        test_seeds_per_type=config.test_seeds_per_type,
        scheduler_policy="failure_aware",
    )
    ours_dir = failurebench_dir / ours_run_id
    (ours_dir / "config.json").write_text(
        json.dumps({"system": "rolloutos", "scheduler_policy": "failure_aware"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (ours_dir / "schedule.json").write_text(
        scheduler_summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    save_metrics(ours_dir, [decision.metrics for decision in scheduler_summary.decisions])
    validation = validate_run_artifacts(ours_dir)
    save_artifact_report(ours_dir, validation)
    replay_report = replay_run_artifacts(ours_dir)
    save_batch_replay_report(ours_dir, replay_report)
    failurebench_runs["rolloutos"] = str(ours_dir / "metrics.json")

    lifecycle_dir = suite_dir / "lifecycle"
    lifecycle_runs: dict[str, str] = {}
    for policy in [ReusePolicy.RECREATE, ReusePolicy.BLIND_REUSE, ReusePolicy.CONTAMINATION_AWARE]:
        run_id = policy.value
        run_lifecycle_benchmark(
            policy=policy,
            episodes=config.lifecycle_episodes,
            output_dir=lifecycle_dir,
            run_id=run_id,
        )
        lifecycle_runs[policy.value] = str(lifecycle_dir / run_id / "lifecycle_summary.json")

    rows = build_summary_rows({name: Path(path) for name, path in failurebench_runs.items()})
    table_dir = suite_dir / "tables"
    csv_path = table_dir / "failurebench_summary.csv"
    tex_path = table_dir / "failurebench_summary.tex"
    write_summary_csv(rows, csv_path)
    write_summary_latex(rows, tex_path)

    lifecycle_rows = build_lifecycle_rows(
        {name: Path(path) for name, path in lifecycle_runs.items()}
    )
    lifecycle_csv_path = table_dir / "lifecycle_summary.csv"
    lifecycle_tex_path = table_dir / "lifecycle_summary.tex"
    write_lifecycle_csv(lifecycle_rows, lifecycle_csv_path)
    write_lifecycle_latex(lifecycle_rows, lifecycle_tex_path)

    report = ExperimentSuiteReport(
        suite_dir=str(suite_dir),
        failurebench_runs=failurebench_runs,
        lifecycle_runs=lifecycle_runs,
        tables={
            "failurebench_summary_csv": str(csv_path),
            "failurebench_summary_tex": str(tex_path),
            "lifecycle_summary_csv": str(lifecycle_csv_path),
            "lifecycle_summary_tex": str(lifecycle_tex_path),
        },
    )
    (suite_dir / "suite_report.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return report
