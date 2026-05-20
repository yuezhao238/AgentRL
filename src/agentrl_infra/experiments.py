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
from .benchmarks.model_provenance import DEFAULT_MODEL_IDS, run_model_provenance_audit
from .benchmarks.throughput import ThroughputPolicy, run_throughput_benchmark
from .integrations.miniwob import MINIWOB_20_TASKS, run_miniwob_contract_subset
from .metrics import save_metrics
from .tables import (
    build_lifecycle_rows,
    build_miniwob_contract_rows,
    build_model_provenance_rows,
    build_summary_rows,
    build_throughput_rows,
    write_lifecycle_csv,
    write_lifecycle_latex,
    write_miniwob_contract_csv,
    write_miniwob_contract_latex,
    write_model_provenance_csv,
    write_model_provenance_latex,
    write_summary_csv,
    write_summary_latex,
    write_throughput_csv,
    write_throughput_latex,
)


class ExperimentSuiteConfig(BaseModel):
    suite_id: str = "local-suite"
    output_dir: Path = Path("runs/experiments")
    split: str = "dev"
    dev_seeds_per_type: int = 5
    test_seeds_per_type: int = 0
    lifecycle_episodes: int = 50
    miniwob_tasks: list[str] = Field(default_factory=lambda: list(MINIWOB_20_TASKS))
    miniwob_seeds: list[int] = Field(default_factory=lambda: [1000, 1001, 1002, 1003, 1004])
    miniwob_policies: list[str] = Field(
        default_factory=lambda: [
            "oracle",
            "stale_dom",
            "invalid_selector",
            "wait_loop",
            "repeated_action",
        ]
    )
    throughput_workers: int = 8
    model_provenance_model_ids: list[str] = Field(default_factory=lambda: list(DEFAULT_MODEL_IDS))


class ExperimentSuiteReport(BaseModel):
    suite_dir: str
    failurebench_runs: dict[str, str] = Field(default_factory=dict)
    lifecycle_runs: dict[str, str] = Field(default_factory=dict)
    miniwob_runs: dict[str, str] = Field(default_factory=dict)
    throughput_runs: dict[str, str] = Field(default_factory=dict)
    model_provenance_run: str | None = None
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

    miniwob_dir = suite_dir / "miniwob_contract"
    miniwob_runs: dict[str, str] = {}
    for policy_name in config.miniwob_policies:
        run_miniwob_contract_subset(
            output_dir=miniwob_dir,
            run_id=policy_name,
            task_names=config.miniwob_tasks,
            seeds=config.miniwob_seeds,
            policy_name=policy_name,
        )
        miniwob_runs[policy_name] = str(miniwob_dir / policy_name / "summary.json")

    throughput_dir = suite_dir / "throughput"
    throughput_runs: dict[str, str] = {}
    for policy in ThroughputPolicy:
        run_throughput_benchmark(
            policy=policy,
            worker_count=config.throughput_workers,
            output_dir=throughput_dir,
            run_id=policy.value,
        )
        throughput_runs[policy.value] = str(
            throughput_dir / policy.value / "throughput_summary.json"
        )

    model_provenance_dir = suite_dir / "model_provenance"
    run_model_provenance_audit(
        output_dir=model_provenance_dir,
        run_id="tokenizer_audit",
        model_ids=config.model_provenance_model_ids,
    )
    model_provenance_summary_path = (
        model_provenance_dir / "tokenizer_audit" / "model_provenance_summary.json"
    )

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

    miniwob_rows = build_miniwob_contract_rows(
        {name: Path(path) for name, path in miniwob_runs.items()}
    )
    miniwob_csv_path = table_dir / "miniwob_contract_summary.csv"
    miniwob_tex_path = table_dir / "miniwob_contract_summary.tex"
    write_miniwob_contract_csv(miniwob_rows, miniwob_csv_path)
    write_miniwob_contract_latex(miniwob_rows, miniwob_tex_path)

    throughput_rows = build_throughput_rows(
        {name: Path(path) for name, path in throughput_runs.items()}
    )
    throughput_csv_path = table_dir / "throughput_summary.csv"
    throughput_tex_path = table_dir / "throughput_summary.tex"
    write_throughput_csv(throughput_rows, throughput_csv_path)
    write_throughput_latex(throughput_rows, throughput_tex_path)

    model_rows = build_model_provenance_rows(model_provenance_summary_path)
    model_csv_path = table_dir / "model_provenance_summary.csv"
    model_tex_path = table_dir / "model_provenance_summary.tex"
    write_model_provenance_csv(model_rows, model_csv_path)
    write_model_provenance_latex(model_rows, model_tex_path)

    report = ExperimentSuiteReport(
        suite_dir=str(suite_dir),
        failurebench_runs=failurebench_runs,
        lifecycle_runs=lifecycle_runs,
        miniwob_runs=miniwob_runs,
        throughput_runs=throughput_runs,
        model_provenance_run=str(model_provenance_summary_path),
        tables={
            "failurebench_summary_csv": str(csv_path),
            "failurebench_summary_tex": str(tex_path),
            "lifecycle_summary_csv": str(lifecycle_csv_path),
            "lifecycle_summary_tex": str(lifecycle_tex_path),
            "miniwob_contract_summary_csv": str(miniwob_csv_path),
            "miniwob_contract_summary_tex": str(miniwob_tex_path),
            "throughput_summary_csv": str(throughput_csv_path),
            "throughput_summary_tex": str(throughput_tex_path),
            "model_provenance_summary_csv": str(model_csv_path),
            "model_provenance_summary_tex": str(model_tex_path),
        },
    )
    (suite_dir / "suite_report.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return report
