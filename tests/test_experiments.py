from __future__ import annotations

import json

from agentrl_infra.experiments import (
    ExperimentSuiteConfig,
    load_experiment_suite_config,
    run_experiment_suite,
)


def test_run_experiment_suite_smoke(tmp_path) -> None:
    report = run_experiment_suite(
        ExperimentSuiteConfig(
            suite_id="smoke",
            output_dir=tmp_path,
            split="dev",
            dev_seeds_per_type=1,
            test_seeds_per_type=0,
            lifecycle_episodes=5,
            miniwob_tasks=["click-button", "enter-text"],
            miniwob_seeds=[1000],
            throughput_workers=2,
        )
    )

    suite_dir = tmp_path / "smoke"
    assert (suite_dir / "suite_report.json").exists()
    assert (suite_dir / "tables" / "failurebench_summary.csv").exists()
    assert (suite_dir / "tables" / "failurebench_summary.tex").exists()
    assert (suite_dir / "tables" / "miniwob_contract_summary.csv").exists()
    assert (suite_dir / "tables" / "miniwob_contract_summary.tex").exists()
    assert (suite_dir / "tables" / "throughput_summary.csv").exists()
    assert (suite_dir / "tables" / "throughput_summary.tex").exists()
    assert "rolloutos" in report.failurebench_runs
    assert "oracle" in report.miniwob_runs
    assert "failure_aware" in report.throughput_runs


def test_load_experiment_suite_config(tmp_path) -> None:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "suite_id": "cfg",
                "output_dir": str(tmp_path),
                "split": "dev",
                "dev_seeds_per_type": 1,
                "test_seeds_per_type": 0,
                "lifecycle_episodes": 5,
                "miniwob_tasks": ["click-button"],
                "miniwob_seeds": [1000, 1001],
                "throughput_workers": 2,
            }
        ),
        encoding="utf-8",
    )

    config = load_experiment_suite_config(path)

    assert config.suite_id == "cfg"
    assert config.dev_seeds_per_type == 1
    assert config.miniwob_tasks == ["click-button"]
    assert config.miniwob_seeds == [1000, 1001]
    assert config.throughput_workers == 2
