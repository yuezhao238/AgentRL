from __future__ import annotations

from typer.testing import CliRunner

from agentrl_infra.cli import app


def test_run_failurebench_cli_smoke(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-failurebench",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "smoke",
            "--split",
            "dev",
            "--dev-seeds-per-type",
            "1",
            "--test-seeds-per-type",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "smoke" / "metrics.json").exists()
    assert (tmp_path / "smoke" / "summary.json").exists()


def test_run_miniwob_contract_cli_smoke(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-miniwob-contract",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "smoke",
            "--tasks",
            "click-button,enter-text",
            "--seeds",
            "1000,1001",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "smoke" / "metrics.json").exists()
    assert (tmp_path / "smoke" / "summary.json").exists()
    assert len(list((tmp_path / "smoke" / "traces").glob("*.jsonl"))) == 4


def test_run_throughput_cli_smoke(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-throughput-bench",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "smoke",
            "--policy",
            "failure_aware",
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "smoke" / "throughput_metrics.json").exists()
    assert (tmp_path / "smoke" / "throughput_summary.json").exists()


def test_run_model_provenance_cli_empty_smoke(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-model-provenance",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "smoke",
            "--models",
            "",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "smoke" / "model_provenance_summary.json").exists()


def test_run_model_generation_cli_empty_smoke(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-model-generation-smoke",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "smoke",
            "--models",
            "",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "smoke" / "model_generation_summary.json").exists()
