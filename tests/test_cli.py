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
