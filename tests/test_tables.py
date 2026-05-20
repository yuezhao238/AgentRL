from __future__ import annotations

from agentrl_infra.tables import TableRow, render_summary_latex, write_summary_csv


def test_render_summary_latex_escapes_names() -> None:
    latex = render_summary_latex(
        [
            TableRow(
                name="raw_timeout",
                episodes=8,
                macro_f1=0.0,
                detection_accuracy=0.0,
                mean_wasted_turns=1.0,
                failed_rollout_cost_turns=8,
                replayable_failure_rate=0.0,
                useful_trajectories_per_hour=0.0,
            )
        ]
    )

    assert "raw\\_timeout" in latex
    assert "\\begin{tabular}" in latex


def test_write_summary_csv(tmp_path) -> None:
    path = tmp_path / "table.csv"
    write_summary_csv([], path)

    assert path.exists()
    assert "name" in path.read_text(encoding="utf-8")
