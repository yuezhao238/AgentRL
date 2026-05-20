from __future__ import annotations

from agentrl_infra.tables import (
    LifecycleTableRow,
    MiniWoBContractTableRow,
    TableRow,
    ThroughputTableRow,
    render_lifecycle_latex,
    render_miniwob_contract_latex,
    render_summary_latex,
    render_throughput_latex,
    write_summary_csv,
)


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
                failed_rollout_cost_units=8.0,
                replayable_failure_rate=0.0,
                useful_trajectories_per_cost_unit=0.0,
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


def test_render_lifecycle_latex() -> None:
    latex = render_lifecycle_latex(
        [
            LifecycleTableRow(
                policy="blind_reuse",
                episodes=5,
                successes=1,
                resets=1,
                restores=0,
                reuses=4,
                contamination_failures=4,
                total_cost_units=1.5,
                cost_per_success=1.5,
            )
        ]
    )

    assert "blind\\_reuse" in latex
    assert "Contam. Fail" in latex


def test_render_miniwob_contract_latex() -> None:
    latex = render_miniwob_contract_latex(
        [
            MiniWoBContractTableRow(
                policy="stale_dom",
                episodes=10,
                successes=0,
                failures=10,
                success_rate=0.0,
                replayable_rate=1.0,
                mean_turn_count=1.0,
                agent_invalid_action=10,
            )
        ]
    )

    assert "stale\\_dom" in latex
    assert "No Prog." in latex


def test_render_throughput_latex() -> None:
    latex = render_throughput_latex(
        [
            ThroughputTableRow(
                policy="failure_aware",
                episodes=240,
                attempts=240,
                successes=100,
                useful=200,
                makespan_units=100.0,
                useful_per_hour=7200.0,
                success_per_hour=3600.0,
                failed_cost_units=100.0,
                zombie_rate=0.0,
                p95_latency_units=4.0,
                utilization=0.7,
            )
        ]
    )

    assert "failure\\_aware" in latex
    assert "Useful/hr" in latex
