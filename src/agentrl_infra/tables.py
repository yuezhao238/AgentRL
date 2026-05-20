from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel

from .metrics import RunSummary, load_metrics, summarize_metrics


class TableRow(BaseModel):
    name: str
    episodes: int
    macro_f1: float
    detection_accuracy: float
    mean_wasted_turns: float
    failed_rollout_cost_turns: int
    failed_rollout_cost_units: float
    replayable_failure_rate: float
    useful_trajectories_per_cost_unit: float
    useful_trajectories_per_hour: float


def build_summary_rows(inputs: dict[str, Path]) -> list[TableRow]:
    rows: list[TableRow] = []
    for name, metrics_path in inputs.items():
        summary = summarize_metrics(load_metrics(metrics_path))
        rows.append(table_row_from_summary(name, summary))
    return rows


def table_row_from_summary(name: str, summary: RunSummary) -> TableRow:
    return TableRow(
        name=name,
        episodes=summary.episode_count,
        macro_f1=summary.macro_f1,
        detection_accuracy=summary.detection_accuracy,
        mean_wasted_turns=summary.mean_turns_wasted_after_oracle,
        failed_rollout_cost_turns=summary.failed_rollout_cost_turns,
        failed_rollout_cost_units=summary.failed_rollout_cost_units,
        replayable_failure_rate=summary.replayable_failure_rate,
        useful_trajectories_per_cost_unit=summary.useful_trajectories_per_cost_unit,
        useful_trajectories_per_hour=summary.useful_trajectories_per_hour,
    )


def write_summary_csv(rows: list[TableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TableRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def write_summary_latex(rows: list[TableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_summary_latex(rows), encoding="utf-8")


def render_summary_latex(rows: list[TableRow]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        (
            "System & Episodes & Macro F1 & Acc. & Wasted Turns & "
            "Failed Cost & Replayable & Useful/Cost \\\\"
        ),
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_latex(row.name)} & {row.episodes} & {row.macro_f1:.3f} & "
            f"{row.detection_accuracy:.3f} & {row.mean_wasted_turns:.2f} & "
            f"{row.failed_rollout_cost_units:.1f} & {row.replayable_failure_rate:.3f} & "
            f"{row.useful_trajectories_per_cost_unit:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _escape_latex(value: str) -> str:
    return value.replace("_", "\\_")
