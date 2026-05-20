from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel

from .benchmarks.lifecycle import LifecycleSummary
from .benchmarks.model_action import ModelActionSummary
from .benchmarks.model_generation import ModelGenerationSummary
from .benchmarks.model_provenance import ModelProvenanceSummary
from .benchmarks.throughput import ThroughputSummary
from .integrations.miniwob import MiniWoBRunSummary
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


class LifecycleTableRow(BaseModel):
    policy: str
    episodes: int
    successes: int
    resets: int
    restores: int
    reuses: int
    contamination_failures: int
    total_cost_units: float
    cost_per_success: float


class MiniWoBContractTableRow(BaseModel):
    policy: str
    episodes: int
    successes: int
    failures: int
    success_rate: float
    replayable_rate: float
    mean_turn_count: float
    agent_invalid_action: int = 0
    repetitive_loop: int = 0
    no_progress: int = 0
    environment_contamination: int = 0


class ThroughputTableRow(BaseModel):
    policy: str
    episodes: int
    attempts: int
    successes: int
    useful: int
    makespan_units: float
    useful_per_hour: float
    success_per_hour: float
    failed_cost_units: float
    zombie_rate: float
    p95_latency_units: float
    utilization: float


class ModelProvenanceTableRow(BaseModel):
    model_id: str
    tokenizer_class: str
    vocab_size: int
    prompt_count: int
    mean_prompt_tokens: float
    max_prompt_tokens: int
    drift_validated: str
    load_seconds: float
    tokenizer_hash_prefix: str


class ModelGenerationTableRow(BaseModel):
    model_id: str
    prompts: int
    prompt_tokens: int
    generated_tokens: int
    load_seconds: float
    latency_seconds: float
    tokens_per_second: float
    mean_logprob: float
    min_logprob: float


class ModelActionTableRow(BaseModel):
    model_id: str
    episodes: int
    successes: int
    parsed_actions: int
    success_rate: float
    parse_rate: float
    invalid_actions: int
    no_progress: int
    generated_tokens: int
    tokens_per_second: float
    mean_logprob: float


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


def build_lifecycle_rows(inputs: dict[str, Path]) -> list[LifecycleTableRow]:
    rows: list[LifecycleTableRow] = []
    for name, summary_path in inputs.items():
        summary = LifecycleSummary.model_validate(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        rows.append(
            LifecycleTableRow(
                policy=name,
                episodes=summary.episode_count,
                successes=summary.success_count,
                resets=summary.reset_count,
                restores=summary.restore_count,
                reuses=summary.reuse_count,
                contamination_failures=summary.contamination_induced_failures,
                total_cost_units=summary.total_cost_units,
                cost_per_success=summary.cost_per_success,
            )
        )
    return rows


def write_lifecycle_csv(rows: list[LifecycleTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LifecycleTableRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def write_lifecycle_latex(rows: list[LifecycleTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_lifecycle_latex(rows), encoding="utf-8")


def build_miniwob_contract_rows(inputs: dict[str, Path]) -> list[MiniWoBContractTableRow]:
    rows: list[MiniWoBContractTableRow] = []
    for name, summary_path in inputs.items():
        summary = MiniWoBRunSummary.model_validate(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        episodes = summary.episode_count
        rows.append(
            MiniWoBContractTableRow(
                policy=name,
                episodes=episodes,
                successes=summary.success_count,
                failures=summary.failure_count,
                success_rate=summary.success_count / episodes if episodes else 0.0,
                replayable_rate=summary.replayable_count / episodes if episodes else 0.0,
                mean_turn_count=summary.mean_turn_count,
                agent_invalid_action=summary.by_failure_type.get("agent_invalid_action", 0),
                repetitive_loop=summary.by_failure_type.get("repetitive_loop", 0),
                no_progress=summary.by_failure_type.get("no_progress", 0),
                environment_contamination=summary.by_failure_type.get(
                    "environment_contamination", 0
                ),
            )
        )
    return rows


def write_miniwob_contract_csv(rows: list[MiniWoBContractTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MiniWoBContractTableRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def write_miniwob_contract_latex(rows: list[MiniWoBContractTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_miniwob_contract_latex(rows), encoding="utf-8")


def build_throughput_rows(inputs: dict[str, Path]) -> list[ThroughputTableRow]:
    rows: list[ThroughputTableRow] = []
    for name, summary_path in inputs.items():
        summary = ThroughputSummary.model_validate(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        rows.append(
            ThroughputTableRow(
                policy=name,
                episodes=summary.episode_count,
                attempts=summary.attempt_count,
                successes=summary.success_count,
                useful=summary.useful_count,
                makespan_units=summary.makespan_units,
                useful_per_hour=summary.useful_trajectories_per_hour,
                success_per_hour=summary.successful_trajectories_per_hour,
                failed_cost_units=summary.failed_cost_units,
                zombie_rate=summary.zombie_session_rate,
                p95_latency_units=summary.p95_latency_units,
                utilization=summary.worker_utilization,
            )
        )
    return rows


def write_throughput_csv(rows: list[ThroughputTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ThroughputTableRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def write_throughput_latex(rows: list[ThroughputTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_throughput_latex(rows), encoding="utf-8")


def build_model_provenance_rows(path: Path) -> list[ModelProvenanceTableRow]:
    summaries = [
        ModelProvenanceSummary.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    rows: list[ModelProvenanceTableRow] = []
    for summary in summaries:
        rows.append(
            ModelProvenanceTableRow(
                model_id=summary.model_id,
                tokenizer_class=summary.tokenizer_class,
                vocab_size=summary.vocab_size,
                prompt_count=summary.prompt_count,
                mean_prompt_tokens=summary.mean_prompt_tokens,
                max_prompt_tokens=summary.max_prompt_tokens,
                drift_validated=f"{summary.drift_validated_count}/{summary.prompt_count}",
                load_seconds=summary.load_seconds,
                tokenizer_hash_prefix=summary.tokenizer_hash[:12],
            )
        )
    return rows


def write_model_provenance_csv(rows: list[ModelProvenanceTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ModelProvenanceTableRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def write_model_provenance_latex(rows: list[ModelProvenanceTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_provenance_latex(rows), encoding="utf-8")


def build_model_generation_rows(path: Path) -> list[ModelGenerationTableRow]:
    summaries = [
        ModelGenerationSummary.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    return [
        ModelGenerationTableRow(
            model_id=summary.model_id,
            prompts=summary.prompt_count,
            prompt_tokens=summary.prompt_tokens,
            generated_tokens=summary.generated_tokens,
            load_seconds=summary.load_seconds,
            latency_seconds=summary.total_latency_seconds,
            tokens_per_second=summary.tokens_per_second,
            mean_logprob=summary.mean_logprob,
            min_logprob=summary.min_logprob,
        )
        for summary in summaries
    ]


def write_model_generation_csv(rows: list[ModelGenerationTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ModelGenerationTableRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def write_model_generation_latex(rows: list[ModelGenerationTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_generation_latex(rows), encoding="utf-8")


def build_model_action_rows(path: Path) -> list[ModelActionTableRow]:
    summaries = [
        ModelActionSummary.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    rows: list[ModelActionTableRow] = []
    for summary in summaries:
        episodes = summary.episode_count
        rows.append(
            ModelActionTableRow(
                model_id=summary.model_id,
                episodes=episodes,
                successes=summary.success_count,
                parsed_actions=summary.parsed_action_count,
                success_rate=summary.success_count / episodes if episodes else 0.0,
                parse_rate=summary.parsed_action_count / episodes if episodes else 0.0,
                invalid_actions=summary.invalid_action_count,
                no_progress=summary.no_progress_count,
                generated_tokens=summary.generated_tokens,
                tokens_per_second=summary.tokens_per_second,
                mean_logprob=summary.mean_logprob,
            )
        )
    return rows


def write_model_action_csv(rows: list[ModelActionTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ModelActionTableRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def write_model_action_latex(rows: list[ModelActionTableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_action_latex(rows), encoding="utf-8")


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


def render_lifecycle_latex(rows: list[LifecycleTableRow]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Policy & Success & Reset & Restore & Reuse & Contam. Fail & Cost/Success \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_latex(row.policy)} & {row.successes}/{row.episodes} & "
            f"{row.resets} & {row.restores} & {row.reuses} & "
            f"{row.contamination_failures} & {row.cost_per_success:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def render_miniwob_contract_latex(rows: list[MiniWoBContractTableRow]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Policy & Success & Succ. Rate & Replayable & Turns & Invalid & Loop & No Prog. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_latex(row.policy)} & {row.successes}/{row.episodes} & "
            f"{row.success_rate:.3f} & {row.replayable_rate:.3f} & "
            f"{row.mean_turn_count:.2f} & {row.agent_invalid_action} & "
            f"{row.repetitive_loop} & {row.no_progress} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def render_throughput_latex(rows: list[ThroughputTableRow]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Policy & Useful/hr & Succ./hr & Failed Cost & Zombie & P95 Lat. & Util. & Attempts \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_latex(row.policy)} & {row.useful_per_hour:.1f} & "
            f"{row.success_per_hour:.1f} & {row.failed_cost_units:.1f} & "
            f"{row.zombie_rate:.3f} & {row.p95_latency_units:.1f} & "
            f"{row.utilization:.3f} & {row.attempts} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def render_model_provenance_latex(rows: list[ModelProvenanceTableRow]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & Vocab & Prompts & Mean Tok. & Max Tok. & Drift Valid \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_latex(_short_model_id(row.model_id))} & {row.vocab_size} & "
            f"{row.prompt_count} & {row.mean_prompt_tokens:.1f} & "
            f"{row.max_prompt_tokens} & {row.drift_validated} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def render_model_generation_latex(rows: list[ModelGenerationTableRow]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & Prompts & In Tok. & Out Tok. & Load s & Tok/s & Mean Logp \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_latex(_short_model_id(row.model_id))} & {row.prompts} & "
            f"{row.prompt_tokens} & {row.generated_tokens} & {row.load_seconds:.1f} & "
            f"{row.tokens_per_second:.2f} & {row.mean_logprob:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def render_model_action_latex(rows: list[ModelActionTableRow]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & Success & Parse & Invalid & No Prog. & Tok/s & Mean Logp \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_latex(_short_model_id(row.model_id))} & "
            f"{row.successes}/{row.episodes} & {row.parsed_actions}/{row.episodes} & "
            f"{row.invalid_actions} & {row.no_progress} & "
            f"{row.tokens_per_second:.2f} & {row.mean_logprob:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _escape_latex(value: str) -> str:
    return value.replace("_", "\\_")


def _short_model_id(value: str) -> str:
    return value.split("/")[-1]
