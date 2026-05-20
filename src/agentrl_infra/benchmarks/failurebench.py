from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..events import EventLog
from ..failures import FailureRecord, FailureType
from ..metrics import EpisodeMetrics, metrics_from_result
from ..orchestrator import BatchOrchestrator, SchedulerRunSummary, fifo_decisions
from ..replay import ReplayMode, ReplayReport
from ..runner import EpisodeResult, EpisodeRunner, StepResult
from ..scheduler import FailureAwareScheduler, RolloutRequest, TaskStats
from ..session import SessionRuntimeConfig


class FailureBenchScenario(StrEnum):
    AGENT_INVALID_ACTION = "agent_invalid_action"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    ENVIRONMENT_CRASH = "environment_crash"
    ENVIRONMENT_CONTAMINATION = "environment_contamination"
    CONTEXT_LIMIT = "context_limit"
    REPETITIVE_LOOP = "repetitive_loop"
    RATE_LIMIT = "rate_limit"

    @property
    def failure_type(self) -> FailureType:
        return FailureType(self.value)


class FailureBenchCase(BaseModel):
    scenario: FailureBenchScenario
    seed: int
    split: str
    task_id: str
    sample_id: str
    oracle_failure_type: FailureType
    oracle_failure_turn: int
    oracle_attribution: str
    oracle_replayable: bool
    expected_salvageable: bool


@dataclass
class FailureBenchPolicy:
    case: FailureBenchCase

    def next_action(self, observation: dict[str, Any], log: Any) -> dict[str, Any]:
        turn = len([event for event in log.events if event.type.value == "action_proposed"]) + 1
        scenario = self.case.scenario

        if scenario == FailureBenchScenario.AGENT_INVALID_ACTION:
            return {"tool": "missing_tool", "args": {"seed": self.case.seed}}
        if scenario == FailureBenchScenario.TOOL_TIMEOUT:
            return {"tool": "slow_search", "args": {"timeout_ms": 1, "seed": self.case.seed}}
        if scenario == FailureBenchScenario.TOOL_EXECUTION_ERROR:
            return {"tool": "calculator", "args": {"expr": "1/0", "seed": self.case.seed}}
        if scenario == FailureBenchScenario.ENVIRONMENT_CRASH:
            return {"tool": "browser", "args": {"url": "about:crash", "seed": self.case.seed}}
        if scenario == FailureBenchScenario.ENVIRONMENT_CONTAMINATION:
            return {"tool": "mutate_global_state", "args": {"seed": self.case.seed}}
        if scenario == FailureBenchScenario.CONTEXT_LIMIT:
            return {"tool": "noop", "args": {"turn": turn, "seed": self.case.seed}}
        if scenario == FailureBenchScenario.REPETITIVE_LOOP:
            return {"tool": "search", "args": {"query": "same query"}}
        if scenario == FailureBenchScenario.RATE_LIMIT:
            return {"tool": "external_api", "args": {"burst": 10_000, "seed": self.case.seed}}

        return {"tool": "noop", "args": {"turn": turn}}


@dataclass
class FailureBenchEnvironment:
    case: FailureBenchCase
    turn: int = 0

    def reset(self) -> dict[str, Any]:
        self.turn = 0
        return {
            "scenario": self.case.scenario.value,
            "seed": self.case.seed,
            "state": "ready",
        }

    def step(self, action: dict[str, Any]) -> StepResult:
        self.turn += 1
        scenario = self.case.scenario
        details = {"oracle_failure_turn": self.case.oracle_failure_turn, "action": action}

        if scenario == FailureBenchScenario.AGENT_INVALID_ACTION:
            return self._failure(
                FailureType.AGENT_INVALID_ACTION,
                "action references a tool that is not registered",
                "tool_validator",
                details,
            )
        if scenario == FailureBenchScenario.TOOL_TIMEOUT:
            return self._failure(
                FailureType.TOOL_TIMEOUT,
                "tool call exceeded execution budget",
                "tool_runtime",
                details,
            )
        if scenario == FailureBenchScenario.TOOL_EXECUTION_ERROR:
            return self._failure(
                FailureType.TOOL_EXECUTION_ERROR,
                "tool raised a typed execution error",
                "tool_runtime",
                details,
            )
        if scenario == FailureBenchScenario.ENVIRONMENT_CRASH:
            return self._failure(
                FailureType.ENVIRONMENT_CRASH,
                "environment became unavailable",
                "environment",
                details,
            )
        if scenario == FailureBenchScenario.ENVIRONMENT_CONTAMINATION:
            return self._failure(
                FailureType.ENVIRONMENT_CONTAMINATION,
                "environment state failed isolation invariant",
                "environment_health_check",
                details,
            )
        if scenario == FailureBenchScenario.RATE_LIMIT:
            return self._failure(
                FailureType.RATE_LIMIT,
                "external API returned a rate-limit response",
                "external_api",
                details,
            )

        return StepResult(
            observation={
                "scenario": scenario.value,
                "seed": self.case.seed,
                "state": (
                    "unchanged"
                    if scenario == FailureBenchScenario.REPETITIVE_LOOP
                    else "growing"
                ),
                "turn": self.turn,
            },
            reward=0.0,
            done=False,
            info={"synthetic": True, "scenario": scenario.value},
        )

    @staticmethod
    def _failure(
        failure_type: FailureType,
        message: str,
        source: str,
        details: dict[str, Any],
    ) -> StepResult:
        return StepResult(
            observation={"state": "failed", "failure_type": failure_type.value},
            reward=0.0,
            done=False,
            failure=FailureRecord(
                type=failure_type,
                message=message,
                source=source,
                details=details,
            ),
            info={"failure_type": failure_type.value},
        )


def build_failurebench_cases(
    *,
    split: str = "all",
    dev_seeds_per_type: int = 20,
    test_seeds_per_type: int = 80,
) -> list[FailureBenchCase]:
    if split not in {"dev", "test", "all"}:
        raise ValueError(f"unknown split: {split}")

    cases: list[FailureBenchCase] = []
    for scenario in FailureBenchScenario:
        if split in {"dev", "all"}:
            cases.extend(_cases_for_scenario(scenario, "dev", range(dev_seeds_per_type)))
        if split in {"test", "all"}:
            start = dev_seeds_per_type
            stop = dev_seeds_per_type + test_seeds_per_type
            cases.extend(_cases_for_scenario(scenario, "test", range(start, stop)))
    return cases


def run_failurebench(
    *,
    output_dir: Path,
    run_id: str,
    split: str = "all",
    dev_seeds_per_type: int = 20,
    test_seeds_per_type: int = 80,
) -> list[EpisodeMetrics]:
    run_dir = output_dir / run_id
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    cases = build_failurebench_cases(
        split=split,
        dev_seeds_per_type=dev_seeds_per_type,
        test_seeds_per_type=test_seeds_per_type,
    )
    metrics: list[EpisodeMetrics] = []
    for case in cases:
        result = run_failurebench_case(case)
        trace_path = trace_dir / f"{case.sample_id}.jsonl"
        result.event_log.save_jsonl(trace_path)
        report = ReplayReport.from_log(result.event_log, ReplayMode.EXACT)
        metrics.append(
            metrics_from_result(
                result,
                oracle_failure_type=case.oracle_failure_type,
                oracle_failure_turn=case.oracle_failure_turn,
                scenario=case.scenario.value,
                split=case.split,
                seed=case.seed,
                trace_path=trace_path,
                replayable=report.replayable,
            )
        )
    return metrics


def run_failurebench_scheduled(
    *,
    output_dir: Path,
    run_id: str,
    split: str = "all",
    dev_seeds_per_type: int = 20,
    test_seeds_per_type: int = 80,
    scheduler_policy: str = "failure_aware",
) -> SchedulerRunSummary:
    run_dir = output_dir / run_id
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    cases = build_failurebench_cases(
        split=split,
        dev_seeds_per_type=dev_seeds_per_type,
        test_seeds_per_type=test_seeds_per_type,
    )
    cases_by_sample = {case.sample_id: case for case in cases}
    requests = [
        RolloutRequest(
            task_id=case.task_id,
            sample_id=case.sample_id,
            priority=1.0,
            estimated_cost=float(case.oracle_failure_turn),
        )
        for case in cases
    ]

    def execute(request: RolloutRequest) -> EpisodeMetrics:
        case = cases_by_sample[request.sample_id]
        result = run_failurebench_case(case)
        trace_path = trace_dir / f"{case.sample_id}.jsonl"
        result.event_log.save_jsonl(trace_path)
        report = ReplayReport.from_log(result.event_log, ReplayMode.EXACT)
        return metrics_from_result(
            result,
            oracle_failure_type=case.oracle_failure_type,
            oracle_failure_turn=case.oracle_failure_turn,
            scenario=case.scenario.value,
            split=case.split,
            seed=case.seed,
            trace_path=trace_path,
            replayable=report.replayable,
        )

    if scheduler_policy == "fifo":
        return fifo_decisions(requests, execute)
    if scheduler_policy != "failure_aware":
        raise ValueError(f"unknown scheduler policy: {scheduler_policy}")
    return BatchOrchestrator(FailureAwareScheduler()).run(
        requests,
        _initial_task_stats(cases),
        execute,
    )


def run_failurebench_case(case: FailureBenchCase) -> EpisodeResult:
    config = SessionRuntimeConfig(
        max_turns=4 if case.scenario == FailureBenchScenario.CONTEXT_LIMIT else 16,
        max_wall_time_seconds=60,
        max_repeated_actions=3,
        max_no_progress_observations=0,
    )
    runner = EpisodeRunner(config)
    return runner.run(
        session_id=f"fb-{case.sample_id}",
        task_id=case.task_id,
        sample_id=case.sample_id,
        policy=FailureBenchPolicy(case),
        environment=FailureBenchEnvironment(case),
        metadata={
            "benchmark": "failurebench",
            "scenario": case.scenario.value,
            "seed": case.seed,
            "split": case.split,
        },
    )


def _initial_task_stats(cases: list[FailureBenchCase]) -> dict[str, TaskStats]:
    stats: dict[str, TaskStats] = {}
    for case in cases:
        stats.setdefault(case.task_id, TaskStats(capacity=1, mean_cost=case.oracle_failure_turn))
    return stats


def rerun_failurebench_log(log: Any) -> EventLog:
    case = case_from_failurebench_log(log)
    return run_failurebench_case(case).event_log


def case_from_failurebench_log(log: Any) -> FailureBenchCase:
    if not log.events:
        raise ValueError("cannot reconstruct FailureBench case from empty log")
    start = log.events[0]
    metadata = start.payload.get("metadata", {})
    if metadata.get("benchmark") != "failurebench":
        raise ValueError("trace does not contain FailureBench metadata")
    scenario = FailureBenchScenario(metadata["scenario"])
    seed = int(metadata["seed"])
    split = str(metadata["split"])
    return FailureBenchCase(
        scenario=scenario,
        seed=seed,
        split=split,
        task_id=f"failurebench/{scenario.value}",
        sample_id=log.sample_id or f"{split}-{scenario.value}-{seed:04d}",
        oracle_failure_type=scenario.failure_type,
        oracle_failure_turn=_oracle_failure_turn(scenario),
        oracle_attribution=_oracle_attribution(scenario.failure_type),
        oracle_replayable=_oracle_replayable(scenario.failure_type),
        expected_salvageable=_oracle_salvageable(scenario.failure_type),
    )


def _cases_for_scenario(
    scenario: FailureBenchScenario,
    split: str,
    seeds: range,
) -> list[FailureBenchCase]:
    return [
        FailureBenchCase(
            scenario=scenario,
            seed=seed,
            split=split,
            task_id=f"failurebench/{scenario.value}",
            sample_id=f"{split}-{scenario.value}-{seed:04d}",
            oracle_failure_type=scenario.failure_type,
            oracle_failure_turn=_oracle_failure_turn(scenario),
            oracle_attribution=_oracle_attribution(scenario.failure_type),
            oracle_replayable=_oracle_replayable(scenario.failure_type),
            expected_salvageable=_oracle_salvageable(scenario.failure_type),
        )
        for seed in seeds
    ]


def _oracle_failure_turn(scenario: FailureBenchScenario) -> int:
    if scenario == FailureBenchScenario.CONTEXT_LIMIT:
        return 5
    if scenario == FailureBenchScenario.REPETITIVE_LOOP:
        return 3
    return 1


def _oracle_attribution(failure_type: FailureType) -> str:
    semantics = FailureRecord(type=failure_type).resolved_semantics()
    return "policy" if semantics.attributable_to_policy else "runtime"


def _oracle_replayable(failure_type: FailureType) -> bool:
    semantics = FailureRecord(type=failure_type).resolved_semantics()
    return not semantics.reset_environment


def _oracle_salvageable(failure_type: FailureType) -> bool:
    return FailureRecord(type=failure_type).resolved_semantics().salvageable
