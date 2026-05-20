from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..events import EventLog
from ..failures import FailureRecord, FailureType
from ..resources import HealthCheckResult
from ..runner import EpisodeResult, EpisodeRunner, StepResult
from ..session import SessionRuntimeConfig, SessionState


class BrowserActionType(StrEnum):
    CLICK = "click"
    TYPE_TEXT = "type_text"
    WAIT = "wait"


class BrowserAction(BaseModel):
    action_type: BrowserActionType
    selector: str | None = None
    text: str | None = None
    observed_dom_hash: str | None = None


class BrowserElement(BaseModel):
    selector: str
    tag: str
    text: str = ""
    value: str = ""
    enabled: bool = True
    visible: bool = True


class BrowserObservation(BaseModel):
    task_name: str
    seed: int
    dom_hash: str
    url: str
    elements: list[BrowserElement]
    action_hint: BrowserAction
    progress_key: str
    done: bool = False

    def to_runner_observation(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class MiniWoBTaskSpec(BaseModel):
    name: str
    target_selector: str
    required_action: BrowserActionType
    expected_text: str | None = None
    max_turns: int = 15
    tags: list[str] = Field(default_factory=list)


class MiniWoBEpisodeMetrics(BaseModel):
    session_id: str
    task_name: str
    seed: int
    success: bool
    failure_type: FailureType | None
    turn_count: int
    event_count: int
    total_reward: float
    replayable: bool
    trace_path: str


class MiniWoBRunSummary(BaseModel):
    policy: str
    episode_count: int
    success_count: int
    failure_count: int
    replayable_count: int
    mean_turn_count: float
    by_failure_type: dict[str, int] = Field(default_factory=dict)


MINIWOB_20_TASKS: tuple[str, ...] = (
    "click-button",
    "click-checkboxes",
    "click-checkboxes-large",
    "click-collapsible",
    "click-dialog",
    "click-link",
    "click-menu",
    "click-option",
    "click-pie",
    "enter-date",
    "enter-password",
    "enter-text",
    "focus-text",
    "login-user",
    "multi-layouts",
    "navigate-tree",
    "search-engine",
    "social-media",
    "use-autocomplete",
    "use-spinner",
)


def build_miniwob_task_specs() -> dict[str, MiniWoBTaskSpec]:
    specs: dict[str, MiniWoBTaskSpec] = {}
    for name in MINIWOB_20_TASKS:
        if name.startswith("enter-") or name in {
            "focus-text",
            "login-user",
            "search-engine",
            "use-autocomplete",
            "use-spinner",
        }:
            specs[name] = MiniWoBTaskSpec(
                name=name,
                target_selector="#target-input",
                required_action=BrowserActionType.TYPE_TEXT,
                expected_text=f"{name}-value",
                tags=["form"],
            )
        else:
            specs[name] = MiniWoBTaskSpec(
                name=name,
                target_selector="#target",
                required_action=BrowserActionType.CLICK,
                tags=["click"],
            )
    return specs


@dataclass
class MiniWoBContractEnvironment:
    """Deterministic MiniWoB-like browser contract used before binding a live browser."""

    task_spec: MiniWoBTaskSpec
    seed: int
    environment_id: str | None = None
    dirty: bool = False
    closed: bool = False
    _done: bool = False
    _snapshots: dict[str, tuple[bool, bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.environment_id is None:
            self.environment_id = f"miniwob-contract-{self.task_spec.name}-{self.seed}"

    def reset(self) -> dict[str, Any]:
        self.dirty = False
        self.closed = False
        self._done = False
        return self._observation().to_runner_observation()

    def step(self, action: dict[str, Any]) -> StepResult:
        if self.closed:
            return StepResult(
                observation={"closed": True},
                failure=FailureRecord(
                    type=FailureType.ENVIRONMENT_CRASH,
                    message="browser environment is closed",
                    source="miniwob_contract",
                ),
            )
        if self.dirty:
            return StepResult(
                observation=self._observation().to_runner_observation(),
                failure=FailureRecord(
                    type=FailureType.ENVIRONMENT_CONTAMINATION,
                    message="browser state leaked across episodes without reset/restore",
                    source="miniwob_contract",
                    details={"task_name": self.task_spec.name, "seed": self.seed},
                ),
            )

        parsed, failure = self._parse_action(action)
        if failure:
            return StepResult(
                observation=self._observation().to_runner_observation(),
                failure=failure,
            )
        assert parsed is not None

        current = self._observation()
        if parsed.action_type == BrowserActionType.WAIT:
            return StepResult(
                observation=current.to_runner_observation(),
                reward=0.0,
                done=False,
                info={"progress_key": current.progress_key, "reason": "wait"},
            )

        if parsed.observed_dom_hash and parsed.observed_dom_hash != current.dom_hash:
            return StepResult(
                observation=current.to_runner_observation(),
                failure=FailureRecord(
                    type=FailureType.AGENT_INVALID_ACTION,
                    message="action was produced for a stale DOM",
                    source="miniwob_contract",
                    details={
                        "expected_dom_hash": current.dom_hash,
                        "observed_dom_hash": parsed.observed_dom_hash,
                    },
                ),
            )

        if parsed.selector != self.task_spec.target_selector:
            return StepResult(
                observation=current.to_runner_observation(),
                failure=FailureRecord(
                    type=FailureType.AGENT_INVALID_ACTION,
                    message="selector does not target an actionable MiniWoB element",
                    source="miniwob_contract",
                    details={"selector": parsed.selector, "target": self.task_spec.target_selector},
                ),
            )
        if parsed.action_type != self.task_spec.required_action:
            return StepResult(
                observation=current.to_runner_observation(),
                failure=FailureRecord(
                    type=FailureType.AGENT_INVALID_ACTION,
                    message="browser action type does not match task contract",
                    source="miniwob_contract",
                    details={
                        "action_type": parsed.action_type,
                        "required_action": self.task_spec.required_action,
                    },
                ),
            )
        if self.task_spec.expected_text is not None and parsed.text != self.task_spec.expected_text:
            return StepResult(
                observation=current.to_runner_observation(),
                reward=0.0,
                done=False,
                info={"progress_key": current.progress_key, "reason": "text_mismatch"},
            )

        self._done = True
        self.dirty = True
        done_observation = self._observation()
        return StepResult(
            observation=done_observation.to_runner_observation(),
            reward=1.0,
            done=True,
            info={
                "dom_hash": done_observation.dom_hash,
                "progress_key": done_observation.progress_key,
            },
        )

    def snapshot(self) -> str:
        snapshot_id = f"{self.environment_id}:dirty={int(self.dirty)}:done={int(self._done)}"
        self._snapshots[snapshot_id] = (self.dirty, self._done)
        return snapshot_id

    def restore(self, snapshot_id: str) -> None:
        dirty, done = self._snapshots.get(snapshot_id, (False, False))
        self.dirty = dirty
        self._done = done
        self.closed = False

    def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(
            healthy=not self.closed and not self.dirty,
            details={"closed": self.closed, "dirty": self.dirty, "done": self._done},
        )

    def contamination_check(self) -> HealthCheckResult:
        return HealthCheckResult(
            healthy=not self.dirty,
            details={"dirty": self.dirty, "task_name": self.task_spec.name, "seed": self.seed},
        )

    def close(self) -> None:
        self.closed = True

    def _parse_action(
        self, action: dict[str, Any]
    ) -> tuple[BrowserAction | None, FailureRecord | None]:
        try:
            return BrowserAction.model_validate(action), None
        except Exception as exc:
            return None, FailureRecord(
                type=FailureType.AGENT_INVALID_ACTION,
                message="action does not match browser action schema",
                source="miniwob_contract",
                details={"error": str(exc), "action": action},
            )

    def _observation(self) -> BrowserObservation:
        target = BrowserElement(
            selector=self.task_spec.target_selector,
            tag=(
                "input"
                if self.task_spec.required_action == BrowserActionType.TYPE_TEXT
                else "button"
            ),
            text=self.task_spec.name,
            value="" if not self._done else (self.task_spec.expected_text or ""),
        )
        distractor = BrowserElement(selector="#distractor", tag="button", text="decoy")
        dom_hash = self._dom_hash(done=self._done)
        return BrowserObservation(
            task_name=self.task_spec.name,
            seed=self.seed,
            dom_hash=dom_hash,
            url=f"miniwob://{self.task_spec.name}?seed={self.seed}",
            elements=[target, distractor],
            action_hint=BrowserAction(
                action_type=self.task_spec.required_action,
                selector=self.task_spec.target_selector,
                text=self.task_spec.expected_text,
                observed_dom_hash=dom_hash,
            ),
            progress_key=f"{self.task_spec.name}:{self.seed}:done={int(self._done)}",
            done=self._done,
        )

    def _dom_hash(self, *, done: bool) -> str:
        raw = f"{self.task_spec.name}|{self.seed}|{self.task_spec.target_selector}|{done}"
        return sha256(raw.encode("utf-8")).hexdigest()


class MiniWoBOraclePolicy:
    def next_action(self, observation: dict[str, Any], log: EventLog) -> dict[str, Any]:
        return dict(observation["action_hint"])


class MiniWoBRepeatedActionPolicy:
    def next_action(self, observation: dict[str, Any], log: EventLog) -> dict[str, Any]:
        hint = observation["action_hint"]
        return {
            "action_type": hint["action_type"],
            "selector": hint["selector"],
            "text": "__wrong_repeated_text__",
            "observed_dom_hash": observation["dom_hash"],
        }


class MiniWoBStaleDomPolicy:
    def next_action(self, observation: dict[str, Any], log: EventLog) -> dict[str, Any]:
        action = dict(observation["action_hint"])
        action["observed_dom_hash"] = "stale-dom-hash"
        return action


class MiniWoBInvalidSelectorPolicy:
    def next_action(self, observation: dict[str, Any], log: EventLog) -> dict[str, Any]:
        action = dict(observation["action_hint"])
        action["selector"] = "#distractor"
        return action


class MiniWoBWaitLoopPolicy:
    def next_action(self, observation: dict[str, Any], log: EventLog) -> dict[str, Any]:
        return {
            "action_type": BrowserActionType.WAIT.value,
            "observed_dom_hash": observation["dom_hash"],
        }


def run_miniwob_contract_subset(
    *,
    output_dir: Path,
    run_id: str,
    task_names: list[str] | None = None,
    seeds: list[int] | None = None,
    policy_name: str = "oracle",
) -> tuple[list[MiniWoBEpisodeMetrics], MiniWoBRunSummary]:
    specs = build_miniwob_task_specs()
    selected_tasks = task_names or list(MINIWOB_20_TASKS)
    selected_seeds = seeds or [1000]
    run_dir = output_dir / run_id
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    runner = EpisodeRunner(
        SessionRuntimeConfig(
            max_turns=15,
            max_repeated_actions=3,
            max_no_progress_observations=3,
        )
    )
    metrics: list[MiniWoBEpisodeMetrics] = []

    for task_name in selected_tasks:
        if task_name not in specs:
            raise ValueError(f"unknown MiniWoB task: {task_name}")
        for seed in selected_seeds:
            env = MiniWoBContractEnvironment(specs[task_name], seed=seed)
            policy = _policy_from_name(policy_name)
            session_id = f"miniwob-{task_name}-{seed}-{policy_name}"
            result = runner.run(
                session_id=session_id,
                task_id="miniwob_contract",
                sample_id=f"{task_name}:{seed}",
                policy=policy,
                environment=env,
                metadata={"task_name": task_name, "seed": seed, "policy": policy_name},
            )
            trace_path = trace_dir / f"{task_name}-{seed}.jsonl"
            result.event_log.save_jsonl(trace_path)
            metrics.append(_metrics_from_result(result, task_name, seed, trace_path))

    summary = summarize_miniwob_metrics(metrics, policy_name=policy_name)
    _write_miniwob_run(run_dir, selected_tasks, selected_seeds, policy_name, metrics, summary)
    return metrics, summary


def summarize_miniwob_metrics(
    metrics: list[MiniWoBEpisodeMetrics],
    *,
    policy_name: str = "unknown",
) -> MiniWoBRunSummary:
    by_failure: dict[str, int] = {}
    for metric in metrics:
        if metric.failure_type:
            by_failure[metric.failure_type.value] = by_failure.get(metric.failure_type.value, 0) + 1
    return MiniWoBRunSummary(
        policy=policy_name,
        episode_count=len(metrics),
        success_count=sum(1 for metric in metrics if metric.success),
        failure_count=sum(1 for metric in metrics if not metric.success),
        replayable_count=sum(1 for metric in metrics if metric.replayable),
        mean_turn_count=(
            sum(metric.turn_count for metric in metrics) / len(metrics) if metrics else 0.0
        ),
        by_failure_type=by_failure,
    )


def _policy_from_name(
    policy_name: str,
) -> (
    MiniWoBOraclePolicy
    | MiniWoBRepeatedActionPolicy
    | MiniWoBStaleDomPolicy
    | MiniWoBInvalidSelectorPolicy
    | MiniWoBWaitLoopPolicy
):
    if policy_name == "oracle":
        return MiniWoBOraclePolicy()
    if policy_name == "repeated_action":
        return MiniWoBRepeatedActionPolicy()
    if policy_name == "stale_dom":
        return MiniWoBStaleDomPolicy()
    if policy_name == "invalid_selector":
        return MiniWoBInvalidSelectorPolicy()
    if policy_name == "wait_loop":
        return MiniWoBWaitLoopPolicy()
    raise ValueError(f"unknown MiniWoB policy: {policy_name}")


def _metrics_from_result(
    result: EpisodeResult,
    task_name: str,
    seed: int,
    trace_path: Path,
) -> MiniWoBEpisodeMetrics:
    failure = result.failure
    return MiniWoBEpisodeMetrics(
        session_id=result.session_id,
        task_name=task_name,
        seed=seed,
        success=result.state == SessionState.COMPLETED,
        failure_type=failure.type if failure else None,
        turn_count=result.turn_count,
        event_count=len(result.event_log.events),
        total_reward=result.total_reward,
        replayable=True,
        trace_path=str(trace_path),
    )


def _write_miniwob_run(
    run_dir: Path,
    task_names: list[str],
    seeds: list[int],
    policy_name: str,
    metrics: list[MiniWoBEpisodeMetrics],
    summary: MiniWoBRunSummary,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "benchmark": "miniwob_contract",
                "tasks": task_names,
                "seeds": seeds,
                "policy": policy_name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps([metric.model_dump(mode="json") for metric in metrics], indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
