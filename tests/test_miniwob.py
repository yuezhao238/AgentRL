from __future__ import annotations

from agentrl_infra.failures import FailureType
from agentrl_infra.integrations.miniwob import (
    BrowserActionType,
    MiniWoBContractEnvironment,
    build_miniwob_task_specs,
    run_miniwob_contract_subset,
)
from agentrl_infra.resources import EnvironmentAdapter


def test_miniwob_task_specs_cover_fixed_subset() -> None:
    specs = build_miniwob_task_specs()

    assert len(specs) == 20
    assert specs["click-button"].required_action == BrowserActionType.CLICK
    assert specs["enter-text"].required_action == BrowserActionType.TYPE_TEXT


def test_miniwob_contract_environment_is_adapter_compatible() -> None:
    specs = build_miniwob_task_specs()
    env = MiniWoBContractEnvironment(specs["click-button"], seed=1000)

    assert isinstance(env, EnvironmentAdapter)


def test_miniwob_contract_accepts_oracle_action_and_detects_contamination() -> None:
    specs = build_miniwob_task_specs()
    env = MiniWoBContractEnvironment(specs["enter-text"], seed=1000)
    observation = env.reset()

    step = env.step(observation["action_hint"])
    assert step.done
    assert step.reward == 1.0

    contaminated = env.step(observation["action_hint"])
    assert contaminated.failure is not None
    assert contaminated.failure.type == FailureType.ENVIRONMENT_CONTAMINATION


def test_miniwob_contract_detects_stale_dom() -> None:
    specs = build_miniwob_task_specs()
    env = MiniWoBContractEnvironment(specs["click-button"], seed=1000)
    observation = env.reset()
    action = dict(observation["action_hint"])
    action["observed_dom_hash"] = "stale"

    step = env.step(action)

    assert step.failure is not None
    assert step.failure.type == FailureType.AGENT_INVALID_ACTION


def test_run_miniwob_contract_subset_writes_traces(tmp_path) -> None:
    _, summary = run_miniwob_contract_subset(
        output_dir=tmp_path,
        run_id="smoke",
        task_names=["click-button", "enter-text"],
        seeds=[1000, 1001],
    )

    assert summary.episode_count == 4
    assert summary.success_count == 4
    assert (tmp_path / "smoke" / "summary.json").exists()
    assert len(list((tmp_path / "smoke" / "traces").glob("*.jsonl"))) == 4


def test_run_miniwob_contract_repeated_action_exercises_no_progress_detector(tmp_path) -> None:
    _, summary = run_miniwob_contract_subset(
        output_dir=tmp_path,
        run_id="loop",
        task_names=["enter-text"],
        seeds=[1000],
        policy_name="repeated_action",
    )

    assert summary.success_count == 0
    assert summary.by_failure_type[FailureType.NO_PROGRESS.value] == 1
