from __future__ import annotations

from agentrl_infra import (
    EpisodeRunner,
    FailureRecord,
    FailureType,
    SessionRuntimeConfig,
    StepResult,
)


class OneStepPolicy:
    def next_action(self, observation, log):
        return {"tool": "finish", "observation_state": observation["state"]}


class OneStepEnvironment:
    def reset(self):
        return {"state": "ready"}

    def step(self, action):
        return StepResult(observation={"state": "done"}, reward=1.0, done=True)


class FailingEnvironment:
    def reset(self):
        return {"state": "ready"}

    def step(self, action):
        return StepResult(
            observation={"state": "failed"},
            failure=FailureRecord(type=FailureType.TOOL_EXECUTION_ERROR),
        )


def test_episode_runner_completes_successful_episode() -> None:
    runner = EpisodeRunner(SessionRuntimeConfig(max_turns=4))

    result = runner.run(
        session_id="s1",
        task_id="unit",
        sample_id="a",
        policy=OneStepPolicy(),
        environment=OneStepEnvironment(),
    )

    assert result.failure is None
    assert result.total_reward == 1.0
    assert result.turn_count == 1


def test_episode_runner_records_environment_failure() -> None:
    runner = EpisodeRunner(SessionRuntimeConfig(max_turns=4))

    result = runner.run(
        session_id="s1",
        task_id="unit",
        sample_id="a",
        policy=OneStepPolicy(),
        environment=FailingEnvironment(),
    )

    assert result.failure is not None
    assert result.failure.type == FailureType.TOOL_EXECUTION_ERROR
