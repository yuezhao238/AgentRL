from __future__ import annotations

from agentrl_infra.baselines import RuntimeBaseline, evaluate_failurebench_baseline


def test_failurebench_baselines_have_lower_macro_f1_than_typed_runtime() -> None:
    _, raw = evaluate_failurebench_baseline(
        baseline=RuntimeBaseline.RAW_TIMEOUT,
        split="dev",
        dev_seeds_per_type=1,
        test_seeds_per_type=0,
    )
    _, message = evaluate_failurebench_baseline(
        baseline=RuntimeBaseline.MESSAGE_TRACE,
        split="dev",
        dev_seeds_per_type=1,
        test_seeds_per_type=0,
    )

    assert raw.macro_f1 < 1.0
    assert message.macro_f1 < 1.0
    assert raw.mean_turns_wasted_after_oracle > 0
