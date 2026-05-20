from agentrl_infra import FailureRecord, FailureType


def test_policy_attributable_failures_emit_training_signal() -> None:
    record = FailureRecord(type=FailureType.REPETITIVE_LOOP)

    semantics = record.resolved_semantics()

    assert semantics.attributable_to_policy is True
    assert semantics.emit_training_signal is True
    assert semantics.salvageable is True


def test_environment_crash_requires_reset_and_no_training_signal() -> None:
    record = FailureRecord(type=FailureType.ENVIRONMENT_CRASH)

    semantics = record.resolved_semantics()

    assert semantics.reset_environment is True
    assert semantics.emit_training_signal is False
    assert semantics.salvageable is False

