from agentrl_infra import FailureType, SessionRuntime, SessionRuntimeConfig, SessionState


def test_session_detects_repetitive_loop() -> None:
    runtime = SessionRuntime(
        session_id="s1",
        task_id="calculator",
        config=SessionRuntimeConfig(max_repeated_actions=3),
    )
    runtime.start()

    assert runtime.record_action({"tool": "calc", "args": "1+1"}) is None
    assert runtime.record_action({"tool": "calc", "args": "1+1"}) is None
    failure = runtime.record_action({"tool": "calc", "args": "1+1"})

    assert failure is not None
    assert failure.type == FailureType.REPETITIVE_LOOP
    assert runtime.state == SessionState.FAILED
    assert runtime.event_log.failures()[0].type == FailureType.REPETITIVE_LOOP


def test_session_completes_without_failure() -> None:
    runtime = SessionRuntime(session_id="s1", task_id="calculator")
    runtime.start()
    runtime.record_action({"tool": "calc", "args": "1+1"})
    runtime.complete(score=1.0)

    assert runtime.state == SessionState.COMPLETED
    assert runtime.event_log.failures() == []

