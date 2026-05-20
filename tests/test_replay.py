from agentrl_infra import EventLog, EventType, FailureRecord, FailureType, ReplayMode, ReplayReport


def test_replay_report_marks_environment_reset_failure_as_partial() -> None:
    log = EventLog.new("s1", "browser")
    log.append(EventType.SESSION_STARTED)
    log.append_failure(FailureRecord(type=FailureType.ENVIRONMENT_CRASH))

    report = ReplayReport.from_log(log, ReplayMode.EXACT)

    assert report.replayable is False
    assert report.determinism_level == "partial"
    assert report.failure_count == 1


def test_model_substituted_replay_defaults_to_state_equivalent() -> None:
    log = EventLog.new("s1", "browser")
    log.append(EventType.SESSION_STARTED)

    report = ReplayReport.from_log(log, ReplayMode.MODEL_SUBSTITUTED)

    assert report.replayable is True
    assert report.determinism_level == "state-equivalent"

