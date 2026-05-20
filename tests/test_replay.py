from agentrl_infra import (
    EventLog,
    EventType,
    FailureRecord,
    FailureType,
    ReplayEngine,
    ReplayMode,
    ReplayReport,
)
from agentrl_infra.benchmarks.failurebench import (
    build_failurebench_cases,
    rerun_failurebench_log,
    run_failurebench_case,
)
from agentrl_infra.replay import validate_event_log


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


def test_validate_event_log_accepts_normal_log() -> None:
    log = EventLog.new("s1", "browser")
    log.append(EventType.SESSION_STARTED)
    log.append_failure(FailureRecord(type=FailureType.AGENT_INVALID_ACTION))

    assert validate_event_log(log) == []


def test_failurebench_replay_execution_matches_trace() -> None:
    case = build_failurebench_cases(split="dev", dev_seeds_per_type=1, test_seeds_per_type=0)[0]
    result = run_failurebench_case(case)

    report = ReplayEngine(rerun_failurebench_log).execute(result.event_log)

    assert report.replayable is True
    assert report.matched is True
    assert report.mismatch_count == 0
