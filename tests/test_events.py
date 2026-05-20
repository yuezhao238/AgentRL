from agentrl_infra import EventLog, EventType, FailureRecord, FailureType


def test_event_log_jsonl_roundtrip(tmp_path) -> None:
    log = EventLog.new("s1", "webshop", "42")
    log.append(EventType.SESSION_STARTED, model_version="policy-a")
    log.append_failure(FailureRecord(type=FailureType.AGENT_INVALID_ACTION))
    path = tmp_path / "trace.jsonl"

    log.save_jsonl(path)
    loaded = EventLog.load_jsonl(path)

    assert loaded.session_id == "s1"
    assert loaded.task_id == "webshop"
    assert len(loaded.events) == 2
    assert loaded.failures()[0].type == FailureType.AGENT_INVALID_ACTION
    assert loaded.events[1].parent_id == loaded.events[0].id


def test_empty_event_log_load_rejected(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")

    try:
        EventLog.load_jsonl(path)
    except ValueError as exc:
        assert "event log is empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")

