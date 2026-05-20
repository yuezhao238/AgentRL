from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .events import Event, EventLog, EventType


class ReplayMode(StrEnum):
    EXACT = "exact"
    MODEL_SUBSTITUTED = "model_substituted"
    ENVIRONMENT_SUBSTITUTED = "environment_substituted"


class ReplayReport(BaseModel):
    session_id: str
    mode: ReplayMode
    replayable: bool
    event_count: int
    failure_count: int
    determinism_level: str
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_log(cls, log: EventLog, mode: ReplayMode) -> ReplayReport:
        notes: list[str] = []
        failures = log.failures()
        replayable = True
        determinism_level = "exact"

        if mode != ReplayMode.EXACT:
            determinism_level = "state-equivalent"
        if any(f.resolved_semantics().reset_environment for f in failures):
            replayable = False
            determinism_level = "partial"
            notes.append("environment-reset failure present; exact replay may be impossible")
        if not log.events:
            replayable = False
            determinism_level = "none"
            notes.append("empty event log")

        return cls(
            session_id=log.session_id,
            mode=mode,
            replayable=replayable,
            event_count=len(log.events),
            failure_count=len(failures),
            determinism_level=determinism_level,
            notes=notes,
        )


class ReplayMismatch(BaseModel):
    index: int
    field: str
    expected: Any
    actual: Any


class ReplayExecutionReport(BaseModel):
    session_id: str
    mode: ReplayMode
    replayable: bool
    matched: bool
    expected_event_count: int
    actual_event_count: int
    mismatch_count: int
    mismatches: list[ReplayMismatch] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


ReplayRerun = Callable[[EventLog], EventLog]


class ReplayEngine:
    def __init__(self, rerun: ReplayRerun) -> None:
        self.rerun = rerun

    def execute(self, log: EventLog, mode: ReplayMode = ReplayMode.EXACT) -> ReplayExecutionReport:
        static_report = ReplayReport.from_log(log, mode)
        notes = list(static_report.notes)
        if not static_report.replayable:
            notes.append("static replayability check failed before execution")
            return ReplayExecutionReport(
                session_id=log.session_id,
                mode=mode,
                replayable=False,
                matched=False,
                expected_event_count=len(log.events),
                actual_event_count=0,
                mismatch_count=0,
                notes=notes,
            )

        actual = self.rerun(log)
        mismatches = compare_event_logs(log, actual)
        return ReplayExecutionReport(
            session_id=log.session_id,
            mode=mode,
            replayable=True,
            matched=not mismatches,
            expected_event_count=len(log.events),
            actual_event_count=len(actual.events),
            mismatch_count=len(mismatches),
            mismatches=mismatches,
            notes=notes,
        )


def compare_event_logs(expected: EventLog, actual: EventLog) -> list[ReplayMismatch]:
    mismatches: list[ReplayMismatch] = []
    limit = min(len(expected.events), len(actual.events))
    for index in range(limit):
        expected_projection = _event_projection(expected.events[index])
        actual_projection = _event_projection(actual.events[index])
        keys = sorted(set(expected_projection) | set(actual_projection))
        for key in keys:
            if expected_projection.get(key) != actual_projection.get(key):
                mismatches.append(
                    ReplayMismatch(
                        index=index,
                        field=key,
                        expected=expected_projection.get(key),
                        actual=actual_projection.get(key),
                    )
                )
    if len(expected.events) != len(actual.events):
        mismatches.append(
            ReplayMismatch(
                index=limit,
                field="event_count",
                expected=len(expected.events),
                actual=len(actual.events),
            )
        )
    return mismatches


def validate_event_log(log: EventLog) -> list[str]:
    errors: list[str] = []
    if not log.events:
        return ["event log is empty"]
    event_ids: set[str] = set()
    for index, event in enumerate(log.events):
        if event.id in event_ids:
            errors.append(f"duplicate event id at index {index}: {event.id}")
        event_ids.add(event.id)
        if event.session_id != log.session_id:
            errors.append(f"session id mismatch at index {index}: {event.session_id}")
        if event.task_id != log.task_id:
            errors.append(f"task id mismatch at index {index}: {event.task_id}")
        if index == 0 and event.parent_id is not None:
            errors.append("first event must not have parent_id")
        if index > 0 and event.parent_id not in event_ids:
            errors.append(f"parent id missing before index {index}: {event.parent_id}")
    terminal_events = {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_CANCELLED,
        EventType.FAILURE_DETECTED,
    }
    if not any(event.type in terminal_events for event in log.events):
        errors.append("event log has no terminal completion/cancellation/failure event")
    return errors


def _event_projection(event: Event) -> dict[str, Any]:
    projection: dict[str, Any] = {"type": event.type.value}
    if event.failure:
        projection["failure_type"] = event.failure.type.value
        projection["failure_source"] = event.failure.source
    if event.type in {EventType.ACTION_PROPOSED, EventType.MODEL_RESPONDED}:
        action = event.payload.get("action")
        if action is not None:
            projection["action"] = action
    if event.type == EventType.OBSERVATION_RETURNED:
        projection["observation"] = event.payload.get("observation")
    if event.type == EventType.REWARD_EMITTED:
        projection["reward"] = event.payload.get("reward")
    return projection
