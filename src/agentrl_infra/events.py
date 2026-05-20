from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .failures import FailureRecord


class EventType(StrEnum):
    SESSION_STARTED = "session_started"
    MODEL_REQUESTED = "model_requested"
    MODEL_RESPONDED = "model_responded"
    ACTION_PROPOSED = "action_proposed"
    TOOL_CALL_VALIDATED = "tool_call_validated"
    TOOL_CALL_EXECUTED = "tool_call_executed"
    OBSERVATION_RETURNED = "observation_returned"
    REWARD_EMITTED = "reward_emitted"
    ENVIRONMENT_SNAPSHOT_CREATED = "environment_snapshot_created"
    FAILURE_DETECTED = "failure_detected"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_COMPLETED = "session_completed"


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    session_id: str
    task_id: str
    sample_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parent_id: str | None = None
    model_version: str | None = None
    tokenizer_hash: str | None = None
    environment_id: str | None = None
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    loss_mask: list[int] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    failure: FailureRecord | None = None

    def to_json_line(self) -> str:
        return self.model_dump_json() + "\n"


class EventLog(BaseModel):
    session_id: str
    task_id: str
    sample_id: str | None = None
    events: list[Event] = Field(default_factory=list)

    @classmethod
    def new(cls, session_id: str, task_id: str, sample_id: str | None = None) -> EventLog:
        return cls(session_id=session_id, task_id=task_id, sample_id=sample_id)

    @property
    def last_event_id(self) -> str | None:
        return self.events[-1].id if self.events else None

    def append(self, event_type: EventType, **kwargs: Any) -> Event:
        event = Event(
            type=event_type,
            session_id=self.session_id,
            task_id=self.task_id,
            sample_id=self.sample_id,
            parent_id=kwargs.pop("parent_id", self.last_event_id),
            **kwargs,
        )
        self.events.append(event)
        return event

    def append_failure(self, failure: FailureRecord, **kwargs: Any) -> Event:
        return self.append(EventType.FAILURE_DETECTED, failure=failure, **kwargs)

    def failures(self) -> list[FailureRecord]:
        return [event.failure for event in self.events if event.failure is not None]

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(event.to_json_line())

    @classmethod
    def load_jsonl(cls, path: str | Path) -> EventLog:
        path = Path(path)
        events = [
            Event.model_validate(json.loads(line))
            for line in path.read_text().splitlines()
            if line
        ]
        if not events:
            raise ValueError(f"event log is empty: {path}")
        first = events[0]
        return cls(
            session_id=first.session_id,
            task_id=first.task_id,
            sample_id=first.sample_id,
            events=events,
        )
