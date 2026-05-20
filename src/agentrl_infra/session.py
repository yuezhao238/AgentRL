from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any

from .events import EventLog, EventType
from .failures import FailureRecord, FailureType


class SessionState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SessionRuntimeConfig:
    max_turns: int = 32
    max_wall_time_seconds: float = 600.0
    max_repeated_actions: int = 3


@dataclass
class SessionRuntime:
    session_id: str
    task_id: str
    sample_id: str | None = None
    config: SessionRuntimeConfig = field(default_factory=SessionRuntimeConfig)
    event_log: EventLog = field(init=False)
    state: SessionState = SessionState.CREATED
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    turn_count: int = 0
    _recent_action_hashes: deque[str] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.event_log = EventLog.new(self.session_id, self.task_id, self.sample_id)

    def start(self, **payload: Any) -> None:
        self.state = SessionState.RUNNING
        self.event_log.append(EventType.SESSION_STARTED, payload=payload)

    def record_action(self, action: dict[str, Any]) -> FailureRecord | None:
        if self.state != SessionState.RUNNING:
            raise RuntimeError(f"session is not running: {self.state}")
        self.turn_count += 1
        action_hash = sha256(repr(sorted(action.items())).encode("utf-8")).hexdigest()
        self._recent_action_hashes.append(action_hash)
        while len(self._recent_action_hashes) > self.config.max_repeated_actions:
            self._recent_action_hashes.popleft()

        self.event_log.append(EventType.ACTION_PROPOSED, payload={"action": action})
        failure = self._check_watchdogs()
        if failure:
            self.fail(failure)
        return failure

    def complete(self, **payload: Any) -> None:
        self.state = SessionState.COMPLETED
        self.event_log.append(EventType.SESSION_COMPLETED, payload=payload)

    def cancel(self, message: str | None = None) -> None:
        self.state = SessionState.CANCELLED
        failure = FailureRecord(type=FailureType.SCHEDULER_CANCELLED, message=message)
        self.event_log.append_failure(failure)
        self.event_log.append(EventType.SESSION_CANCELLED)

    def fail(self, failure: FailureRecord) -> None:
        self.state = SessionState.FAILED
        self.event_log.append_failure(failure)

    def _check_watchdogs(self) -> FailureRecord | None:
        if self.turn_count > self.config.max_turns:
            return FailureRecord(
                type=FailureType.CONTEXT_LIMIT,
                message=f"turn budget exceeded: {self.turn_count}>{self.config.max_turns}",
            )

        elapsed = datetime.now(UTC) - self.started_at
        if elapsed > timedelta(seconds=self.config.max_wall_time_seconds):
            return FailureRecord(
                type=FailureType.SCHEDULER_CANCELLED,
                message=f"wall-clock budget exceeded: {elapsed.total_seconds():.3f}s",
            )

        if len(self._recent_action_hashes) == self.config.max_repeated_actions:
            if len(set(self._recent_action_hashes)) == 1:
                return FailureRecord(
                    type=FailureType.REPETITIVE_LOOP,
                    message="same action repeated beyond loop threshold",
                    details={"repeat_count": self.config.max_repeated_actions},
                )
        return None

