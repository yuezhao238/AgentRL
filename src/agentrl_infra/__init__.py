"""Failure-aware and replayable rollout infrastructure primitives."""

from .events import Event, EventLog, EventType
from .failures import FailureRecord, FailureSemantics, FailureType
from .replay import ReplayMode, ReplayReport
from .scheduler import RolloutRequest, SchedulerDecision, TaskStats
from .session import SessionRuntime, SessionRuntimeConfig, SessionState

__all__ = [
    "Event",
    "EventLog",
    "EventType",
    "FailureRecord",
    "FailureSemantics",
    "FailureType",
    "ReplayMode",
    "ReplayReport",
    "RolloutRequest",
    "SchedulerDecision",
    "SessionRuntime",
    "SessionRuntimeConfig",
    "SessionState",
    "TaskStats",
]

