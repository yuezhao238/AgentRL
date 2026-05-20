"""Failure-aware and replayable rollout infrastructure primitives."""

from .events import Event, EventLog, EventType
from .experiments import ExperimentSuiteConfig, ExperimentSuiteReport
from .failures import FailureRecord, FailureSemantics, FailureType
from .metrics import EpisodeMetrics, RunSummary
from .orchestrator import BatchOrchestrator, SchedulerRunSummary
from .replay import ReplayEngine, ReplayExecutionReport, ReplayMode, ReplayReport
from .resources import (
    EnvironmentLease,
    HealthCheckResult,
    InMemoryEnvironmentPool,
    ResourceState,
    ToolResult,
)
from .runner import EpisodeResult, EpisodeRunner, StepResult
from .scheduler import RolloutRequest, SchedulerDecision, TaskStats
from .session import SessionRuntime, SessionRuntimeConfig, SessionState

__all__ = [
    "Event",
    "EventLog",
    "EventType",
    "ExperimentSuiteConfig",
    "ExperimentSuiteReport",
    "EpisodeMetrics",
    "EpisodeResult",
    "EpisodeRunner",
    "EnvironmentLease",
    "FailureRecord",
    "FailureSemantics",
    "FailureType",
    "BatchOrchestrator",
    "HealthCheckResult",
    "InMemoryEnvironmentPool",
    "ReplayMode",
    "ReplayEngine",
    "ReplayExecutionReport",
    "ReplayReport",
    "RolloutRequest",
    "ResourceState",
    "SchedulerRunSummary",
    "SchedulerDecision",
    "SessionRuntime",
    "SessionRuntimeConfig",
    "SessionState",
    "StepResult",
    "TaskStats",
    "ToolResult",
    "RunSummary",
]
