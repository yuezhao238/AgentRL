from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .events import EventLog


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

