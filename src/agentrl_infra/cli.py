from __future__ import annotations

from pathlib import Path

import typer

from .events import EventLog
from .replay import ReplayMode, ReplayReport

app = typer.Typer(help="AgentRL Infra research utilities.")


@app.command()
def inspect_trace(path: Path, replay_mode: ReplayMode = ReplayMode.EXACT) -> None:
    """Inspect a JSONL event trace and print a replayability report."""
    log = EventLog.load_jsonl(path)
    report = ReplayReport.from_log(log, replay_mode)
    typer.echo(report.model_dump_json(indent=2))

