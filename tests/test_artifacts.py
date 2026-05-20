from __future__ import annotations

from agentrl_infra.artifacts import replay_run_artifacts, validate_run_artifacts
from agentrl_infra.benchmarks.failurebench import run_failurebench
from agentrl_infra.metrics import save_metrics


def test_validate_and_replay_run_artifacts(tmp_path) -> None:
    metrics = run_failurebench(
        output_dir=tmp_path,
        run_id="r1",
        split="dev",
        dev_seeds_per_type=1,
        test_seeds_per_type=0,
    )
    save_metrics(tmp_path / "r1", metrics)
    (tmp_path / "r1" / "config.json").write_text("{}\n", encoding="utf-8")

    validation = validate_run_artifacts(tmp_path / "r1")
    replay = replay_run_artifacts(tmp_path / "r1")

    assert validation.valid is True
    assert validation.trace_count == 8
    assert replay.attempted == 8
    assert replay.matched == 6
    assert replay.mismatched == 0
