from __future__ import annotations

import json

from agentrl_infra.benchmarks.model_action import ModelActionEpisodeRecord, ModelActionSummary
from agentrl_infra.benchmarks.throughput import (
    ThroughputPolicy,
    build_model_action_throughput_workload,
    run_throughput_benchmark,
)
from agentrl_infra.failures import FailureType


def test_throughput_failure_aware_improves_useful_yield(tmp_path) -> None:
    fifo = run_throughput_benchmark(
        policy=ThroughputPolicy.FIFO,
        worker_count=8,
        output_dir=tmp_path,
        run_id="fifo",
    )
    failure_aware = run_throughput_benchmark(
        policy=ThroughputPolicy.FAILURE_AWARE,
        worker_count=8,
        output_dir=tmp_path,
        run_id="failure_aware",
    )

    assert failure_aware.summary.useful_trajectories_per_hour > (
        fifo.summary.useful_trajectories_per_hour
    )
    assert failure_aware.summary.zombie_session_rate < fifo.summary.zombie_session_rate
    assert (tmp_path / "failure_aware" / "throughput_summary.json").exists()


def test_throughput_retry_only_adds_attempts() -> None:
    fifo = run_throughput_benchmark(policy=ThroughputPolicy.FIFO, worker_count=8)
    retry = run_throughput_benchmark(policy=ThroughputPolicy.RETRY_ONLY, worker_count=8)

    assert retry.summary.attempt_count > fifo.summary.attempt_count
    assert retry.summary.failed_cost_units > 0


def test_model_action_summary_drives_empirical_throughput_workload(tmp_path) -> None:
    summary_path = tmp_path / "model_action_summary.json"
    records = [
        ModelActionEpisodeRecord(
            model_id="Qwen/Qwen3-4B",
            prompt_protocol="default",
            max_new_tokens=96,
            task_name="click-button",
            seed=1000,
            success=False,
            parsed_action=False,
            failure_type=FailureType.AGENT_INVALID_ACTION,
            prompt_tokens=128,
            generated_tokens=96,
            latency_seconds=3.0,
            tokens_per_second=32.0,
            mean_logprob=-0.1,
            output_text="<think>",
        ),
        ModelActionEpisodeRecord(
            model_id="Qwen/Qwen3-4B",
            prompt_protocol="default",
            max_new_tokens=96,
            task_name="enter-text",
            seed=1001,
            success=True,
            parsed_action=True,
            failure_type=None,
            prompt_tokens=128,
            generated_tokens=70,
            latency_seconds=2.0,
            tokens_per_second=35.0,
            mean_logprob=-0.01,
            output_text='{"action_type":"click"}',
            action={"action_type": "click", "selector": "#submit"},
        ),
    ]
    summary = ModelActionSummary(
        model_id="Qwen/Qwen3-4B",
        prompt_protocol="default",
        max_new_tokens=96,
        episode_count=2,
        success_count=1,
        parsed_action_count=1,
        invalid_action_count=1,
        no_progress_count=0,
        prompt_tokens=256,
        generated_tokens=166,
        total_latency_seconds=5.0,
        tokens_per_second=33.2,
        mean_logprob=-0.055,
        trace_path="trace.jsonl",
        records=records,
    )
    summary_path.write_text(
        json.dumps([summary.model_dump(mode="json")], indent=2) + "\n",
        encoding="utf-8",
    )

    workload = build_model_action_throughput_workload(summary_path, episodes_per_cell=10)
    fifo = run_throughput_benchmark(
        policy=ThroughputPolicy.FIFO,
        worker_count=2,
        workload=workload,
    )
    failure_aware = run_throughput_benchmark(
        policy=ThroughputPolicy.FAILURE_AWARE,
        worker_count=2,
        workload=workload,
    )

    assert workload[0].failure_probability == 0.5
    assert workload[0].replayable_failure_probability == 1.0
    assert fifo.summary.episode_count == 10
    assert failure_aware.summary.useful_count >= fifo.summary.useful_count
