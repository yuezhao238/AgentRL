from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..events import EventLog, EventType
from ..failures import FailureRecord, FailureType
from ..integrations.miniwob import (
    BrowserAction,
    MiniWoBContractEnvironment,
    build_miniwob_task_specs,
)
from .model_generation import DEFAULT_GENERATION_MODEL_IDS, LoadedCausalLM, _load_causal_lm
from .model_provenance import encode_prompt, tokenizer_fingerprint

DEFAULT_ACTION_TASKS = ["click-button", "enter-text", "search-engine"]
DEFAULT_ACTION_SEEDS = [1000]
ActionPromptProtocol = Literal["default", "no_thinking"]
DEFAULT_ACTION_PROMPT_PROTOCOLS: list[ActionPromptProtocol] = ["no_thinking"]


class ModelActionEpisodeRecord(BaseModel):
    model_id: str
    prompt_protocol: str
    max_new_tokens: int
    task_name: str
    seed: int
    success: bool
    parsed_action: bool
    failure_type: FailureType | None
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float
    tokens_per_second: float
    mean_logprob: float
    output_text: str
    action: dict[str, Any] | None = None


class ModelActionSummary(BaseModel):
    model_id: str
    prompt_protocol: str
    max_new_tokens: int
    episode_count: int
    success_count: int
    parsed_action_count: int
    invalid_action_count: int
    no_progress_count: int
    prompt_tokens: int
    generated_tokens: int
    total_latency_seconds: float
    tokens_per_second: float
    mean_logprob: float
    trace_path: str
    records: list[ModelActionEpisodeRecord] = Field(default_factory=list)


def run_model_action_benchmark(
    *,
    output_dir: Path,
    run_id: str,
    model_ids: list[str] | None = None,
    task_names: list[str] | None = None,
    seeds: list[int] | None = None,
    prompt_protocols: list[ActionPromptProtocol] | None = None,
    max_new_tokens: int = 96,
) -> list[ModelActionSummary]:
    model_ids = DEFAULT_GENERATION_MODEL_IDS if model_ids is None else model_ids
    task_names = DEFAULT_ACTION_TASKS if task_names is None else task_names
    seeds = DEFAULT_ACTION_SEEDS if seeds is None else seeds
    prompt_protocols = (
        DEFAULT_ACTION_PROMPT_PROTOCOLS if prompt_protocols is None else prompt_protocols
    )
    run_dir = output_dir / run_id
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[ModelActionSummary] = []
    for model_id in model_ids:
        loaded = _load_causal_lm(model_id)
        for prompt_protocol in prompt_protocols:
            summaries.append(
                run_model_action_with_loaded_model(
                    loaded=loaded,
                    model_id=model_id,
                    prompt_protocol=prompt_protocol,
                    task_names=task_names,
                    seeds=seeds,
                    max_new_tokens=max_new_tokens,
                    trace_dir=trace_dir,
                )
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_action_summary.json").write_text(
        json.dumps([summary.model_dump(mode="json") for summary in summaries], indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model_ids": model_ids,
                "task_names": task_names,
                "seeds": seeds,
                "prompt_protocols": prompt_protocols,
                "max_new_tokens": max_new_tokens,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summaries


def run_model_action_for_model(
    *,
    model_id: str,
    prompt_protocol: ActionPromptProtocol,
    task_names: list[str],
    seeds: list[int],
    max_new_tokens: int,
    trace_dir: Path,
) -> ModelActionSummary:
    loaded = _load_causal_lm(model_id)
    return run_model_action_with_loaded_model(
        loaded=loaded,
        model_id=model_id,
        prompt_protocol=prompt_protocol,
        task_names=task_names,
        seeds=seeds,
        max_new_tokens=max_new_tokens,
        trace_dir=trace_dir,
    )


def run_model_action_with_loaded_model(
    *,
    loaded: LoadedCausalLM,
    model_id: str,
    prompt_protocol: ActionPromptProtocol,
    task_names: list[str],
    seeds: list[int],
    max_new_tokens: int,
    trace_dir: Path,
) -> ModelActionSummary:
    tokenizer_hash = tokenizer_fingerprint(loaded.tokenizer)
    trace_path = trace_dir / f"{_safe_model_id(model_id)}__{prompt_protocol}.jsonl"
    log = EventLog.new(
        session_id=f"model-action-{_safe_model_id(model_id)}",
        task_id="model_action_miniwob_contract",
        sample_id=model_id,
    )
    log.append(
        EventType.SESSION_STARTED,
        model_version=model_id,
        tokenizer_hash=tokenizer_hash,
        payload={
            "model_id": model_id,
            "prompt_protocol": prompt_protocol,
            "max_new_tokens": max_new_tokens,
        },
    )
    specs = build_miniwob_task_specs()
    records: list[ModelActionEpisodeRecord] = []
    for task_name in task_names:
        for seed in seeds:
            env = MiniWoBContractEnvironment(specs[task_name], seed=seed)
            observation = env.reset()
            record = _run_action_episode(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                model_id=model_id,
                prompt_protocol=prompt_protocol,
                tokenizer_hash=tokenizer_hash,
                log=log,
                task_name=task_name,
                seed=seed,
                observation=observation,
                env=env,
                max_new_tokens=max_new_tokens,
            )
            records.append(record)
    log.append(EventType.SESSION_COMPLETED, payload={"episode_count": len(records)})
    log.save_jsonl(trace_path)
    logprobs = [
        record.mean_logprob for record in records if record.generated_tokens > 0
    ]
    generated_tokens = sum(record.generated_tokens for record in records)
    total_latency = sum(record.latency_seconds for record in records)
    return ModelActionSummary(
        model_id=model_id,
        prompt_protocol=prompt_protocol,
        max_new_tokens=max_new_tokens,
        episode_count=len(records),
        success_count=sum(1 for record in records if record.success),
        parsed_action_count=sum(1 for record in records if record.parsed_action),
        invalid_action_count=sum(
            1 for record in records if record.failure_type == FailureType.AGENT_INVALID_ACTION
        ),
        no_progress_count=sum(
            1 for record in records if record.failure_type == FailureType.NO_PROGRESS
        ),
        prompt_tokens=sum(record.prompt_tokens for record in records),
        generated_tokens=generated_tokens,
        total_latency_seconds=total_latency,
        tokens_per_second=generated_tokens / total_latency if total_latency else 0.0,
        mean_logprob=mean(logprobs) if logprobs else 0.0,
        trace_path=str(trace_path),
        records=records,
    )


def _run_action_episode(
    *,
    model: Any,
    tokenizer: Any,
    model_id: str,
    prompt_protocol: ActionPromptProtocol,
    tokenizer_hash: str,
    log: EventLog,
    task_name: str,
    seed: int,
    observation: dict[str, Any],
    env: MiniWoBContractEnvironment,
    max_new_tokens: int,
) -> ModelActionEpisodeRecord:
    generated = _generate_action_text(
        model=model,
        tokenizer=tokenizer,
        prompt_protocol=prompt_protocol,
        prompt=_format_action_prompt(observation),
        max_new_tokens=max_new_tokens,
    )
    log.append(
        EventType.MODEL_REQUESTED,
        model_version=model_id,
        tokenizer_hash=tokenizer_hash,
        token_ids=generated["prompt_token_ids"],
        loss_mask=[0] * len(generated["prompt_token_ids"]),
        payload={"task_name": task_name, "seed": seed, "prompt_protocol": prompt_protocol},
    )
    log.append(
        EventType.MODEL_RESPONDED,
        model_version=model_id,
        tokenizer_hash=tokenizer_hash,
        token_ids=generated["generated_token_ids"],
        logprobs=generated["generated_logprobs"],
        loss_mask=[1] * len(generated["generated_token_ids"]),
        payload={
            "task_name": task_name,
            "seed": seed,
            "output_text": generated["output_text"],
            "latency_seconds": generated["latency_seconds"],
        },
    )
    action, parse_failure = parse_browser_action(generated["output_text"])
    if parse_failure:
        log.append_failure(parse_failure, payload={"task_name": task_name, "seed": seed})
        return _record(
            model_id=model_id,
            prompt_protocol=prompt_protocol,
            max_new_tokens=max_new_tokens,
            task_name=task_name,
            seed=seed,
            generated=generated,
            success=False,
            parsed_action=False,
            failure_type=parse_failure.type,
            action=None,
        )
    assert action is not None
    log.append(
        EventType.ACTION_PROPOSED,
        payload={"task_name": task_name, "seed": seed, "action": action},
    )
    step = env.step(action)
    log.append(
        EventType.TOOL_CALL_EXECUTED,
        payload={"task_name": task_name, "seed": seed, "info": step.info},
    )
    log.append(EventType.REWARD_EMITTED, payload={"reward": step.reward, "task_name": task_name})
    if step.failure:
        log.append_failure(step.failure, payload={"task_name": task_name, "seed": seed})
    elif step.done:
        log.append(EventType.SESSION_COMPLETED, payload={"task_name": task_name, "seed": seed})
    return _record(
        model_id=model_id,
        prompt_protocol=prompt_protocol,
        max_new_tokens=max_new_tokens,
        task_name=task_name,
        seed=seed,
        generated=generated,
        success=step.failure is None and step.done and step.reward > 0,
        parsed_action=True,
        failure_type=step.failure.type if step.failure else None,
        action=action,
    )


def parse_browser_action(output_text: str) -> tuple[dict[str, Any] | None, FailureRecord | None]:
    candidate = _extract_json_object(output_text)
    if candidate is None:
        return None, FailureRecord(
            type=FailureType.AGENT_INVALID_ACTION,
            message="model output did not contain a JSON browser action",
            source="model_action_parser",
            details={"output_text": output_text},
        )
    try:
        action = BrowserAction.model_validate(json.loads(candidate))
    except Exception as exc:
        return None, FailureRecord(
            type=FailureType.AGENT_INVALID_ACTION,
            message="model JSON did not match browser action schema",
            source="model_action_parser",
            details={"error": str(exc), "output_text": output_text},
        )
    return action.model_dump(mode="json", exclude_none=True), None


def _generate_action_text(
    *,
    model: Any,
    tokenizer: Any,
    prompt_protocol: ActionPromptProtocol,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for local model action benchmarks") from exc

    prompt_token_ids = _encode_action_prompt(tokenizer, prompt, prompt_protocol)
    input_ids = torch.tensor([prompt_token_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    started = perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = perf_counter() - started
    sequences = output.sequences
    generated_token_ids = sequences[0, input_ids.shape[-1] :].tolist()
    transition_scores = model.compute_transition_scores(
        sequences,
        output.scores,
        normalize_logits=True,
    )
    generated_logprobs = [
        float(score) for score in transition_scores[0, -len(generated_token_ids) :].tolist()
    ]
    return {
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_logprobs": generated_logprobs,
        "output_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "latency_seconds": latency,
    }


def _format_action_prompt(observation: dict[str, Any]) -> str:
    elements = [
        {
            "selector": item["selector"],
            "tag": item["tag"],
            "text": item.get("text", ""),
            "value": item.get("value", ""),
        }
        for item in observation["elements"]
    ]
    return (
        "You are controlling a browser. Return exactly one JSON object and no extra text. "
        "The JSON schema is: {\"action_type\":\"click|type_text|wait\","
        "\"selector\":\"css selector\",\"text\":\"optional text\","
        "\"observed_dom_hash\":\"current dom hash\"}.\n"
        f"Task name: {observation['task_name']}\n"
        f"Current DOM hash: {observation['dom_hash']}\n"
        f"Actionable elements: {json.dumps(elements, separators=(',', ':'))}\n"
        f"Known target hint for this contract benchmark: "
        f"{json.dumps(observation['action_hint'], separators=(',', ':'))}\n"
        "Return only the JSON action."
    )


def _encode_action_prompt(
    tokenizer: Any,
    prompt: str,
    prompt_protocol: ActionPromptProtocol,
) -> list[int]:
    chat = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        if prompt_protocol == "no_thinking":
            try:
                rendered = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return list(tokenizer.encode(rendered, add_special_tokens=False))
        if prompt_protocol == "default":
            rendered = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            return list(tokenizer.encode(rendered, add_special_tokens=False))
        raise ValueError(f"unsupported action prompt protocol: {prompt_protocol}")
    return encode_prompt(tokenizer, prompt)


def _record(
    *,
    model_id: str,
    prompt_protocol: str,
    max_new_tokens: int,
    task_name: str,
    seed: int,
    generated: dict[str, Any],
    success: bool,
    parsed_action: bool,
    failure_type: FailureType | None,
    action: dict[str, Any] | None,
) -> ModelActionEpisodeRecord:
    logprobs = generated["generated_logprobs"]
    generated_tokens = len(generated["generated_token_ids"])
    latency = generated["latency_seconds"]
    return ModelActionEpisodeRecord(
        model_id=model_id,
        prompt_protocol=prompt_protocol,
        max_new_tokens=max_new_tokens,
        task_name=task_name,
        seed=seed,
        success=success,
        parsed_action=parsed_action,
        failure_type=failure_type,
        prompt_tokens=len(generated["prompt_token_ids"]),
        generated_tokens=generated_tokens,
        latency_seconds=latency,
        tokens_per_second=generated_tokens / latency if latency else 0.0,
        mean_logprob=mean(logprobs) if logprobs else 0.0,
        output_text=generated["output_text"],
        action=action,
    )


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")
