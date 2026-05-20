from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from ..events import EventLog, EventType
from .model_provenance import DEFAULT_PROMPTS, encode_prompt, tokenizer_fingerprint

DEFAULT_GENERATION_MODEL_IDS = ["Qwen/Qwen3-4B"]


class GenerationPromptRecord(BaseModel):
    prompt_index: int
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float
    tokens_per_second: float
    mean_logprob: float
    min_logprob: float
    output_text: str
    generated_token_ids: list[int]
    generated_logprobs: list[float]


class ModelGenerationSummary(BaseModel):
    model_id: str
    prompt_count: int
    prompt_tokens: int
    generated_tokens: int
    load_seconds: float
    total_latency_seconds: float
    tokens_per_second: float
    mean_logprob: float
    min_logprob: float
    tokenizer_hash: str
    trace_path: str
    records: list[GenerationPromptRecord] = Field(default_factory=list)


@dataclass(frozen=True)
class LoadedCausalLM:
    tokenizer: Any
    model: Any
    load_seconds: float


def run_model_generation_smoke(
    *,
    output_dir: Path,
    run_id: str,
    model_ids: list[str] | None = None,
    prompts: list[str] | None = None,
    max_new_tokens: int = 32,
) -> list[ModelGenerationSummary]:
    model_ids = DEFAULT_GENERATION_MODEL_IDS if model_ids is None else model_ids
    prompts = DEFAULT_PROMPTS if prompts is None else prompts
    run_dir = output_dir / run_id
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[ModelGenerationSummary] = []
    for model_id in model_ids:
        summaries.append(
            generate_with_model(
                model_id=model_id,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                trace_dir=trace_dir,
            )
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_generation_summary.json").write_text(
        json.dumps([summary.model_dump(mode="json") for summary in summaries], indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model_ids": model_ids,
                "prompt_count": len(prompts),
                "max_new_tokens": max_new_tokens,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summaries


def generate_with_model(
    *,
    model_id: str,
    prompts: list[str],
    max_new_tokens: int,
    trace_dir: Path,
) -> ModelGenerationSummary:
    loaded = _load_causal_lm(model_id)
    tokenizer = loaded.tokenizer
    model = loaded.model
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    trace_path = trace_dir / f"{_safe_model_id(model_id)}.jsonl"
    log = EventLog.new(
        session_id=f"model-generation-{_safe_model_id(model_id)}",
        task_id="model_generation_smoke",
        sample_id=model_id,
    )
    log.append(
        EventType.SESSION_STARTED,
        model_version=model_id,
        tokenizer_hash=tokenizer_hash,
        payload={"model_id": model_id, "max_new_tokens": max_new_tokens},
    )
    records: list[GenerationPromptRecord] = []
    for index, prompt in enumerate(prompts):
        records.append(
            _generate_one(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                tokenizer_hash=tokenizer_hash,
                log=log,
                prompt=prompt,
                prompt_index=index,
                max_new_tokens=max_new_tokens,
            )
        )
    log.append(EventType.SESSION_COMPLETED, payload={"prompt_count": len(records)})
    log.save_jsonl(trace_path)
    generated_tokens = sum(record.generated_tokens for record in records)
    total_latency = sum(record.latency_seconds for record in records)
    logprobs = [
        logprob for record in records for logprob in record.generated_logprobs
    ]
    return ModelGenerationSummary(
        model_id=model_id,
        prompt_count=len(records),
        prompt_tokens=sum(record.prompt_tokens for record in records),
        generated_tokens=generated_tokens,
        load_seconds=loaded.load_seconds,
        total_latency_seconds=total_latency,
        tokens_per_second=generated_tokens / total_latency if total_latency else 0.0,
        mean_logprob=mean(logprobs) if logprobs else 0.0,
        min_logprob=min(logprobs) if logprobs else 0.0,
        tokenizer_hash=tokenizer_hash,
        trace_path=str(trace_path),
        records=records,
    )


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    model_id: str,
    tokenizer_hash: str,
    log: EventLog,
    prompt: str,
    prompt_index: int,
    max_new_tokens: int,
) -> GenerationPromptRecord:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("torch is required for local generation smoke tests") from exc

    input_ids_list = encode_prompt(tokenizer, prompt)
    input_ids = torch.tensor([input_ids_list], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    log.append(
        EventType.MODEL_REQUESTED,
        model_version=model_id,
        tokenizer_hash=tokenizer_hash,
        token_ids=input_ids_list,
        loss_mask=[0] * len(input_ids_list),
        payload={"prompt_index": prompt_index, "prompt": prompt},
    )
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
    generated_ids = sequences[0, input_ids.shape[-1] :].tolist()
    transition_scores = model.compute_transition_scores(
        sequences,
        output.scores,
        normalize_logits=True,
    )
    generated_logprobs = [
        float(score) for score in transition_scores[0, -len(generated_ids) :].tolist()
    ]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    log.append(
        EventType.MODEL_RESPONDED,
        model_version=model_id,
        tokenizer_hash=tokenizer_hash,
        token_ids=generated_ids,
        logprobs=generated_logprobs,
        loss_mask=[1] * len(generated_ids),
        payload={
            "prompt_index": prompt_index,
            "output_text": output_text,
            "generated_tokens": len(generated_ids),
            "latency_seconds": latency,
            "tokens_per_second": len(generated_ids) / latency if latency else 0.0,
        },
    )
    return GenerationPromptRecord(
        prompt_index=prompt_index,
        prompt=prompt,
        prompt_tokens=len(input_ids_list),
        generated_tokens=len(generated_ids),
        latency_seconds=latency,
        tokens_per_second=len(generated_ids) / latency if latency else 0.0,
        mean_logprob=mean(generated_logprobs) if generated_logprobs else 0.0,
        min_logprob=min(generated_logprobs) if generated_logprobs else 0.0,
        output_text=output_text,
        generated_token_ids=generated_ids,
        generated_logprobs=generated_logprobs,
    )


def _load_causal_lm(model_id: str) -> LoadedCausalLM:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("transformers is required for local generation smoke tests") from exc

    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return LoadedCausalLM(tokenizer=tokenizer, model=model, load_seconds=perf_counter() - started)


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")
