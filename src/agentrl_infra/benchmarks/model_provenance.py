from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from ..events import EventLog, EventType

DEFAULT_MODEL_IDS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]

DEFAULT_PROMPTS = [
    "Click the blue button and then report success.",
    "Search for a waterproof hiking backpack under 80 dollars.",
    "If the page stops changing after an action, explain the likely failure type.",
]


class PromptTokenRecord(BaseModel):
    prompt_index: int
    prompt: str
    token_count: int
    token_ids: list[int]
    roundtrip_text_hash: str
    drift_validated: bool


class ModelProvenanceSummary(BaseModel):
    model_id: str
    tokenizer_class: str
    vocab_size: int
    tokenizer_hash: str
    chat_template_hash: str | None = None
    prompt_count: int
    min_prompt_tokens: int
    mean_prompt_tokens: float
    max_prompt_tokens: int
    drift_validated_count: int
    load_seconds: float
    trace_path: str
    records: list[PromptTokenRecord] = Field(default_factory=list)


def run_model_provenance_audit(
    *,
    output_dir: Path,
    run_id: str,
    model_ids: list[str] | None = None,
    prompts: list[str] | None = None,
) -> list[ModelProvenanceSummary]:
    model_ids = DEFAULT_MODEL_IDS if model_ids is None else model_ids
    prompts = DEFAULT_PROMPTS if prompts is None else prompts
    run_dir = output_dir / run_id
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[ModelProvenanceSummary] = []

    for model_id in model_ids:
        summary = audit_model_tokenizer(model_id=model_id, prompts=prompts, trace_dir=trace_dir)
        summaries.append(summary)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_provenance_summary.json").write_text(
        json.dumps([summary.model_dump(mode="json") for summary in summaries], indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps({"model_ids": model_ids, "prompt_count": len(prompts)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summaries


def audit_model_tokenizer(
    *,
    model_id: str,
    prompts: list[str],
    trace_dir: Path,
) -> ModelProvenanceSummary:
    tokenizer, load_seconds = _load_tokenizer(model_id)
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_hash = sha256(chat_template.encode()).hexdigest() if chat_template else None
    trace_path = trace_dir / f"{_safe_model_id(model_id)}.jsonl"
    log = EventLog.new(
        session_id=f"model-provenance-{_safe_model_id(model_id)}",
        task_id="model_provenance_audit",
        sample_id=model_id,
    )
    log.append(
        EventType.SESSION_STARTED,
        model_version=model_id,
        tokenizer_hash=tokenizer_hash,
        payload={
            "model_id": model_id,
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "chat_template_hash": chat_template_hash,
        },
    )

    records: list[PromptTokenRecord] = []
    for index, prompt in enumerate(prompts):
        token_ids = encode_prompt(tokenizer, prompt)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        reencoded = tokenizer.encode(decoded, add_special_tokens=False)
        record = PromptTokenRecord(
            prompt_index=index,
            prompt=prompt,
            token_count=len(token_ids),
            token_ids=token_ids,
            roundtrip_text_hash=sha256(decoded.encode()).hexdigest(),
            drift_validated=token_ids == reencoded,
        )
        records.append(record)
        loss_mask = [1] * len(token_ids)
        log.append(
            EventType.MODEL_RESPONDED,
            model_version=model_id,
            tokenizer_hash=tokenizer_hash,
            token_ids=token_ids,
            loss_mask=loss_mask,
            payload={
                "prompt_index": index,
                "prompt": prompt,
                "token_count": len(token_ids),
                "drift_validated": record.drift_validated,
                "roundtrip_text_hash": record.roundtrip_text_hash,
            },
        )

    log.append(EventType.SESSION_COMPLETED, payload={"prompt_count": len(records)})
    log.save_jsonl(trace_path)
    token_counts = [record.token_count for record in records]
    return ModelProvenanceSummary(
        model_id=model_id,
        tokenizer_class=tokenizer.__class__.__name__,
        vocab_size=len(tokenizer),
        tokenizer_hash=tokenizer_hash,
        chat_template_hash=chat_template_hash,
        prompt_count=len(records),
        min_prompt_tokens=min(token_counts) if token_counts else 0,
        mean_prompt_tokens=sum(token_counts) / len(token_counts) if token_counts else 0.0,
        max_prompt_tokens=max(token_counts) if token_counts else 0,
        drift_validated_count=sum(1 for record in records if record.drift_validated),
        load_seconds=load_seconds,
        trace_path=str(trace_path),
        records=records,
    )


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    chat = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        return list(tokenizer.encode(rendered, add_special_tokens=False))
    return list(tokenizer.encode(prompt, add_special_tokens=True))


def tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        payload = backend.to_str()
    else:
        payload = json.dumps(tokenizer.get_vocab(), sort_keys=True)
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    special_tokens = json.dumps(tokenizer.special_tokens_map, sort_keys=True)
    return sha256(f"{payload}\n{chat_template}\n{special_tokens}".encode()).hexdigest()


def _load_tokenizer(model_id: str) -> tuple[Any, float]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("transformers is required for model provenance audits") from exc

    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    return tokenizer, perf_counter() - started


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")
