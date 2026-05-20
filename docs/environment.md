# AgentRL Infra Environment

This repository is prepared with three optional dependency groups:

- `dev`: tests and linting.
- `experiments`: Hugging Face, Transformers, datasets, OpenAI-compatible clients, and utility packages.
- `sglang`: preferred local inference backend for the current machine.
- `vllm`: backup local inference backend.

`sglang` and `vllm` are declared as conflicting extras because their transitive
runtime dependencies pin different versions of some serving packages.

## Recommended setup

```bash
uv sync --extra dev --extra experiments --extra sglang
```

To switch to vLLM:

```bash
uv sync --extra dev --extra experiments --extra vllm
```

To return to SGLang:

```bash
uv sync --extra dev --extra experiments --extra sglang
```

## Model cache

The project uses the Hugging Face cache under:

```text
/root/.cache/huggingface/hub
```

The following models are expected for the paper experiments:

- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-32B`
- `meta-llama/Llama-3.1-8B-Instruct`

Download or refresh them with:

```bash
HF_ENDPOINT="https://hf-mirror.com" uv run python scripts/download_models.py
```

`meta-llama/Llama-3.1-8B-Instruct` is gated on Hugging Face. The local account
must have access before the download can succeed.

## Current machine notes

The current machine exposes 8 NVIDIA A100-SXM4-80GB GPUs. The SGLang environment
uses `torch==2.8.0+cu128`, which works with the installed driver. Avoid
unconstrained latest vLLM installs on this machine: newer releases can pull
CUDA 13 builds of PyTorch that require a newer NVIDIA driver.
