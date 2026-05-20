from __future__ import annotations

import os

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.1-8B-Instruct",
]

EXCLUDES = [
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "onnx/*",
    "tf_model*",
    "flax_model*",
]


def main() -> int:
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    for repo_id in MODELS:
        print(f"Downloading {repo_id} from {env['HF_ENDPOINT']}", flush=True)
        snapshot_download(
            repo_id,
            endpoint=env["HF_ENDPOINT"],
            max_workers=16,
            ignore_patterns=EXCLUDES,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
