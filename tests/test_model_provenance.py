from __future__ import annotations

from agentrl_infra.benchmarks.model_provenance import tokenizer_fingerprint


class _FakeTokenizer:
    chat_template = "template"
    special_tokens_map = {"eos_token": "<eos>"}

    def get_vocab(self) -> dict[str, int]:
        return {"hello": 1, "world": 2}


def test_tokenizer_fingerprint_is_stable() -> None:
    first = tokenizer_fingerprint(_FakeTokenizer())
    second = tokenizer_fingerprint(_FakeTokenizer())

    assert first == second
    assert len(first) == 64
