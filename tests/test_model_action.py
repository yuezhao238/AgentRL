from __future__ import annotations

from agentrl_infra.benchmarks.model_action import browser_action_json_schema, parse_browser_action
from agentrl_infra.failures import FailureType


def test_parse_browser_action_extracts_json_object() -> None:
    action, failure = parse_browser_action(
        'Reasoning... {"action_type":"click","selector":"#target","observed_dom_hash":"abc"}'
    )

    assert failure is None
    assert action == {
        "action_type": "click",
        "selector": "#target",
        "observed_dom_hash": "abc",
    }


def test_parse_browser_action_returns_typed_failure() -> None:
    action, failure = parse_browser_action("no json here")

    assert action is None
    assert failure is not None
    assert failure.type == FailureType.AGENT_INVALID_ACTION


def test_browser_action_json_schema_requires_runtime_fields() -> None:
    schema = browser_action_json_schema()

    assert schema["properties"]["action_type"]["enum"] == ["click", "type_text", "wait"]
    assert schema["required"] == ["action_type", "selector", "observed_dom_hash"]
    assert schema["additionalProperties"] is False
