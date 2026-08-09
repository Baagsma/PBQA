"""Unit tests for the NInfer backend. No live server required.

Covers the ninfer-specific wire behavior: schema-in-prompt injection (no
grammar support server-side), client-side schema validation with one jittered
retry, served-alias discovery and strict-model recovery, warm-on-link
prefilling with schema parity, and engine detection.

Usage:
    python -m pytest tests/test_ninfer_backend.py
"""

import json

import pytest

from PBQA import LLM
from PBQA.backends.ninfer import SCHEMA_INSTRUCTION
from tests.mock_transport import (
    HOST,
    MODEL,
    PORT,
    Reply,
    StubDB,
    Weather,
)

SERVED_ID = "qwen3.6-27b-nvfp4-ninfer"


def make_ninfer_llm(transport, schema=None, db=None, **connect_kwargs):
    server = transport.add_ninfer_server(HOST, PORT, model_id=SERVED_ID)
    llm = LLM(db=db or StubDB(schema=schema), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, engine="ninfer", **connect_kwargs)
    return llm, server


def expected_instruction(schema):
    return SCHEMA_INSTRUCTION.format(schema=json.dumps(schema))


# =============================================================================
# Connection & detection
# =============================================================================


def test_connect_discovers_served_model(transport):
    llm, server = make_ninfer_llm(transport)
    backend = llm.models[MODEL]

    assert backend.connected
    assert backend.model_id == SERVED_ID
    assert backend.is_rerank is False
    assert backend.store_cache is False  # no client-side cache management
    assert backend.warm_on_link is False  # warming can never hit (see module doc)

    paths = [path for _, path, _ in server.calls]
    assert "/health" in paths
    assert "/v1/models" in paths


def test_connect_unreachable_server(transport):
    llm = LLM(db=StubDB(), host="downhost")
    with pytest.raises(ValueError, match="Failed to connect to NInfer server"):
        llm.connect_model(model=MODEL, port=PORT, host="downhost", engine="ninfer")


def test_auto_detection_picks_ninfer(transport):
    transport.add_ninfer_server(HOST, PORT, model_id=SERVED_ID)
    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, engine="auto")

    from PBQA.backends.ninfer import NinferBackend

    assert isinstance(llm.models[MODEL], NinferBackend)


# =============================================================================
# Payload shape
# =============================================================================


def test_payload_renders_schema_into_system_message(transport):
    llm, server = make_ninfer_llm(transport)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")

    # No grammar fields on the wire: ninfer rejects response_format and PBQA
    # never sends structured_outputs/json_schema to it
    assert set(payload.keys()) == {
        "model", "messages", "stop", "temperature", "min_p", "top_p",
        "max_tokens",
    }
    assert payload["model"] == SERVED_ID

    system = payload["messages"][0]
    assert system["role"] == "system"
    # Original system prompt kept, schema instruction appended
    assert system["content"].startswith("You report the weather.")
    assert expected_instruction(Weather.model_json_schema()) in system["content"]

    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_schema_instruction_prepended_without_system_prompt(transport):
    db = StubDB(system_prompt=None)
    llm, server = make_ninfer_llm(transport, db=db)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    system = payload["messages"][0]
    assert system["role"] == "system"
    assert system["content"] == expected_instruction(Weather.model_json_schema())


def test_single_string_schema_skips_injection(transport):
    llm, server = make_ninfer_llm(transport, schema=Reply.model_json_schema())
    server.chat_content = "plain text reply"
    result = llm.ask(input="hi", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert all(
        "JSON Schema" not in m["content"] for m in payload["messages"]
    )
    assert result["response"] == {"reply": "plain text reply"}


# =============================================================================
# Client-side validation
# =============================================================================


def test_schema_violation_retried_with_jitter(transport):
    llm, server = make_ninfer_llm(transport)
    server.chat_contents = [
        json.dumps({"temperature": "warm"}),  # wrong type, missing key
        json.dumps({"temperature": 21.5, "condition": "cloudy"}),
    ]

    result = llm.ask(input="weather?", pattern="weather", model=MODEL)

    calls = server.calls_to("/v1/chat/completions")
    assert len(calls) == 2
    # Retry jitters sampling to break the attractor
    retry_payload = calls[1][2]
    assert retry_payload["temperature"] >= 0.4
    assert result["response"] == {"temperature": 21.5, "condition": "cloudy"}


def test_schema_violation_after_retry_raises(transport):
    llm, server = make_ninfer_llm(transport)
    server.chat_content = json.dumps({"temperature": "warm"})

    with pytest.raises(ValueError, match="schema-violating"):
        llm.ask(input="weather?", pattern="weather", model=MODEL)

    assert len(server.calls_to("/v1/chat/completions")) == 2


def test_code_fences_stripped(transport):
    llm, server = make_ninfer_llm(transport)
    server.chat_content = (
        "```json\n" + json.dumps({"temperature": 20.0, "condition": "sunny"}) + "\n```"
    )

    result = llm.ask(input="weather?", pattern="weather", model=MODEL)
    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    assert len(server.calls_to("/v1/chat/completions")) == 1


# =============================================================================
# Strict model alias
# =============================================================================


def test_alias_change_rediscovered_and_retried(transport):
    llm, server = make_ninfer_llm(transport)
    # A different --model-id was deployed on the same port after connect
    server.model_id = "qwen3.6-27b-ninfer-v2"

    result = llm.ask(input="weather?", pattern="weather", model=MODEL)

    calls = server.calls_to("/v1/chat/completions")
    assert len(calls) == 2  # 404 model_not_found, rediscover, retry
    assert calls[1][2]["model"] == "qwen3.6-27b-ninfer-v2"
    assert result["response"]["condition"] == "sunny"


# =============================================================================
# No warm on link
# =============================================================================


def test_link_does_not_warm(transport):
    # NInfer's restore checkpoint sits past the assistant header of the
    # warmed prompt; every real query diverges at its user turn before
    # reaching it, so a warm request can never produce a hit — link() must
    # not spend a prefill on one
    llm, server = make_ninfer_llm(transport)
    llm.link(pattern="weather", model=MODEL)

    assert llm.models[MODEL].warm_on_link is False
    assert server.calls_to("/v1/chat/completions") == []
