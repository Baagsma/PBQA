"""Unit tests for the vLLM backend. No live server required.

Covers the vLLM-specific wire behavior: structured_outputs schema placement
(vLLM >= 0.12 format), served-model-id discovery, no client-side cache
management, warm-on-link prefilling, and mixed-engine fallback chains.

Usage:
    python -m pytest tests/test_vllm_backend.py
"""

import json

import pytest
import requests as real_requests

from PBQA import LLM
from tests.mock_transport import (
    HOST,
    MODEL,
    PORT,
    Reply,
    StubDB,
    Weather,
)

SERVED_ID = "qwen3.6-27b-nvfp4"


def make_vllm_llm(transport, schema=None, db=None, **connect_kwargs):
    server = transport.add_vllm_server(HOST, PORT, model_id=SERVED_ID)
    llm = LLM(db=db or StubDB(schema=schema), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, engine="vllm", **connect_kwargs)
    return llm, server


# =============================================================================
# Connection
# =============================================================================


def test_connect_discovers_served_model(transport):
    llm, server = make_vllm_llm(transport)
    backend = llm.models[MODEL]

    assert backend.connected
    assert backend.model_id == SERVED_ID
    assert backend.is_rerank is False
    assert backend.store_cache is False  # vLLM caching is automatic
    assert backend.warm_on_link is True

    paths = [path for _, path, _ in server.calls]
    assert "/health" in paths
    assert "/v1/models" in paths


def test_connect_unreachable_server(transport):
    llm = LLM(db=StubDB(), host="downhost")
    with pytest.raises(ValueError, match="Failed to connect to vLLM server"):
        llm.connect_model(model=MODEL, port=PORT, host="downhost", engine="vllm")


# =============================================================================
# Payload shape
# =============================================================================


def test_payload_uses_structured_outputs_and_served_model(transport):
    llm, server = make_vllm_llm(transport)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")

    # Golden payload shape: exactly these keys, nothing more. Notably absent:
    # id_slot and cache_prompt (llama.cpp-only), any internal config.
    assert set(payload.keys()) == {
        "model", "messages", "structured_outputs",
        "json_schema",  # dual-emitted so a router failover to llama.cpp keeps enforcement
        "stop", "temperature", "min_p", "top_p", "max_tokens",
    }
    # The wire model is the served model id, not PBQA's model name
    assert payload["model"] == SERVED_ID
    # vLLM >= 0.12 structured output format (guided_json was removed)
    assert payload["structured_outputs"] == {"json": Weather.model_json_schema()}
    assert payload["json_schema"] == Weather.model_json_schema()

    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_use_cache_false_is_ignored(transport):
    llm, server = make_vllm_llm(transport)
    llm.ask(input="hi", pattern="weather", model=MODEL, use_cache=False)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert "cache_prompt" not in payload
    assert "id_slot" not in payload


def test_single_string_schema_unwrapped(transport):
    llm, server = make_vllm_llm(transport, schema=Reply.model_json_schema())
    server.chat_content = "plain text reply"

    result = llm.ask(input="say something", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert "structured_outputs" not in payload
    assert result["response"] == {"reply": "plain text reply"}


def test_openai_style_error_raises(transport):
    llm, server = make_vllm_llm(transport)
    server.error_payload = {
        "object": "error",
        "message": "bad request",
        "type": "BadRequestError",
        "code": 400,
    }

    with pytest.raises(ValueError, match="LLM error"):
        llm.ask(input="hi", pattern="weather", model=MODEL)


def test_model_swap_rediscovers_served_id(transport):
    # A different model deployed on the same port after connect: the backend
    # should rediscover the served id and retry instead of failing
    llm, server = make_vllm_llm(transport)
    server.strict_model = True
    server.model_id = "qwen3.6-35b-a3b-nvfp4"  # server swapped after connect

    result = llm.ask(input="how hot?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    assert llm.models[MODEL].model_id == "qwen3.6-35b-a3b-nvfp4"
    chat_calls = server.calls_to("/v1/chat/completions")
    assert len(chat_calls) == 2  # stale-id attempt + retry
    assert chat_calls[-1][2]["model"] == "qwen3.6-35b-a3b-nvfp4"


def test_unknown_model_error_without_swap_still_raises(transport):
    # Same 404 shape, but /v1/models still reports the id we already have —
    # no retry loop, the error propagates
    llm, server = make_vllm_llm(transport)
    server.error_payload = {
        "object": "error",
        "message": f"The model `{SERVED_ID}` does not exist.",
        "type": "NotFoundError",
        "code": 404,
    }

    with pytest.raises(ValueError, match="LLM error"):
        llm.ask(input="hi", pattern="weather", model=MODEL)


# =============================================================================
# Warm on link
# =============================================================================


def test_link_warms_prefix(transport):
    llm, server = make_vllm_llm(transport)
    llm.link(pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert payload["max_tokens"] == 1
    assert payload["model"] == SERVED_ID
    assert "structured_outputs" not in payload
    # Prefix (system + base examples) plus the filler user turn that chat
    # templates require to close the conversation
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert payload["messages"][-1]["content"] == "."


def test_warm_failure_does_not_break_link(transport):
    llm, server = make_vllm_llm(transport)
    server.chat_error = real_requests.exceptions.ConnectionError("not up yet")

    llm.link(pattern="weather", model=MODEL)  # must not raise
    assert llm.pattern_models["weather"] == MODEL

    # Recovers once the server responds again
    server.chat_error = None
    result = llm.ask(input="what's the weather?", pattern="weather")
    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}


def test_link_skips_warm_when_no_prefix(transport):
    db = StubDB(system_prompt=None, base_examples=[])
    llm, server = make_vllm_llm(transport, db=db)

    llm.link(pattern="weather", model=MODEL)
    assert server.calls_to("/v1/chat/completions") == []


def test_explicit_warm(transport):
    llm, server = make_vllm_llm(transport)
    llm.link(pattern="weather", model=MODEL)
    before = len(server.calls_to("/v1/chat/completions"))

    llm.warm("weather")  # e.g. after a known server restart
    assert len(server.calls_to("/v1/chat/completions")) == before + 1


# =============================================================================
# Mixed-engine fallback
# =============================================================================


def test_vllm_fallback_for_llamacpp_primary(transport):
    primary_server = transport.add_server(HOST, PORT)
    vllm_server = transport.add_vllm_server("vllmhost", 8001, model_id=SERVED_ID)

    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model=MODEL, port=PORT)  # llama.cpp primary
    llm.add_fallback(model=MODEL, host="vllmhost", port=8001, engine="vllm")

    primary_server.chat_error = real_requests.exceptions.ConnectionError("down")

    result = llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    ((_, _, payload),) = vllm_server.calls_to("/v1/chat/completions")
    assert payload["model"] == SERVED_ID
    assert "structured_outputs" in payload
    assert "id_slot" not in payload


def test_rerank_raises_on_vllm(transport):
    llm, _ = make_vllm_llm(transport)
    with pytest.raises(ValueError, match="not a reranking model"):
        llm.rerank(input="q", model=MODEL, documents=["a"])
