"""Unit tests for the llama.cpp backend and engine-agnostic routing. No live
server required.

The HTTP transport used by backends is replaced with an in-memory fake (see
tests/mock_transport.py), so these tests pin down the exact wire behavior:
payload shape, cache slot save/restore sequencing, capability probing,
fallback routing, and schema preprocessing.

Usage:
    python -m pytest tests/test_backends.py
"""

import json
import logging

import pytest
import requests as real_requests

from PBQA import LLM
from tests.mock_transport import (
    HOST,
    MODEL,
    PORT,
    FakeResponse,
    Item,
    Reply,
    StubDB,
    Weather,
)


def make_llm(transport, schema=None, **connect_kwargs):
    server = transport.add_server(HOST, PORT)
    llm = LLM(db=StubDB(schema=schema), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, **connect_kwargs)
    return llm, server


# =============================================================================
# Connection & capability probing
# =============================================================================


def test_connect_probes_capabilities(transport):
    llm, server = make_llm(transport)
    backend = llm.models[MODEL]

    assert backend.connected
    assert backend.total_slots == 1
    assert backend.is_rerank is False
    assert backend.store_cache is True  # slot restore probe answered 400

    paths = [path for _, path, _ in server.calls]
    assert "/props" in paths
    assert "/v1/rerank" in paths
    assert "/slots/0?action=restore" in paths


def test_connect_unknown_engine(transport):
    transport.add_server(HOST, PORT)
    llm = LLM(db=StubDB(), host=HOST)
    with pytest.raises(ValueError, match="Unknown engine"):
        llm.connect_model(model=MODEL, port=PORT, engine="nonsense")


def test_connect_unreachable_server(transport):
    llm = LLM(db=StubDB(), host="downhost")
    with pytest.raises(ValueError, match="Could not detect"):
        llm.connect_model(model=MODEL, port=PORT, host="downhost")


def test_connect_unreachable_server_explicit_engine(transport):
    llm = LLM(db=StubDB(), host="downhost")
    with pytest.raises(ValueError, match="Failed to get properties"):
        llm.connect_model(
            model=MODEL, port=PORT, host="downhost", engine="llamacpp"
        )


# =============================================================================
# Engine auto-detection
# =============================================================================


def test_auto_detects_llamacpp(transport):
    from PBQA.backends import LlamaCppBackend

    transport.add_server(HOST, PORT)
    llm = LLM(db=StubDB(), host=HOST)
    backend = llm.connect_model(model=MODEL, port=PORT)

    assert isinstance(backend, LlamaCppBackend)


def test_auto_detects_vllm(transport):
    from PBQA.backends import VLLMBackend

    transport.add_vllm_server(HOST, PORT)
    llm = LLM(db=StubDB(), host=HOST)
    backend = llm.connect_model(model=MODEL, port=PORT)

    assert isinstance(backend, VLLMBackend)
    assert backend.model_id == "qwen3.6-27b-nvfp4"


def test_detect_probes_are_mutually_exclusive(transport):
    from PBQA.backends import LlamaCppBackend, VLLMBackend, BackendConfig

    transport.add_server(HOST, PORT)
    transport.add_vllm_server(HOST, PORT + 1)
    llama_config = BackendConfig(host=HOST, port=PORT)
    vllm_config = BackendConfig(host=HOST, port=PORT + 1)

    assert LlamaCppBackend.detect(llama_config) is True
    assert LlamaCppBackend.detect(vllm_config) is False
    assert VLLMBackend.detect(vllm_config) is True
    assert VLLMBackend.detect(llama_config) is False


def test_router_answering_props_still_detects_vllm(transport):
    """A router can answer /props on behalf of a sibling llama.cpp backend
    while completions for the served model actually go to vLLM. The exact
    vLLM discriminator (owned_by on /v1/models) must win over the /props
    probe, or schemas get sent in a field vLLM silently ignores."""
    from PBQA.backends import VLLMBackend

    server = transport.add_vllm_server(HOST, PORT)
    original_handle = server.handle

    def handle(method, path, body):
        if method == "GET" and path == "/props":
            return FakeResponse({"total_slots": 4})  # llama.cpp-style answer
        return original_handle(method, path, body)

    server.handle = handle

    llm = LLM(db=StubDB(), host=HOST)
    backend = llm.connect_model(model=MODEL, port=PORT)

    assert isinstance(backend, VLLMBackend)


def test_explicit_llamacpp_engine_rejects_vllm_server(transport):
    # The pre-2.0 bug: vLLM's 404 body for /props parsed as valid props
    transport.add_vllm_server(HOST, PORT)
    llm = LLM(db=StubDB(), host=HOST)
    with pytest.raises(ValueError, match="Failed to connect"):
        llm.connect_model(model=MODEL, port=PORT, engine="llamacpp")


def test_fallback_engine_auto(transport):
    from PBQA.backends import VLLMBackend

    transport.add_server(HOST, PORT)
    transport.add_vllm_server(HOST, PORT + 1)
    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model=MODEL, port=PORT)
    llm.add_fallback(model=MODEL, host=HOST, port=PORT + 1, engine="auto")

    assert isinstance(llm._fallbacks[MODEL][0], VLLMBackend)


# =============================================================================
# Payload shape
# =============================================================================


def test_payload_contains_no_internal_config(transport):
    llm, server = make_llm(transport)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")

    forbidden = {
        "host", "port", "is_rerank", "store_cache", "strict_schema",
        "total_slots", "lazy", "connected", "request_defaults",
    }
    assert not forbidden & set(payload.keys()), (
        f"Internal config leaked into payload: {forbidden & set(payload.keys())}"
    )

    # Golden payload shape: exactly these keys, nothing more
    assert set(payload.keys()) == {
        "model", "id_slot", "cache_prompt", "messages", "json_schema",
        "structured_outputs",  # dual-emitted so a router failover to vLLM keeps enforcement
        "stop", "temperature", "min_p", "top_p", "max_tokens",
    }
    assert payload["model"] == MODEL
    assert payload["id_slot"] == 0
    assert payload["cache_prompt"] is True
    assert payload["temperature"] == 1.0
    assert payload["min_p"] == 0.02
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 4096
    assert payload["json_schema"] == Weather.model_json_schema()
    assert payload["structured_outputs"] == {"json": Weather.model_json_schema()}

    # Messages: system + base example pair + input
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert payload["messages"][-1] == {
        "role": "user",
        "content": "what's the weather?",
    }


def test_connect_kwargs_reach_payload(transport):
    llm, server = make_llm(transport, top_k=40)
    llm.ask(input="hi", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert payload["top_k"] == 40


def test_stop_strings_merge(transport):
    llm, server = make_llm(transport, stop=["<A>"])
    llm.ask(input="hi", pattern="weather", model=MODEL, stop=["<B>"])

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert payload["stop"] == ["<A>", "<B>"]


def test_response_parsing_with_schema(transport):
    llm, server = make_llm(transport)
    result = llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    assert result["metadata"]["total_tokens"] == 15
    assert "total_time" in result["metadata"]


# =============================================================================
# Schema preprocessing
# =============================================================================


def test_single_string_schema_unwrapped(transport):
    llm, server = make_llm(transport, schema=Reply.model_json_schema())
    server.chat_content = "just plain text"

    result = llm.ask(input="say something", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert "json_schema" not in payload
    assert result["response"] == {"reply": "just plain text"}


def test_refs_resolved_in_payload_schema(transport):
    llm, server = make_llm(transport, schema=Item.model_json_schema())
    server.chat_content = json.dumps({"color": "red", "note": "n"})

    llm.ask(input="pick a color", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    dumped = json.dumps(payload["json_schema"])
    assert "$ref" not in dumped
    assert "$defs" not in dumped
    assert '"enum"' in dumped  # the Color enum survived inlining


def test_strict_schema_locks_objects(transport):
    llm, server = make_llm(transport, strict_schema=True)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert payload["json_schema"]["additionalProperties"] is False


def test_no_strict_schema_by_default(transport):
    llm, server = make_llm(transport)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert "additionalProperties" not in payload["json_schema"]


# =============================================================================
# Cache slot save/restore
# =============================================================================


def test_cache_restore_before_and_save_after_generation(transport):
    llm, server = make_llm(transport)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    # Filter out the connect-time probe calls
    request_calls = [
        (method, path, body)
        for method, path, body in server.calls
        if path.startswith("/slots/") and body["filename"] == f"weather-{MODEL}.bin"
        or path == "/v1/chat/completions"
    ]
    paths = [path for _, path, _ in request_calls]
    assert paths == [
        "/slots/0?action=restore",
        "/v1/chat/completions",
        "/slots/0?action=save",
    ]


def test_no_cache_calls_when_store_cache_disabled(transport):
    llm, server = make_llm(transport, store_cache=False)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    cache_calls = [
        (method, path, body)
        for method, path, body in server.calls
        if path.startswith("/slots/")
        and body["filename"] == f"weather-{MODEL}.bin"
    ]
    assert cache_calls == []


def test_use_cache_false_sets_cache_prompt(transport):
    llm, server = make_llm(transport)
    llm.ask(input="hi", pattern="weather", model=MODEL, use_cache=False)

    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert payload["cache_prompt"] is False


# =============================================================================
# Deprecations
# =============================================================================


def test_cache_slot_deprecated_on_ask(transport, caplog):
    llm, server = make_llm(transport)
    with caplog.at_level(logging.WARNING, logger="PBQA.llm"):
        llm.ask(input="hi", pattern="weather", model=MODEL, cache_slot=3)
    assert any("cache_slot" in r.message for r in caplog.records)

    # And it must not reach the wire
    ((_, _, payload),) = server.calls_to("/v1/chat/completions")
    assert "cache_slot" not in payload
    assert payload["id_slot"] == 0


def test_cache_slot_deprecated_on_link(transport, caplog):
    llm, _ = make_llm(transport)
    with caplog.at_level(logging.WARNING, logger="PBQA.llm"):
        llm.link(pattern="weather", model=MODEL, cache_slot=5)
    assert any("cache_slot" in r.message for r in caplog.records)
    assert llm.pattern_models["weather"] == MODEL


def test_link_then_ask_without_model(transport):
    llm, server = make_llm(transport)
    llm.link(pattern="weather", model=MODEL)
    result = llm.ask(input="what's the weather?", pattern="weather")
    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}


def test_link_does_not_warm_llamacpp(transport):
    llm, server = make_llm(transport)
    llm.link(pattern="weather", model=MODEL)
    # Durable slot caches: no prefill request on link
    assert server.calls_to("/v1/chat/completions") == []


# =============================================================================
# Fallback routing
# =============================================================================


def test_fallback_used_when_primary_fails(transport):
    llm, primary = make_llm(transport)
    fallback_server = transport.add_server("fallbackhost", 9090)
    llm.add_fallback(model=MODEL, host="fallbackhost", port=9090)

    primary.chat_error = real_requests.exceptions.ConnectionError("primary down")

    result = llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    assert len(fallback_server.calls_to("/v1/chat/completions")) == 1
    # Primary got marked unhealthy for subsequent requests
    healthy, _ = llm._health_cache[(HOST, PORT)]
    assert healthy is False


def test_all_backends_failing_raises(transport):
    llm, primary = make_llm(transport)
    primary.chat_error = real_requests.exceptions.ConnectionError("down")

    with pytest.raises(ValueError, match="All backends failed"):
        llm.ask(input="hi", pattern="weather", model=MODEL)


def test_server_reported_error_aborts_without_failover(transport):
    llm, primary = make_llm(transport)
    fallback_server = transport.add_server("fallbackhost", 9090)
    llm.add_fallback(model=MODEL, host="fallbackhost", port=9090)

    def bad_handle(method, path, body, _orig=primary.handle):
        if path == "/v1/chat/completions":
            primary.calls.append((method, path, body))
            return FakeResponse({"error": {"code": 500, "message": "exploded"}})
        return _orig(method, path, body)

    primary.handle = bad_handle

    # Server-reported errors (vs. transport failures) abort instead of
    # silently falling back — same behavior as before the refactor
    with pytest.raises(ValueError, match="LLM error"):
        llm.ask(input="hi", pattern="weather", model=MODEL)
    assert fallback_server.calls_to("/v1/chat/completions") == []


# =============================================================================
# Rerank
# =============================================================================


def test_rerank_scores_and_sorts(transport):
    server = transport.add_server(HOST, 8090, rerank=True)
    server.rerank_results = [
        {"index": 0, "relevance_score": -2.0},
        {"index": 1, "relevance_score": 3.0},
    ]

    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model="reranker", port=8090)
    assert llm.models["reranker"].is_rerank is True

    results = llm.rerank(input="query", model="reranker", documents=["a", "b"])

    assert [r["document"] for r in results] == ["b", "a"]
    assert results[0]["score"] > 0.9  # sigmoid(3.0)
    assert results[1]["score"] < 0.2  # sigmoid(-2.0)


def test_ask_on_rerank_model_raises(transport):
    server = transport.add_server(HOST, 8090, rerank=True)
    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model="reranker", port=8090)

    with pytest.raises(ValueError, match="reranking model"):
        llm.ask(input="hi", pattern="weather", model="reranker")
