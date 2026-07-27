"""Engine drift: the server behind a connected backend's address changes
engines (deploy swap behind a proxy/router).

Observed live (AVA, 2026-07-21): a proxy answered probes during vLLM's
startup window, the backend was built as llama.cpp, and every request then
spoke llama.cpp dialect (alias model name, slot restores) to a vLLM server —
which rejects unknown model names with a NotFoundError 404. Nothing retried,
because the failure surfaced as a plain ValueError.

These tests pin the healing path: the llama.cpp backend classifies that
response shape as EngineDriftError (llama.cpp itself never rejects a model
name), and the LLM layer redetects the engine, rebuilds the backend in
place, and retries once. Plus the vLLM-side guards that close the original
hole: connect() waits for a model to register, and generate() refuses to
send the alias name when no served id is known.

Usage:
    python -m pytest tests/test_engine_drift.py
"""

import pytest

import PBQA.backends.vllm as vllm_module
from PBQA import LLM
from PBQA.backends import EngineDriftError, LlamaCppBackend, VLLMBackend
from PBQA.backends.base import BackendConfig
from tests.mock_transport import HOST, MODEL, PORT, StubDB

SERVED_ID = "qwen3.6-27b-nvfp4"


def make_llamacpp_llm(transport):
    server = transport.add_server(HOST, PORT)
    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, engine="llamacpp")
    return llm, server


# =============================================================================
# Drift classification (backend level)
# =============================================================================


def test_llamacpp_classifies_model_rejection_as_drift(transport):
    llm, server = make_llamacpp_llm(transport)

    # The server behind the address is now vLLM: it rejects the alias name
    vllm_server = transport.add_vllm_server(
        HOST, PORT, model_id=SERVED_ID, strict_model=True
    )

    backend = llm.models[MODEL]
    with pytest.raises(EngineDriftError, match="rejected the model name"):
        backend.generate(
            messages=[{"role": "user", "content": "hi"}],
            schema=None,
            pattern="weather",
            model=MODEL,
            overrides={},
        )


def test_llamacpp_classifies_nested_error_shape_as_drift(transport):
    # The shape observed live: vLLM's 404 wrapped in {"error": {...}}
    llm, server = make_llamacpp_llm(transport)
    vllm_server = transport.add_vllm_server(HOST, PORT, model_id=SERVED_ID)
    vllm_server.error_payload = {
        "error": {
            "message": f"The model `{MODEL}` does not exist.",
            "type": "NotFoundError",
            "param": "model",
            "code": 404,
        }
    }

    backend = llm.models[MODEL]
    with pytest.raises(EngineDriftError, match="rejected the model name"):
        backend.generate(
            messages=[{"role": "user", "content": "hi"}],
            schema=None,
            pattern="weather",
            model=MODEL,
            overrides={},
        )


# =============================================================================
# Self-healing (LLM level)
# =============================================================================


def test_drift_rebuilds_backend_and_retries(transport):
    llm, server = make_llamacpp_llm(transport)
    assert isinstance(llm.models[MODEL], LlamaCppBackend)

    vllm_server = transport.add_vllm_server(
        HOST, PORT, model_id=SERVED_ID, strict_model=True
    )

    result = llm.ask(input="how hot?", pattern="weather", model=MODEL)

    # The request succeeded on the rebuilt backend, in vLLM dialect
    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    assert isinstance(llm.models[MODEL], VLLMBackend)
    assert llm.models[MODEL].model_id == SERVED_ID
    chat_calls = vllm_server.calls_to("/v1/chat/completions")
    assert chat_calls[-1][2]["model"] == SERVED_ID

    # And the healing is durable: the next request goes straight through
    vllm_server.calls.clear()
    result = llm.ask(input="again?", pattern="weather", model=MODEL)
    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    assert len(vllm_server.calls_to("/v1/chat/completions")) == 1


def test_drift_with_unreachable_replacement_fails_loud(transport):
    llm, server = make_llamacpp_llm(transport)

    # Swap to a strict vLLM server, then make redetection impossible by
    # removing the server entirely after the drift error fires
    class VanishingServer:
        def __init__(self, inner):
            self.inner = inner
            self.chat_done = False

        def handle(self, method, path, body):
            response = self.inner.handle(method, path, body)
            if path == "/v1/chat/completions":
                # After answering the drift-triggering 404, vanish
                transport.servers.pop(f"{HOST}:{PORT}")
            return response

    strict = transport.add_vllm_server(
        HOST, PORT, model_id=SERVED_ID, strict_model=True
    )
    transport.servers[f"{HOST}:{PORT}"] = VanishingServer(strict)

    with pytest.raises(ValueError, match="All backends failed"):
        llm.ask(input="hi", pattern="weather", model=MODEL)


# =============================================================================
# vLLM startup-window guards
# =============================================================================


def test_vllm_connect_waits_for_model_registration(transport, monkeypatch):
    monkeypatch.setattr(vllm_module, "MODEL_LOAD_POLL", 0.0)
    server = transport.add_vllm_server(
        HOST, PORT, model_id=SERVED_ID, model_available_after=3
    )

    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, engine="vllm")

    assert llm.models[MODEL].model_id == SERVED_ID
    # First discovery saw the empty startup window, then polled through it
    assert server._models_requests >= 4


def test_vllm_connect_times_out_when_model_never_registers(
    transport, monkeypatch
):
    monkeypatch.setattr(vllm_module, "MODEL_LOAD_POLL", 0.0)
    monkeypatch.setattr(vllm_module, "MODEL_LOAD_TIMEOUT", 0.05)
    transport.add_vllm_server(
        HOST, PORT, model_id=SERVED_ID, model_available_after=10_000
    )

    llm = LLM(db=StubDB(), host=HOST)
    with pytest.raises(ValueError, match="did not register a model"):
        llm.connect_model(model=MODEL, port=PORT, engine="vllm")


def test_vllm_generate_refuses_alias_without_served_id(transport):
    server = transport.add_vllm_server(HOST, PORT, model_id=SERVED_ID)
    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, engine="vllm")

    backend = llm.models[MODEL]
    backend.model_id = None  # simulate a discovery that found nothing
    server.model_available_after = 10_000
    server._models_requests = 0

    with pytest.raises(ValueError, match="no model registered"):
        backend.generate(
            messages=[{"role": "user", "content": "hi"}],
            schema=None,
            pattern="weather",
            model=MODEL,
            overrides={},
        )
