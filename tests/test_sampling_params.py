"""Sampling parameters the server refuses: strip, retry, remember.

Observed live (2026-07-31): a vLLM deployment running speculative decoding
answers any sampled request carrying min_p with a 400 naming the parameter —
and PBQA sends min_p on every request by default. Nothing recovered, because
the refusal surfaced as a plain ValueError. Worse, the malformed-JSON retry
jitters temperature above 0 with min_p still aboard, so the recovery path
itself 400s on such a server.

These tests pin the healing path: the named parameters are dropped, the
request is retried once, and the restriction is remembered on the backend so
every later request omits them without a failed round-trip.

Usage:
    python -m pytest tests/test_sampling_params.py
"""

import json
import logging

import pytest

from PBQA import LLM
from tests.mock_transport import HOST, MODEL, PORT, StubDB

SERVED_ID = "qwen3.6-35b-a3b-nvfp4-acc"


def make_vllm_llm(transport, **connect_kwargs):
    server = transport.add_vllm_server(
        HOST, PORT, model_id=SERVED_ID, speculative_decoding=True
    )
    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, engine="vllm", **connect_kwargs)
    return llm, server


def chat_payloads(server):
    return [body for _, _, body in server.calls_to("/v1/chat/completions")]


# =============================================================================
# Strip and retry
# =============================================================================


def test_refused_param_stripped_and_retried(transport):
    llm, server = make_vllm_llm(transport)

    result = llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    refused, retried = chat_payloads(server)
    assert "min_p" in refused
    assert "min_p" not in retried
    # Only the refused parameter goes; the rest of the request is untouched
    assert retried["temperature"] == refused["temperature"]
    assert retried["top_p"] == refused["top_p"]
    assert retried["messages"] == refused["messages"]


def test_restriction_is_remembered(transport):
    llm, server = make_vllm_llm(transport)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)
    server.calls.clear()

    result = llm.ask(input="and now?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    # No failed round-trip the second time around
    ((payload),) = chat_payloads(server)
    assert "min_p" not in payload
    assert llm.models[MODEL]._suppressed_params == {"min_p"}


def test_warning_logged_once(transport, caplog):
    llm, server = make_vllm_llm(transport)

    with caplog.at_level(logging.WARNING, logger="PBQA.backends"):
        llm.ask(input="what's the weather?", pattern="weather", model=MODEL)
        llm.ask(input="and now?", pattern="weather", model=MODEL)

    learned = [r for r in caplog.records if "does not support" in r.message]
    assert len(learned) == 1
    assert "min_p" in learned[0].message


def test_only_sent_params_are_suppressed(transport):
    # The refusal names logit_bias too, but PBQA never sent it — it is prose,
    # not a parameter this backend has to give up
    llm, _ = make_vllm_llm(transport)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    assert llm.models[MODEL]._suppressed_params == {"min_p"}


def test_greedy_requests_are_unaffected(transport):
    # min_p is inert under greedy decoding and the live server accepts it
    # there, so nothing is learned and nothing is stripped
    llm, server = make_vllm_llm(transport, temperature=0)

    result = llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    ((payload),) = chat_payloads(server)
    assert payload["min_p"] == 0.02
    assert llm.models[MODEL]._suppressed_params == set()


# =============================================================================
# Boundaries
# =============================================================================


def test_unrelated_error_still_raises(transport):
    llm, server = make_vllm_llm(transport)
    server.error_payload = {
        "object": "error",
        "message": "This model's maximum context length is 4096 tokens.",
        "type": "BadRequestError",
        "code": 400,
    }

    with pytest.raises(ValueError, match="LLM error"):
        llm.ask(input="hi", pattern="weather", model=MODEL)

    assert llm.models[MODEL]._suppressed_params == set()
    assert len(chat_payloads(server)) == 1  # no retry on an unrelated error


def test_server_refusing_after_strip_raises(transport):
    # A server that keeps refusing over a parameter we no longer send has
    # nothing left to strip: raise instead of retrying forever
    llm, server = make_vllm_llm(transport)
    llm.ask(input="what's the weather?", pattern="weather", model=MODEL)
    server.error_payload = {
        "error": {
            "message": (
                "The min_p and logit_bias sampling parameters are not yet "
                "supported with speculative decoding."
            ),
            "type": "BadRequestError",
            "code": 400,
        }
    }
    server.calls.clear()

    with pytest.raises(ValueError, match="LLM error"):
        llm.ask(input="and now?", pattern="weather", model=MODEL)

    assert len(chat_payloads(server)) == 1


def test_llamacpp_dialect_recovers_too(transport):
    # The restriction belongs to the deployment, not the engine: a llama.cpp
    # backend talking to a server that refuses min_p heals the same way
    server = transport.add_server(HOST, PORT, speculative_decoding=True)
    llm = LLM(db=StubDB(), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, engine="llamacpp")

    result = llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    refused, retried = chat_payloads(server)
    assert "min_p" in refused
    assert "min_p" not in retried
    assert retried["id_slot"] == refused["id_slot"]


# =============================================================================
# The path that broke live: malformed JSON -> sampling jitter -> refusal
# =============================================================================


def test_jittered_retry_survives_refusal(transport):
    llm, server = make_vllm_llm(transport, temperature=0)
    # Greedy generation runs into a repetition loop and comes back truncated;
    # the jittered retry raises temperature above 0, which is what turns the
    # min_p the request has always carried into a 400
    server.chat_contents = ['{"temperature": 20.0, "condition": "sun']

    result = llm.ask(input="what's the weather?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    malformed, refused, retried = chat_payloads(server)
    assert malformed["temperature"] == 0 and "min_p" in malformed
    assert refused["temperature"] >= 0.4 and "min_p" in refused
    assert retried["temperature"] == refused["temperature"]
    assert "min_p" not in retried
