"""Out-of-schema responses are caught, retried, and never returned.

Observed live (PINE, 2026-07-16): a structured-output request came back as
clean JSON with the right fields but an out-of-enum value ("combat" through
a List[enum] grammar). Grammar enforcement can silently fail — engine drift
behind a router, a request field the server ignores — so ask() verifies
every structured response against the schema and treats a violation exactly
like malformed JSON: one jittered uncached retry, then a loud failure.
Verified, never assumed.

Usage:
    python -m pytest tests/test_response_validation.py
"""

import pytest

from PBQA import LLM
from PBQA.schema import validate_response
from tests.mock_transport import HOST, MODEL, PORT, StubDB

ENUM_SCHEMA = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number"},
        "condition": {"type": "string", "enum": ["sunny", "rain"]},
    },
    "required": ["temperature", "condition"],
}


def make_llm(transport):
    server = transport.add_vllm_server(HOST, PORT, model_id="m")
    llm = LLM(db=StubDB(schema=ENUM_SCHEMA), host=HOST)
    llm.connect_model(model=MODEL, port=PORT, engine="vllm", temperature=0)
    return llm, server


def chat_payloads(server):
    return [body for _, _, body in server.calls_to("/v1/chat/completions")]


# =============================================================================
# The validator itself
# =============================================================================


def test_validate_response_accepts_conforming():
    assert validate_response({"temperature": 20.0, "condition": "rain"}, ENUM_SCHEMA) is None


def test_validate_response_reports_path_and_counts_rest():
    err = validate_response({"temperature": "hot", "condition": "combat"}, ENUM_SCHEMA)
    assert "$.condition" in err
    assert "+1 more" in err


def test_validate_response_catches_missing_required():
    err = validate_response({"temperature": 20.0}, ENUM_SCHEMA)
    assert "condition" in err


def test_validate_response_catches_array_item_enum():
    # The live failure shape: an enum inside array items
    schema = {
        "type": "object",
        "properties": {
            "skills": {"type": "array", "items": {"type": "string", "enum": ["agility", "power"]}},
        },
        "required": ["skills"],
    }
    err = validate_response({"skills": ["agility", "combat"]}, schema)
    assert "$.skills[1]" in err and "combat" in err


# =============================================================================
# The ask() guarantee
# =============================================================================


def test_out_of_schema_is_retried_then_accepted(transport):
    llm, server = make_llm(transport)
    server.chat_contents = [
        '{"temperature": 20.0, "condition": "combat"}',
        '{"temperature": 20.0, "condition": "sunny"}',
    ]

    result = llm.ask(input="weather?", pattern="weather", model=MODEL)

    assert result["response"] == {"temperature": 20.0, "condition": "sunny"}
    first, second = chat_payloads(server)
    assert first["temperature"] == 0
    assert second["temperature"] == 0  # temperature is only ever user-instigated
    assert second["repetition_penalty"] == 1.1  # the deterministic loop-breaker


def test_persistent_violation_raises_loudly(transport):
    llm, server = make_llm(transport)
    server.chat_contents = ['{"temperature": 20.0, "condition": "combat"}'] * 2

    with pytest.raises(ValueError, match="schema violation"):
        llm.ask(input="weather?", pattern="weather", model=MODEL)

    assert len(chat_payloads(server)) == 2  # exactly one retry, never silent


def test_conforming_response_passes_untouched(transport):
    llm, server = make_llm(transport)

    result = llm.ask(input="weather?", pattern="weather", model=MODEL)

    assert result["response"]["condition"] in ("sunny", "rain")
    assert len(chat_payloads(server)) == 1  # no retry when the schema holds
