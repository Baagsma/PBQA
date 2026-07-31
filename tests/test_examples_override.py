"""Unit tests for the explicit `examples=` override and `example_offset`
placement in LLM._format_messages.

These run fully offline: message assembly only touches self.db, so a
minimal fake DB stands in for qdrant + the embedder.
"""

from PBQA.llm import LLM


class FakeDB:
    """Covers exactly the surface _format_messages uses."""

    def __init__(self, base_examples=None, queried_examples=None):
        # Stored newest-first, like db.where(); _format_messages reverses.
        self.base_examples = base_examples or []
        self.queried_examples = queried_examples or []
        self.query_calls = 0

    def get_metadata(self, pattern):
        return {"system_prompt": "You are a test pattern."}

    def where(self, collection_name=None, n=None, base_example=None, **kwargs):
        return list(self.base_examples)

    def query(self, pattern, query_input, n=0, min_d=None, **kwargs):
        self.query_calls += 1
        if n is not None and n < 1:
            return []
        return list(self.queried_examples)[:n]


def doc(tag: str) -> dict:
    return {"input": f"in-{tag}", "response": f"out-{tag}", "metadata": {}}


def make_llm(base=None, queried=None) -> LLM:
    llm = LLM.__new__(LLM)  # skip __init__ (no server/model needed)
    llm.db = FakeDB(base_examples=base, queried_examples=queried)
    return llm


def contents(messages):
    return [m["content"] for m in messages]


# Base docs passed newest-first (b3, b2, b1) so the rendered file order
# after the internal reverse() is b1, b2, b3.
BASE = [doc("b3"), doc("b2"), doc("b1")]


def test_default_retrieval_unchanged():
    llm = make_llm(base=BASE, queried=[doc("ret")])
    msgs = llm._format_messages("p", input="live", n_example=1)
    assert llm.db.query_calls == 1
    assert contents(msgs) == [
        "You are a test pattern.",
        "in-b1", "out-b1", "in-b2", "out-b2", "in-b3", "out-b3",
        "in-ret", "out-ret",
        "live",
    ]


def test_override_skips_retrieval():
    llm = make_llm(base=BASE, queried=[doc("ret")])
    msgs = llm._format_messages("p", input="live", examples=[doc("dyn")])
    assert llm.db.query_calls == 0
    assert contents(msgs) == [
        "You are a test pattern.",
        "in-b1", "out-b1", "in-b2", "out-b2", "in-b3", "out-b3",
        "in-dyn", "out-dyn",
        "live",
    ]


def test_override_empty_list_suppresses_retrieval():
    llm = make_llm(base=BASE, queried=[doc("ret")])
    msgs = llm._format_messages("p", input="live", examples=[], n_example=1)
    assert llm.db.query_calls == 0
    assert contents(msgs) == [
        "You are a test pattern.",
        "in-b1", "out-b1", "in-b2", "out-b2", "in-b3", "out-b3",
        "live",
    ]


def test_offset_places_before_last_base_example():
    llm = make_llm(base=BASE)
    msgs = llm._format_messages(
        "p", input="live", examples=[doc("dyn")], example_offset=1
    )
    assert contents(msgs) == [
        "You are a test pattern.",
        "in-b1", "out-b1", "in-b2", "out-b2",
        "in-dyn", "out-dyn",
        "in-b3", "out-b3",
        "live",
    ]


def test_offset_beyond_base_clamps_to_front():
    llm = make_llm(base=BASE)
    msgs = llm._format_messages(
        "p", input="live", examples=[doc("dyn")], example_offset=99
    )
    assert contents(msgs) == [
        "You are a test pattern.",
        "in-dyn", "out-dyn",
        "in-b1", "out-b1", "in-b2", "out-b2", "in-b3", "out-b3",
        "live",
    ]


def test_offset_without_examples_is_inert():
    llm = make_llm(base=BASE)
    msgs = llm._format_messages("p", input="live", example_offset=1)
    assert contents(msgs) == [
        "You are a test pattern.",
        "in-b1", "out-b1", "in-b2", "out-b2", "in-b3", "out-b3",
        "live",
    ]


def test_multiple_provided_examples_keep_order():
    llm = make_llm(base=BASE)
    msgs = llm._format_messages(
        "p", input="live", examples=[doc("d1"), doc("d2")]
    )
    assert contents(msgs) == [
        "You are a test pattern.",
        "in-b1", "out-b1", "in-b2", "out-b2", "in-b3", "out-b3",
        "in-d1", "out-d1", "in-d2", "out-d2",
        "live",
    ]
