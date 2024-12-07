import logging
import os
import sys
from pathlib import Path

from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB, LLM  # run with python -m tests.convo

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class Conversation(BaseModel):
    reply: str


messages = [
    "Hey there",
    "My day is going well",
    "Actually, what's 1 + 1?",
    "Now what's the capital of France?",
]

db = DB(host="localhost", port=6333, reset=True)
db.load_pattern(
    schema=Conversation,
    system_prompt="You are a virtual assistant. You are here to help where you can or simply engage in conversation.",
)

llm = LLM(db=db, host="192.168.0.91")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
)

for message in messages:
    log.info(f"Sending message: {message}")
    exchange = llm.ask(
        input=message,
        pattern="conversation",
        model="llama",
        n_hist=50,
    )
    assert (
        exchange["response"]["reply"] != ""
    ), f"Expected a response, got {exchange['reply']}"

    n = db.n("conversation")
    db.add(
        input=message,
        collection_name="conversation",
        **exchange["response"],
    )
    assert (
        db.n("conversation") == n + 1
    ), f"Expected {n + 1} entries, got {db.n('conversation')}"

    log.info(f"Response: {exchange['response']['reply']}")

history = db.where(collection_name="conversation")
assert history == sorted(
    history, key=lambda x: x["metadata"]["time_added"], reverse=True
), f"Expected the history to be sorted by time_added, got {history}"

log.info("All tests passed")
db.delete_collection("conversation")
