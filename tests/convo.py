import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB, LLM  # run with python -m tests.convo

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

messages = [
    "Hey there",
    "My day is going well",
    "Actually, what's 1 + 1?",
    "Now what's the capital of France?",
]

db = DB(host="localhost", port=6333, reset=True)
db.load_pattern("examples/conversation.yaml")

llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
)

for message in messages:
    log.info(f"Sending message: {message}")
    response = llm.ask(
        input=message,
        pattern="conversation",
        model="llama",
        n_hist=50,
    )
    assert response["reply"] != "", f"Expected a response, got {response['reply']}"

    n = db.n("conversation")
    db.add(
        input=message,
        collection_name="conversation",
        **response,
    )
    assert (
        db.n("conversation") == n + 1
    ), f"Expected {n + 1} entries, got {db.n('conversation')}"

    log.info(f"Response: {response['reply']}")

history = db.where(collection_name="conversation")
assert history == sorted(
    history, key=lambda x: x["time_added"], reverse=True
), f"Expected the history to be sorted by time_added, got {history}"

log.info("All tests passed")
db.delete_collection("conversation")
