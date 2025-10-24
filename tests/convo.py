import json
import logging
import os
import sys
from pathlib import Path
from random import random

from dotenv import load_dotenv
from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB, LLM  # run with python -m tests.convo

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("TEST_LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
log = logging.getLogger()


class OpenReply(BaseModel):
    reply: str


messages = [
    "Hey there",
    "My day is going well",
    "Actually, what's 1 + 1?",
    "Now what's the capital of France?",
]

# Load configuration from environment
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
reset_db = os.getenv("TEST_RESET_DB", "true").lower() == "true"
llm_host = os.getenv("LLM_HOST", "localhost")
llm_port = int(os.getenv("LLM_PORT", 8080))

db = DB(host=qdrant_host, port=qdrant_port, reset=reset_db)
collections = db.get_collections()
if "openreply" in collections:
    db.delete_collection("openreply")
if "convo" in collections:
    db.delete_collection("convo")
db.load_pattern(
    schema=OpenReply,
    system_prompt="You are a virtual assistant. You are here to help where you can or simply engage in conversation.",
)
db.create_collection(
    "convo",
    metadata={"additional_metadata": "test"},
)

llm = LLM(db=db, host=llm_host)
llm.connect_model(
    model="llama",
    port=llm_port,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
)

for message in messages:
    log.info(f"Sending message: {message}")
    exchange = llm.ask(
        input=message,
        pattern="openreply",
        model="llama",
        history_name="convo",
        n_hist=50,
        log_input=True,
    )
    assert (
        exchange["response"]["reply"] != ""
    ), f"Expected a response, got {exchange['reply']}"

    n = db.n("convo")
    db.add(
        input=message,
        collection_name="convo",
        rand_nummer=random(),
        **exchange["response"],
    )
    assert db.n("convo") == n + 1, f"Expected {n + 1} entries, got {db.n('convo')}"

    log.info(f"Response: {exchange['response']['reply']}")

history = db.where(collection_name="convo")
history.reverse()
log.info(f"History:\n{json.dumps(history, indent=4)}")

ordered_history = sorted(history, key=lambda x: x["time_added"])
log.info(f"Ordered History:\n{json.dumps(ordered_history, indent=4)}")

assert (
    history == ordered_history
), f"Expected the history to be sorted by time_added, got {[exchange['time_added'] for exchange in history]}"

log.info("All tests passed")
db.delete_collection("openreply")
db.delete_collection("convo")
