import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB  # run with python -m tests.non_pattern_collection

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

entries = [
    {
        "input": "What is the capital of France?",
        "text": "Paris",
        "country": "France",
    },
    {
        "input": "What is the capital of Germany?",
        "text": "Berlin",
        "country": "Germany",
    },
]

db = DB(host="localhost", port=6333, reset=True)
db.create_collection("non_pattern_collection")
assert db.get_collections() == ["non_pattern_collection"]

for entry in entries:
    db.add(**entry, collection_name="non_pattern_collection")
    log.info(f"Added entry: {entry}")

assert (
    db.n("non_pattern_collection") == 2
), f"Expected 2 entries, got {db.n('non_pattern_collection')}"

response = db.query(
    "non_pattern_collection",
    "What's France's capital?",
    n=1,
)[0]
assert (
    response["input"] == "What is the capital of France?"
), f"Expected 'What is the capital of France?', got {response['input']}"
assert response["text"] == "Paris", f"Expected 'Paris', got {response['text']}"
log.info(f"Query successful: {response['text']}")

db.index("non_pattern_collection", "country", "keyword")
response = db.where(
    "non_pattern_collection",
    n=1,
    country="France",
)[0]
assert (
    response["input"] == "What is the capital of France?"
), f"Expected 'What is the capital of France?', got {response['input']}"
assert response["text"] == "Paris", f"Expected 'Paris', got {response['text']}"

log.info(f"All tests passed")
db.delete_collection("non_pattern_collection")
