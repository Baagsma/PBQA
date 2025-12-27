import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB  # run with python -m tests.non_pattern_collection

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("TEST_LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
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

# Load configuration from environment
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
reset_db = os.getenv("TEST_RESET_DB", "true").lower() == "true"

db = DB(host=qdrant_host, port=qdrant_port, reset=reset_db)
db.create_collection("non_pattern_collection")
assert db.get_collections() == ["non_pattern_collection"]

for entry in entries:
    db.add(**entry, collection_name="non_pattern_collection")
    log.info(f"Added entry: {entry}")

assert (
    db.n("non_pattern_collection") == 2
), f"Expected 2 entries, got {db.n('non_pattern_collection')}"

exchange = db.query(
    "non_pattern_collection",
    "What's France's capital?",
    n=1,
)[0]
assert (
    exchange["input"] == "What is the capital of France?"
), f"Expected 'What is the capital of France?', got {exchange['input']}"
assert exchange["text"] == "Paris", f"Expected 'Paris', got {exchange['text']}"
log.info(f"Query successful: {exchange['text']}")

db.index("non_pattern_collection", "country", "keyword")
whole_exchange = db.where(
    "non_pattern_collection",
    n=1,
    country="France",
)
log.info(f"Where successful:\n{json.dumps(whole_exchange, indent=4)}")
exchange = whole_exchange[0]
assert (
    exchange["input"] == "What is the capital of France?"
), f"Expected 'What is the capital of France?', got {exchange['input']}"
assert exchange["text"] == "Paris", f"Expected 'Paris', got {exchange['text']}"

log.info(f"All tests passed")
db.delete_collection("non_pattern_collection")
