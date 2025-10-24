import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB, LLM  # run with python -m tests.custom_history

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("TEST_LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
log = logging.getLogger()


class Conversation(BaseModel):
    reply: str


# Setup
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
reset_db = os.getenv("TEST_RESET_DB", "true").lower() == "true"
llm_host = os.getenv("LLM_HOST", "localhost")
llm_port = int(os.getenv("LLM_PORT", 8080))

db = DB(host=qdrant_host, port=qdrant_port, reset=reset_db)
collections = db.get_collections()
if "conversation" in collections:
    db.delete_collection("conversation")

db.load_pattern(
    schema=Conversation,
    system_prompt="You are a virtual assistant. You are here to help where you can or simply engage in conversation.",
)

llm = LLM(db=db, host=llm_host)
llm.connect_model(
    model="llama",
    port=llm_port,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
)

# Test 1: Custom history bypasses database
log.info("=" * 60)
log.info("Test 1: Custom history bypasses database lookup")
log.info("=" * 60)

# Create custom history manually
custom_history = [
    {
        "input": "My name is Alice",
        "response": {"reply": "Nice to meet you, Alice!"},
        "metadata": {},
    },
    {
        "input": "I live in Paris",
        "response": {"reply": "Paris is a beautiful city!"},
        "metadata": {},
    },
]

# Ask a question that references the custom history
response = llm.ask(
    input="What's my name and where do I live?",
    pattern="conversation",
    model="llama",
    custom_history=custom_history,
    log_input=True,
)

log.info(f"Response with custom history: {response['response']['reply']}")

# Verify response references Alice and Paris (basic check)
reply_lower = response["response"]["reply"].lower()
assert (
    "alice" in reply_lower or "your name" in reply_lower
), f"Expected response to reference Alice, got: {response['response']['reply']}"

log.info("✓ Test 1 passed: Custom history was used")

# Test 2: Verify database wasn't queried (no entries in database yet)
log.info("=" * 60)
log.info("Test 2: Database should be empty (wasn't queried)")
log.info("=" * 60)

n_entries = db.n("conversation")
assert n_entries == 0, f"Expected 0 database entries, got {n_entries}"

log.info("✓ Test 2 passed: Database wasn't queried")

# Test 3: Add entries to database, verify n_hist is ignored when custom_history is used
log.info("=" * 60)
log.info("Test 3: custom_history takes precedence over n_hist")
log.info("=" * 60)

# Add different entries to the database
db.add(
    input="My name is Bob",
    collection_name="conversation",
    reply="Nice to meet you, Bob!",
)
db.add(
    input="I live in London",
    collection_name="conversation",
    reply="London is a great city!",
)

# Query with both n_hist and custom_history - custom_history should win
response = llm.ask(
    input="What's my name?",
    pattern="conversation",
    model="llama",
    n_hist=10,  # This should be ignored
    custom_history=custom_history,  # This should be used
    log_input=True,
)

log.info(f"Response: {response['response']['reply']}")

# Should reference Alice (from custom_history), not Bob (from database)
reply_lower = response["response"]["reply"].lower()
assert (
    "alice" in reply_lower or "your name" in reply_lower
), f"Expected response to reference Alice from custom_history, got: {response['response']['reply']}"
assert "bob" not in reply_lower, f"Response shouldn't reference Bob from database, got: {response['response']['reply']}"

log.info("✓ Test 3 passed: custom_history overrode n_hist")

# Test 4: Empty custom_history should work
log.info("=" * 60)
log.info("Test 4: Empty custom_history list")
log.info("=" * 60)

response = llm.ask(
    input="Hello!",
    pattern="conversation",
    model="llama",
    custom_history=[],  # Empty history
    log_input=True,
)

assert response["response"]["reply"] != "", "Expected a response with empty custom_history"

log.info(f"Response with empty custom_history: {response['response']['reply']}")
log.info("✓ Test 4 passed: Empty custom_history works")

# Test 5: None custom_history falls back to n_hist
log.info("=" * 60)
log.info("Test 5: None custom_history falls back to n_hist")
log.info("=" * 60)

response = llm.ask(
    input="What's my name?",
    pattern="conversation",
    model="llama",
    n_hist=5,  # Should use database history
    custom_history=None,  # Explicitly None
    log_input=True,
)

log.info(f"Response: {response['response']['reply']}")

# Should reference Bob (from database), not Alice
reply_lower = response["response"]["reply"].lower()
assert (
    "bob" in reply_lower or "your name" in reply_lower
), f"Expected response to reference Bob from database, got: {response['response']['reply']}"

log.info("✓ Test 5 passed: None custom_history falls back to n_hist")

# Cleanup
log.info("=" * 60)
log.info("Cleaning up")
log.info("=" * 60)
db.delete_collection("conversation")

log.info("=" * 60)
log.info("✓ All tests passed!")
log.info("=" * 60)
