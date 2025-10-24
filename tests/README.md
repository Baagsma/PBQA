# PBQA Tests

This directory contains test files for the PBQA library functionality.

## Setup

### 1. Environment Configuration

Tests use environment variables for configuration. Copy the `.env.example` file to `.env` in the project root:

```bash
cp ../.env.example ../.env
```

Then edit `.env` to match your environment:

```bash
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LLM Configuration (for tests that use LLM)
LLM_HOST=localhost
LLM_PORT=8080

# Rerank Configuration (for rerank tests)
RERANK_HOST=localhost
RERANK_PORT=8090

# Test Configuration
TEST_RESET_DB=true
TEST_LOG_LEVEL=INFO
```

### 2. Prerequisites

Make sure you have:
- **Qdrant server** running (required for most tests)
  - Default: `localhost:6333`
  - See [Qdrant installation](https://qdrant.tech/documentation/quick-start/)
- **llama.cpp server** running (required for LLM tests)
  - Default: `localhost:8080`
  - See [llama.cpp server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
- **Rerank model server** running (required for rerank tests only)
  - Default: `localhost:8090` (can be same as LLM server or separate)
  - Configure with `RERANK_HOST` and `RERANK_PORT` if using a different endpoint

### 3. Install Dependencies

```bash
pip install -e .
```

This will install `python-dotenv` along with other dependencies.

## Running Tests

### Run Individual Tests

```bash
# Test upsert and delete functionality
python -m tests.upsert_delete

# Test non-pattern collections
python -m tests.non_pattern_collection

# Test nested path resolution
python -m tests.nested_paths

# Test custom history
python -m tests.custom_history

# Test conversation patterns
python -m tests.convo

# Test tool use patterns
python -m tests.tool_use

# Test narrative breakdown
python -m tests.narrative_breakdown

# Test reranking
python -m tests.rerank
```

### Run All Tests

```bash
# From project root
for test in tests/*.py; do
    python -m "tests.$(basename $test .py)" || exit 1
done
```

## Test Files

### Core Functionality Tests

- **`upsert_delete.py`** - Tests for upsert and delete operations
  - Custom document IDs (UUID format)
  - Upsert (insert/update) behavior
  - Delete by ID
  - Schema collection upsert
  - Complete workflow testing
  - **Note:** Document IDs must be valid UUIDs or unsigned integers (Qdrant requirement)

- **`non_pattern_collection.py`** - Tests for non-pattern collections
  - Basic add/query operations
  - Indexing and filtering

- **`nested_paths.py`** - Tests for nested path resolution
  - Dot notation (`user.query`)
  - Array indexing (`history[0]`, `history[-1]`)
  - Complex nested access (`user.history[0].input`)

### Advanced Feature Tests

- **`custom_history.py`** - Tests for custom history handling
- **`convo.py`** - Conversation pattern tests
- **`tool_use.py`** - Tool use pattern tests
- **`narrative_breakdown.py`** - Narrative breakdown tests
- **`rerank.py`** - Reranking functionality tests

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `LLM_HOST` | `localhost` | LLM server hostname |
| `LLM_PORT` | `8080` | LLM server port |
| `RERANK_HOST` | `localhost` | Rerank model server hostname |
| `RERANK_PORT` | `8090` | Rerank model server port |
| `TEST_RESET_DB` | `true` | Reset database before tests |
| `TEST_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Troubleshooting

### "Failed to connect to Qdrant server"

Make sure Qdrant is running:
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or check if it's running
curl http://localhost:6333/collections
```

### Tests are not using .env values

Make sure:
1. The `.env` file exists in the project root (not in the tests directory)
2. `python-dotenv` is installed: `pip install python-dotenv`
3. The `.env` file has the correct format (no quotes around values needed)

### Database not resetting between tests

Set `TEST_RESET_DB=true` in your `.env` file, or run with explicit reset:
```python
db = DB(host="localhost", port=6333, reset=True)
```

## Contributing

When adding new tests:
1. Import and use `dotenv.load_dotenv()` at the top
2. Load configuration from environment variables with sensible defaults
3. Use the logging configuration from `TEST_LOG_LEVEL`
4. Clean up any collections/resources created during the test
5. Add documentation to this README
