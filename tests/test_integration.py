"""Runs the integration scripts as one matrix. Requires live services.

Skipped unless PBQA_INTEGRATION=1, since every script talks to a real Qdrant
and (mostly) a real inference server. DB(reset=True) WIPES the target Qdrant,
so point QDRANT_HOST/QDRANT_PORT at a throwaway instance:

    docker run -d --rm --name pbqa-test-qdrant -p 6335:6333 qdrant/qdrant
    PBQA_INTEGRATION=1 QDRANT_HOST=localhost QDRANT_PORT=6335 \\
        LLM_HOST=... LLM_PORT=... RERANK_HOST=... RERANK_PORT=... \\
        python -m pytest tests/test_integration.py -v

Each script is a module run in its own process, exactly as documented in
tests/README.md — this only collects them into a single command.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

MODULES = [
    "upsert_delete",
    "non_pattern_collection",
    "nested_paths",
    "custom_history",
    "convo",
    "tool_use",
    "narrative_breakdown",
    "rerank",
]

pytestmark = pytest.mark.skipif(
    os.getenv("PBQA_INTEGRATION") != "1",
    reason="live services required; set PBQA_INTEGRATION=1 to run",
)


@pytest.mark.parametrize("module", MODULES)
def test_integration_script(module):
    result = subprocess.run(
        [sys.executable, "-m", f"tests.{module}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"tests.{module} exited {result.returncode}\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
    )
