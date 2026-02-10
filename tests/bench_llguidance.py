"""
Benchmark structured generation speed across schema complexity levels.

Usage:
    python -m tests.bench_llguidance [--strict]

Run against the standard llama.cpp server, then swap to the llguidance build
and run again with --strict to compare.
"""

import json
import logging
import os
import sys
from pathlib import Path
from time import time
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB, LLM

logging.basicConfig(level=logging.WARNING)

# Load configuration from environment
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
llm_host = os.getenv("LLM_HOST", "localhost")
llm_port = int(os.getenv("LLM_PORT", "8080"))
strict = "--strict" in sys.argv
iterations = 3


# =============================================================================
# Schemas
# =============================================================================


class CourseOfAction(BaseModel):
    reflection: str
    course_of_action: str


class StepDefinition(BaseModel):
    description: str
    intent: Literal["retrieve", "execute", "reflect"]


class DecomposeSteps(BaseModel):
    steps: List[StepDefinition]


class ControlAttributes(BaseModel):
    brightness: Optional[int] = Field(default=None, ge=0, le=100)
    color_temp: Optional[int] = Field(default=None, ge=2000, le=6500)
    rgb_color: Optional[List[int]] = Field(default=None)
    temperature: Optional[float] = Field(default=None, ge=10, le=35)
    position: Optional[int] = Field(default=None, ge=0, le=100)
    hvac_mode: Optional[Literal["auto", "heat", "cool", "heat_cool", "off"]] = Field(
        default=None
    )
    transition: Optional[int] = Field(default=None, ge=0, le=300)


class HomeAssistantControl(BaseModel):
    entity_id: str
    action: Literal["turn_on", "turn_off", "toggle", "set_value"]
    attributes: Optional[ControlAttributes] = Field(default=None)


# =============================================================================
# Setup
# =============================================================================

db = DB(host=qdrant_host, port=qdrant_port, reset=True)

db.load_pattern(
    schema=CourseOfAction,
    examples="tests/bench_course_of_action.yaml",
    system_prompt=(
        "You are a strategic planning assistant. Given a context summary of "
        "an ongoing task, provide a brief reflection on the current state and "
        "a course of action for what to do next."
    ),
    input_key="context_summary",
)

db.load_pattern(
    schema=DecomposeSteps,
    examples="tests/bench_decompose_steps.yaml",
    system_prompt=(
        "You are a task decomposition assistant. Given an objective and a "
        "breakdown of the task, decompose it into concrete steps. Each step "
        "has a description and an intent: 'retrieve' (gather information), "
        "'execute' (perform action), or 'reflect' (analyze/decide)."
    ),
    input_key="objective",
)

db.load_pattern(
    schema=HomeAssistantControl,
    examples="tests/bench_ha_control.yaml",
    system_prompt=(
        "You are a smart home control assistant. Given a user request about "
        "controlling a device, extract the entity ID, action, and any "
        "control attributes. Use null for attributes that are not mentioned."
    ),
    input_key="request",
)

llm = LLM(db=db, host=llm_host)
llm.connect_model(
    model="qwen-coder",
    port=llm_port,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
    temperature=0,
    store_cache=False,
    strict_schema=strict,
)
llm.link("courseofaction", "qwen-coder")
llm.link("decomposesteps", "qwen-coder")
llm.link("homeassistantcontrol", "qwen-coder")


# =============================================================================
# Benchmark
# =============================================================================

scenarios = [
    (
        "simple",
        "courseofaction",
        (
            "The original objective was to adjust a desk lamp's RGB indexes and check "
            "Utrecht weather to set lamp color conditionally. The weather check completed "
            "successfully: cloudy and cold (2.5C), high of 5C, no precipitation. Based on "
            "cloudy conditions the lamp should be set to blue. However, the lamp adjustment "
            "cannot proceed due to missing device identifier. The user was informed and "
            "asked to provide the lamp's entity ID."
        ),
    ),
    (
        "medium",
        "decomposesteps",
        {
            "objective": (
                "Adjust desk lamp RGB indexes one position to the right, then check "
                "Utrecht weather and set lamp color to red if warm tomorrow, blue otherwise"
            ),
            "breakdown": (
                "Two-part request: 1) adjust RGB lamp settings (move indexes right by "
                "one), 2) weather check and conditional lamp color change. First part is "
                "a device control command with specific adjustment. Second part is "
                "conditional logic: check weather in Utrecht, then set lamp color based "
                "on temperature. 'Warm' is subjective - need to define thresholds. "
                "'Tomorrow' is the day for weather check. Device control and weather "
                "logic are independent but both need to be executed."
            ),
        },
    ),
    (
        "complex",
        "homeassistantcontrol",
        (
            "Turn the standing lamp to an aesthetic blue and purple combination "
            "and set it to 60% brightness with a 3 second transition"
        ),
    ),
]

print(f"strict_schema: {strict}\n")

results = {}
for name, pattern, input in scenarios:
    times = []
    token_counts = []
    failures = 0

    for i in range(iterations):
        try:
            result = llm.ask(input=input, pattern=pattern)
        except Exception as e:
            failures += 1
            print(f"  [{name}] iter {i + 1}/{iterations}: FAILED - {e}")
            continue

        meta = result["metadata"]
        t = meta["total_time"]
        tokens = meta.get("completion_tokens", 0)
        times.append(t)
        token_counts.append(tokens)

        tps = tokens / t if t > 0 else 0
        print(f"  [{name}] iter {i + 1}/{iterations}: {t:.2f}s, {tokens} tokens, {tps:.1f} tok/s")

    avg_time = sum(times) / len(times) if times else 0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    avg_tps = avg_tokens / avg_time if avg_time > 0 else 0

    results[name] = {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "avg_tps": avg_tps,
        "failures": failures,
    }

# Summary
print("\n" + "=" * 72)
print(f"{'Schema':<12} {'Avg Time':>10} {'Avg Tokens':>12} {'Avg tok/s':>12} {'Failures':>10}")
print("-" * 72)
for name, data in results.items():
    print(
        f"{name:<12} {data['avg_time']:>9.2f}s {data['avg_tokens']:>12.0f} "
        f"{data['avg_tps']:>12.1f} {data['failures']:>10}"
    )
print("=" * 72)

# Cleanup
db.delete_collection("courseofaction")
db.delete_collection("decomposesteps")
db.delete_collection("homeassistantcontrol")
