import json
import logging
import os
import sys
from pathlib import Path

from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB  # run with python -m tests.nested_paths

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class NestedResponse(BaseModel):
    response: str


class CourseOfAction(BaseModel):
    thought: str
    objective: str


# Test nested path resolution functionality
def test_path_resolution():
    """Test the path resolution functionality directly"""
    from PBQA.db import resolve_path, path_exists
    
    log.info("Testing path resolution functionality...")
    
    # Test data structure similar to CourseOfAction
    test_data = {
        "query": "simple key",
        "user": {
            "name": "John",
            "history": [
                {"input": "Research neighborhoods in Denver", "thought": "initial assessment"},
                {"input": "Check weather forecast", "objective": "weather info"},
            ]
        },
        "history": [
            {"input": "msg1", "output": "response1"},
            {"input": "msg2", "output": "response2"},
        ],
        "todo": ["Look up gyms", "Create checklist", "Research schools"]
    }
    
    # Test cases covering all scenarios
    test_cases = [
        # Backward compatibility
        ("query", "simple key"),
        
        # Dot notation
        ("user.name", "John"),
        
        # Array indexing
        ("history[0]", {"input": "msg1", "output": "response1"}),
        ("todo[0]", "Look up gyms"),
        ("todo[-1]", "Research schools"),
        
        # CourseOfAction scenarios
        ("user.history[0].input", "Research neighborhoods in Denver"),
        ("user.history[-1].objective", "weather info"),
        ("history[0].input", "msg1"),
        ("history[-1].output", "response2"),
    ]
    
    for path, expected in test_cases:
        result = resolve_path(test_data, path)
        assert result == expected, f"Path {path}: expected {expected}, got {result}"
        log.info(f"Path resolution: {path} -> {result}")
    
    # Test path existence
    assert path_exists(test_data, "user.history[0].input")
    assert not path_exists(test_data, "nonexistent.path")
    assert not path_exists(test_data, "history[10]")
    
    log.info("Path resolution tests passed")


def test_db_integration():
    """Test nested paths with actual DB operations"""
    log.info("Testing DB integration with nested paths...")
    
    try:
        db = DB(host="localhost", port=6333, reset=True)
    except ValueError as e:
        if "Failed to connect to Qdrant server" in str(e):
            log.warning("Skipping DB integration tests - Qdrant server not available")
            return
        raise
    
    # Clean up any existing collections
    collections = db.get_collections()
    for collection in ["nestedresponse", "courseofaction", "simple_test"]:
        if collection in collections:
            db.delete_collection(collection)
    
    # Test 1: Backward compatibility with simple keys
    log.info("Test 1: Simple key (backward compatibility)")
    db.load_pattern(
        schema=NestedResponse,
        input_key="query",
        collection_name="simple_test"
    )
    
    doc = db.add(
        input={"query": "test message"},
        collection_name="simple_test",
        response="test response"
    )
    assert doc["metadata"]["embedding_text"] == "test message"
    log.info("Simple key works")
    
    # Test 2: Nested dot notation
    log.info("Test 2: Nested dot notation")
    db.load_pattern(
        schema=NestedResponse,
        input_key="user.query",
        collection_name="nestedresponse"
    )
    
    nested_input = {
        "user": {"query": "nested message"},
        "metadata": {"timestamp": "2024-01-01"}
    }
    
    doc = db.add(
        input=nested_input,
        collection_name="nestedresponse",
        response="nested response"
    )
    assert doc["metadata"]["embedding_text"] == "nested message"
    log.info("Nested dot notation works")
    
    # Test 3: CourseOfAction scenario
    log.info("Test 3: CourseOfAction scenario")
    db.load_pattern(
        schema=CourseOfAction,
        input_key="history[0].input",
        collection_name="courseofaction"
    )
    
    # Simulate AVA CourseOfAction structure
    courseofaction_input = {
        "history": [
            {"input": "Research neighborhoods in Denver", "thought": "User planning move"},
            {"thought": "analysis complete", "objective": "weather check", "state": "completed"}
        ],
        "todo": ["Look up gyms", "Create checklist"]
    }
    
    doc = db.add(
        input=courseofaction_input,
        collection_name="courseofaction",
        thought="I need to help with the research",
        objective="Provide neighborhood information"
    )
    assert doc["metadata"]["embedding_text"] == "Research neighborhoods in Denver"
    log.info("CourseOfAction scenario works")
    
    # Test 4: Array indexing with negative indices
    log.info("Test 4: Array indexing with negative indices")
    db.load_pattern(
        schema=NestedResponse,
        input_key="messages[-1]",
        collection_name="array_test"
    )
    
    array_input = {
        "messages": ["first", "second", "last message"],
        "count": 3
    }
    
    doc = db.add(
        input=array_input,
        collection_name="array_test",
        response="processed last message"
    )
    assert doc["metadata"]["embedding_text"] == "last message"
    log.info("Array indexing with negative indices works")
    
    # Test 5: Error handling
    log.info("Test 5: Error handling")
    try:
        db.add(
            input={"wrong": "structure"},
            collection_name="courseofaction",  # Expects history[0].input
            thought="should fail",
            objective="error test"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "history[0].input" in str(e)
        log.info("Error handling works correctly")
    
    # Clean up
    for collection in ["nestedresponse", "courseofaction", "simple_test", "array_test"]:
        if collection in db.get_collections():
            db.delete_collection(collection)
    
    log.info("DB integration tests passed")


def test_complex_courseofaction():
    """Test the exact CourseOfAction use case mentioned in the issue"""
    log.info("Testing complex CourseOfAction structure...")
    
    try:
        db = DB(host="localhost", port=6333, reset=True)
    except ValueError as e:
        if "Failed to connect to Qdrant server" in str(e):
            log.warning("Skipping complex CourseOfAction tests - Qdrant server not available")
            return
        raise
    
    # Clean up
    if "complex_coa" in db.get_collections():
        db.delete_collection("complex_coa")
    
    # Load pattern with nested path
    db.load_pattern(
        schema=CourseOfAction,
        input_key="user.history[0].input",
        collection_name="complex_coa"
    )
    
    # Complex nested structure as described in the issue
    complex_input = {
        "user": {
            "history": [
                {"input": "Research neighborhoods in Denver...", "thought": "User is planning a major life transition..."},
                {"input": "What about schools?", "thought": "Education is important"}
            ],
            "preferences": {"location": "Denver", "budget": 500000}
        },
        "history": [
            {"thought": "analysis thought", "objective": "neighborhood research", "state": "completed", "results": {"found": 5}},
            {"thought": "school analysis", "objective": "education options", "state": "in_progress"}
        ],
        "todo": ["Look up local gyms", "Create moving checklist", "Research schools"]
    }
    
    # Add document
    doc = db.add(
        input=complex_input,
        collection_name="complex_coa",
        thought="I need to help with Denver research",
        objective="Provide comprehensive neighborhood information"
    )
    
    # Verify the correct input was extracted for embedding
    assert doc["metadata"]["embedding_text"] == "Research neighborhoods in Denver..."
    log.info("Complex CourseOfAction structure works")
    
    # Test that we can also access other paths if needed
    from PBQA.db import resolve_path
    
    # Test other potential access patterns
    assert resolve_path(complex_input, "history[-1].objective") == "education options"
    assert resolve_path(complex_input, "todo[0]") == "Look up local gyms"
    assert resolve_path(complex_input, "user.preferences.location") == "Denver"
    
    log.info("All CourseOfAction path access patterns work")
    
    # Clean up
    db.delete_collection("complex_coa")


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PBQA Nested Path Resolution Test Suite")
    log.info("=" * 60)
    
    try:
        test_path_resolution()
        test_db_integration()
        test_complex_courseofaction()
        
        log.info("=" * 60)
        log.info("ALL TESTS PASSED! Nested path resolution is working correctly.")
        log.info("CourseOfAction patterns are now supported!")
        log.info("=" * 60)
    except Exception as e:
        log.error(f"TEST FAILED: {e}")
        raise