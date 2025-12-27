import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB  # run with python -m tests.upsert_delete

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("TEST_LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
log = logging.getLogger()


def test_add_with_custom_id():
    """Test adding documents with custom IDs"""
    log.info("Testing add() with custom doc_id...")

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    reset_db = os.getenv("TEST_RESET_DB", "true").lower() == "true"

    db = DB(host=qdrant_host, port=qdrant_port, reset=reset_db)
    db.create_collection("test_custom_id")

    # Test 1: Add with custom UUID ID
    custom_uuid = "12345678-1234-1234-1234-123456789abc"
    doc1 = db.add(
        input="First document",
        collection_name="test_custom_id",
        doc_id=custom_uuid,
        metadata_field="value1"
    )

    assert doc1["id"] == custom_uuid, f"Expected '{custom_uuid}', got {doc1['id']}"
    assert doc1["input"] == "First document"
    log.info(f"Added document with custom UUID: {doc1['id']}")

    # Test 2: Add without custom ID (should auto-generate UUID)
    doc2 = db.add(
        input="Second document",
        collection_name="test_custom_id",
        metadata_field="value2"
    )

    assert "id" in doc2, "Document should have an ID field"
    assert len(doc2["id"]) == 36, "Auto-generated ID should be a UUID (36 chars)"
    log.info(f"Added document with auto-generated UUID: {doc2['id']}")

    # Verify collection has 2 documents
    assert db.n("test_custom_id") == 2, f"Expected 2 documents, got {db.n('test_custom_id')}"

    db.delete_collection("test_custom_id")
    log.info("test_add_with_custom_id passed")


def test_upsert():
    """Test upsert functionality (insert and update)"""
    log.info("Testing upsert() functionality...")

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    reset_db = os.getenv("TEST_RESET_DB", "true").lower() == "true"

    db = DB(host=qdrant_host, port=qdrant_port, reset=reset_db)
    db.create_collection("test_upsert")

    # Test 1: Insert new document with upsert
    test_uuid = "aaaaaaaa-bbbb-cccc-dddd-111111111111"
    doc1 = db.upsert(
        input="Original content",
        collection_name="test_upsert",
        doc_id=test_uuid,
        version=1
    )

    assert doc1["id"] == test_uuid
    assert doc1["input"] == "Original content"
    assert doc1["version"] == 1
    log.info(f"Upserted (inserted) document: {doc1['id']}")

    # Verify collection has 1 document
    assert db.n("test_upsert") == 1, f"Expected 1 document, got {db.n('test_upsert')}"

    # Test 2: Update existing document with upsert
    doc2 = db.upsert(
        input="Updated content",
        collection_name="test_upsert",
        doc_id=test_uuid,  # Same ID as before
        version=2
    )

    assert doc2["id"] == test_uuid
    assert doc2["input"] == "Updated content"
    assert doc2["version"] == 2
    log.info(f"Upserted (updated) document: {doc2['id']}")

    # Verify collection still has only 1 document (updated, not added)
    assert db.n("test_upsert") == 1, f"Expected 1 document after update, got {db.n('test_upsert')}"

    # Test 3: Query to verify the document was actually updated
    results = db.query("test_upsert", "Updated content", n=1)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0]["input"] == "Updated content"
    assert results[0]["version"] == 2
    log.info("Query confirmed document was updated")

    db.delete_collection("test_upsert")
    log.info("test_upsert passed")


def test_delete():
    """Test delete functionality"""
    log.info("Testing delete() functionality...")

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    reset_db = os.getenv("TEST_RESET_DB", "true").lower() == "true"

    db = DB(host=qdrant_host, port=qdrant_port, reset=reset_db)
    db.create_collection("test_delete")

    # Add multiple documents with valid UUIDs
    doc1_id = "11111111-1111-1111-1111-111111111111"
    doc2_id = "22222222-2222-2222-2222-222222222222"
    doc3_id = "33333333-3333-3333-3333-333333333333"

    db.add(input="Document 1", collection_name="test_delete", doc_id=doc1_id, tag="keep")
    db.add(input="Document 2", collection_name="test_delete", doc_id=doc2_id, tag="delete")
    db.add(input="Document 3", collection_name="test_delete", doc_id=doc3_id, tag="keep")

    # Verify we have 3 documents
    assert db.n("test_delete") == 3, f"Expected 3 documents, got {db.n('test_delete')}"
    log.info("Added 3 documents")

    # Test 1: Delete one document
    db.delete(collection_name="test_delete", doc_id=doc2_id)
    log.info(f"Deleted document: {doc2_id}")

    # Verify we now have 2 documents
    assert db.n("test_delete") == 2, f"Expected 2 documents after delete, got {db.n('test_delete')}"

    # Test 2: Verify the correct document was deleted
    results = db.query("test_delete", "Document", n=10)
    remaining_ids = {doc["id"] for doc in results}

    assert doc1_id in remaining_ids, f"Document {doc1_id} should still exist"
    assert doc2_id not in remaining_ids, f"Document {doc2_id} should be deleted"
    assert doc3_id in remaining_ids, f"Document {doc3_id} should still exist"
    log.info("Verified correct document was deleted")

    # Test 3: Delete remaining documents
    db.delete(collection_name="test_delete", doc_id=doc1_id)
    db.delete(collection_name="test_delete", doc_id=doc3_id)

    # Verify collection is now empty
    assert db.n("test_delete") == 0, f"Expected 0 documents, got {db.n('test_delete')}"
    log.info("All documents deleted successfully")

    db.delete_collection("test_delete")
    log.info("test_delete passed")


def test_delete_error_handling():
    """Test delete error handling with invalid collections"""
    log.info("Testing delete() error handling...")

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    reset_db = os.getenv("TEST_RESET_DB", "true").lower() == "true"

    db = DB(host=qdrant_host, port=qdrant_port, reset=reset_db)

    # Test: Try to delete from non-existent collection
    try:
        db.delete(collection_name="nonexistent_collection", doc_id="some-id")
        assert False, "Should have raised ValueError for non-existent collection"
    except ValueError as e:
        assert "not found in collections" in str(e)
        log.info(f"Correctly raised ValueError: {e}")

    log.info("test_delete_error_handling passed")


def test_upsert_delete_workflow():
    """Test a complete workflow with upsert and delete operations"""
    log.info("Testing complete upsert/delete workflow...")

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    reset_db = os.getenv("TEST_RESET_DB", "true").lower() == "true"

    db = DB(host=qdrant_host, port=qdrant_port, reset=reset_db)
    db.create_collection("test_workflow")

    # Workflow: Simulate a document lifecycle
    doc_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"

    # 1. Create initial version
    db.upsert(
        input="Version 1 content",
        collection_name="test_workflow",
        doc_id=doc_id,
        version=1,
        status="draft"
    )
    log.info("Created version 1")

    # 2. Update to version 2
    db.upsert(
        input="Version 2 content - reviewed",
        collection_name="test_workflow",
        doc_id=doc_id,
        version=2,
        status="reviewed"
    )
    log.info("Updated to version 2")

    # 3. Update to version 3 (published)
    db.upsert(
        input="Version 3 content - published",
        collection_name="test_workflow",
        doc_id=doc_id,
        version=3,
        status="published"
    )
    log.info("Updated to version 3")

    # Verify still only 1 document
    assert db.n("test_workflow") == 1, f"Expected 1 document, got {db.n('test_workflow')}"

    # Query and verify latest version
    results = db.query("test_workflow", "content", n=1)
    assert results[0]["version"] == 3
    assert results[0]["status"] == "published"
    log.info("Verified latest version is v3")

    # 4. Delete the document
    db.delete(collection_name="test_workflow", doc_id=doc_id)
    log.info("Deleted document")

    # Verify collection is empty
    assert db.n("test_workflow") == 0, f"Expected 0 documents, got {db.n('test_workflow')}"

    db.delete_collection("test_workflow")
    log.info("test_upsert_delete_workflow passed")


def test_upsert_with_schema():
    """Test upsert functionality with schema collections"""
    log.info("Testing upsert() with schema collections...")

    from pydantic import BaseModel

    class TestSchema(BaseModel):
        result: str
        confidence: float

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    reset_db = os.getenv("TEST_RESET_DB", "true").lower() == "true"

    db = DB(host=qdrant_host, port=qdrant_port, reset=reset_db)

    # Load pattern with schema
    db.load_pattern(
        schema=TestSchema,
        input_key="query",
        collection_name="test_schema_upsert"
    )

    # Test 1: Initial insert with upsert
    schema_doc_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    doc1 = db.upsert(
        input={"query": "test query"},
        collection_name="test_schema_upsert",
        doc_id=schema_doc_id,
        result="initial result",
        confidence=0.7
    )

    assert doc1["metadata"]["id"] == schema_doc_id
    assert doc1["response"]["result"] == "initial result"
    assert doc1["response"]["confidence"] == 0.7
    log.info("Upserted schema document (insert)")

    # Test 2: Update with upsert
    doc2 = db.upsert(
        input={"query": "test query"},
        collection_name="test_schema_upsert",
        doc_id=schema_doc_id,  # Same ID
        result="updated result",
        confidence=0.9
    )

    assert doc2["metadata"]["id"] == schema_doc_id
    assert doc2["response"]["result"] == "updated result"
    assert doc2["response"]["confidence"] == 0.9
    log.info("Upserted schema document (update)")

    # Verify only 1 document exists
    assert db.n("test_schema_upsert") == 1, f"Expected 1 document, got {db.n('test_schema_upsert')}"

    db.delete_collection("test_schema_upsert")
    log.info("test_upsert_with_schema passed")


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PBQA Upsert/Delete Functionality Test Suite")
    log.info("=" * 60)
    log.info(f"Qdrant: {os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', 6333)}")
    log.info("=" * 60)

    try:
        test_add_with_custom_id()
        test_upsert()
        test_delete()
        test_delete_error_handling()
        test_upsert_delete_workflow()
        test_upsert_with_schema()

        log.info("=" * 60)
        log.info("ALL TESTS PASSED! Upsert/Delete functionality is working correctly.")
        log.info("=" * 60)
    except Exception as e:
        log.error(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
