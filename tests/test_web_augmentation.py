"""
Tests for augmentation API routes and service.

This module tests the dataset augmentation functionality including:
- File upload and field extraction
- Field validation
- Augmentation preview and full processing
- Download functionality
- Error handling

Requirements satisfied:
- 2.1: Upload and validate dataset files
- 2.2: Validate target field exists in dataset
- 2.3: Preview augmentation before full processing
- 2.4: Provide download link for augmented dataset
- 2.5: Preserve original data on failure
"""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from httpx._transports.asgi import ASGITransport
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ollaforge.models import OutputLanguage
from ollaforge.web.server import app
from ollaforge.web.services.augmentation import AugmentationService


def create_test_client():
    """Create an async test client for the FastAPI app using httpx with ASGITransport."""
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def augmentation_service():
    """Create an augmentation service instance."""
    return AugmentationService()


# ============================================================================
# Strategies for Property-Based Testing
# ============================================================================

# Strategy for valid field names (simple ASCII identifiers)
field_name_strategy = st.text(
    alphabet='abcdefghijklmnopqrstuvwxyz_',
    min_size=1,
    max_size=20
).filter(lambda x: x[0].isalpha())

# Strategy for valid field values
field_value_strategy = st.text(
    alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-',
    min_size=1,
    max_size=100
).filter(lambda x: x.strip())

# Strategy for model names
model_strategy = st.sampled_from([
    "llama3.2", "mistral", "gpt-oss:20b", "qwen2.5:7b"
])

# Strategy for output languages
language_strategy = st.sampled_from([
    OutputLanguage.EN,
    OutputLanguage.ZH_TW
])

# Strategy for download formats
format_strategy = st.sampled_from(["jsonl", "json", "csv", "tsv"])


# ============================================================================
# Helper Functions
# ============================================================================

def create_jsonl_content(entries: list) -> bytes:
    """Create JSONL file content from entries."""
    lines = [json.dumps(entry) for entry in entries]
    return '\n'.join(lines).encode('utf-8')


def create_test_entries(field_names: list, count: int = 5) -> list:
    """Create test entries with given field names."""
    entries = []
    for i in range(count):
        entry = {field: f"value_{field}_{i}" for field in field_names}
        entries.append(entry)
    return entries


# ============================================================================
# Property Test 4: File upload extracts fields
# ============================================================================

@given(
    field_names=st.lists(
        field_name_strategy,
        min_size=1,
        max_size=5,
        unique=True
    ),
    entry_count=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=20, deadline=60000)
def test_file_upload_extracts_fields(field_names, entry_count):
    """
    **Feature: web-interface, Property 4: File upload extracts fields**
    **Validates: Requirements 2.1**

    For any valid dataset file upload, the system should extract and return
    all field names present in the dataset.
    """
    async def run_test():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Create test entries
            entries = create_test_entries(field_names, entry_count)
            content = create_jsonl_content(entries)

            # Upload file
            files = {"file": ("test.jsonl", content, "application/x-ndjson")}
            response = await client.post("/api/augment/upload", files=files)

            # Should return 200 OK
            assert response.status_code == 200, \
                f"Valid file upload should return 200, got {response.status_code}: {response.text}"

            data = response.json()

            # Should have required fields
            assert "file_id" in data, "Response should include file_id"
            assert "entry_count" in data, "Response should include entry_count"
            assert "fields" in data, "Response should include fields"
            assert "preview" in data, "Response should include preview"

            # File ID should be non-empty string
            assert isinstance(data["file_id"], str)
            assert len(data["file_id"]) > 0

            # Entry count should match
            assert data["entry_count"] == entry_count

            # All field names should be extracted
            returned_fields = set(data["fields"])
            expected_fields = set(field_names)
            assert returned_fields == expected_fields, \
                f"Expected fields {expected_fields}, got {returned_fields}"

            # Preview should have entries (up to 3)
            expected_preview_count = min(entry_count, 3)
            assert len(data["preview"]) == expected_preview_count

    asyncio.run(run_test())


# ============================================================================
# Property Test 5: Field validation works correctly
# ============================================================================

@given(
    field_names=st.lists(
        field_name_strategy,
        min_size=2,
        max_size=5,
        unique=True
    ),
    target_index=st.integers(min_value=0, max_value=4)
)
@settings(max_examples=20, deadline=60000)
def test_field_validation_works_correctly(field_names, target_index):
    """
    **Feature: web-interface, Property 5: Field validation works correctly**
    **Validates: Requirements 2.2**

    For any uploaded dataset, field validation should correctly identify
    whether a target field exists in the dataset.
    """
    # Ensure target_index is within bounds
    assume(target_index < len(field_names))

    # Test using the service directly instead of HTTP to avoid mock issues
    service = AugmentationService()

    async def run_test():
        # Create and upload test file
        entries = create_test_entries(field_names, 3)
        content = create_jsonl_content(entries)

        result = await service.upload_file(content, "test.jsonl")
        file_id = result["file_id"]
        valid_field = field_names[target_index]

        # Test with valid field - should pass validation
        is_valid, error = service.validate_field(file_id, valid_field)
        assert is_valid is True, \
            f"Valid field '{valid_field}' should pass validation, got error: {error}"
        assert error is None

        # Test with invalid field - should fail validation
        is_valid, error = service.validate_field(file_id, "nonexistent_field_xyz")
        assert is_valid is False, \
            "Invalid field should fail validation"
        assert error is not None
        assert "not found" in error.lower(), \
            f"Error should mention field not found, got: {error}"

        # Test with invalid field but create_new_field=True - should pass
        is_valid, error = service.validate_field(
            file_id, "new_field", create_new_field=True
        )
        assert is_valid is True, \
            "New field with create_new_field=True should pass validation"

        # Test with non-existent file
        is_valid, error = service.validate_field("nonexistent_file", valid_field)
        assert is_valid is False
        assert "not found" in error.lower()

    asyncio.run(run_test())


# ============================================================================
# Property Test 6: Completed augmentation provides download
# ============================================================================

@given(
    field_names=st.lists(
        field_name_strategy,
        min_size=1,
        max_size=3,
        unique=True
    ),
    entry_count=st.integers(min_value=1, max_value=5),
    format=format_strategy
)
@settings(max_examples=20, deadline=10000)
@patch('ollaforge.web.routes.augmentation.augmentation_service')
def test_completed_augmentation_provides_download(
    mock_service,
    field_names,
    entry_count,
    format
):
    """
    **Feature: web-interface, Property 6: Completed augmentation provides download**
    **Validates: Requirements 2.4**

    For any augmentation that completes successfully, the system should provide
    a download endpoint that returns the augmented dataset in the requested format.
    """
    async def run_test():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Create mock task with completed status
            task_id = f"aug_test_{hash(str(field_names)) % 10000}"

            # Create mock entries
            mock_entries = create_test_entries(field_names, entry_count)

            mock_task = {
                "status": "completed",
                "progress": entry_count,
                "total": entry_count,
                "result": {
                    "entries": mock_entries,
                    "total": entry_count,
                    "success_count": entry_count,
                    "failure_count": 0,
                    "duration": 5.5,
                    "errors": []
                },
                "error": None
            }

            mock_service.get_task.return_value = mock_task

            # Request download
            response = await client.get(f"/api/augment/{task_id}/download?format={format}")

            # Should return 200 OK
            assert response.status_code == 200, \
                f"Download should return 200 for completed task, got {response.status_code}"

            # Should have appropriate content type
            content_type = response.headers.get("content-type", "")
            assert len(content_type) > 0, "Response should have content-type header"

            # Should have content-disposition header for download
            content_disposition = response.headers.get("content-disposition", "")
            assert "attachment" in content_disposition, \
                "Response should have attachment content-disposition"
            assert format in content_disposition, \
                f"Filename should include format {format}"

            # Should have content
            assert len(response.content) > 0, "Download should have content"

    asyncio.run(run_test())


# ============================================================================
# Property Test 7: Partial augmentation failures preserve data
# ============================================================================

@given(
    field_names=st.lists(
        field_name_strategy,
        min_size=1,
        max_size=3,
        unique=True
    ),
    total_count=st.integers(min_value=5, max_value=10),
    failure_count=st.integers(min_value=1, max_value=4)
)
@settings(max_examples=20, deadline=10000)
@patch('ollaforge.web.routes.augmentation.augmentation_service')
def test_partial_augmentation_failures_preserve_data(
    mock_service,
    field_names,
    total_count,
    failure_count
):
    """
    **Feature: web-interface, Property 7: Partial augmentation failures preserve data**
    **Validates: Requirements 2.5**

    For any augmentation with partial failures, the system should preserve
    successfully augmented entries and report failure statistics.
    """
    # Ensure failure_count is less than total_count
    assume(failure_count < total_count)
    success_count = total_count - failure_count

    async def run_test():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            task_id = f"aug_partial_{hash(str(field_names)) % 10000}"

            # Create mock entries (only successful ones)
            mock_entries = create_test_entries(field_names, success_count)

            # Create mock errors
            mock_errors = [
                f"Failed to augment entry {i}: Model error"
                for i in range(failure_count)
            ]

            mock_task = {
                "status": "completed",
                "progress": total_count,
                "total": total_count,
                "result": {
                    "entries": mock_entries,
                    "total": total_count,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "duration": 10.0,
                    "errors": mock_errors
                },
                "error": None
            }

            mock_service.get_task.return_value = mock_task

            # Query task status
            response = await client.get(f"/api/augment/{task_id}")

            assert response.status_code == 200
            data = response.json()

            # Task should be completed (not failed)
            assert data["status"] == "completed", \
                "Partial failures should not mark task as failed"

            # Result should contain statistics
            result = data.get("result", {})
            assert result.get("success_count") == success_count, \
                f"Success count should be {success_count}"
            assert result.get("failure_count") == failure_count, \
                f"Failure count should be {failure_count}"

            # Successful entries should be preserved
            assert len(result.get("entries", [])) == success_count, \
                f"Should have {success_count} preserved entries"

            # Download should still work
            download_response = await client.get(f"/api/augment/{task_id}/download")
            assert download_response.status_code == 200, \
                "Download should work for partial success"

    asyncio.run(run_test())


# ============================================================================
# Unit Tests for Augmentation Routes
# ============================================================================

@pytest.mark.asyncio
async def test_upload_unsupported_format():
    """
    Test that unsupported file formats are rejected.

    Requirements satisfied:
    - 4.4: Unsupported format error
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Try to upload unsupported format
        content = b"some random content"
        files = {"file": ("test.xyz", content, "application/octet-stream")}

        response = await client.post("/api/augment/upload", files=files)

        assert response.status_code == 415, \
            f"Unsupported format should return 415, got {response.status_code}"

        data = response.json()
        assert "detail" in data
        assert "UnsupportedFormat" in str(data["detail"])


@pytest.mark.asyncio
async def test_upload_invalid_jsonl():
    """
    Test that invalid JSONL content is rejected.

    Requirements satisfied:
    - 2.1: File validation
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Upload invalid JSONL
        content = b"not valid json\nalso not valid"
        files = {"file": ("test.jsonl", content, "application/x-ndjson")}

        response = await client.post("/api/augment/upload", files=files)

        assert response.status_code == 400, \
            f"Invalid JSONL should return 400, got {response.status_code}"


@pytest.mark.asyncio
async def test_preview_file_not_found():
    """
    Test that preview with non-existent file returns 404.

    Requirements satisfied:
    - 2.3: Error handling
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {
            "file_id": "nonexistent_file_id",
            "target_field": "output",
            "instruction": "Test",
            "model": "llama3.2"
        }

        response = await client.post("/api/augment/preview", json=payload)

        assert response.status_code == 404, \
            f"Non-existent file should return 404, got {response.status_code}"


@pytest.mark.asyncio
async def test_augmentation_file_not_found():
    """
    Test that augmentation with non-existent file returns 404.

    Requirements satisfied:
    - 2.2: Error handling
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {
            "file_id": "nonexistent_file_id",
            "target_field": "output",
            "instruction": "Test",
            "model": "llama3.2"
        }

        response = await client.post("/api/augment", json=payload)

        assert response.status_code == 404, \
            f"Non-existent file should return 404, got {response.status_code}"


@pytest.mark.asyncio
async def test_augmentation_status_not_found():
    """
    Test that querying non-existent task returns 404.

    Requirements satisfied:
    - Error handling
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/augment/nonexistent_task_id")

        assert response.status_code == 404, \
            f"Non-existent task should return 404, got {response.status_code}"


@pytest.mark.asyncio
async def test_download_not_completed_task():
    """
    Test that downloading from non-completed task returns error.

    Requirements satisfied:
    - 2.4: Download only for completed tasks
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/augment/nonexistent_task/download")

        assert response.status_code == 404


@pytest.mark.asyncio
async def test_download_invalid_format():
    """
    Test that invalid format parameter is rejected.

    Requirements satisfied:
    - 4.4: Unsupported format error
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/augment/some_task/download?format=invalid")

        assert response.status_code == 422, \
            f"Invalid format should return 422, got {response.status_code}"


# ============================================================================
# Unit Tests for Augmentation Service
# ============================================================================

@pytest.mark.asyncio
async def test_augmentation_service_creates_task(augmentation_service):
    """
    Test that augmentation service can create tasks.
    """
    task_id = augmentation_service.create_task()

    assert isinstance(task_id, str)
    assert len(task_id) > 0
    assert task_id.startswith("aug_")

    task = augmentation_service.get_task(task_id)
    assert task is not None
    assert task["status"] == "pending"


@pytest.mark.asyncio
async def test_augmentation_service_updates_task(augmentation_service):
    """
    Test that augmentation service can update task status.
    """
    task_id = augmentation_service.create_task()

    augmentation_service.update_task(
        task_id,
        status="running",
        progress=5,
        total=10
    )

    task = augmentation_service.get_task(task_id)
    assert task["status"] == "running"
    assert task["progress"] == 5
    assert task["total"] == 10


@pytest.mark.asyncio
async def test_augmentation_service_deletes_task(augmentation_service):
    """
    Test that augmentation service can delete tasks.
    """
    task_id = augmentation_service.create_task()

    assert augmentation_service.get_task(task_id) is not None

    augmentation_service.delete_task(task_id)

    assert augmentation_service.get_task(task_id) is None


@pytest.mark.asyncio
async def test_augmentation_service_upload_file(augmentation_service):
    """
    Test that augmentation service can upload and process files.

    Requirements satisfied:
    - 2.1: Upload and validate dataset files
    """
    entries = [
        {"instruction": "Task 1", "input": "Input 1", "output": "Output 1"},
        {"instruction": "Task 2", "input": "Input 2", "output": "Output 2"},
    ]
    content = create_jsonl_content(entries)

    result = await augmentation_service.upload_file(content, "test.jsonl")

    assert "file_id" in result
    assert result["entry_count"] == 2
    assert set(result["fields"]) == {"instruction", "input", "output"}
    assert len(result["preview"]) == 2


@pytest.mark.asyncio
async def test_augmentation_service_validate_field(augmentation_service):
    """
    Test that augmentation service validates fields correctly.

    Requirements satisfied:
    - 2.2: Validate target field exists
    """
    entries = [{"field_a": "value", "field_b": "value"}]
    content = create_jsonl_content(entries)

    result = await augmentation_service.upload_file(content, "test.jsonl")
    file_id = result["file_id"]

    # Valid field
    is_valid, error = augmentation_service.validate_field(file_id, "field_a")
    assert is_valid is True
    assert error is None

    # Invalid field
    is_valid, error = augmentation_service.validate_field(file_id, "nonexistent")
    assert is_valid is False
    assert error is not None
    assert "not found" in error.lower()

    # Invalid field with create_new_field=True
    is_valid, error = augmentation_service.validate_field(
        file_id, "new_field", create_new_field=True
    )
    assert is_valid is True


@pytest.mark.asyncio
async def test_augmentation_service_delete_file(augmentation_service):
    """
    Test that augmentation service can delete uploaded files.
    """
    entries = [{"field": "value"}]
    content = create_jsonl_content(entries)

    result = await augmentation_service.upload_file(content, "test.jsonl")
    file_id = result["file_id"]

    # File should exist
    assert augmentation_service.get_uploaded_file(file_id) is not None

    # Delete file
    augmentation_service.delete_file(file_id)

    # File should be gone
    assert augmentation_service.get_uploaded_file(file_id) is None
