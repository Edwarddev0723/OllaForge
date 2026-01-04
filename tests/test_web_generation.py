"""
Tests for generation API routes and service.

This module tests the dataset generation functionality including:
- Generation initiation with valid parameters
- Progress tracking and status updates
- Download functionality
- Error handling

Requirements satisfied:
- 1.2: Initiate dataset generation with valid parameters
- 1.3: Provide download link for generated dataset
- 1.4: Display clear error messages
"""

import asyncio
from unittest.mock import patch

import httpx
import pytest
from httpx._transports.asgi import ASGITransport
from hypothesis import given, settings
from hypothesis import strategies as st

from ollaforge.client import OllamaConnectionError, OllamaGenerationError
from ollaforge.models import DataEntry, DatasetType, OutputLanguage
from ollaforge.web.server import app
from ollaforge.web.services.generation import GenerationService


def create_test_client():
    """Create an async test client for the FastAPI app using httpx with ASGITransport."""
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def generation_service():
    """Create a generation service instance."""
    return GenerationService()


# ============================================================================
# Strategies for Property-Based Testing
# ============================================================================

# Strategy for valid topics (simple ASCII strings to avoid encoding issues)
topic_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-",
    min_size=1,
    max_size=100,
).filter(lambda x: x.strip())

# Strategy for valid counts (1-100 for testing)
count_strategy = st.integers(min_value=1, max_value=100)

# Strategy for model names
model_strategy = st.sampled_from(["llama3.2", "mistral", "gpt-oss:20b", "qwen2.5:7b"])

# Strategy for dataset types
dataset_type_strategy = st.sampled_from(
    [
        DatasetType.SFT,
        DatasetType.PRETRAIN,
        DatasetType.SFT_CONVERSATION,
        DatasetType.DPO,
    ]
)

# Strategy for output languages
language_strategy = st.sampled_from([OutputLanguage.EN, OutputLanguage.ZH_TW])

# Strategy for QC settings
qc_enabled_strategy = st.booleans()
qc_confidence_strategy = st.floats(min_value=0.0, max_value=1.0)


# ============================================================================
# Property Test 1: Valid generation parameters initiate processing
# ============================================================================


@given(
    topic=topic_strategy,
    count=st.integers(min_value=1, max_value=50),  # Reduced range for faster tests
    model=model_strategy,
    dataset_type=dataset_type_strategy,
    language=language_strategy,
    qc_enabled=qc_enabled_strategy,
    qc_confidence=qc_confidence_strategy,
)
@settings(max_examples=20, deadline=60000)  # 60 second deadline per example
def test_valid_generation_parameters_initiate_processing(
    topic, count, model, dataset_type, language, qc_enabled, qc_confidence
):
    """
    **Feature: web-interface, Property 1: Valid generation parameters initiate processing**
    **Validates: Requirements 1.2**

    For any valid generation configuration (topic, count, model, dataset type, language),
    submitting the form should initiate dataset generation and return a task ID with
    pending status.
    """

    async def run_test():
        # Create async client
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            # Prepare request payload
            payload = {
                "topic": topic,
                "count": count,
                "model": model,
                "dataset_type": dataset_type.value,
                "language": language.value,
                "qc_enabled": qc_enabled,
                "qc_confidence": qc_confidence,
            }

            # Make request to start generation
            response = await client.post("/api/generate", json=payload)

            # Should return 200 OK
            assert (
                response.status_code == 200
            ), f"Valid generation request should return 200, got {response.status_code}: {response.text}"

            # Parse response
            data = response.json()

            # Should have required fields
            assert "task_id" in data, "Response should include task_id"
            assert "status" in data, "Response should include status"
            assert "message" in data, "Response should include message"

            # Task ID should be non-empty string
            assert isinstance(data["task_id"], str), "task_id should be a string"
            assert len(data["task_id"]) > 0, "task_id should not be empty"

            # Status should be pending (task just created)
            assert data["status"] in [
                "pending",
                "running",
            ], f"Initial status should be pending or running, got {data['status']}"

            # Should be able to query task status
            task_id = data["task_id"]
            status_response = await client.get(f"/api/generate/{task_id}")

            # Status endpoint should work
            assert (
                status_response.status_code == 200
            ), f"Status query should return 200, got {status_response.status_code}"

            status_data = status_response.json()
            assert status_data["task_id"] == task_id, "Task ID should match"
            assert "status" in status_data, "Status response should include status"
            assert "progress" in status_data, "Status response should include progress"
            assert "total" in status_data, "Status response should include total"

    # Run the async test
    asyncio.run(run_test())


# ============================================================================
# Property Test 2: Completed generation provides download
# ============================================================================


@given(
    topic=topic_strategy,
    count=st.integers(min_value=1, max_value=10),  # Small count for faster tests
    format=st.sampled_from(["jsonl", "json", "csv", "tsv"]),
)
@settings(max_examples=20, deadline=10000)
@patch("ollaforge.web.routes.generation.generation_service")
def test_completed_generation_provides_download(mock_service, topic, count, format):
    """
    **Feature: web-interface, Property 2: Completed generation provides download**
    **Validates: Requirements 1.3**

    For any dataset generation that completes successfully, the system should provide
    a download endpoint that returns the generated dataset in the requested format.
    """

    async def run_test():
        # Create async client
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            # Create mock task with completed status
            task_id = f"gen_test_{hash(topic) % 10000}"

            # Create mock entries
            mock_entries = [
                DataEntry(
                    instruction=f"Task {i}", input=f"Input {i}", output=f"Output {i}"
                )
                for i in range(count)
            ]

            mock_task = {
                "status": "completed",
                "progress": count,
                "total": count,
                "result": {"entries": mock_entries, "total": count, "duration": 10.5},
                "error": None,
            }

            # Configure mock service
            mock_service.get_task.return_value = mock_task

            # Request download
            response = await client.get(
                f"/api/generate/{task_id}/download?format={format}"
            )

            # Should return 200 OK
            assert (
                response.status_code == 200
            ), f"Download should return 200 for completed task, got {response.status_code}"

            # Should have appropriate content type
            content_type = response.headers.get("content-type", "")
            assert len(content_type) > 0, "Response should have content-type header"

            # Should have content-disposition header for download
            content_disposition = response.headers.get("content-disposition", "")
            assert (
                "attachment" in content_disposition
            ), "Response should have attachment content-disposition"
            assert (
                format in content_disposition
            ), f"Filename should include format {format}"

            # Should have content
            assert len(response.content) > 0, "Download should have content"

    asyncio.run(run_test())


# ============================================================================
# Property Test 3: Generation failures display errors
# ============================================================================


@given(
    topic=topic_strategy,
    count=count_strategy,
    error_type=st.sampled_from(["connection", "generation", "unexpected"]),
)
@settings(max_examples=20, deadline=10000)
@patch("ollaforge.web.routes.generation.generation_service")
def test_generation_failures_display_errors(mock_service, topic, count, error_type):
    """
    **Feature: web-interface, Property 3: Generation failures display errors**
    **Validates: Requirements 1.4**

    For any dataset generation that fails, the system should display a clear error
    message indicating what went wrong.
    """

    async def run_test():
        # Create async client
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            # Create mock task with failed status
            task_id = f"gen_test_{hash(topic) % 10000}"

            # Create appropriate error message based on error type
            error_messages = {
                "connection": "Ollama service unavailable: Connection refused",
                "generation": "Generation failed: Model not found",
                "unexpected": "Unexpected error: Internal server error",
            }

            mock_task = {
                "status": "failed",
                "progress": 0,
                "total": count,
                "result": None,
                "error": error_messages[error_type],
            }

            # Configure mock service
            mock_service.get_task.return_value = mock_task

            # Query task status
            response = await client.get(f"/api/generate/{task_id}")

            # Should return 200 OK (status query succeeds even if task failed)
            assert (
                response.status_code == 200
            ), f"Status query should return 200, got {response.status_code}"

            # Parse response
            data = response.json()

            # Should indicate failure
            assert (
                data["status"] == "failed"
            ), f"Status should be 'failed', got {data['status']}"

            # Should have error message
            assert "error" in data, "Failed task should include error field"
            assert data["error"] is not None, "Error field should not be None"
            assert isinstance(data["error"], str), "Error should be a string"
            assert len(data["error"]) > 0, "Error message should not be empty"

            # Error message should be descriptive
            error_msg = data["error"].lower()
            if error_type == "connection":
                assert (
                    "ollama" in error_msg
                    or "connection" in error_msg
                    or "unavailable" in error_msg
                ), "Connection error should mention Ollama or connection"
            elif error_type == "generation":
                assert (
                    "generation" in error_msg
                    or "failed" in error_msg
                    or "model" in error_msg
                ), "Generation error should mention generation or model"

    asyncio.run(run_test())


# ============================================================================
# Unit Tests for Generation Routes
# ============================================================================


@pytest.mark.asyncio
async def test_generation_route_parameter_validation():
    """
    Test that generation route validates parameters correctly.

    Requirements satisfied:
    - 1.2: Parameter validation
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        # Test with missing required field (topic)
        payload = {"count": 10, "model": "llama3.2"}

        response = await client.post("/api/generate", json=payload)

        # Should return 422 Unprocessable Entity (validation error)
        assert (
            response.status_code == 422
        ), f"Missing required field should return 422, got {response.status_code}"


@pytest.mark.asyncio
async def test_generation_route_count_validation():
    """
    Test that count parameter is validated.

    Requirements satisfied:
    - 1.2: Parameter validation
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        # Test with invalid count (0)
        payload = {"topic": "Test topic", "count": 0, "model": "llama3.2"}

        response = await client.post("/api/generate", json=payload)

        # Should return 422 (validation error)
        assert (
            response.status_code == 422
        ), f"Invalid count should return 422, got {response.status_code}"

        # Test with negative count
        payload["count"] = -5
        response = await client.post("/api/generate", json=payload)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_generation_status_not_found():
    """
    Test that querying non-existent task returns 404.

    Requirements satisfied:
    - 1.4: Error handling
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get("/api/generate/nonexistent_task_id")

        # Should return 404 Not Found
        assert (
            response.status_code == 404
        ), f"Non-existent task should return 404, got {response.status_code}"

        # Should have error details
        data = response.json()
        assert "detail" in data


@pytest.mark.asyncio
async def test_download_not_completed_task():
    """
    Test that downloading from non-completed task returns error.

    Requirements satisfied:
    - 1.3: Download only for completed tasks
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        # This will fail because the task doesn't exist
        response = await client.get("/api/generate/nonexistent_task/download")

        # Should return 404
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_download_invalid_format():
    """
    Test that invalid format parameter is rejected.

    Requirements satisfied:
    - 4.4: Unsupported format error
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        # Try to download with invalid format
        response = await client.get("/api/generate/some_task/download?format=invalid")

        # Should return 422 (validation error) due to regex constraint
        assert (
            response.status_code == 422
        ), f"Invalid format should return 422, got {response.status_code}"


# ============================================================================
# Unit Tests for Generation Service
# ============================================================================


@pytest.mark.asyncio
async def test_generation_service_creates_task(generation_service):
    """
    Test that generation service can create tasks.
    """
    task_id = generation_service.create_task()

    # Should return a task ID
    assert isinstance(task_id, str)
    assert len(task_id) > 0
    assert task_id.startswith("gen_")

    # Should be able to retrieve task
    task = generation_service.get_task(task_id)
    assert task is not None
    assert task["status"] == "pending"


@pytest.mark.asyncio
async def test_generation_service_updates_task(generation_service):
    """
    Test that generation service can update task status.
    """
    task_id = generation_service.create_task()

    # Update task
    generation_service.update_task(task_id, status="running", progress=5, total=10)

    # Verify update
    task = generation_service.get_task(task_id)
    assert task["status"] == "running"
    assert task["progress"] == 5
    assert task["total"] == 10


@pytest.mark.asyncio
async def test_generation_service_deletes_task(generation_service):
    """
    Test that generation service can delete tasks.
    """
    task_id = generation_service.create_task()

    # Verify task exists
    assert generation_service.get_task(task_id) is not None

    # Delete task
    generation_service.delete_task(task_id)

    # Verify task is gone
    assert generation_service.get_task(task_id) is None


@pytest.mark.asyncio
@patch("ollaforge.web.services.generation.generate_data_concurrent")
@patch("ollaforge.web.services.generation.process_model_response")
async def test_generation_service_handles_ollama_connection_error(
    mock_process, mock_generate, generation_service
):
    """
    Test that generation service handles Ollama connection errors.

    Requirements satisfied:
    - 7.5: Clear error message for Ollama unavailability
    """
    # Configure mock to raise connection error
    mock_generate.side_effect = OllamaConnectionError("Connection refused")

    # Attempt generation
    with pytest.raises(OllamaConnectionError):
        await generation_service.generate_dataset(
            topic="Test", count=5, model="llama3.2"
        )


@pytest.mark.asyncio
@patch("ollaforge.web.services.generation.generate_data_concurrent")
async def test_generation_service_handles_generation_error(
    mock_generate, generation_service
):
    """
    Test that generation service handles generation errors.

    Requirements satisfied:
    - 1.4: Error handling
    """
    # Configure mock to raise generation error
    mock_generate.side_effect = OllamaGenerationError("Model not found")

    # Attempt generation
    with pytest.raises(OllamaGenerationError):
        await generation_service.generate_dataset(
            topic="Test", count=5, model="nonexistent_model"
        )
