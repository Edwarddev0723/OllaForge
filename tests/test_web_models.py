"""
Tests for model management API routes.

This module tests the Ollama model management functionality including:
- Listing available models
- Getting model information
- Model validation
- Error handling for Ollama unavailability

Requirements satisfied:
- 10.1: Fetch and display available Ollama models
- 10.2: Show model names with size information
- 10.3: Display warning when Ollama service is not available
- 10.4: Validate model exists before starting generation
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, settings, assume
import httpx
from httpx._transports.asgi import ASGITransport
from unittest.mock import patch, MagicMock

from ollaforge.web.server import app
from ollaforge.web.routes.models import _format_size, _get_model_details
from ollaforge.client import OllamaConnectionError


def create_test_client():
    """Create an async test client for the FastAPI app using httpx with ASGITransport."""
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ============================================================================
# Strategies for Property-Based Testing
# ============================================================================

# Strategy for model names
model_name_strategy = st.text(
    alphabet='abcdefghijklmnopqrstuvwxyz0123456789-_.:',
    min_size=1,
    max_size=30
).filter(lambda x: x[0].isalpha())

# Strategy for model sizes in bytes
model_size_strategy = st.integers(min_value=1024 * 1024, max_value=100 * 1024 * 1024 * 1024)

# Strategy for timestamps
timestamp_strategy = st.text(
    alphabet='0123456789-T:Z',
    min_size=10,
    max_size=30
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_mock_model(name: str, size: int = None, modified_at: str = None) -> dict:
    """Create a mock model dict as returned by Ollama."""
    model = {"name": name}
    if size is not None:
        model["size"] = size
    if modified_at is not None:
        model["modified_at"] = modified_at
    return model


def create_mock_ollama_response(models: list) -> dict:
    """Create a mock Ollama list response."""
    return {"models": models}


# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================

class TestFormatSize:
    """Tests for the _format_size helper function."""
    
    def test_format_size_none(self):
        """Test that None input returns None."""
        assert _format_size(None) is None
    
    def test_format_size_gigabytes(self):
        """Test formatting sizes in gigabytes."""
        # 1 GB
        assert _format_size(1024 ** 3) == "1.0GB"
        # 3.2 GB
        assert _format_size(int(3.2 * 1024 ** 3)) == "3.2GB"
        # 7 GB
        assert _format_size(7 * 1024 ** 3) == "7.0GB"
    
    def test_format_size_megabytes(self):
        """Test formatting sizes in megabytes."""
        # 500 MB
        assert _format_size(500 * 1024 ** 2) == "500MB"
        # 100 MB
        assert _format_size(100 * 1024 ** 2) == "100MB"


# ============================================================================
# Property Test 33: Model information includes size
# ============================================================================

@given(
    model_names=st.lists(
        model_name_strategy,
        min_size=1,
        max_size=5,
        unique=True
    ),
    model_sizes=st.lists(
        model_size_strategy,
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=20, deadline=60000)
@patch('ollaforge.web.routes.models._get_ollama_client')
def test_model_information_includes_size(mock_get_client, model_names, model_sizes):
    """
    **Feature: web-interface, Property 33: Model information includes size**
    **Validates: Requirements 10.2**
    
    For any Ollama model displayed when the service is running, the system
    should show the model name with size information.
    """
    # Ensure we have matching sizes for names
    assume(len(model_sizes) >= len(model_names))
    
    async def run_test():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Create mock models with sizes
            mock_models = [
                create_mock_model(name, size)
                for name, size in zip(model_names, model_sizes)
            ]
            mock_client = MagicMock()
            mock_client.list.return_value = create_mock_ollama_response(mock_models)
            mock_get_client.return_value = mock_client
            
            # Request model list
            response = await client.get("/api/models")
            
            assert response.status_code == 200, \
                f"Model list should return 200, got {response.status_code}"
            
            data = response.json()
            assert "models" in data
            
            # Each model should have name and size
            for model in data["models"]:
                assert "name" in model, "Model should have name"
                assert "size" in model, "Model should have size"
                
                # Size should be formatted (e.g., "3.2GB", "500MB")
                size = model["size"]
                if size is not None:
                    assert "GB" in size or "MB" in size, \
                        f"Size should be formatted with GB or MB, got {size}"
    
    asyncio.run(run_test())


# ============================================================================
# Property Test 34: Model validation before generation
# ============================================================================

@given(
    existing_models=st.lists(
        model_name_strategy,
        min_size=1,
        max_size=5,
        unique=True
    ),
    model_index=st.integers(min_value=0, max_value=4)
)
@settings(max_examples=20, deadline=60000)
@patch('ollaforge.web.routes.models._get_ollama_client')
@patch('ollaforge.web.routes.models.get_available_models')
def test_model_validation_before_generation(
    mock_get_models,
    mock_get_client,
    existing_models,
    model_index
):
    """
    **Feature: web-interface, Property 34: Model validation before generation**
    **Validates: Requirements 10.4**
    
    For any model selected by the user, the system should validate that
    the model exists before starting generation.
    """
    assume(model_index < len(existing_models))
    
    async def run_test():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Create mock models
            mock_models = [create_mock_model(name, 1024**3) for name in existing_models]
            mock_client = MagicMock()
            mock_client.list.return_value = create_mock_ollama_response(mock_models)
            mock_get_client.return_value = mock_client
            mock_get_models.return_value = existing_models
            
            valid_model = existing_models[model_index]
            
            # Test validation of existing model
            response = await client.get(f"/api/models/{valid_model}/validate")
            
            assert response.status_code == 200, \
                f"Valid model should return 200, got {response.status_code}"
            
            data = response.json()
            assert data["valid"] is True
            assert data["model"] == valid_model
            
            # Test validation of non-existing model
            response = await client.get("/api/models/nonexistent_model_xyz/validate")
            
            assert response.status_code == 404, \
                f"Non-existent model should return 404, got {response.status_code}"
    
    asyncio.run(run_test())


# ============================================================================
# Property Test 25: Ollama unavailability shows clear error
# ============================================================================

@given(
    error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip())
)
@settings(max_examples=10, deadline=60000)
@patch('ollaforge.web.routes.models._get_ollama_client')
def test_ollama_unavailability_shows_clear_error(mock_get_client, error_message):
    """
    **Feature: web-interface, Property 25: Ollama unavailability shows clear error**
    **Validates: Requirements 7.5**
    
    For any situation where the Ollama service is unavailable, the system
    should return a clear error message.
    """
    async def run_test():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Simulate connection error
            mock_client = MagicMock()
            mock_client.list.side_effect = Exception(f"Connection refused: {error_message}")
            mock_get_client.return_value = mock_client
            
            # Request model list
            response = await client.get("/api/models")
            
            assert response.status_code == 503, \
                f"Ollama unavailable should return 503, got {response.status_code}"
            
            data = response.json()
            assert "detail" in data
            
            detail = data["detail"]
            assert "error" in detail
            assert "message" in detail
            
            # Error message should be clear and helpful
            message = detail["message"].lower()
            assert "ollama" in message or "connect" in message, \
                f"Error should mention Ollama or connection, got: {detail['message']}"
    
    asyncio.run(run_test())


# ============================================================================
# Unit Tests for Model Routes
# ============================================================================

@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
async def test_list_models_success(mock_get_client):
    """
    Test successful model listing.
    
    Requirements satisfied:
    - 10.1: Fetch and display available Ollama models
    """
    mock_models = [
        create_mock_model("llama3.2", 3 * 1024**3, "2024-01-15T10:30:00Z"),
        create_mock_model("mistral", 7 * 1024**3, "2024-01-10T08:20:00Z"),
    ]
    mock_client = MagicMock()
    mock_client.list.return_value = create_mock_ollama_response(mock_models)
    mock_get_client.return_value = mock_client
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert len(data["models"]) == 2
        
        # Check first model
        model1 = data["models"][0]
        assert model1["name"] == "llama3.2"
        assert model1["size"] == "3.0GB"
        assert model1["modified_at"] == "2024-01-15T10:30:00Z"


@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
async def test_list_models_empty(mock_get_client):
    """
    Test listing when no models are available.
    
    Requirements satisfied:
    - 10.5: Display instructions for installing Ollama models
    """
    mock_client = MagicMock()
    mock_client.list.return_value = create_mock_ollama_response([])
    mock_get_client.return_value = mock_client
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert len(data["models"]) == 0


@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
async def test_list_models_connection_error(mock_get_client):
    """
    Test model listing when Ollama is unavailable.
    
    Requirements satisfied:
    - 10.3: Display warning when Ollama service is not available
    """
    mock_client = MagicMock()
    mock_client.list.side_effect = Exception("Connection refused")
    mock_get_client.return_value = mock_client
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models")
        
        assert response.status_code == 503
        data = response.json()
        
        assert "detail" in data
        assert "OllamaConnectionError" in str(data["detail"]) or "OllamaError" in str(data["detail"])


@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
async def test_get_model_info_success(mock_get_client):
    """
    Test getting info for a specific model.
    
    Requirements satisfied:
    - 10.2: Show model names with size information
    """
    mock_models = [
        create_mock_model("llama3.2", 3 * 1024**3, "2024-01-15T10:30:00Z"),
    ]
    mock_client = MagicMock()
    mock_client.list.return_value = create_mock_ollama_response(mock_models)
    mock_get_client.return_value = mock_client
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models/llama3.2")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "llama3.2"
        assert data["size"] == "3.0GB"
        assert data["modified_at"] == "2024-01-15T10:30:00Z"


@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
@patch('ollaforge.web.routes.models.get_available_models')
async def test_get_model_info_not_found(mock_get_models, mock_get_client):
    """
    Test getting info for a non-existent model.
    
    Requirements satisfied:
    - 10.4: Validate model exists
    """
    mock_models = [
        create_mock_model("llama3.2", 3 * 1024**3),
    ]
    mock_client = MagicMock()
    mock_client.list.return_value = create_mock_ollama_response(mock_models)
    mock_get_client.return_value = mock_client
    mock_get_models.return_value = ["llama3.2"]
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models/nonexistent_model")
        
        assert response.status_code == 404
        data = response.json()
        
        assert "detail" in data
        assert "ModelNotFound" in str(data["detail"])


@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
async def test_get_model_info_connection_error(mock_get_client):
    """
    Test getting model info when Ollama is unavailable.
    
    Requirements satisfied:
    - 10.3: Display warning when Ollama service is not available
    """
    mock_client = MagicMock()
    mock_client.list.side_effect = Exception("Connection refused")
    mock_get_client.return_value = mock_client
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models/llama3.2")
        
        assert response.status_code == 503


@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
async def test_validate_model_success(mock_get_client):
    """
    Test validating an existing model.
    
    Requirements satisfied:
    - 10.4: Validate model exists before starting generation
    """
    mock_models = [
        create_mock_model("llama3.2", 3 * 1024**3),
    ]
    mock_client = MagicMock()
    mock_client.list.return_value = create_mock_ollama_response(mock_models)
    mock_get_client.return_value = mock_client
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models/llama3.2/validate")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["valid"] is True
        assert data["model"] == "llama3.2"


@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
@patch('ollaforge.web.routes.models.get_available_models')
async def test_validate_model_not_found(mock_get_models, mock_get_client):
    """
    Test validating a non-existent model.
    
    Requirements satisfied:
    - 10.4: Validate model exists before starting generation
    """
    mock_models = [
        create_mock_model("llama3.2", 3 * 1024**3),
    ]
    mock_client = MagicMock()
    mock_client.list.return_value = create_mock_ollama_response(mock_models)
    mock_get_client.return_value = mock_client
    mock_get_models.return_value = ["llama3.2"]
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models/nonexistent/validate")
        
        assert response.status_code == 404


@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
async def test_list_models_with_missing_size(mock_get_client):
    """
    Test listing models when some don't have size info.
    """
    mock_models = [
        create_mock_model("llama3.2", 3 * 1024**3),
        create_mock_model("custom-model"),  # No size
    ]
    mock_client = MagicMock()
    mock_client.list.return_value = create_mock_ollama_response(mock_models)
    mock_get_client.return_value = mock_client
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["models"]) == 2
        
        # First model has size
        assert data["models"][0]["size"] == "3.0GB"
        
        # Second model has no size
        assert data["models"][1]["size"] is None


@pytest.mark.asyncio
@patch('ollaforge.web.routes.models._get_ollama_client')
async def test_list_models_invalid_response(mock_get_client):
    """
    Test handling of invalid Ollama response.
    """
    mock_client = MagicMock()
    mock_client.list.return_value = "invalid response"
    mock_get_client.return_value = mock_client
    
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/models")
        
        assert response.status_code == 503
