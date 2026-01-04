"""
Model management API routes for OllaForge web interface.

This module provides REST API endpoints for Ollama model management:
- GET /api/models: List available Ollama models
- GET /api/models/{model_name}: Get information about a specific model

Requirements satisfied:
- 10.1: Fetch and display available Ollama models
- 10.2: Show model names with size information
- 10.3: Display warning when Ollama service is not available
- 10.4: Validate model exists before starting generation
"""

import os
from typing import Optional

import ollama
from fastapi import APIRouter, HTTPException

from ...client import OllamaConnectionError, get_available_models
from ..models import ErrorResponse, ModelInfo, ModelListResponse

# Create router
router = APIRouter(prefix="/api/models", tags=["models"])


def _get_ollama_client() -> ollama.Client:
    """Get ollama client with configured host (lazy initialization)."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return ollama.Client(host=host)


def _format_size(size_bytes: Optional[int]) -> Optional[str]:
    """
    Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "3.2GB", "7.0GB")
    """
    if size_bytes is None:
        return None

    # Convert to GB
    size_gb = size_bytes / (1024 ** 3)

    if size_gb >= 1:
        return f"{size_gb:.1f}GB"

    # Convert to MB if less than 1GB
    size_mb = size_bytes / (1024 ** 2)
    return f"{size_mb:.0f}MB"


def _get_model_details(model_name: str) -> Optional[dict]:
    """
    Get detailed information about a specific model from Ollama.

    Args:
        model_name: Name of the model

    Returns:
        Dict with model details or None if not found

    Raises:
        OllamaConnectionError: If connection to Ollama fails
    """
    try:
        client = _get_ollama_client()
        response = client.list()
        if not isinstance(response, dict) or 'models' not in response:
            raise OllamaConnectionError("Invalid response from Ollama API")

        for model in response['models']:
            if isinstance(model, dict) and model.get('name') == model_name:
                return model

        return None
    except OllamaConnectionError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            raise OllamaConnectionError(f"Unable to connect to Ollama service: {str(e)}") from e
        raise


@router.get("", response_model=ModelListResponse, responses={
    503: {"model": ErrorResponse, "description": "Ollama service unavailable"}
},
    summary="List Available Models",
    description="""
List all available Ollama models.

Returns a list of all models available in the local Ollama installation,
including their names, sizes, and last modified timestamps.

### Prerequisites

Ollama must be running locally. Start it with:
```bash
ollama serve
```

If Ollama is not running, this endpoint returns a 503 error with
instructions for starting the service.
"""
)
async def list_models() -> ModelListResponse:
    """
    List available Ollama models.

    Returns a list of all models available in the local Ollama installation,
    including their names and size information.

    Requirements satisfied:
    - 10.1: Fetch and display available Ollama models
    - 10.2: Show model names with size information
    - 10.3: Display warning when Ollama service is not available

    Returns:
        ModelListResponse with list of available models

    Raises:
        HTTPException 503: If Ollama service is unavailable
    """
    try:
        # Get raw response from Ollama to extract size info
        client = _get_ollama_client()
        response = client.list()

        if not isinstance(response, dict) or 'models' not in response:
            raise OllamaConnectionError("Invalid response from Ollama API")

        models = []
        for model in response['models']:
            if isinstance(model, dict) and 'name' in model:
                model_info = ModelInfo(
                    name=model['name'],
                    size=_format_size(model.get('size')),
                    modified_at=model.get('modified_at')
                )
                models.append(model_info)

        return ModelListResponse(models=models)

    except OllamaConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OllamaConnectionError",
                "message": f"Unable to connect to Ollama service: {str(e)}. Please ensure Ollama is running with 'ollama serve'."
            }
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OllamaConnectionError",
                    "message": "Unable to connect to Ollama service. Please ensure Ollama is running with 'ollama serve'."
                }
            )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OllamaError",
                "message": f"Failed to list models: {str(e)}"
            }
        )


@router.get("/{model_name}", response_model=ModelInfo, responses={
    404: {"model": ErrorResponse, "description": "Model not found"},
    503: {"model": ErrorResponse, "description": "Ollama service unavailable"}
},
    summary="Get Model Information",
    description="""
Get detailed information about a specific Ollama model.

Returns the model's name, size, and last modified timestamp.
If the model is not found, returns a 404 error with a list of available models.
"""
)
async def get_model_info(model_name: str) -> ModelInfo:
    """
    Get information about a specific Ollama model.

    Returns detailed information about the specified model including
    its name, size, and last modified timestamp.

    Requirements satisfied:
    - 10.2: Show model names with size information
    - 10.4: Validate model exists before starting generation

    Args:
        model_name: Name of the model to get information for

    Returns:
        ModelInfo with model details

    Raises:
        HTTPException 404: If model not found
        HTTPException 503: If Ollama service is unavailable
    """
    try:
        model_details = _get_model_details(model_name)

        if model_details is None:
            # Try to get available models to provide helpful error
            try:
                available = get_available_models()
                available_str = ", ".join(available[:5])
                if len(available) > 5:
                    available_str += f" (and {len(available) - 5} more)"
                message = f"Model '{model_name}' not found. Available models: {available_str}"
            except Exception:
                message = f"Model '{model_name}' not found"

            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ModelNotFound",
                    "message": message
                }
            )

        return ModelInfo(
            name=model_details['name'],
            size=_format_size(model_details.get('size')),
            modified_at=model_details.get('modified_at')
        )

    except HTTPException:
        raise
    except OllamaConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OllamaConnectionError",
                "message": f"Unable to connect to Ollama service: {str(e)}. Please ensure Ollama is running with 'ollama serve'."
            }
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OllamaConnectionError",
                    "message": "Unable to connect to Ollama service. Please ensure Ollama is running with 'ollama serve'."
                }
            )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalError",
                "message": f"Failed to get model info: {str(e)}"
            }
        )


@router.get("/{model_name}/validate", responses={
    200: {"description": "Model is valid and available"},
    404: {"model": ErrorResponse, "description": "Model not found"},
    503: {"model": ErrorResponse, "description": "Ollama service unavailable"}
},
    summary="Validate Model Availability",
    description="""
Validate that a model exists and is available for use.

Use this endpoint to check if a model is available before starting
a generation or augmentation task. This helps provide better error
messages to users before they start a long-running operation.
"""
)
async def validate_model(model_name: str) -> dict:
    """
    Validate that a model exists and is available for use.

    This endpoint can be used to check if a model is available
    before starting a generation task.

    Requirements satisfied:
    - 10.4: Validate model exists before starting generation

    Args:
        model_name: Name of the model to validate

    Returns:
        Dict with validation result

    Raises:
        HTTPException 404: If model not found
        HTTPException 503: If Ollama service is unavailable
    """
    # Use get_model_info to validate - it will raise appropriate exceptions
    model_info = await get_model_info(model_name)

    return {
        "valid": True,
        "model": model_info.name,
        "message": f"Model '{model_name}' is available"
    }
