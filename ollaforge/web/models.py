"""
API request/response models for OllaForge web interface.

This module defines Pydantic models for API communication between
the frontend and backend. All models use JSON serialization and
include validation rules.

Requirements satisfied:
- 9.2: JSON format for request and response bodies
"""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Import core models from ollaforge
from ..models import DatasetType, OutputLanguage

# ============================================================================
# Generation Models
# ============================================================================


class GenerationRequest(BaseModel):
    """
    Request model for dataset generation.

    Requirements satisfied:
    - 1.1: Form fields for topic, count, model, dataset type
    - 1.5: QC filtering options for Traditional Chinese
    """

    topic: str = Field(
        ..., description="Description of the dataset content to generate"
    )
    count: int = Field(
        ..., ge=1, le=10000, description="Number of data entries to generate"
    )
    model: str = Field("llama3.2", description="Ollama model to use")
    dataset_type: DatasetType = Field(
        DatasetType.SFT, description="Type of dataset to generate"
    )
    language: OutputLanguage = Field(
        OutputLanguage.EN, description="Output language for generated content"
    )
    qc_enabled: bool = Field(
        True, description="Enable QC for Traditional Chinese (Taiwan)"
    )
    qc_confidence: float = Field(
        0.9, ge=0.0, le=1.0, description="QC confidence threshold for Taiwan Chinese"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Customer service conversations for an e-commerce platform",
                "count": 100,
                "model": "llama3.2",
                "dataset_type": "sft",
                "language": "en",
                "qc_enabled": True,
                "qc_confidence": 0.9,
            }
        }


class GenerationResponse(BaseModel):
    """
    Response model for generation start.

    Requirements satisfied:
    - 1.2: Initiate dataset generation
    """

    task_id: str = Field(..., description="Unique identifier for the generation task")
    status: str = Field(..., description="Current status of the task")
    message: str = Field(..., description="Human-readable status message")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "gen_abc123",
                "status": "pending",
                "message": "Generation task created successfully",
            }
        }


# ============================================================================
# Augmentation Models
# ============================================================================


class AugmentUploadResponse(BaseModel):
    """
    Response after dataset upload.

    Requirements satisfied:
    - 2.1: Display available fields after upload
    - 5.3: Display first 3 entries
    """

    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    entry_count: int = Field(..., description="Total number of entries in the dataset")
    fields: list[str] = Field(..., description="List of available field names")
    preview: list[dict[str, Any]] = Field(
        ..., description="Preview of first few entries"
    )
    source_type: str = Field("file", description="Source type: 'file' or 'huggingface'")

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "file_xyz789",
                "entry_count": 150,
                "fields": ["instruction", "input", "output"],
                "preview": [
                    {"instruction": "Translate", "input": "Hello", "output": "Bonjour"}
                ],
                "source_type": "file",
            }
        }


class HuggingFaceLoadRequest(BaseModel):
    """
    Request to load a HuggingFace dataset.
    """

    dataset_name: str = Field(
        ...,
        description="HuggingFace dataset identifier (e.g., 'renhehuang/govQA-database-zhtw')",
    )
    config_name: Optional[str] = Field(None, description="Dataset configuration name")
    split: str = Field("train", description="Dataset split to load")
    max_entries: Optional[int] = Field(
        None, ge=1, le=100000, description="Maximum number of entries to load"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "renhehuang/govQA-database-zhtw",
                "config_name": None,
                "split": "train",
                "max_entries": 1000,
            }
        }


class AugmentPreviewRequest(BaseModel):
    """
    Request for augmentation preview.

    Requirements satisfied:
    - 2.3: Preview mode before full processing
    """

    file_id: str = Field(..., description="ID of the uploaded file")
    target_field: str = Field(..., description="Field to augment")
    instruction: str = Field(..., description="Augmentation instruction")
    model: str = Field("llama3.2", description="Ollama model to use")
    create_new_field: bool = Field(False, description="Whether to create a new field")
    context_fields: list[str] = Field(
        default_factory=list, description="Additional fields to include as context"
    )
    preview_count: int = Field(
        3, ge=1, le=10, description="Number of entries to preview"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "file_xyz789",
                "target_field": "output",
                "instruction": "Make the response more formal",
                "model": "llama3.2",
                "create_new_field": False,
                "context_fields": ["instruction", "input"],
                "preview_count": 3,
            }
        }


class AugmentPreviewResponse(BaseModel):
    """
    Response with augmentation preview.

    Requirements satisfied:
    - 5.2: Show before and after comparison
    """

    previews: list[dict[str, Any]] = Field(
        ..., description="List of preview entries with original and augmented versions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "previews": [
                    {
                        "original": {"instruction": "Say hi", "output": "Hey!"},
                        "augmented": {
                            "instruction": "Say hi",
                            "output": "Good day, how may I assist you?",
                        },
                    }
                ]
            }
        }


class AugmentationRequest(BaseModel):
    """
    Request for full augmentation.

    Requirements satisfied:
    - 2.2: Augmentation parameters specification
    """

    file_id: str = Field(..., description="ID of the uploaded file")
    target_field: str = Field(..., description="Field to augment")
    instruction: str = Field(..., description="Augmentation instruction")
    model: str = Field("llama3.2", description="Ollama model to use")
    language: OutputLanguage = Field(OutputLanguage.EN, description="Output language")
    create_new_field: bool = Field(False, description="Whether to create a new field")
    context_fields: list[str] = Field(
        default_factory=list, description="Additional fields to include as context"
    )
    concurrency: int = Field(
        5, ge=1, le=20, description="Number of concurrent requests"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "file_xyz789",
                "target_field": "output",
                "instruction": "Make the response more formal",
                "model": "llama3.2",
                "language": "en",
                "create_new_field": False,
                "context_fields": ["instruction", "input"],
                "concurrency": 5,
            }
        }


# ============================================================================
# Task Status Models
# ============================================================================


class TaskStatusEnum(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(BaseModel):
    """
    Task status model.

    Requirements satisfied:
    - 3.1: Progress bar showing completion percentage
    - 3.3: Total duration and success statistics
    """

    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatusEnum = Field(..., description="Current task status")
    progress: int = Field(0, ge=0, description="Number of completed items")
    total: int = Field(0, ge=0, description="Total number of items")
    result: Optional[dict[str, Any]] = Field(None, description="Task result data")
    error: Optional[str] = Field(None, description="Error message if task failed")
    duration: Optional[float] = Field(
        None, ge=0, description="Task duration in seconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "gen_abc123",
                "status": "running",
                "progress": 45,
                "total": 100,
                "result": None,
                "error": None,
                "duration": None,
            }
        }


# ============================================================================
# Model Information Models
# ============================================================================


class ModelInfo(BaseModel):
    """
    Ollama model information.

    Requirements satisfied:
    - 10.2: Model names with size information
    """

    name: str = Field(..., description="Model name")
    size: Optional[str] = Field(None, description="Model size (e.g., '7B', '13B')")
    modified_at: Optional[str] = Field(None, description="Last modified timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "llama3.2",
                "size": "3.2B",
                "modified_at": "2024-01-15T10:30:00Z",
            }
        }


class ModelListResponse(BaseModel):
    """Response containing list of available models."""

    models: list[ModelInfo] = Field(..., description="List of available Ollama models")

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "name": "llama3.2",
                        "size": "3.2B",
                        "modified_at": "2024-01-15T10:30:00Z",
                    },
                    {
                        "name": "mistral",
                        "size": "7B",
                        "modified_at": "2024-01-10T08:20:00Z",
                    },
                ]
            }
        }


# ============================================================================
# Error Response Models
# ============================================================================


class ErrorResponse(BaseModel):
    """
    Standard error response.

    Requirements satisfied:
    - 1.4: Clear error messages
    - 7.5: Clear error message for Ollama unavailability
    """

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(
        None, description="Additional error details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "OllamaConnectionError",
                "message": "Unable to connect to Ollama service. Please ensure Ollama is running.",
                "details": {"host": "localhost:11434"},
            }
        }


# ============================================================================
# Download Models
# ============================================================================


class DownloadRequest(BaseModel):
    """Request for dataset download with format specification."""

    format: Literal["jsonl", "json", "csv", "tsv", "parquet"] = Field(
        "jsonl", description="Output format for the dataset"
    )

    class Config:
        json_schema_extra = {"example": {"format": "jsonl"}}
