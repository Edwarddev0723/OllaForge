"""
Generation API routes for OllaForge web interface.

This module provides REST API endpoints for dataset generation:
- POST /api/generate: Start a new generation task
- GET /api/generate/{task_id}: Get task status
- GET /api/generate/{task_id}/download: Download generated dataset

Requirements satisfied:
- 1.2: Initiate dataset generation with valid parameters
- 1.3: Provide download link for generated dataset
- 1.4: Display clear error messages
- 3.1: Display progress bar showing completion percentage
- 3.2: Update progress in real-time during processing
"""

import asyncio
import io
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...client import OllamaConnectionError, OllamaGenerationError
from ...formats import FileFormat, write_file
from ..models import (
    ErrorResponse,
    GenerationRequest,
    GenerationResponse,
    TaskStatus,
    TaskStatusEnum,
)
from ..services.generation import GenerationService
from .websocket import ws_manager

# Create router
router = APIRouter(prefix="/api/generate", tags=["generation"])

# Global service instance
generation_service = GenerationService()


@router.post("", response_model=GenerationResponse, responses={
    503: {"model": ErrorResponse, "description": "Ollama service unavailable"},
    400: {"model": ErrorResponse, "description": "Invalid request parameters"}
},
    summary="Start Dataset Generation",
    description="""
Start a new dataset generation task.

This endpoint initiates dataset generation in the background and returns
a task ID that can be used to track progress via WebSocket or polling,
and to download results when complete.

### Supported Dataset Types

- **sft**: Supervised Fine-tuning format (instruction, input, output)
- **pretrain**: Pre-training format (text only)
- **sft_conv**: Conversational SFT format (multi-turn conversations)
- **dpo**: Direct Preference Optimization format (chosen/rejected pairs)

### Quality Control

When generating Traditional Chinese (zh-tw) content, QC filtering can be enabled
to ensure content quality. The confidence threshold controls how strict the filtering is.
"""
)
async def start_generation(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
) -> GenerationResponse:
    """
    Start a new dataset generation task.

    This endpoint initiates dataset generation in the background and returns
    a task ID that can be used to track progress and download results.

    Requirements satisfied:
    - 1.2: Initiate dataset generation with valid parameters
    - 3.1: Display progress bar during generation

    Args:
        request: Generation configuration
        background_tasks: FastAPI background tasks

    Returns:
        GenerationResponse with task_id and status

    Raises:
        HTTPException 503: If Ollama service is unavailable
        HTTPException 400: If request parameters are invalid
    """
    # Create task
    task_id = generation_service.create_task()

    # Start generation in background
    background_tasks.add_task(
        _run_generation_task,
        task_id,
        request
    )

    return GenerationResponse(
        task_id=task_id,
        status="pending",
        message="Generation task created successfully"
    )


async def _run_generation_task(task_id: str, request: GenerationRequest):
    """
    Run generation task in background.

    Args:
        task_id: Task identifier
        request: Generation configuration
    """
    try:
        # Update task status to running
        generation_service.update_task(
            task_id,
            status="running",
            total=request.count
        )

        # Emit initial progress via WebSocket
        await ws_manager.emit_progress(
            task_id=task_id,
            progress=0,
            total=request.count,
            status="running",
            message="Starting generation..."
        )

        # Progress callback that updates both task state and WebSocket
        async def async_progress_callback(completed: int, total: int):
            generation_service.update_task(
                task_id,
                progress=completed,
                total=total
            )
            await ws_manager.emit_progress(
                task_id=task_id,
                progress=completed,
                total=total,
                status="running",
                message=f"Generated {completed}/{total} entries"
            )

        # Sync wrapper for progress callback
        def progress_callback(completed: int, total: int):
            generation_service.update_task(
                task_id,
                progress=completed,
                total=total
            )
            # Schedule async emit
            asyncio.create_task(ws_manager.emit_progress(
                task_id=task_id,
                progress=completed,
                total=total,
                status="running",
                message=f"Generated {completed}/{total} entries"
            ))

        # Run generation
        result = await generation_service.generate_dataset(
            topic=request.topic,
            count=request.count,
            model=request.model,
            dataset_type=request.dataset_type,
            language=request.language,
            qc_enabled=request.qc_enabled,
            qc_confidence=request.qc_confidence,
            progress_callback=progress_callback
        )

        # Update task with result
        generation_service.update_task(
            task_id,
            status="completed",
            progress=result["total"],
            total=result["total"],
            result=result
        )

        # Emit completion via WebSocket
        await ws_manager.emit_completed(
            task_id=task_id,
            total=result["total"],
            success_count=result["total"],
            failure_count=0,
            duration=result.get("duration", 0),
            message="Generation completed successfully"
        )

    except OllamaConnectionError as e:
        # Ollama connection error
        error_msg = f"Ollama service unavailable: {str(e)}"
        generation_service.update_task(
            task_id,
            status="failed",
            error=error_msg
        )
        await ws_manager.emit_failed(
            task_id=task_id,
            error=error_msg,
            details={"error_type": "connection"}
        )
    except OllamaGenerationError as e:
        # Generation error
        error_msg = f"Generation failed: {str(e)}"
        generation_service.update_task(
            task_id,
            status="failed",
            error=error_msg
        )
        await ws_manager.emit_failed(
            task_id=task_id,
            error=error_msg,
            details={"error_type": "generation"}
        )
    except Exception as e:
        # Unexpected error
        error_msg = f"Unexpected error: {str(e)}"
        generation_service.update_task(
            task_id,
            status="failed",
            error=error_msg
        )
        await ws_manager.emit_failed(
            task_id=task_id,
            error=error_msg,
            details={"error_type": "unexpected"}
        )


@router.get("/{task_id}", response_model=TaskStatus, responses={
    404: {"model": ErrorResponse, "description": "Task not found"}
},
    summary="Get Generation Task Status",
    description="""
Get the current status of a generation task.

Returns progress information including:
- Current progress (number of entries generated)
- Total entries to generate
- Task status (pending, running, completed, failed)
- Duration (when completed)
- Error message (if failed)

Use this endpoint to poll for task status, or use WebSocket for real-time updates.
"""
)
async def get_generation_status(task_id: str) -> TaskStatus:
    """
    Get the status of a generation task.

    Requirements satisfied:
    - 3.1: Display progress bar showing completion percentage
    - 3.3: Display total duration and success statistics

    Args:
        task_id: Task identifier

    Returns:
        TaskStatus with current progress and status

    Raises:
        HTTPException 404: If task not found
    """
    task = generation_service.get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "TaskNotFound", "message": f"Task {task_id} not found"}
        )

    # Calculate duration if task is completed
    duration = None
    if task["status"] == "completed" and task.get("result"):
        duration = task["result"].get("duration")

    return TaskStatus(
        task_id=task_id,
        status=TaskStatusEnum(task["status"]),
        progress=task["progress"],
        total=task["total"],
        result=task.get("result"),
        error=task.get("error"),
        duration=duration
    )


@router.get("/{task_id}/download", responses={
    200: {"description": "Dataset file download"},
    404: {"model": ErrorResponse, "description": "Task not found or not completed"},
    400: {"model": ErrorResponse, "description": "Invalid format parameter"}
},
    summary="Download Generated Dataset",
    description="""
Download the generated dataset in the specified format.

### Supported Formats

- **jsonl**: JSON Lines format (default) - one JSON object per line
- **json**: Standard JSON array format
- **csv**: Comma-separated values
- **tsv**: Tab-separated values
- **parquet**: Apache Parquet binary format

The file will be downloaded with an appropriate filename based on the task ID.
"""
)
async def download_generated_dataset(
    task_id: str,
    format: str = Query("jsonl", regex="^(jsonl|json|csv|tsv|parquet)$", description="Output format for the dataset")
) -> StreamingResponse:
    """
    Download the generated dataset.

    Requirements satisfied:
    - 1.3: Provide download link for generated dataset
    - 4.2: Format selection options for download

    Args:
        task_id: Task identifier
        format: Output format (jsonl, json, csv, tsv, parquet)

    Returns:
        StreamingResponse with dataset file

    Raises:
        HTTPException 404: If task not found or not completed
        HTTPException 400: If format is invalid
    """
    task = generation_service.get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "TaskNotFound", "message": f"Task {task_id} not found"}
        )

    if task["status"] != "completed":
        raise HTTPException(
            status_code=404,
            detail={
                "error": "TaskNotCompleted",
                "message": f"Task {task_id} is not completed yet"
            }
        )

    if not task.get("result") or not task["result"].get("entries"):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NoData",
                "message": "No data available for download"
            }
        )

    # Get entries
    entries = task["result"]["entries"]

    # Convert entries to dictionaries
    entries_dict = [entry.model_dump() if hasattr(entry, 'model_dump') else entry for entry in entries]

    # Convert to requested format
    try:
        # Create temporary file in memory
        output = io.BytesIO()

        # Map format string to FileFormat enum
        format_map = {
            "jsonl": FileFormat.JSONL,
            "json": FileFormat.JSON,
            "csv": FileFormat.CSV,
            "tsv": FileFormat.TSV,
            "parquet": FileFormat.PARQUET
        }

        if format not in format_map:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "InvalidFormat",
                    "message": f"Unsupported format: {format}"
                }
            )

        # Write to BytesIO using formats module
        # For text formats, write to string first then encode
        file_format = format_map[format]

        if format in ["jsonl", "json", "csv", "tsv"]:
            # Text formats - write to string buffer first
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix=f'.{format}', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                write_file(entries_dict, tmp_path, file_format)
                with open(tmp_path, 'rb') as f:
                    output.write(f.read())
            finally:
                os.unlink(tmp_path)
        else:
            # Binary format (parquet)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                write_file(entries_dict, tmp_path, file_format)
                with open(tmp_path, 'rb') as f:
                    output.write(f.read())
            finally:
                os.unlink(tmp_path)

        # Reset buffer position
        output.seek(0)

        # Determine media type
        media_types = {
            "jsonl": "application/x-ndjson",
            "json": "application/json",
            "csv": "text/csv",
            "tsv": "text/tab-separated-values",
            "parquet": "application/octet-stream"
        }

        # Return streaming response
        return StreamingResponse(
            output,
            media_type=media_types[format],
            headers={
                "Content-Disposition": f"attachment; filename=dataset_{task_id}.{format}"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ConversionError",
                "message": f"Failed to convert dataset: {str(e)}"
            }
        )
