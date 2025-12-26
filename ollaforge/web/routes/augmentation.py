"""
Augmentation API routes for OllaForge web interface.

This module provides REST API endpoints for dataset augmentation:
- POST /api/augment/upload: Upload a dataset file
- POST /api/augment/preview: Preview augmentation on sample entries
- POST /api/augment: Start full augmentation
- GET /api/augment/{task_id}: Get task status
- GET /api/augment/{task_id}/download: Download augmented dataset

Requirements satisfied:
- 2.1: Upload and validate dataset files
- 2.2: Validate target field exists in dataset
- 2.3: Preview augmentation before full processing
- 2.4: Provide download link for augmented dataset
- 2.5: Preserve original data on failure
- 3.1: Display progress bar showing completion percentage
- 3.2: Update progress in real-time during processing
"""

import asyncio
import os
import io
import tempfile
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import StreamingResponse

from ..models import (
    AugmentUploadResponse,
    AugmentPreviewRequest,
    AugmentPreviewResponse,
    AugmentationRequest,
    GenerationResponse,
    TaskStatus,
    TaskStatusEnum,
    ErrorResponse
)
from ..services.augmentation import AugmentationService
from .websocket import ws_manager
from ...client import OllamaConnectionError, OllamaGenerationError
from ...file_manager import FileOperationError
from ...formats import write_file, FileFormat


# Create router
router = APIRouter(prefix="/api/augment", tags=["augmentation"])

# Global service instance
augmentation_service = AugmentationService()


@router.post("/upload", response_model=AugmentUploadResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid file or format"},
    415: {"model": ErrorResponse, "description": "Unsupported file format"}
},
    summary="Upload Dataset for Augmentation",
    description="""
Upload a dataset file for augmentation.

### Supported Formats

- **JSONL**: JSON Lines format (.jsonl)
- **JSON**: Standard JSON array format (.json)
- **CSV**: Comma-separated values (.csv)
- **TSV**: Tab-separated values (.tsv)
- **Parquet**: Apache Parquet format (.parquet)

Returns file information including:
- Unique file ID for subsequent operations
- Total entry count
- List of available field names
- Preview of first few entries
"""
)
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file to upload")
) -> AugmentUploadResponse:
    """
    Upload a dataset file for augmentation.
    
    Supports JSONL, JSON, CSV, TSV, and Parquet formats.
    Returns file info including available fields and preview entries.
    
    Requirements satisfied:
    - 2.1: Upload and validate dataset files
    - 5.3: Display first 3 entries
    
    Args:
        file: Uploaded file
        
    Returns:
        AugmentUploadResponse with file_id, fields, and preview
        
    Raises:
        HTTPException 400: If file cannot be parsed
        HTTPException 415: If file format is not supported
    """
    # Validate file extension
    filename = file.filename or "unknown"
    valid_extensions = [".jsonl", ".json", ".csv", ".tsv", ".parquet"]
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "UnsupportedFormat",
                "message": f"Unsupported file format: {ext}. Supported formats: {', '.join(valid_extensions)}"
            }
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Upload and process file
        result = await augmentation_service.upload_file(content, filename)
        
        return AugmentUploadResponse(
            file_id=result["file_id"],
            entry_count=result["entry_count"],
            fields=result["fields"],
            preview=result["preview"]
        )
        
    except FileOperationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "FileProcessingError",
                "message": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "UploadError",
                "message": f"Failed to process uploaded file: {str(e)}"
            }
        )


@router.post("/preview", response_model=AugmentPreviewResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid request parameters"},
    404: {"model": ErrorResponse, "description": "File not found"},
    503: {"model": ErrorResponse, "description": "Ollama service unavailable"}
},
    summary="Preview Augmentation",
    description="""
Preview augmentation on sample entries before full processing.

This endpoint generates augmented versions of a few entries to preview
the effect of the augmentation instruction. Use this to verify the
augmentation will produce the desired results before processing the
entire dataset.

Returns original and augmented versions side-by-side for comparison.
"""
)
async def preview_augmentation(
    request: AugmentPreviewRequest
) -> AugmentPreviewResponse:
    """
    Preview augmentation on sample entries.
    
    Generates augmented versions of a few entries to preview
    the effect of the augmentation before full processing.
    
    Requirements satisfied:
    - 2.3: Preview augmentation before full processing
    - 5.2: Show before and after comparison
    
    Args:
        request: Preview configuration
        
    Returns:
        AugmentPreviewResponse with original and augmented entries
        
    Raises:
        HTTPException 400: If request parameters are invalid
        HTTPException 404: If file not found
        HTTPException 503: If Ollama service is unavailable
    """
    # Validate file exists
    file_info = augmentation_service.get_uploaded_file(request.file_id)
    if not file_info:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "FileNotFound",
                "message": f"File {request.file_id} not found"
            }
        )
    
    # Validate target field
    is_valid, error_msg = augmentation_service.validate_field(
        request.file_id,
        request.target_field,
        request.create_new_field
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "InvalidField",
                "message": error_msg
            }
        )
    
    try:
        # Generate preview
        previews = await augmentation_service.preview_augmentation(
            file_id=request.file_id,
            target_field=request.target_field,
            instruction=request.instruction,
            model=request.model,
            create_new_field=request.create_new_field,
            context_fields=request.context_fields,
            preview_count=request.preview_count
        )
        
        return AugmentPreviewResponse(previews=previews)
        
    except OllamaConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OllamaConnectionError",
                "message": f"Unable to connect to Ollama service: {str(e)}"
            }
        )
    except FileOperationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "PreviewError",
                "message": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PreviewError",
                "message": f"Preview failed: {str(e)}"
            }
        )


@router.post("", response_model=GenerationResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid request parameters"},
    404: {"model": ErrorResponse, "description": "File not found"},
    503: {"model": ErrorResponse, "description": "Ollama service unavailable"}
},
    summary="Start Dataset Augmentation",
    description="""
Start a full augmentation task on an uploaded dataset.

Initiates augmentation in the background and returns a task ID
that can be used to track progress and download results.

### Augmentation Options

- **target_field**: The field to augment (must exist unless create_new_field is true)
- **instruction**: Natural language instruction for the augmentation
- **create_new_field**: If true, creates a new field instead of modifying existing
- **context_fields**: Additional fields to include as context for the LLM
- **concurrency**: Number of concurrent requests (1-20)

### Error Handling

If some entries fail during augmentation, the original data is preserved
and failure statistics are reported in the result.
"""
)
async def start_augmentation(
    request: AugmentationRequest,
    background_tasks: BackgroundTasks
) -> GenerationResponse:
    """
    Start a full augmentation task.
    
    Initiates augmentation in the background and returns a task ID
    that can be used to track progress and download results.
    
    Requirements satisfied:
    - 2.2: Augmentation parameters specification
    - 2.4: Provide download link for augmented dataset
    - 3.1: Display progress bar during augmentation
    
    Args:
        request: Augmentation configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        GenerationResponse with task_id and status
        
    Raises:
        HTTPException 400: If request parameters are invalid
        HTTPException 404: If file not found
    """
    # Validate file exists
    file_info = augmentation_service.get_uploaded_file(request.file_id)
    if not file_info:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "FileNotFound",
                "message": f"File {request.file_id} not found"
            }
        )
    
    # Validate target field
    is_valid, error_msg = augmentation_service.validate_field(
        request.file_id,
        request.target_field,
        request.create_new_field
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "InvalidField",
                "message": error_msg
            }
        )
    
    # Create task
    task_id = augmentation_service.create_task()
    
    # Start augmentation in background
    background_tasks.add_task(
        _run_augmentation_task,
        task_id,
        request
    )
    
    return GenerationResponse(
        task_id=task_id,
        status="pending",
        message="Augmentation task created successfully"
    )


async def _run_augmentation_task(task_id: str, request: AugmentationRequest):
    """
    Run augmentation task in background.
    
    Args:
        task_id: Task identifier
        request: Augmentation configuration
    """
    try:
        # Get file info for total count
        file_info = augmentation_service.get_uploaded_file(request.file_id)
        total = file_info["entry_count"] if file_info else 0
        
        # Update task status to running
        augmentation_service.update_task(
            task_id,
            status="running",
            total=total
        )
        
        # Emit initial progress via WebSocket
        await ws_manager.emit_progress(
            task_id=task_id,
            progress=0,
            total=total,
            status="running",
            message="Starting augmentation..."
        )
        
        # Progress callback that updates both task state and WebSocket
        def progress_callback(completed: int, total: int):
            augmentation_service.update_task(
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
                message=f"Augmented {completed}/{total} entries"
            ))
        
        # Run augmentation
        result = await augmentation_service.augment_dataset(
            file_id=request.file_id,
            target_field=request.target_field,
            instruction=request.instruction,
            model=request.model,
            language=request.language,
            create_new_field=request.create_new_field,
            context_fields=request.context_fields,
            concurrency=request.concurrency,
            progress_callback=progress_callback
        )
        
        # Update task with result
        augmentation_service.update_task(
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
            success_count=result.get("success_count", result["total"]),
            failure_count=result.get("failure_count", 0),
            duration=result.get("duration", 0),
            message="Augmentation completed successfully"
        )
        
        # Emit item errors if any
        if result.get("errors"):
            for error in result["errors"][:5]:  # Limit to first 5 errors
                await ws_manager.emit_error(
                    task_id=task_id,
                    error=error,
                    error_type="item_error"
                )
        
    except OllamaConnectionError as e:
        error_msg = f"Ollama service unavailable: {str(e)}"
        augmentation_service.update_task(
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
        error_msg = f"Augmentation failed: {str(e)}"
        augmentation_service.update_task(
            task_id,
            status="failed",
            error=error_msg
        )
        await ws_manager.emit_failed(
            task_id=task_id,
            error=error_msg,
            details={"error_type": "generation"}
        )
    except FileOperationError as e:
        error_msg = f"File error: {str(e)}"
        augmentation_service.update_task(
            task_id,
            status="failed",
            error=error_msg
        )
        await ws_manager.emit_failed(
            task_id=task_id,
            error=error_msg,
            details={"error_type": "file"}
        )
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        augmentation_service.update_task(
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
    summary="Get Augmentation Task Status",
    description="""
Get the current status of an augmentation task.

Returns progress information including:
- Current progress (number of entries augmented)
- Total entries to augment
- Task status (pending, running, completed, failed)
- Duration (when completed)
- Success/failure counts
- Error message (if failed)
"""
)
async def get_augmentation_status(task_id: str) -> TaskStatus:
    """
    Get the status of an augmentation task.
    
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
    task = augmentation_service.get_task(task_id)
    
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
    summary="Download Augmented Dataset",
    description="""
Download the augmented dataset in the specified format.

### Supported Formats

- **jsonl**: JSON Lines format (default)
- **json**: Standard JSON array format
- **csv**: Comma-separated values
- **tsv**: Tab-separated values
- **parquet**: Apache Parquet binary format
"""
)
async def download_augmented_dataset(
    task_id: str,
    format: str = Query("jsonl", regex="^(jsonl|json|csv|tsv|parquet)$", description="Output format for the dataset")
) -> StreamingResponse:
    """
    Download the augmented dataset.
    
    Requirements satisfied:
    - 2.4: Provide download link for augmented dataset
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
    task = augmentation_service.get_task(task_id)
    
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
    
    # Convert to requested format
    try:
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
        
        file_format = format_map[format]
        
        # Write to temp file then read back
        with tempfile.NamedTemporaryFile(
            mode='wb' if format == 'parquet' else 'w+',
            suffix=f'.{format}',
            delete=False
        ) as tmp:
            tmp_path = tmp.name
        
        try:
            write_file(entries, tmp_path, file_format)
            with open(tmp_path, 'rb') as f:
                output.write(f.read())
        finally:
            os.unlink(tmp_path)
        
        output.seek(0)
        
        # Determine media type
        media_types = {
            "jsonl": "application/x-ndjson",
            "json": "application/json",
            "csv": "text/csv",
            "tsv": "text/tab-separated-values",
            "parquet": "application/octet-stream"
        }
        
        return StreamingResponse(
            output,
            media_type=media_types[format],
            headers={
                "Content-Disposition": f"attachment; filename=augmented_{task_id}.{format}"
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
