"""
Task management API routes for OllaForge web interface.

This module provides REST API endpoints for task management:
- GET /api/tasks: List all tasks
- GET /api/tasks/stats: Get task manager statistics
- GET /api/tasks/{task_id}: Get task status
- DELETE /api/tasks/{task_id}: Cancel or delete a task

Requirements satisfied:
- 7.1: Process multiple requests independently
- 7.2: Operations don't block other endpoints
- 7.3: Queue requests when resources are limited
- 7.4: Return appropriate error responses for timeouts
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..models import ErrorResponse
from ..services.task_manager import TaskStatus as TMTaskStatus
from ..services.task_manager import TaskType, task_manager

# Create router
router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get(
    "",
    responses={200: {"description": "List of tasks"}},
    summary="List All Tasks",
    description="""
List all tasks with optional filtering.

Returns a list of all generation and augmentation tasks, optionally
filtered by type and/or status.

### Filters

- **task_type**: Filter by task type (generation, augmentation)
- **status**: Filter by status (pending, queued, running, completed, failed, timeout, cancelled)
- **limit**: Maximum number of tasks to return (default 100)
""",
)
async def list_tasks(
    task_type: Optional[str] = Query(
        None, regex="^(generation|augmentation)$", description="Filter by task type"
    ),
    status: Optional[str] = Query(
        None,
        regex="^(pending|queued|running|completed|failed|timeout|cancelled)$",
        description="Filter by status",
    ),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of tasks to return"
    ),
) -> dict:
    """
    List all tasks with optional filtering.

    Args:
        task_type: Filter by task type (generation, augmentation)
        status: Filter by status
        limit: Maximum number of tasks to return

    Returns:
        Dict with list of tasks
    """
    # Convert string filters to enums
    type_filter = None
    if task_type:
        type_filter = (
            TaskType.GENERATION if task_type == "generation" else TaskType.AUGMENTATION
        )

    status_filter = None
    if status:
        status_map = {
            "pending": TMTaskStatus.PENDING,
            "queued": TMTaskStatus.QUEUED,
            "running": TMTaskStatus.RUNNING,
            "completed": TMTaskStatus.COMPLETED,
            "failed": TMTaskStatus.FAILED,
            "timeout": TMTaskStatus.TIMEOUT,
            "cancelled": TMTaskStatus.CANCELLED,
        }
        status_filter = status_map.get(status)

    tasks = task_manager.list_tasks(
        task_type=type_filter, status=status_filter, limit=limit
    )

    return {"tasks": tasks, "count": len(tasks)}


@router.get(
    "/stats",
    responses={200: {"description": "Task manager statistics"}},
    summary="Get Task Statistics",
    description="""
Get task manager statistics.

Returns statistics about the task queue and processing:
- Total tasks by status
- Queue length
- Active tasks count
- Resource utilization
""",
)
async def get_stats() -> dict:
    """
    Get task manager statistics.

    Returns:
        Dict with statistics about tasks and queue

    Requirements satisfied:
    - 7.3: Queue requests when resources are limited (shows queue status)
    """
    return task_manager.get_stats()


@router.get(
    "/{task_id}",
    responses={
        200: {"description": "Task status"},
        404: {"model": ErrorResponse, "description": "Task not found"},
    },
    summary="Get Task by ID",
    description="Get detailed status information for a specific task.",
)
async def get_task(task_id: str) -> dict:
    """
    Get task status by ID.

    Args:
        task_id: Task identifier

    Returns:
        Task status dict

    Raises:
        HTTPException 404: If task not found
    """
    task = task_manager.get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "TaskNotFound", "message": f"Task {task_id} not found"},
        )

    return task


@router.post(
    "/{task_id}/cancel",
    responses={
        200: {"description": "Task cancelled"},
        404: {"model": ErrorResponse, "description": "Task not found"},
        400: {"model": ErrorResponse, "description": "Task cannot be cancelled"},
    },
    summary="Cancel Task",
    description="""
Cancel a running or queued task.

Only tasks in 'pending', 'queued', or 'running' status can be cancelled.
Completed or failed tasks cannot be cancelled.
""",
)
async def cancel_task(task_id: str) -> dict:
    """
    Cancel a running or queued task.

    Args:
        task_id: Task identifier

    Returns:
        Cancellation result

    Raises:
        HTTPException 404: If task not found
        HTTPException 400: If task cannot be cancelled
    """
    task = task_manager.get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "TaskNotFound", "message": f"Task {task_id} not found"},
        )

    success = await task_manager.cancel_task(task_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "CannotCancel",
                "message": f"Task {task_id} cannot be cancelled (status: {task['status']})",
            },
        )

    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Task cancelled successfully",
    }


@router.delete(
    "/{task_id}",
    responses={
        200: {"description": "Task deleted"},
        404: {"model": ErrorResponse, "description": "Task not found"},
        400: {"model": ErrorResponse, "description": "Task cannot be deleted"},
    },
    summary="Delete Task",
    description="""
Delete a completed or failed task.

Only tasks that are no longer running can be deleted.
This removes the task from the task list and frees associated resources.
""",
)
async def delete_task(task_id: str) -> dict:
    """
    Delete a completed/failed task.

    Args:
        task_id: Task identifier

    Returns:
        Deletion result

    Raises:
        HTTPException 404: If task not found
        HTTPException 400: If task cannot be deleted (still running)
    """
    task = task_manager.get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "TaskNotFound", "message": f"Task {task_id} not found"},
        )

    success = task_manager.delete_task(task_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "CannotDelete",
                "message": f"Task {task_id} cannot be deleted (status: {task['status']})",
            },
        )

    return {"task_id": task_id, "message": "Task deleted successfully"}


@router.post(
    "/cleanup",
    responses={200: {"description": "Cleanup result"}},
    summary="Cleanup Old Tasks",
    description="""
Clean up old completed and failed tasks.

Removes tasks older than the specified age to free up memory.
Default is 24 hours.
""",
)
async def cleanup_old_tasks(
    max_age_hours: float = Query(
        24.0, ge=0.1, le=720, description="Maximum age in hours for tasks to keep"
    )
) -> dict:
    """
    Clean up old completed/failed tasks.

    Args:
        max_age_hours: Maximum age in hours (default 24)

    Returns:
        Number of tasks cleaned up
    """
    max_age_seconds = max_age_hours * 3600
    count = await task_manager.cleanup_old_tasks(max_age_seconds)

    return {"cleaned_up": count, "message": f"Cleaned up {count} old tasks"}
