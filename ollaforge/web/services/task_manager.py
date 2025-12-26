"""
Task manager for OllaForge web interface.

This module provides centralized task management with:
- Task queue and status tracking
- Concurrent request handling with resource limits
- Request queueing when resources are limited
- Timeout handling for long-running tasks

Requirements satisfied:
- 7.1: Process multiple requests independently
- 7.2: Operations don't block other endpoints
- 7.3: Queue requests when resources are limited
- 7.4: Return appropriate error responses for timeouts
"""

import asyncio
import uuid
from typing import Optional, Callable, Dict, Any, Awaitable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import time


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Task type enumeration."""
    GENERATION = "generation"
    AUGMENTATION = "augmentation"


@dataclass
class Task:
    """Task data class."""
    task_id: str
    task_type: TaskType
    status: TaskStatus
    progress: int = 0
    total: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    timeout_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class TaskTimeoutError(Exception):
    """Raised when a task exceeds its timeout."""
    pass


class TaskQueueFullError(Exception):
    """Raised when the task queue is full."""
    pass


class TaskManager:
    """
    Centralized task manager for handling concurrent requests.
    
    Features:
    - Task creation and tracking
    - Concurrent execution with configurable limits
    - Request queueing when at capacity
    - Timeout handling
    - Task cancellation
    
    Requirements satisfied:
    - 7.1: Process multiple requests independently
    - 7.2: Operations don't block other endpoints
    - 7.3: Queue requests when resources are limited
    - 7.4: Return appropriate error responses for timeouts
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        max_queue_size: int = 100,
        default_timeout: float = 3600.0  # 1 hour default
    ):
        """
        Initialize the task manager.
        
        Args:
            max_concurrent_tasks: Maximum number of tasks running concurrently
            max_queue_size: Maximum number of tasks in the queue
            default_timeout: Default timeout in seconds for tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._running_count = 0
        self._queued_count = 0
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        # Running task futures for cancellation
        self._running_futures: Dict[str, asyncio.Task] = {}
    
    def create_task(
        self,
        task_type: TaskType,
        timeout_seconds: Optional[float] = None
    ) -> str:
        """
        Create a new task and return its ID.
        
        Args:
            task_type: Type of task (generation or augmentation)
            timeout_seconds: Optional timeout override
            
        Returns:
            Task ID string
        """
        prefix = "gen" if task_type == TaskType.GENERATION else "aug"
        task_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            timeout_seconds=timeout_seconds or self.default_timeout
        )
        
        self.tasks[task_id] = task
        return task_id
    
    async def submit_task(
        self,
        task_id: str,
        coroutine: Callable[[], Awaitable[Dict[str, Any]]],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Submit a task for execution.
        
        The task will be queued if max concurrent tasks is reached.
        
        Args:
            task_id: Task identifier
            coroutine: Async function to execute
            on_progress: Optional progress callback
            
        Raises:
            TaskQueueFullError: If queue is full
            KeyError: If task_id not found
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task not found: {task_id}")
        
        task = self.tasks[task_id]
        
        # Check queue capacity
        async with self._lock:
            if self._queued_count >= self.max_queue_size:
                task.status = TaskStatus.FAILED
                task.error = "Task queue is full. Please try again later."
                raise TaskQueueFullError("Task queue is full")
            
            # Mark as queued if we need to wait
            if self._running_count >= self.max_concurrent_tasks:
                task.status = TaskStatus.QUEUED
                self._queued_count += 1
        
        # Execute task with semaphore for concurrency control
        asyncio.create_task(self._execute_task(task_id, coroutine, on_progress))
    
    async def _execute_task(
        self,
        task_id: str,
        coroutine: Callable[[], Awaitable[Dict[str, Any]]],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Execute a task with concurrency control and timeout.
        
        Args:
            task_id: Task identifier
            coroutine: Async function to execute
            on_progress: Optional progress callback
        """
        task = self.tasks.get(task_id)
        if not task:
            return
        
        # Wait for semaphore (respects max_concurrent_tasks)
        async with self._semaphore:
            async with self._lock:
                if task.status == TaskStatus.QUEUED:
                    self._queued_count -= 1
                self._running_count += 1
            
            # Update task status to running
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow().isoformat()
            task.updated_at = task.started_at
            
            try:
                # Create the task with timeout
                coro = coroutine()
                
                # Wrap in asyncio.Task for cancellation support
                future = asyncio.create_task(coro)
                self._running_futures[task_id] = future
                
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        future,
                        timeout=task.timeout_seconds
                    )
                    
                    # Task completed successfully
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.utcnow().isoformat()
                    
                except asyncio.TimeoutError:
                    # Task timed out
                    task.status = TaskStatus.TIMEOUT
                    task.error = f"Task timed out after {task.timeout_seconds} seconds"
                    task.completed_at = datetime.utcnow().isoformat()
                    
                except asyncio.CancelledError:
                    # Task was cancelled
                    task.status = TaskStatus.CANCELLED
                    task.error = "Task was cancelled"
                    task.completed_at = datetime.utcnow().isoformat()
                    
                except Exception as e:
                    # Task failed with error
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.utcnow().isoformat()
                
                finally:
                    # Remove from running futures
                    self._running_futures.pop(task_id, None)
                    
            finally:
                # Update running count
                async with self._lock:
                    self._running_count -= 1
                
                task.updated_at = datetime.utcnow().isoformat()
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        total: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update task status.
        
        Args:
            task_id: Task identifier
            status: New status
            progress: Current progress count
            total: Total items count
            result: Result data
            error: Error message
            
        Returns:
            True if task was updated, False if not found
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if status is not None:
            task.status = status
        if progress is not None:
            task.progress = progress
        if total is not None:
            task.total = total
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        
        task.updated_at = datetime.utcnow().isoformat()
        return True
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status as dictionary.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task data dict or None if not found
        """
        task = self.tasks.get(task_id)
        if task:
            return task.to_dict()
        return None
    
    def get_task_object(self, task_id: str) -> Optional[Task]:
        """
        Get task object directly.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task object or None if not found
        """
        return self.tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running or queued task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # Can only cancel pending, queued, or running tasks
        if task.status not in [TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING]:
            return False
        
        # Cancel running future if exists
        future = self._running_futures.get(task_id)
        if future and not future.done():
            future.cancel()
        
        task.status = TaskStatus.CANCELLED
        task.error = "Task was cancelled by user"
        task.completed_at = datetime.utcnow().isoformat()
        task.updated_at = task.completed_at
        
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from storage.
        
        Only completed, failed, timeout, or cancelled tasks can be deleted.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was deleted, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # Don't delete running or queued tasks
        if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING]:
            return False
        
        del self.tasks[task_id]
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get task manager statistics.
        
        Returns:
            Dict with statistics about tasks and queue
        """
        status_counts = {}
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "running_count": self._running_count,
            "queued_count": self._queued_count,
            "max_concurrent": self.max_concurrent_tasks,
            "max_queue_size": self.max_queue_size,
            "status_counts": status_counts
        }
    
    def list_tasks(
        self,
        task_type: Optional[TaskType] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> list[Dict[str, Any]]:
        """
        List tasks with optional filtering.
        
        Args:
            task_type: Filter by task type
            status: Filter by status
            limit: Maximum number of tasks to return
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        for task in self.tasks.values():
            if task_type and task.task_type != task_type:
                continue
            if status and task.status != status:
                continue
            tasks.append(task.to_dict())
            if len(tasks) >= limit:
                break
        
        # Sort by created_at descending
        tasks.sort(key=lambda t: t["created_at"], reverse=True)
        return tasks
    
    async def cleanup_old_tasks(self, max_age_seconds: float = 86400) -> int:
        """
        Clean up old completed/failed tasks.
        
        Args:
            max_age_seconds: Maximum age in seconds (default 24 hours)
            
        Returns:
            Number of tasks cleaned up
        """
        now = datetime.utcnow()
        to_delete = []
        
        for task_id, task in self.tasks.items():
            # Only clean up finished tasks
            if task.status not in [
                TaskStatus.COMPLETED, TaskStatus.FAILED,
                TaskStatus.TIMEOUT, TaskStatus.CANCELLED
            ]:
                continue
            
            # Check age
            if task.completed_at:
                completed = datetime.fromisoformat(task.completed_at)
                age = (now - completed).total_seconds()
                if age > max_age_seconds:
                    to_delete.append(task_id)
        
        # Delete old tasks
        for task_id in to_delete:
            del self.tasks[task_id]
        
        return len(to_delete)


# Global task manager instance
task_manager = TaskManager()
