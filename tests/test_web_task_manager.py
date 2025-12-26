"""
Tests for task management and concurrency in OllaForge web interface.

This module tests the task manager functionality including:
- Task creation and tracking
- Concurrent request handling
- Request queueing for resource limits
- Timeout handling

Requirements satisfied:
- 7.1: Process multiple requests independently
- 7.2: Operations don't block other endpoints
- 7.3: Queue requests when resources are limited
- 7.4: Return appropriate error responses for timeouts
"""

import pytest
import asyncio
import time
from hypothesis import given, strategies as st, settings, assume
import httpx
from httpx._transports.asgi import ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock

from ollaforge.web.server import app
from ollaforge.web.services.task_manager import (
    TaskManager,
    TaskStatus,
    TaskType,
    Task,
    TaskTimeoutError,
    TaskQueueFullError,
    task_manager
)


def create_test_client():
    """Create an async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ============================================================================
# Strategies for Property-Based Testing
# ============================================================================

# Strategy for task counts
task_count_strategy = st.integers(min_value=1, max_value=10)

# Strategy for timeout values
timeout_strategy = st.floats(min_value=0.1, max_value=5.0)

# Strategy for task types
task_type_strategy = st.sampled_from([TaskType.GENERATION, TaskType.AUGMENTATION])


# ============================================================================
# Property Test 21: Concurrent requests are independent
# ============================================================================

@given(
    task_count=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=10, deadline=30000)
def test_concurrent_requests_are_independent(task_count):
    """
    **Feature: web-interface, Property 21: Concurrent requests are independent**
    **Validates: Requirements 7.1**
    
    For any multiple generation requests submitted simultaneously, the system
    should process each request independently without interference.
    """
    async def run_test():
        # Create a fresh task manager for this test
        manager = TaskManager(max_concurrent_tasks=task_count)
        
        # Track results for each task
        results = {}
        
        async def mock_task(task_id: str, value: int):
            """Mock task that returns a unique value."""
            await asyncio.sleep(0.05)  # Small delay to simulate work
            return {"task_id": task_id, "value": value}
        
        # Create and submit multiple tasks
        task_ids = []
        for i in range(task_count):
            task_id = manager.create_task(TaskType.GENERATION)
            task_ids.append(task_id)
            
            # Submit task with unique value
            await manager.submit_task(
                task_id,
                lambda tid=task_id, val=i: mock_task(tid, val)
            )
        
        # Wait for all tasks to complete
        await asyncio.sleep(0.5)
        
        # Verify each task completed independently
        for i, task_id in enumerate(task_ids):
            task = manager.get_task(task_id)
            assert task is not None, f"Task {task_id} should exist"
            assert task["status"] == "completed", \
                f"Task {task_id} should be completed, got {task['status']}"
            
            # Verify result contains correct value
            result = task.get("result", {})
            assert result.get("task_id") == task_id, \
                f"Task {task_id} result should have correct task_id"
            assert result.get("value") == i, \
                f"Task {task_id} should have value {i}, got {result.get('value')}"
    
    asyncio.run(run_test())


# ============================================================================
# Property Test 22: Operations don't block endpoints
# ============================================================================

@given(
    slow_task_delay=st.floats(min_value=0.1, max_value=0.3)
)
@settings(max_examples=10, deadline=30000)
def test_operations_dont_block_endpoints(slow_task_delay):
    """
    **Feature: web-interface, Property 22: Operations don't block endpoints**
    **Validates: Requirements 7.2**
    
    For any generation or augmentation in progress, other API endpoints
    should remain responsive and not be blocked.
    """
    async def run_test():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Start a slow task in the background
            slow_task_started = asyncio.Event()
            
            async def slow_operation():
                slow_task_started.set()
                await asyncio.sleep(slow_task_delay)
                return {"status": "done"}
            
            # Create task in task manager
            task_id = task_manager.create_task(TaskType.GENERATION)
            
            # Submit slow task
            asyncio.create_task(
                task_manager.submit_task(task_id, slow_operation)
            )
            
            # Wait for slow task to start
            await asyncio.wait_for(slow_task_started.wait(), timeout=1.0)
            
            # While slow task is running, verify other endpoints respond quickly
            start_time = time.time()
            
            # Health check should respond immediately
            response = await client.get("/health")
            health_time = time.time() - start_time
            
            assert response.status_code == 200, "Health endpoint should respond"
            assert health_time < 0.5, \
                f"Health endpoint should respond quickly, took {health_time}s"
            
            # Stats endpoint should respond immediately
            start_time = time.time()
            response = await client.get("/api/tasks/stats")
            stats_time = time.time() - start_time
            
            assert response.status_code == 200, "Stats endpoint should respond"
            assert stats_time < 0.5, \
                f"Stats endpoint should respond quickly, took {stats_time}s"
            
            # Wait for slow task to complete
            await asyncio.sleep(slow_task_delay + 0.1)
    
    asyncio.run(run_test())


# ============================================================================
# Property Test 23: Resource limits trigger queueing
# ============================================================================

@given(
    max_concurrent=st.integers(min_value=1, max_value=3),
    total_tasks=st.integers(min_value=2, max_value=6)
)
@settings(max_examples=10, deadline=30000)
def test_resource_limits_trigger_queueing(max_concurrent, total_tasks):
    """
    **Feature: web-interface, Property 23: Resource limits trigger queueing**
    **Validates: Requirements 7.3**
    
    For any situation where system resources are limited, the system should
    queue requests and process them sequentially.
    """
    # Ensure we have more tasks than concurrent limit
    assume(total_tasks > max_concurrent)
    
    async def run_test():
        # Create task manager with limited concurrency
        manager = TaskManager(max_concurrent_tasks=max_concurrent)
        
        # Track execution order
        execution_order = []
        execution_lock = asyncio.Lock()
        
        async def tracked_task(task_num: int):
            """Task that tracks when it starts executing."""
            async with execution_lock:
                execution_order.append(task_num)
            await asyncio.sleep(0.05)
            return {"task_num": task_num}
        
        # Submit more tasks than max_concurrent
        task_ids = []
        for i in range(total_tasks):
            task_id = manager.create_task(TaskType.GENERATION)
            task_ids.append(task_id)
            await manager.submit_task(
                task_id,
                lambda num=i: tracked_task(num)
            )
        
        # Give tasks time to start
        await asyncio.sleep(0.02)
        
        # Check stats - should show queueing
        stats = manager.get_stats()
        
        # At most max_concurrent should be running
        assert stats["running_count"] <= max_concurrent, \
            f"Running count {stats['running_count']} should not exceed {max_concurrent}"
        
        # Wait for all tasks to complete
        await asyncio.sleep(0.5)
        
        # All tasks should complete eventually
        for task_id in task_ids:
            task = manager.get_task(task_id)
            assert task["status"] == "completed", \
                f"Task {task_id} should complete, got {task['status']}"
        
        # All tasks should have executed
        assert len(execution_order) == total_tasks, \
            f"All {total_tasks} tasks should have executed"
    
    asyncio.run(run_test())


# ============================================================================
# Property Test 24: Timeouts return error responses
# ============================================================================

@given(
    timeout_seconds=st.floats(min_value=0.05, max_value=0.2)
)
@settings(max_examples=10, deadline=30000)
def test_timeouts_return_error_responses(timeout_seconds):
    """
    **Feature: web-interface, Property 24: Timeouts return error responses**
    **Validates: Requirements 7.4**
    
    For any request that times out, the system should return an appropriate
    error response.
    """
    async def run_test():
        # Create task manager with short timeout
        manager = TaskManager(default_timeout=timeout_seconds)
        
        async def slow_task():
            """Task that takes longer than timeout."""
            await asyncio.sleep(timeout_seconds * 3)
            return {"status": "done"}
        
        # Create and submit task
        task_id = manager.create_task(TaskType.GENERATION)
        await manager.submit_task(task_id, slow_task)
        
        # Wait for timeout to occur
        await asyncio.sleep(timeout_seconds * 2)
        
        # Task should be in timeout status
        task = manager.get_task(task_id)
        assert task is not None, "Task should exist"
        assert task["status"] == "timeout", \
            f"Task should be timeout, got {task['status']}"
        assert task["error"] is not None, "Task should have error message"
        assert "timed out" in task["error"].lower() or "timeout" in task["error"].lower(), \
            f"Error should mention timeout: {task['error']}"
    
    asyncio.run(run_test())


# ============================================================================
# Unit Tests for Task Manager
# ============================================================================

class TestTaskManagerBasics:
    """Unit tests for basic task manager operations."""
    
    def test_create_task_generation(self):
        """Test creating a generation task."""
        manager = TaskManager()
        task_id = manager.create_task(TaskType.GENERATION)
        
        assert task_id.startswith("gen_")
        assert len(task_id) > 4
        
        task = manager.get_task(task_id)
        assert task is not None
        assert task["status"] == "pending"
        assert task["task_type"] == "generation"
    
    def test_create_task_augmentation(self):
        """Test creating an augmentation task."""
        manager = TaskManager()
        task_id = manager.create_task(TaskType.AUGMENTATION)
        
        assert task_id.startswith("aug_")
        
        task = manager.get_task(task_id)
        assert task["task_type"] == "augmentation"
    
    def test_update_task(self):
        """Test updating task status."""
        manager = TaskManager()
        task_id = manager.create_task(TaskType.GENERATION)
        
        manager.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            progress=5,
            total=10
        )
        
        task = manager.get_task(task_id)
        assert task["status"] == "running"
        assert task["progress"] == 5
        assert task["total"] == 10
    
    def test_get_nonexistent_task(self):
        """Test getting a task that doesn't exist."""
        manager = TaskManager()
        task = manager.get_task("nonexistent_task_id")
        assert task is None
    
    def test_delete_completed_task(self):
        """Test deleting a completed task."""
        manager = TaskManager()
        task_id = manager.create_task(TaskType.GENERATION)
        
        # Mark as completed
        manager.update_task(task_id, status=TaskStatus.COMPLETED)
        
        # Delete should succeed
        success = manager.delete_task(task_id)
        assert success is True
        assert manager.get_task(task_id) is None
    
    def test_delete_running_task_fails(self):
        """Test that deleting a running task fails."""
        manager = TaskManager()
        task_id = manager.create_task(TaskType.GENERATION)
        
        # Mark as running
        manager.update_task(task_id, status=TaskStatus.RUNNING)
        
        # Delete should fail
        success = manager.delete_task(task_id)
        assert success is False
        assert manager.get_task(task_id) is not None


class TestTaskManagerConcurrency:
    """Unit tests for task manager concurrency features."""
    
    @pytest.mark.asyncio
    async def test_submit_and_complete_task(self):
        """Test submitting and completing a task."""
        manager = TaskManager()
        task_id = manager.create_task(TaskType.GENERATION)
        
        async def simple_task():
            return {"result": "success"}
        
        await manager.submit_task(task_id, simple_task)
        
        # Wait for completion
        await asyncio.sleep(0.1)
        
        task = manager.get_task(task_id)
        assert task["status"] == "completed"
        assert task["result"]["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_task_failure_handling(self):
        """Test that task failures are handled correctly."""
        manager = TaskManager()
        task_id = manager.create_task(TaskType.GENERATION)
        
        async def failing_task():
            raise ValueError("Test error")
        
        await manager.submit_task(task_id, failing_task)
        
        # Wait for failure
        await asyncio.sleep(0.1)
        
        task = manager.get_task(task_id)
        assert task["status"] == "failed"
        assert "Test error" in task["error"]
    
    @pytest.mark.asyncio
    async def test_cancel_running_task(self):
        """Test cancelling a running task."""
        manager = TaskManager()
        task_id = manager.create_task(TaskType.GENERATION)
        
        started = asyncio.Event()
        
        async def long_task():
            started.set()
            await asyncio.sleep(10)
            return {"result": "done"}
        
        await manager.submit_task(task_id, long_task)
        
        # Wait for task to start
        await asyncio.wait_for(started.wait(), timeout=1.0)
        
        # Cancel the task
        success = await manager.cancel_task(task_id)
        assert success is True
        
        # Wait for cancellation to process
        await asyncio.sleep(0.1)
        
        task = manager.get_task(task_id)
        assert task["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting task manager statistics."""
        manager = TaskManager(max_concurrent_tasks=2, max_queue_size=10)
        
        # Create some tasks
        manager.create_task(TaskType.GENERATION)
        manager.create_task(TaskType.AUGMENTATION)
        
        stats = manager.get_stats()
        
        assert stats["total_tasks"] == 2
        assert stats["max_concurrent"] == 2
        assert stats["max_queue_size"] == 10
        assert "status_counts" in stats
    
    @pytest.mark.asyncio
    async def test_list_tasks_with_filter(self):
        """Test listing tasks with filters."""
        manager = TaskManager()
        
        # Create tasks of different types
        gen_id = manager.create_task(TaskType.GENERATION)
        aug_id = manager.create_task(TaskType.AUGMENTATION)
        
        # Mark one as completed
        manager.update_task(gen_id, status=TaskStatus.COMPLETED)
        
        # List all
        all_tasks = manager.list_tasks()
        assert len(all_tasks) == 2
        
        # List by type
        gen_tasks = manager.list_tasks(task_type=TaskType.GENERATION)
        assert len(gen_tasks) == 1
        assert gen_tasks[0]["task_id"] == gen_id
        
        # List by status
        completed_tasks = manager.list_tasks(status=TaskStatus.COMPLETED)
        assert len(completed_tasks) == 1
        assert completed_tasks[0]["task_id"] == gen_id


# ============================================================================
# Unit Tests for Task API Routes
# ============================================================================

@pytest.mark.asyncio
async def test_list_tasks_endpoint():
    """Test the list tasks endpoint."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/tasks")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "count" in data


@pytest.mark.asyncio
async def test_get_stats_endpoint():
    """Test the stats endpoint."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/tasks/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_tasks" in data
        assert "running_count" in data
        assert "max_concurrent" in data


@pytest.mark.asyncio
async def test_get_task_not_found():
    """Test getting a non-existent task."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/tasks/nonexistent_task_id")
        
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_cancel_task_not_found():
    """Test cancelling a non-existent task."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/tasks/nonexistent_task_id/cancel")
        
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_task_not_found():
    """Test deleting a non-existent task."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.delete("/api/tasks/nonexistent_task_id")
        
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_cleanup_endpoint():
    """Test the cleanup endpoint."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/tasks/cleanup?max_age_hours=24")
        
        assert response.status_code == 200
        data = response.json()
        assert "cleaned_up" in data


@pytest.mark.asyncio
async def test_list_tasks_with_type_filter():
    """Test listing tasks with type filter."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/tasks?task_type=generation")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data


@pytest.mark.asyncio
async def test_list_tasks_with_status_filter():
    """Test listing tasks with status filter."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/tasks?status=completed")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data


@pytest.mark.asyncio
async def test_list_tasks_invalid_type_filter():
    """Test listing tasks with invalid type filter."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/tasks?task_type=invalid")
        
        # Should return 422 for validation error
        assert response.status_code == 422
