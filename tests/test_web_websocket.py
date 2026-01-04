"""
Tests for WebSocket event handlers and progress tracking.

This module tests the WebSocket functionality including:
- Connection and disconnection handling
- Task subscription mechanism
- Progress event emission
- Error event handling

Requirements satisfied:
- 3.1: Display progress bar showing completion percentage
- 3.2: Update progress in real-time during processing
- 3.3: Display total duration and success/failure statistics
- 3.4: Continue showing progress even when individual items fail
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ollaforge.web.routes.websocket import WebSocketManager


@pytest.fixture
def ws_manager_instance():
    """Create a fresh WebSocket manager instance for testing."""
    manager = WebSocketManager()
    # Create a mock Socket.IO server
    mock_sio = AsyncMock()
    manager.set_sio(mock_sio)
    return manager


# ============================================================================
# Strategies for Property-Based Testing
# ============================================================================

# Strategy for session IDs
sid_strategy = st.text(
    alphabet='abcdefghijklmnopqrstuvwxyz0123456789',
    min_size=8,
    max_size=16
)

# Strategy for task IDs
task_id_strategy = st.text(
    alphabet='abcdefghijklmnopqrstuvwxyz0123456789_',
    min_size=5,
    max_size=20
).filter(lambda x: x.startswith('gen_') or x.startswith('aug_') or len(x) > 5)

# Strategy for progress values
progress_strategy = st.integers(min_value=0, max_value=1000)

# Strategy for status values
status_strategy = st.sampled_from(["pending", "running", "completed", "failed"])


# ============================================================================
# Property Test 8: Progress indicators show during operations
# ============================================================================

@given(
    task_id=task_id_strategy,
    progress=progress_strategy,
    total=st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=20, deadline=10000)
def test_progress_indicators_show_during_operations(task_id, progress, total):
    """
    **Feature: web-interface, Property 8: Progress indicators show during operations**
    **Validates: Requirements 3.1**

    For any task with subscribers, emitting progress should send progress data
    including completion percentage to all subscribers.
    """
    # Ensure progress doesn't exceed total
    progress = min(progress, total)

    async def run_test():
        manager = WebSocketManager()
        mock_sio = AsyncMock()
        manager.set_sio(mock_sio)

        # Connect a client and subscribe to task
        sid = "test_client_001"
        manager.on_connect(sid)
        manager.subscribe(sid, task_id)

        # Emit progress
        await manager.emit_progress(
            task_id=task_id,
            progress=progress,
            total=total,
            status="running"
        )

        # Verify emit was called
        mock_sio.emit.assert_called_once()
        call_args = mock_sio.emit.call_args

        # Check event name
        assert call_args[0][0] == "progress", "Event should be 'progress'"

        # Check event data
        event_data = call_args[0][1]
        assert event_data["task_id"] == task_id
        assert event_data["progress"] == progress
        assert event_data["total"] == total
        assert "percentage" in event_data

        # Verify percentage calculation
        expected_percentage = round(progress / total * 100, 1)
        assert event_data["percentage"] == expected_percentage

        # Verify sent to correct client
        assert call_args[1]["to"] == sid

    asyncio.run(run_test())


# ============================================================================
# Property Test 9: Progress updates in real-time
# ============================================================================

@given(
    task_id=task_id_strategy,
    num_updates=st.integers(min_value=2, max_value=10),
    total=st.integers(min_value=10, max_value=100)
)
@settings(max_examples=20, deadline=15000)
def test_progress_updates_in_real_time(task_id, num_updates, total):
    """
    **Feature: web-interface, Property 9: Progress updates in real-time**
    **Validates: Requirements 3.2**

    For any sequence of progress updates, each update should be emitted
    to subscribers immediately with:
    1. Progress values in increasing order
    2. Timestamps included in each update
    3. Timestamps in chronological order (proving real-time emission)
    4. Each update emitted individually (not batched)
    """
    async def run_test():
        manager = WebSocketManager()
        mock_sio = AsyncMock()
        manager.set_sio(mock_sio)

        # Connect and subscribe
        sid = "test_client_002"
        manager.on_connect(sid)
        manager.subscribe(sid, task_id)

        # Emit multiple progress updates
        progress_values = []
        for i in range(num_updates):
            progress = min((i + 1) * (total // num_updates), total)
            progress_values.append(progress)
            await manager.emit_progress(
                task_id=task_id,
                progress=progress,
                total=total,
                status="running"
            )

        # Verify all updates were emitted (not batched)
        assert mock_sio.emit.call_count == num_updates, \
            f"Expected {num_updates} individual emissions, got {mock_sio.emit.call_count}"

        # Verify progress values and timestamps
        emitted_progress = []
        timestamps = []
        for call in mock_sio.emit.call_args_list:
            event_data = call[0][1]
            emitted_progress.append(event_data["progress"])

            # Each update must contain a timestamp for real-time tracking
            assert "timestamp" in event_data, \
                "Each progress update must include a timestamp"
            timestamps.append(event_data["timestamp"])

        # Progress values should be emitted in order
        assert emitted_progress == progress_values, \
            "Progress values should be emitted in order"

        # Timestamps should be in chronological order (real-time emission)
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1], \
                f"Timestamps should be in chronological order: {timestamps[i-1]} should be <= {timestamps[i]}"

    asyncio.run(run_test())


# ============================================================================
# Property Test 10: Completion shows statistics
# ============================================================================

@given(
    task_id=task_id_strategy,
    total=st.integers(min_value=1, max_value=100),
    success_count=st.integers(min_value=0, max_value=100),
    duration=st.floats(min_value=0.1, max_value=1000.0)
)
@settings(max_examples=20, deadline=10000)
def test_completion_shows_statistics(task_id, total, success_count, duration):
    """
    **Feature: web-interface, Property 10: Completion shows statistics**
    **Validates: Requirements 3.3**

    For any completed task, the completion event should include
    total count, success count, failure count, and duration.
    """
    # Ensure success_count doesn't exceed total
    success_count = min(success_count, total)
    failure_count = total - success_count

    async def run_test():
        manager = WebSocketManager()
        mock_sio = AsyncMock()
        manager.set_sio(mock_sio)

        # Connect and subscribe
        sid = "test_client_003"
        manager.on_connect(sid)
        manager.subscribe(sid, task_id)

        # Emit completion
        await manager.emit_completed(
            task_id=task_id,
            total=total,
            success_count=success_count,
            failure_count=failure_count,
            duration=duration
        )

        # Verify emit was called
        mock_sio.emit.assert_called_once()
        call_args = mock_sio.emit.call_args

        # Check event name
        assert call_args[0][0] == "completed", "Event should be 'completed'"

        # Check event data contains all statistics
        event_data = call_args[0][1]
        assert event_data["task_id"] == task_id
        assert event_data["status"] == "completed"
        assert event_data["total"] == total
        assert event_data["success_count"] == success_count
        assert event_data["failure_count"] == failure_count
        assert event_data["duration"] == round(duration, 2)
        assert "timestamp" in event_data

    asyncio.run(run_test())


# ============================================================================
# Property Test 11: Errors don't stop progress display
# ============================================================================

@given(
    task_id=task_id_strategy,
    error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    error_type=st.sampled_from(["error", "warning", "item_error"])
)
@settings(max_examples=20, deadline=10000)
def test_errors_dont_stop_progress_display(task_id, error_message, error_type):
    """
    **Feature: web-interface, Property 11: Errors don't stop progress display**
    **Validates: Requirements 3.4**

    For any error during processing, the error should be emitted as an event
    without stopping the progress tracking mechanism.
    """
    async def run_test():
        manager = WebSocketManager()
        mock_sio = AsyncMock()
        manager.set_sio(mock_sio)

        # Connect and subscribe
        sid = "test_client_004"
        manager.on_connect(sid)
        manager.subscribe(sid, task_id)

        # Emit progress first
        await manager.emit_progress(
            task_id=task_id,
            progress=5,
            total=10,
            status="running"
        )

        # Emit error (should not affect progress tracking)
        await manager.emit_error(
            task_id=task_id,
            error=error_message,
            error_type=error_type
        )

        # Emit more progress (should still work)
        await manager.emit_progress(
            task_id=task_id,
            progress=6,
            total=10,
            status="running"
        )

        # Verify all events were emitted
        assert mock_sio.emit.call_count == 3

        # Verify event types
        events = [call[0][0] for call in mock_sio.emit.call_args_list]
        assert events == ["progress", "error", "progress"]

        # Verify error event data
        error_call = mock_sio.emit.call_args_list[1]
        error_data = error_call[0][1]
        assert error_data["task_id"] == task_id
        assert error_data["error"] == error_message
        assert error_data["error_type"] == error_type

    asyncio.run(run_test())


# ============================================================================
# Unit Tests for WebSocket Manager
# ============================================================================

@pytest.mark.asyncio
async def test_websocket_manager_connect(ws_manager_instance):
    """
    Test that WebSocket manager handles client connections.

    Requirements satisfied:
    - 3.1: Connection handling
    """
    sid = "test_client_connect"

    # Connect client
    ws_manager_instance.on_connect(sid)

    # Verify client is tracked
    subscriptions = ws_manager_instance.get_subscriptions(sid)
    assert subscriptions == set(), "New client should have no subscriptions"


@pytest.mark.asyncio
async def test_websocket_manager_disconnect(ws_manager_instance):
    """
    Test that WebSocket manager handles client disconnections.

    Requirements satisfied:
    - 3.1: Connection handling
    """
    sid = "test_client_disconnect"
    task_id = "gen_test_task"

    # Connect and subscribe
    ws_manager_instance.on_connect(sid)
    ws_manager_instance.subscribe(sid, task_id)

    # Verify subscription
    assert task_id in ws_manager_instance.get_subscriptions(sid)
    assert sid in ws_manager_instance.get_subscribers(task_id)

    # Disconnect
    ws_manager_instance.on_disconnect(sid)

    # Verify cleanup
    assert ws_manager_instance.get_subscriptions(sid) == set()
    assert sid not in ws_manager_instance.get_subscribers(task_id)


@pytest.mark.asyncio
async def test_websocket_manager_subscribe(ws_manager_instance):
    """
    Test that WebSocket manager handles task subscriptions.

    Requirements satisfied:
    - 3.2: Task subscription mechanism
    """
    sid = "test_client_subscribe"
    task_id = "gen_test_task"

    # Connect
    ws_manager_instance.on_connect(sid)

    # Subscribe
    success = ws_manager_instance.subscribe(sid, task_id)

    assert success is True
    assert task_id in ws_manager_instance.get_subscriptions(sid)
    assert sid in ws_manager_instance.get_subscribers(task_id)


@pytest.mark.asyncio
async def test_websocket_manager_subscribe_without_connect(ws_manager_instance):
    """
    Test that subscription fails for non-connected clients.
    """
    sid = "test_client_not_connected"
    task_id = "gen_test_task"

    # Try to subscribe without connecting
    success = ws_manager_instance.subscribe(sid, task_id)

    assert success is False


@pytest.mark.asyncio
async def test_websocket_manager_unsubscribe(ws_manager_instance):
    """
    Test that WebSocket manager handles task unsubscriptions.

    Requirements satisfied:
    - 3.2: Task subscription mechanism
    """
    sid = "test_client_unsubscribe"
    task_id = "gen_test_task"

    # Connect and subscribe
    ws_manager_instance.on_connect(sid)
    ws_manager_instance.subscribe(sid, task_id)

    # Unsubscribe
    success = ws_manager_instance.unsubscribe(sid, task_id)

    assert success is True
    assert task_id not in ws_manager_instance.get_subscriptions(sid)
    assert sid not in ws_manager_instance.get_subscribers(task_id)


@pytest.mark.asyncio
async def test_websocket_manager_multiple_subscribers(ws_manager_instance):
    """
    Test that multiple clients can subscribe to the same task.

    Requirements satisfied:
    - 3.2: Multiple client support
    """
    task_id = "gen_shared_task"
    clients = ["client_1", "client_2", "client_3"]

    # Connect and subscribe all clients
    for sid in clients:
        ws_manager_instance.on_connect(sid)
        ws_manager_instance.subscribe(sid, task_id)

    # Verify all clients are subscribed
    subscribers = ws_manager_instance.get_subscribers(task_id)
    assert len(subscribers) == 3
    for sid in clients:
        assert sid in subscribers


@pytest.mark.asyncio
async def test_websocket_manager_emit_to_subscribers(ws_manager_instance):
    """
    Test that progress is emitted to all subscribers.

    Requirements satisfied:
    - 3.2: Emit to all subscribers
    """
    task_id = "gen_emit_test"
    clients = ["client_a", "client_b"]

    # Connect and subscribe
    for sid in clients:
        ws_manager_instance.on_connect(sid)
        ws_manager_instance.subscribe(sid, task_id)

    # Emit progress
    await ws_manager_instance.emit_progress(
        task_id=task_id,
        progress=5,
        total=10,
        status="running"
    )

    # Verify emit was called for each subscriber
    mock_sio = ws_manager_instance._sio
    assert mock_sio.emit.call_count == 2


@pytest.mark.asyncio
async def test_websocket_manager_no_emit_without_subscribers(ws_manager_instance):
    """
    Test that no emit occurs when there are no subscribers.
    """
    task_id = "gen_no_subscribers"

    # Emit progress without any subscribers
    await ws_manager_instance.emit_progress(
        task_id=task_id,
        progress=5,
        total=10,
        status="running"
    )

    # Verify no emit was called
    mock_sio = ws_manager_instance._sio
    mock_sio.emit.assert_not_called()


@pytest.mark.asyncio
async def test_websocket_manager_emit_completed(ws_manager_instance):
    """
    Test that completion event is emitted correctly.

    Requirements satisfied:
    - 3.3: Completion statistics
    """
    task_id = "gen_completed_test"
    sid = "client_completed"

    # Connect and subscribe
    ws_manager_instance.on_connect(sid)
    ws_manager_instance.subscribe(sid, task_id)

    # Emit completion
    await ws_manager_instance.emit_completed(
        task_id=task_id,
        total=100,
        success_count=95,
        failure_count=5,
        duration=45.5,
        message="Test completed"
    )

    # Verify emit
    mock_sio = ws_manager_instance._sio
    mock_sio.emit.assert_called_once()

    call_args = mock_sio.emit.call_args
    assert call_args[0][0] == "completed"

    event_data = call_args[0][1]
    assert event_data["total"] == 100
    assert event_data["success_count"] == 95
    assert event_data["failure_count"] == 5
    assert event_data["duration"] == 45.5
    assert event_data["message"] == "Test completed"


@pytest.mark.asyncio
async def test_websocket_manager_emit_error(ws_manager_instance):
    """
    Test that error events are emitted correctly.

    Requirements satisfied:
    - 3.4: Error event handling
    """
    task_id = "gen_error_test"
    sid = "client_error"

    # Connect and subscribe
    ws_manager_instance.on_connect(sid)
    ws_manager_instance.subscribe(sid, task_id)

    # Emit error
    await ws_manager_instance.emit_error(
        task_id=task_id,
        error="Test error message",
        error_type="item_error",
        details={"item_index": 5}
    )

    # Verify emit
    mock_sio = ws_manager_instance._sio
    mock_sio.emit.assert_called_once()

    call_args = mock_sio.emit.call_args
    assert call_args[0][0] == "error"

    event_data = call_args[0][1]
    assert event_data["error"] == "Test error message"
    assert event_data["error_type"] == "item_error"
    assert event_data["details"]["item_index"] == 5


@pytest.mark.asyncio
async def test_websocket_manager_emit_failed(ws_manager_instance):
    """
    Test that failure events are emitted correctly.
    """
    task_id = "gen_failed_test"
    sid = "client_failed"

    # Connect and subscribe
    ws_manager_instance.on_connect(sid)
    ws_manager_instance.subscribe(sid, task_id)

    # Emit failure
    await ws_manager_instance.emit_failed(
        task_id=task_id,
        error="Task failed completely",
        details={"reason": "connection_lost"}
    )

    # Verify emit
    mock_sio = ws_manager_instance._sio
    mock_sio.emit.assert_called_once()

    call_args = mock_sio.emit.call_args
    assert call_args[0][0] == "failed"

    event_data = call_args[0][1]
    assert event_data["status"] == "failed"
    assert event_data["error"] == "Task failed completely"


@pytest.mark.asyncio
async def test_websocket_manager_percentage_calculation(ws_manager_instance):
    """
    Test that percentage is calculated correctly.

    Requirements satisfied:
    - 3.1: Progress percentage
    """
    task_id = "gen_percentage_test"
    sid = "client_percentage"

    # Connect and subscribe
    ws_manager_instance.on_connect(sid)
    ws_manager_instance.subscribe(sid, task_id)

    # Test various progress values
    test_cases = [
        (0, 100, 0.0),
        (50, 100, 50.0),
        (100, 100, 100.0),
        (33, 100, 33.0),
        (1, 3, 33.3),
    ]

    for progress, total, expected_percentage in test_cases:
        ws_manager_instance._sio.reset_mock()

        await ws_manager_instance.emit_progress(
            task_id=task_id,
            progress=progress,
            total=total,
            status="running"
        )

        call_args = ws_manager_instance._sio.emit.call_args
        event_data = call_args[0][1]
        assert event_data["percentage"] == expected_percentage, \
            f"Expected {expected_percentage}% for {progress}/{total}"


@pytest.mark.asyncio
async def test_websocket_manager_handles_emit_exception(ws_manager_instance):
    """
    Test that emit exceptions are handled gracefully.
    """
    task_id = "gen_exception_test"
    sid = "client_exception"

    # Connect and subscribe
    ws_manager_instance.on_connect(sid)
    ws_manager_instance.subscribe(sid, task_id)

    # Make emit raise an exception
    ws_manager_instance._sio.emit.side_effect = Exception("Connection lost")

    # Should not raise
    await ws_manager_instance.emit_progress(
        task_id=task_id,
        progress=5,
        total=10,
        status="running"
    )

    # Test passed if no exception was raised
