"""
WebSocket event handlers for OllaForge web interface.

This module provides Socket.IO event handlers for real-time progress tracking:
- Task subscription mechanism for clients to receive updates
- Progress event emission during generation and augmentation
- Error event handling

Requirements satisfied:
- 3.1: Display progress bar showing completion percentage
- 3.2: Update progress in real-time during processing
- 3.4: Continue showing progress even when individual items fail
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TaskSubscription:
    """Represents a client's subscription to a task."""

    task_id: str
    subscribed_at: datetime = field(default_factory=datetime.utcnow)


class WebSocketManager:
    """
    Manager for WebSocket connections and task subscriptions.

    Handles:
    - Client connection tracking
    - Task subscription management
    - Progress event broadcasting
    """

    def __init__(self):
        """Initialize the WebSocket manager."""
        # Map of session ID to set of subscribed task IDs
        self._subscriptions: dict[str, set[str]] = {}
        # Map of task ID to set of subscribed session IDs
        self._task_subscribers: dict[str, set[str]] = {}
        # Reference to Socket.IO server (set during initialization)
        self._sio = None

    def set_sio(self, sio):
        """
        Set the Socket.IO server reference.

        Args:
            sio: Socket.IO AsyncServer instance
        """
        self._sio = sio

    def on_connect(self, sid: str):
        """
        Handle client connection.

        Args:
            sid: Session ID of the connected client
        """
        self._subscriptions[sid] = set()

    def on_disconnect(self, sid: str):
        """
        Handle client disconnection.

        Cleans up all subscriptions for the disconnected client.

        Args:
            sid: Session ID of the disconnected client
        """
        # Get all tasks this client was subscribed to
        task_ids = self._subscriptions.pop(sid, set())

        # Remove client from task subscriber lists
        for task_id in task_ids:
            if task_id in self._task_subscribers:
                self._task_subscribers[task_id].discard(sid)
                # Clean up empty subscriber sets
                if not self._task_subscribers[task_id]:
                    del self._task_subscribers[task_id]

    def subscribe(self, sid: str, task_id: str) -> bool:
        """
        Subscribe a client to task updates.

        Args:
            sid: Session ID of the client
            task_id: Task ID to subscribe to

        Returns:
            True if subscription was successful, False otherwise
        """
        if sid not in self._subscriptions:
            return False

        # Add to client's subscriptions
        self._subscriptions[sid].add(task_id)

        # Add to task's subscribers
        if task_id not in self._task_subscribers:
            self._task_subscribers[task_id] = set()
        self._task_subscribers[task_id].add(sid)

        return True

    def unsubscribe(self, sid: str, task_id: str) -> bool:
        """
        Unsubscribe a client from task updates.

        Args:
            sid: Session ID of the client
            task_id: Task ID to unsubscribe from

        Returns:
            True if unsubscription was successful, False otherwise
        """
        if sid not in self._subscriptions:
            return False

        # Remove from client's subscriptions
        self._subscriptions[sid].discard(task_id)

        # Remove from task's subscribers
        if task_id in self._task_subscribers:
            self._task_subscribers[task_id].discard(sid)
            if not self._task_subscribers[task_id]:
                del self._task_subscribers[task_id]

        return True

    def get_subscribers(self, task_id: str) -> set[str]:
        """
        Get all subscribers for a task.

        Args:
            task_id: Task ID

        Returns:
            Set of session IDs subscribed to the task
        """
        return self._task_subscribers.get(task_id, set()).copy()

    def get_subscriptions(self, sid: str) -> set[str]:
        """
        Get all task subscriptions for a client.

        Args:
            sid: Session ID

        Returns:
            Set of task IDs the client is subscribed to
        """
        return self._subscriptions.get(sid, set()).copy()

    async def emit_progress(
        self,
        task_id: str,
        progress: int,
        total: int,
        status: str = "running",
        message: Optional[str] = None,
    ):
        """
        Emit progress update to all subscribers of a task.

        Args:
            task_id: Task ID
            progress: Current progress count
            total: Total items count
            status: Task status (pending, running, completed, failed)
            message: Optional status message

        Requirements satisfied:
        - 3.1: Progress bar showing completion percentage
        - 3.2: Update progress in real-time
        """
        if not self._sio:
            return

        subscribers = self.get_subscribers(task_id)
        if not subscribers:
            return

        # Calculate percentage
        percentage = (progress / total * 100) if total > 0 else 0

        event_data = {
            "task_id": task_id,
            "progress": progress,
            "total": total,
            "percentage": round(percentage, 1),
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if message:
            event_data["message"] = message

        # Emit to all subscribers
        for sid in subscribers:
            try:
                await self._sio.emit("progress", event_data, to=sid)
            except Exception:
                # Client may have disconnected
                pass

    async def emit_completed(
        self,
        task_id: str,
        total: int,
        success_count: int,
        failure_count: int,
        duration: float,
        message: Optional[str] = None,
    ):
        """
        Emit task completion event to all subscribers.

        Args:
            task_id: Task ID
            total: Total items processed
            success_count: Number of successful items
            failure_count: Number of failed items
            duration: Task duration in seconds
            message: Optional completion message

        Requirements satisfied:
        - 3.3: Display total duration and success/failure statistics
        """
        if not self._sio:
            return

        subscribers = self.get_subscribers(task_id)
        if not subscribers:
            return

        event_data = {
            "task_id": task_id,
            "status": "completed",
            "total": total,
            "success_count": success_count,
            "failure_count": failure_count,
            "duration": round(duration, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if message:
            event_data["message"] = message

        # Emit to all subscribers
        for sid in subscribers:
            try:
                await self._sio.emit("completed", event_data, to=sid)
            except Exception:
                pass

    async def emit_error(
        self,
        task_id: str,
        error: str,
        error_type: str = "error",
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Emit error event to all subscribers.

        Args:
            task_id: Task ID
            error: Error message
            error_type: Type of error (error, warning, item_error)
            details: Optional error details

        Requirements satisfied:
        - 3.4: Continue showing progress even when individual items fail
        """
        if not self._sio:
            return

        subscribers = self.get_subscribers(task_id)
        if not subscribers:
            return

        event_data = {
            "task_id": task_id,
            "error_type": error_type,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if details:
            event_data["details"] = details

        # Emit to all subscribers
        for sid in subscribers:
            try:
                await self._sio.emit("error", event_data, to=sid)
            except Exception:
                pass

    async def emit_failed(
        self, task_id: str, error: str, details: Optional[dict[str, Any]] = None
    ):
        """
        Emit task failure event to all subscribers.

        Args:
            task_id: Task ID
            error: Error message
            details: Optional error details
        """
        if not self._sio:
            return

        subscribers = self.get_subscribers(task_id)
        if not subscribers:
            return

        event_data = {
            "task_id": task_id,
            "status": "failed",
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if details:
            event_data["details"] = details

        # Emit to all subscribers
        for sid in subscribers:
            try:
                await self._sio.emit("failed", event_data, to=sid)
            except Exception:
                pass


# Global WebSocket manager instance
ws_manager = WebSocketManager()
