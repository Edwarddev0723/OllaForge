"""
API routes for OllaForge web interface.

This package contains route handlers for different API endpoints.
"""

from .generation import router as generation_router
from .augmentation import router as augmentation_router
from .models import router as models_router
from .tasks import router as tasks_router
from .websocket import ws_manager

__all__ = [
    "generation_router",
    "augmentation_router",
    "models_router",
    "tasks_router",
    "ws_manager"
]
