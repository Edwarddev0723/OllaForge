"""
FastAPI server for OllaForge web interface.

This module provides the main FastAPI application with:
- RESTful API endpoints for dataset generation and augmentation
- WebSocket support for real-time progress updates
- CORS configuration for cross-origin requests
- Integration with existing OllaForge core functionality

Requirements satisfied:
- 9.1: RESTful API endpoints for all operations
- 9.3: CORS headers for cross-origin requests
- 3.1: Display progress bar showing completion percentage
- 3.2: Update progress in real-time during processing
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

from .routes.websocket import ws_manager


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging():
    """Configure logging based on environment."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
    logging.getLogger("socketio").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)
    
    return logging.getLogger("ollaforge.web")


# Initialize logger
logger = setup_logging()


# ============================================================================
# Environment Configuration
# ============================================================================

# Debug mode
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# CORS configuration
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8080"
).split(",")

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ============================================================================
# Socket.IO Configuration
# ============================================================================

# Socket.IO server for WebSocket communication
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*' if DEBUG else CORS_ORIGINS,
    logger=DEBUG,
    engineio_logger=DEBUG
)

# Initialize WebSocket manager with Socket.IO server
ws_manager.set_sio(sio)


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 OllaForge Web API starting...")
    logger.info(f"   Debug mode: {DEBUG}")
    logger.info(f"   CORS origins: {CORS_ORIGINS}")
    logger.info(f"   Ollama host: {OLLAMA_HOST}")
    yield
    # Shutdown
    logger.info("👋 OllaForge Web API shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

# Create FastAPI application with comprehensive OpenAPI documentation
app = FastAPI(
    title="OllaForge API",
    description="""
## OllaForge Web API

OllaForge is a tool for generating and augmenting LLM training datasets using local Ollama models.

### Features

- **Dataset Generation**: Generate new training datasets from topic descriptions
- **Dataset Augmentation**: Modify or add fields to existing datasets
- **Real-time Progress**: WebSocket support for live progress updates
- **Multi-format Support**: JSONL, JSON, CSV, TSV, and Parquet formats
- **Quality Control**: Built-in QC for Traditional Chinese (Taiwan) content

### Authentication

Currently, the API does not require authentication. In production deployments,
token-based authentication should be configured.

### WebSocket Events

Connect to `/socket.io` for real-time progress updates:

- **subscribe**: Subscribe to task progress `{task_id: string}`
- **unsubscribe**: Unsubscribe from task `{task_id: string}`
- **progress**: Receive progress updates
- **completed**: Receive completion notification
- **failed**: Receive failure notification

### Rate Limiting

The API implements request queueing when system resources are limited.
Concurrent requests are processed independently.
""",
    version="1.0.0",
    lifespan=lifespan,
    debug=DEBUG,
    openapi_tags=[
        {
            "name": "generation",
            "description": "Dataset generation operations - create new training datasets from topic descriptions"
        },
        {
            "name": "augmentation",
            "description": "Dataset augmentation operations - modify or add fields to existing datasets"
        },
        {
            "name": "models",
            "description": "Ollama model management - list and validate available models"
        },
        {
            "name": "tasks",
            "description": "Task management - monitor and control background tasks"
        },
        {
            "name": "health",
            "description": "Health check and API information endpoints"
        }
    ],
    contact={
        "name": "OllaForge",
        "url": "https://github.com/ollaforge/ollaforge"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)


# ============================================================================
# CORS Configuration
# ============================================================================

# Configure CORS
# In development, allow all origins
# In production, this should be restricted to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Route Registration
# ============================================================================

# Import and register routes
from .routes import generation_router, augmentation_router, models_router, tasks_router

app.include_router(generation_router)
app.include_router(augmentation_router)
app.include_router(models_router)
app.include_router(tasks_router)


# ============================================================================
# Health and Info Endpoints
# ============================================================================


# Health check endpoint
@app.get("/health", tags=["health"], summary="Health Check", description="Check if the API server is running and healthy.")
async def health_check():
    """
    Health check endpoint.
    
    Returns the current health status of the API server.
    Use this endpoint for load balancer health checks or monitoring.
    """
    return {
        "status": "healthy",
        "service": "ollaforge-api",
        "version": "1.0.0",
        "debug": DEBUG
    }


@app.get("/", tags=["health"], summary="API Information", description="Get basic information about the API.")
async def root():
    """
    Root endpoint with API information.
    
    Returns basic information about the API including version and documentation URL.
    """
    return {
        "name": "OllaForge API",
        "version": "1.0.0",
        "description": "API for dataset generation and augmentation using local Ollama models",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json"
    }


# ============================================================================
# Socket.IO ASGI Application
# ============================================================================

# Create ASGI application with Socket.IO
socket_app = socketio.ASGIApp(
    sio,
    app,
    socketio_path="/socket.io"
)


# ============================================================================
# Socket.IO Event Handlers
# ============================================================================
@sio.event
async def connect(sid: str, environ: dict):
    """Handle client connection."""
    logger.debug(f"Client connected: {sid}")
    ws_manager.on_connect(sid)


@sio.event
async def disconnect(sid: str):
    """Handle client disconnection."""
    logger.debug(f"Client disconnected: {sid}")
    ws_manager.on_disconnect(sid)


@sio.event
async def subscribe(sid: str, data: dict):
    """
    Handle task subscription request.
    
    Args:
        sid: Session ID
        data: Dict containing 'task_id' to subscribe to
    """
    task_id = data.get("task_id")
    if task_id:
        success = ws_manager.subscribe(sid, task_id)
        await sio.emit("subscribed", {
            "task_id": task_id,
            "success": success
        }, to=sid)
        logger.debug(f"Client {sid} subscribed to task {task_id}: {success}")


@sio.event
async def unsubscribe(sid: str, data: dict):
    """
    Handle task unsubscription request.
    
    Args:
        sid: Session ID
        data: Dict containing 'task_id' to unsubscribe from
    """
    task_id = data.get("task_id")
    if task_id:
        success = ws_manager.unsubscribe(sid, task_id)
        await sio.emit("unsubscribed", {
            "task_id": task_id,
            "success": success
        }, to=sid)
        logger.debug(f"Client {sid} unsubscribed from task {task_id}: {success}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Determine log level
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    # Run the server
    uvicorn.run(
        "ollaforge.web.server:socket_app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=log_level,
        access_log=DEBUG
    )
