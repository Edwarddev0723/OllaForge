"""Tests for web module setup and imports."""

import pytest


def test_web_module_imports():
    """Test that web module can be imported."""
    try:
        import ollaforge.web
        assert hasattr(ollaforge.web, "__version__")
    except ImportError as e:
        pytest.fail(f"Failed to import ollaforge.web: {e}")


def test_web_services_module_imports():
    """Test that web services module can be imported."""
    try:
        import ollaforge.web.services  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import ollaforge.web.services: {e}")


def test_web_routes_module_imports():
    """Test that web routes module can be imported."""
    try:
        import ollaforge.web.routes  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import ollaforge.web.routes: {e}")


def test_fastapi_dependency():
    """Test that FastAPI is installed and can be imported."""
    try:
        import fastapi
        assert hasattr(fastapi, "FastAPI")
    except ImportError as e:
        pytest.fail(f"Failed to import fastapi: {e}")


def test_socketio_dependency():
    """Test that python-socketio is installed and can be imported."""
    try:
        import socketio
        assert hasattr(socketio, "AsyncServer")
    except ImportError as e:
        pytest.fail(f"Failed to import socketio: {e}")


def test_uvicorn_dependency():
    """Test that uvicorn is installed and can be imported."""
    try:
        import uvicorn
        assert hasattr(uvicorn, "run")
    except ImportError as e:
        pytest.fail(f"Failed to import uvicorn: {e}")


def test_multipart_dependency():
    """Test that python-multipart is installed."""
    try:
        import multipart  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import multipart: {e}")
