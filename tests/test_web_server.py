"""
Tests for OllaForge web server and API endpoints.

This module tests the FastAPI server configuration, including
CORS headers and basic endpoint functionality.
"""

import pytest
from hypothesis import given, strategies as st, settings
from fastapi.testclient import TestClient
from ollaforge.web.server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ============================================================================
# Unit Tests for CORS Configuration
# ============================================================================

def test_cors_headers_in_responses(client):
    """
    Test that CORS headers are present in API responses.
    
    Requirements satisfied:
    - 9.3: CORS headers for cross-origin requests
    """
    # Make a request to the health endpoint
    response = client.get("/health")
    
    # Verify response is successful
    assert response.status_code == 200
    
    # Verify CORS headers are present
    # Note: TestClient doesn't automatically add Origin header, so we need to check
    # that the middleware is configured (headers will be added when Origin is present)
    # We can verify this by making a request with an Origin header
    response_with_origin = client.get(
        "/health",
        headers={"Origin": "http://localhost:3000"}
    )
    
    assert response_with_origin.status_code == 200
    # The CORS middleware should add these headers when Origin is present
    assert "access-control-allow-origin" in response_with_origin.headers


def test_cors_allowed_origins_configuration(client):
    """
    Test that allowed origins are properly configured.
    
    Requirements satisfied:
    - 9.3: CORS configuration for development and production
    """
    # Test with an allowed origin (localhost:3000 is in default config)
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        }
    )
    
    # Should return 200 for preflight request
    assert response.status_code == 200
    
    # Should have CORS headers
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers


def test_cors_allows_credentials(client):
    """
    Test that CORS is configured to allow credentials.
    
    Requirements satisfied:
    - 9.3: CORS configuration with credentials support
    """
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        }
    )
    
    # Should allow credentials
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_allows_all_methods(client):
    """
    Test that CORS allows all HTTP methods.
    
    Requirements satisfied:
    - 9.3: CORS configuration for all methods
    """
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
    )
    
    # Should allow the requested method
    assert response.status_code == 200
    allowed_methods = response.headers.get("access-control-allow-methods", "")
    # Should allow POST (and typically all methods with "*")
    assert "POST" in allowed_methods or "*" in allowed_methods


def test_cors_allows_all_headers(client):
    """
    Test that CORS allows all headers.
    
    Requirements satisfied:
    - 9.3: CORS configuration for all headers
    """
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type,Authorization",
        }
    )
    
    # Should allow the requested headers
    assert response.status_code == 200
    allowed_headers = response.headers.get("access-control-allow-headers", "")
    # Should allow custom headers (typically all with "*")
    assert len(allowed_headers) > 0


# ============================================================================
# Unit Tests for Basic Endpoints
# ============================================================================

def test_health_endpoint(client):
    """Test that the health check endpoint works."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ollaforge-api"


def test_root_endpoint(client):
    """Test that the root endpoint returns API information."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "OllaForge API"
    assert data["version"] == "1.0.0"
    assert "docs" in data


def test_api_returns_json(client):
    """Test that API endpoints return JSON responses."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    
    # Verify it's valid JSON
    data = response.json()
    assert isinstance(data, dict)


# ============================================================================
# Property Test for CORS Headers
# ============================================================================

# Strategy for generating valid HTTP origins
# Only use origins that are in the default CORS_ORIGINS configuration
http_origin_strategy = st.sampled_from([
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
])

# Strategy for generating HTTP methods
http_method_strategy = st.sampled_from([
    "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"
])


@given(
    origin=http_origin_strategy,
    method=http_method_strategy
)
@settings(max_examples=100)
def test_cors_headers_are_present(origin, method):
    """
    **Feature: web-interface, Property 31: CORS headers are present**
    **Validates: Requirements 9.3**
    
    For any valid HTTP origin and method, when making a CORS preflight request,
    the API should return appropriate CORS headers including:
    - access-control-allow-origin
    - access-control-allow-methods
    - access-control-allow-credentials
    """
    client = TestClient(app)
    
    # Make a CORS preflight request (OPTIONS)
    response = client.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
        }
    )
    
    # Should return 200 for preflight
    assert response.status_code == 200, \
        f"Preflight request should return 200, got {response.status_code}"
    
    # Verify CORS headers are present
    headers_lower = {k.lower(): v for k, v in response.headers.items()}
    
    assert "access-control-allow-origin" in headers_lower, \
        f"Missing access-control-allow-origin header for origin {origin}"
    
    assert "access-control-allow-methods" in headers_lower, \
        f"Missing access-control-allow-methods header for method {method}"
    
    assert "access-control-allow-credentials" in headers_lower, \
        f"Missing access-control-allow-credentials header"
    
    # Verify credentials are allowed
    assert headers_lower["access-control-allow-credentials"] == "true", \
        "CORS should allow credentials"


@given(endpoint=st.sampled_from(["/health", "/"]))
@settings(max_examples=50)
def test_cors_headers_on_actual_requests(endpoint):
    """
    **Feature: web-interface, Property 31: CORS headers are present**
    **Validates: Requirements 9.3**
    
    For any API endpoint, when making an actual request with an Origin header,
    the response should include the access-control-allow-origin header.
    """
    client = TestClient(app)
    
    # Make a request with Origin header
    response = client.get(
        endpoint,
        headers={"Origin": "http://localhost:3000"}
    )
    
    # Should be successful
    assert response.status_code == 200, \
        f"Request to {endpoint} should return 200, got {response.status_code}"
    
    # Verify CORS header is present
    headers_lower = {k.lower(): v for k, v in response.headers.items()}
    
    assert "access-control-allow-origin" in headers_lower, \
        f"Missing access-control-allow-origin header for endpoint {endpoint}"
