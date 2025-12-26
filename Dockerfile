# OllaForge Backend Dockerfile
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first for better caching
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[web,formats]"

# ============================================================================
# Stage 2: Production
# ============================================================================
FROM python:3.11-slim as production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r ollaforge && useradd -r -g ollaforge ollaforge

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY ollaforge/ ./ollaforge/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir -e ".[web,formats]"

# Create data directory for temporary files
RUN mkdir -p /app/data && chown -R ollaforge:ollaforge /app

# Switch to non-root user
USER ollaforge

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CORS_ORIGINS="http://localhost:3000,http://localhost:5173,http://localhost:80" \
    OLLAMA_HOST="http://host.docker.internal:11434"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["uvicorn", "ollaforge.web.server:socket_app", "--host", "0.0.0.0", "--port", "8000"]
