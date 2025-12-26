# OllaForge Deployment Guide

This guide covers deploying OllaForge in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start with Docker](#quick-start-with-docker)
- [Development Setup](#development-setup)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Ollama** installed and running (either on host or in container)

### Optional

- **NVIDIA GPU** with CUDA support (for faster inference)
- **Reverse proxy** (nginx, Traefik, etc.) for SSL termination

## Quick Start with Docker

### 1. Clone the Repository

```bash
git clone https://github.com/ollaforge/ollaforge.git
cd ollaforge
```

### 2. Start Ollama

Ensure Ollama is running on your host machine:

```bash
ollama serve
```

Pull at least one model:

```bash
ollama pull llama3.2
```

### 3. Start OllaForge

```bash
docker-compose up -d
```

### 4. Access the Application

- **Frontend**: http://localhost:80
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Development Setup

For development, run the services separately:

### Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[web,formats,dev]"

# Run the server
python -m ollaforge.web.server
```

### Frontend

```bash
cd ollaforge-web

# Install dependencies
npm install

# Run development server
npm run dev
```

## Production Deployment

### Using Docker Compose

1. **Configure environment variables**:

```bash
# Create .env file
cat > .env << EOF
CORS_ORIGINS=https://your-domain.com
OLLAMA_HOST=http://ollama:11434
VITE_API_URL=https://api.your-domain.com
VITE_WS_URL=https://api.your-domain.com
EOF
```

2. **Deploy with production overrides**:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Using Kubernetes

Example Kubernetes manifests are available in the `k8s/` directory (if applicable).

### Behind a Reverse Proxy

Example nginx configuration for SSL termination:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 443 ssl http2;
    server_name api.your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Backend API
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Configuration

### Environment Variables

#### Backend

| Variable | Description | Default |
|----------|-------------|---------|
| `CORS_ORIGINS` | Comma-separated list of allowed origins | `http://localhost:3000,http://localhost:5173` |
| `OLLAMA_HOST` | Ollama API endpoint | `http://localhost:11434` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

#### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_WS_URL` | WebSocket URL | `http://localhost:8000` |

### Resource Limits

Recommended resource limits for production:

| Service | CPU | Memory |
|---------|-----|--------|
| Backend | 2 cores | 4 GB |
| Frontend | 0.5 cores | 256 MB |
| Ollama | 4+ cores | 8+ GB |

## Monitoring

### Health Checks

Both services expose health check endpoints:

- **Backend**: `GET /health`
- **Frontend**: `GET /` (nginx serves index.html)

### Logs

View logs with Docker Compose:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Metrics

The backend exposes basic metrics at `/health`:

```json
{
  "status": "healthy",
  "service": "ollaforge-api"
}
```

For advanced monitoring, consider integrating with:
- Prometheus + Grafana
- Datadog
- New Relic

## Troubleshooting

### Common Issues

#### "Unable to connect to Ollama service"

1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check the `OLLAMA_HOST` environment variable
3. For Docker on Linux, ensure `host.docker.internal` resolves correctly

#### Container fails to start

1. Check logs: `docker-compose logs backend`
2. Verify port 8000 is not in use
3. Ensure sufficient memory is available

#### WebSocket connection fails

1. Check if your reverse proxy supports WebSocket
2. Verify the `VITE_WS_URL` is correct
3. Check browser console for connection errors

#### Frontend shows blank page

1. Check browser console for errors
2. Verify the API URL is accessible
3. Check CORS configuration

### Getting Help

- **GitHub Issues**: Report bugs or request features
- **API Documentation**: Access `/docs` on the backend for Swagger UI
- **Logs**: Check container logs for detailed error messages

## Security Considerations

### Production Checklist

- [ ] Use HTTPS for all traffic
- [ ] Configure proper CORS origins
- [ ] Set up rate limiting
- [ ] Enable logging and monitoring
- [ ] Regular security updates
- [ ] Backup data volumes

### Network Security

- Keep Ollama on a private network
- Use firewall rules to restrict access
- Consider VPN for remote access
