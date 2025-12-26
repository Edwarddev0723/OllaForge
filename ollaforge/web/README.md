# OllaForge Web Interface

Web interface for OllaForge dataset generation and augmentation tool.

## Backend Setup

### Installation

```bash
# Install with web dependencies
pip install -e ".[web]"
```

### Running the Backend

```bash
# From the project root
python -m ollaforge.web.server
```

The backend API will be available at `http://localhost:8000`.

### Configuration

Copy `.env.example` to `.env` and adjust settings as needed:

```bash
cp ollaforge/web/.env.example ollaforge/web/.env
```

## Frontend Setup

### Installation

```bash
cd ollaforge-web
npm install
```

### Running the Frontend

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### Configuration

Copy `.env.example` to `.env` and adjust settings as needed:

```bash
cp ollaforge-web/.env.example ollaforge-web/.env
```

## Development

### Backend Development

The backend uses FastAPI and follows this structure:

- `ollaforge/web/server.py` - Main FastAPI application
- `ollaforge/web/routes/` - API route handlers
- `ollaforge/web/services/` - Business logic services
- `ollaforge/web/models.py` - Pydantic models for API

### Frontend Development

The frontend uses React + TypeScript + Vite:

- `src/pages/` - Page components
- `src/components/` - Reusable UI components
- `src/services/` - API client and WebSocket client
- `src/types/` - TypeScript type definitions

## Testing

### Backend Tests

```bash
pytest tests/
```

### Frontend Tests

```bash
cd ollaforge-web
npm test
```
