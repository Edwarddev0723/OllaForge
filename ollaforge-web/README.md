# OllaForge Web Frontend

React + TypeScript frontend for OllaForge.

## Setup

```bash
npm install
```

## Development

```bash
npm run dev
```

## Build

```bash
npm run build
```

## Testing

### Unit Tests

```bash
npm run test          # Run tests once
npm run test:watch    # Run tests in watch mode
npm run test:ui       # Run tests with UI
```

### E2E Tests

E2E tests use Playwright and require both frontend and backend servers to be running.

```bash
# Install Playwright browsers (first time only)
npx playwright install chromium

# Run E2E tests (starts servers automatically)
npm run test:e2e

# Run E2E tests with UI
npm run test:e2e:ui

# Run E2E tests in headed mode (visible browser)
npm run test:e2e:headed

# Debug E2E tests
npm run test:e2e:debug
```

## Configuration

Copy `.env.example` to `.env` and configure:

- `VITE_API_URL` - Backend API URL (default: http://localhost:8000)
- `VITE_WS_URL` - WebSocket URL (default: http://localhost:8000)

## Tech Stack

- React 18
- TypeScript
- Vite
- Ant Design
- Axios
- Socket.IO Client
- i18next (internationalization)
- Vitest (unit testing)
- Playwright (E2E testing)
