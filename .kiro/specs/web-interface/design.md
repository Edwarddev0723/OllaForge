# Design Document

## Overview

本設計文件描述 OllaForge 網頁介面的架構和實作細節。OllaForge 目前是一個命令列工具，本專案將添加一個現代化的網頁介面，讓使用者能夠透過瀏覽器進行資料集生成和擴增操作。

設計採用前後端分離架構：
- **Backend**: FastAPI 提供 RESTful API，重用現有的 OllaForge 核心邏輯
- **Frontend**: React + TypeScript 提供互動式使用者介面
- **Communication**: WebSocket 用於即時進度更新，REST API 用於資料操作

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           React Frontend (TypeScript)                  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │ Generate │  │ Augment  │  │ Config Manager   │    │  │
│  │  │   Page   │  │   Page   │  │                  │    │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘    │  │
│  │  ┌──────────────────────────────────────────────┐    │  │
│  │  │        API Client (Axios)                     │    │  │
│  │  └──────────────────────────────────────────────┘    │  │
│  │  ┌──────────────────────────────────────────────┐    │  │
│  │  │     WebSocket Client (Socket.IO)             │    │  │
│  │  └──────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/WebSocket
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend Server                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  API Routes                            │  │
│  │  /api/generate  /api/augment  /api/models  /api/ws   │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Business Logic Layer                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │  │
│  │  │  Generation  │  │ Augmentation │  │  Progress  │  │  │
│  │  │   Service    │  │   Service    │  │  Manager   │  │  │
│  │  └──────────────┘  └──────────────┘  └────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           OllaForge Core (Existing)                    │  │
│  │  client.py  augmentor.py  file_manager.py  qc.py     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Ollama Service (localhost:11434)            │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- FastAPI: 現代化的 Python Web 框架，支援非同步處理
- Pydantic: 資料驗證（已在現有程式碼中使用）
- Socket.IO: WebSocket 通訊用於即時進度更新
- CORS Middleware: 支援跨域請求

**Frontend:**
- React 18: UI 框架
- TypeScript: 型別安全
- Ant Design: UI 元件庫（提供完整的表單、表格、進度條等元件）
- Axios: HTTP 客戶端
- Socket.IO Client: WebSocket 客戶端
- i18next: 國際化支援

**Development Tools:**
- Vite: 前端建置工具
- ESLint + Prettier: 程式碼品質工具
- pytest: 後端測試
- Jest + React Testing Library: 前端測試

## Components and Interfaces

### Backend Components

#### 1. API Server (`ollaforge/web/server.py`)

FastAPI 應用程式主入口，負責：
- 初始化 FastAPI app
- 設定 CORS
- 註冊路由
- 啟動 WebSocket 伺服器

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

app = FastAPI(title="OllaForge API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO for WebSocket
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)
```

#### 2. Generation Service (`ollaforge/web/services/generation.py`)

封裝資料集生成邏輯，提供非同步介面：

```python
from typing import AsyncGenerator, Dict, Any
from ..models import GenerationConfig
from ...client import generate_data_concurrent
from ...processor import process_model_response

class GenerationService:
    async def generate_dataset(
        self,
        config: GenerationConfig,
        progress_callback: Callable[[int, int], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate dataset with progress updates.
        
        Yields progress updates and final result.
        """
        # Implementation wraps existing generate_data_concurrent
        pass
```

#### 3. Augmentation Service (`ollaforge/web/services/augmentation.py`)

封裝資料集擴增邏輯：

```python
from ...augmentor import DatasetAugmentor
from ...models import AugmentationConfig

class AugmentationService:
    async def augment_dataset(
        self,
        config: AugmentationConfig,
        progress_callback: Callable[[int, int], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Augment dataset with progress updates.
        """
        pass
    
    async def preview_augmentation(
        self,
        config: AugmentationConfig,
        entries: List[Dict[str, Any]]
    ) -> List[Tuple[Dict, Dict]]:
        """
        Generate preview of augmentation.
        """
        pass
```

#### 4. API Routes (`ollaforge/web/routes/`)

**Generation Routes** (`generation.py`):
```python
@router.post("/api/generate")
async def start_generation(config: GenerationConfig):
    """Start dataset generation task."""
    pass

@router.get("/api/generate/{task_id}")
async def get_generation_status(task_id: str):
    """Get generation task status."""
    pass

@router.get("/api/generate/{task_id}/download")
async def download_generated_dataset(task_id: str, format: str = "jsonl"):
    """Download generated dataset."""
    pass
```

**Augmentation Routes** (`augmentation.py`):
```python
@router.post("/api/augment/upload")
async def upload_dataset(file: UploadFile):
    """Upload dataset for augmentation."""
    pass

@router.post("/api/augment/preview")
async def preview_augmentation(config: AugmentPreviewRequest):
    """Preview augmentation on sample entries."""
    pass

@router.post("/api/augment")
async def start_augmentation(config: AugmentationConfig):
    """Start dataset augmentation task."""
    pass

@router.get("/api/augment/{task_id}/download")
async def download_augmented_dataset(task_id: str, format: str = "jsonl"):
    """Download augmented dataset."""
    pass
```

**Model Routes** (`models.py`):
```python
@router.get("/api/models")
async def list_models():
    """List available Ollama models."""
    pass

@router.get("/api/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    pass
```

**WebSocket Events** (`websocket.py`):
```python
@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    pass

@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    pass

@sio.event
async def subscribe_task(sid, data):
    """Subscribe to task progress updates."""
    pass
```

### Frontend Components

#### 1. Page Components

**GeneratePage** (`src/pages/GeneratePage.tsx`):
- 資料集生成表單
- 參數輸入（topic, count, model, type, language）
- 即時進度顯示
- 結果預覽和下載

**AugmentPage** (`src/pages/AugmentPage.tsx`):
- 檔案上傳
- 欄位選擇和指令輸入
- 預覽功能
- 即時進度顯示
- 結果下載

**ConfigPage** (`src/pages/ConfigPage.tsx`):
- 儲存的設定列表
- 載入/刪除設定
- 設定匯入/匯出

#### 2. Feature Components

**GenerationForm** (`src/components/GenerationForm.tsx`):
```typescript
interface GenerationFormProps {
  onSubmit: (config: GenerationConfig) => void;
  initialValues?: Partial<GenerationConfig>;
}

export const GenerationForm: React.FC<GenerationFormProps> = ({
  onSubmit,
  initialValues
}) => {
  // Form implementation with validation
  return (
    <Form onFinish={onSubmit} initialValues={initialValues}>
      <Form.Item name="topic" label="Topic" rules={[{ required: true }]}>
        <Input.TextArea />
      </Form.Item>
      {/* Other form fields */}
    </Form>
  );
};
```

**ProgressDisplay** (`src/components/ProgressDisplay.tsx`):
```typescript
interface ProgressDisplayProps {
  current: number;
  total: number;
  status: 'active' | 'success' | 'exception';
  message?: string;
}

export const ProgressDisplay: React.FC<ProgressDisplayProps> = ({
  current,
  total,
  status,
  message
}) => {
  const percent = (current / total) * 100;
  return (
    <div>
      <Progress percent={percent} status={status} />
      {message && <p>{message}</p>}
    </div>
  );
};
```

**DatasetPreview** (`src/components/DatasetPreview.tsx`):
```typescript
interface DatasetPreviewProps {
  entries: Array<Record<string, any>>;
  maxEntries?: number;
}

export const DatasetPreview: React.FC<DatasetPreviewProps> = ({
  entries,
  maxEntries = 5
}) => {
  // Display entries in a formatted table
  return <Table dataSource={entries.slice(0, maxEntries)} />;
};
```

#### 3. Service Layer

**API Client** (`src/services/api.ts`):
```typescript
import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const generationAPI = {
  startGeneration: (config: GenerationConfig) =>
    apiClient.post('/api/generate', config),
  
  getStatus: (taskId: string) =>
    apiClient.get(`/api/generate/${taskId}`),
  
  downloadDataset: (taskId: string, format: string) =>
    apiClient.get(`/api/generate/${taskId}/download`, {
      params: { format },
      responseType: 'blob',
    }),
};

export const augmentationAPI = {
  uploadDataset: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/api/augment/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  previewAugmentation: (config: AugmentPreviewRequest) =>
    apiClient.post('/api/augment/preview', config),
  
  startAugmentation: (config: AugmentationConfig) =>
    apiClient.post('/api/augment', config),
};

export const modelsAPI = {
  listModels: () => apiClient.get('/api/models'),
  getModelInfo: (modelName: string) =>
    apiClient.get(`/api/models/${modelName}`),
};
```

**WebSocket Client** (`src/services/websocket.ts`):
```typescript
import io from 'socket.io-client';

class WebSocketClient {
  private socket: Socket;
  
  connect() {
    this.socket = io('http://localhost:8000');
  }
  
  subscribeToTask(taskId: string, callback: (data: any) => void) {
    this.socket.emit('subscribe_task', { task_id: taskId });
    this.socket.on(`task_progress_${taskId}`, callback);
  }
  
  unsubscribeFromTask(taskId: string) {
    this.socket.off(`task_progress_${taskId}`);
  }
  
  disconnect() {
    this.socket.disconnect();
  }
}

export const wsClient = new WebSocketClient();
```

## Data Models

### Backend Models

**API Request/Response Models** (`ollaforge/web/models.py`):

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from ...models import DatasetType, OutputLanguage

class GenerationRequest(BaseModel):
    """Request model for dataset generation."""
    topic: str
    count: int = Field(ge=1, le=10000)
    model: str = "llama3.2"
    dataset_type: DatasetType = DatasetType.SFT
    language: OutputLanguage = OutputLanguage.EN
    qc_enabled: bool = True
    qc_confidence: float = Field(0.9, ge=0.0, le=1.0)

class GenerationResponse(BaseModel):
    """Response model for generation start."""
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    """Task status model."""
    task_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: int
    total: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AugmentUploadResponse(BaseModel):
    """Response after dataset upload."""
    file_id: str
    entry_count: int
    fields: List[str]
    preview: List[Dict[str, Any]]

class AugmentPreviewRequest(BaseModel):
    """Request for augmentation preview."""
    file_id: str
    target_field: str
    instruction: str
    model: str = "llama3.2"
    create_new_field: bool = False
    context_fields: List[str] = []
    preview_count: int = 3

class AugmentPreviewResponse(BaseModel):
    """Response with augmentation preview."""
    previews: List[Dict[str, Any]]  # [{original: {...}, augmented: {...}}]

class AugmentationRequest(BaseModel):
    """Request for full augmentation."""
    file_id: str
    target_field: str
    instruction: str
    model: str = "llama3.2"
    language: OutputLanguage = OutputLanguage.EN
    create_new_field: bool = False
    context_fields: List[str] = []
    concurrency: int = 5

class ModelInfo(BaseModel):
    """Ollama model information."""
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None
```

### Frontend Models

**TypeScript Interfaces** (`src/types/index.ts`):

```typescript
export enum DatasetType {
  SFT = 'sft',
  PRETRAIN = 'pretrain',
  SFT_CONV = 'sft_conv',
  DPO = 'dpo',
}

export enum OutputLanguage {
  EN = 'en',
  ZH_TW = 'zh-tw',
}

export interface GenerationConfig {
  topic: string;
  count: number;
  model: string;
  datasetType: DatasetType;
  language: OutputLanguage;
  qcEnabled: boolean;
  qcConfidence: number;
}

export interface AugmentationConfig {
  fileId: string;
  targetField: string;
  instruction: string;
  model: string;
  language: OutputLanguage;
  createNewField: boolean;
  contextFields: string[];
  concurrency: number;
}

export interface TaskStatus {
  taskId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  total: number;
  result?: any;
  error?: string;
}

export interface ModelInfo {
  name: string;
  size?: string;
  modifiedAt?: string;
}

export interface SavedConfig {
  id: string;
  name: string;
  type: 'generation' | 'augmentation';
  config: GenerationConfig | AugmentationConfig;
  createdAt: string;
}
```

## 

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Valid generation parameters initiate processing
*For any* valid generation configuration (topic, count, model, dataset type, language), submitting the form should initiate dataset generation and display real-time progress updates.
**Validates: Requirements 1.2**

### Property 2: Completed generation provides download
*For any* dataset generation that completes successfully, the system should provide a download link for the generated dataset file.
**Validates: Requirements 1.3**

### Property 3: Generation failures display errors
*For any* dataset generation that fails, the system should display a clear error message to the user.
**Validates: Requirements 1.4**

### Property 4: File upload extracts fields
*For any* valid dataset file uploaded, the system should validate the file format and display the available field names.
**Validates: Requirements 2.1**

### Property 5: Field validation works correctly
*For any* dataset and target field combination, the system should correctly validate whether the target field exists in the dataset, rejecting invalid fields and accepting valid ones.
**Validates: Requirements 2.2**

### Property 6: Completed augmentation provides download
*For any* augmentation that completes successfully, the system should provide a download link for the augmented dataset.
**Validates: Requirements 2.4**

### Property 7: Partial augmentation failures preserve data
*For any* augmentation where some entries fail, the system should preserve the original data for failed entries and report accurate failure statistics.
**Validates: Requirements 2.5**

### Property 8: Progress indicators show during operations
*For any* dataset generation or augmentation in progress, the system should display a progress bar showing the completion percentage.
**Validates: Requirements 3.1**

### Property 9: Progress updates in real-time
*For any* processing operation, the system should update the progress indicator in real-time as entries are processed.
**Validates: Requirements 3.2**

### Property 10: Completion shows statistics
*For any* processing operation that completes, the system should display the total duration and success statistics.
**Validates: Requirements 3.3**

### Property 11: Errors don't stop progress display
*For any* error that occurs during processing, the system should display the error message without stopping the progress display.
**Validates: Requirements 3.4**

### Property 12: Format support is comprehensive
*For any* supported file format (JSONL, JSON, CSV, TSV, Parquet), the system should correctly recognize and parse files in that format.
**Validates: Requirements 4.1**

### Property 13: Format conversion preserves data
*For any* dataset, converting between any two supported formats should preserve all data fields correctly (round-trip property).
**Validates: Requirements 4.3**

### Property 14: Unsupported formats show errors
*For any* unsupported file format uploaded, the system should display a clear error message listing the supported formats.
**Validates: Requirements 4.4**

### Property 15: JSON formatting is readable
*For any* entry displayed in the UI, JSON data should be formatted in a readable structure.
**Validates: Requirements 5.4**

### Property 16: Long text is truncated
*For any* entry containing long text, the system should truncate the display and provide an expand option.
**Validates: Requirements 5.5**

### Property 17: Save option offered after operations
*For any* completed generation or augmentation, the system should offer to save the configuration.
**Validates: Requirements 6.1**

### Property 18: Configurations persist in storage
*For any* configuration saved by the user, the system should store it in browser local storage with the user-provided name.
**Validates: Requirements 6.2**

### Property 19: Configuration round-trip preserves values
*For any* configuration, saving and then loading it should restore all form field values correctly (round-trip property).
**Validates: Requirements 6.3**

### Property 20: Configuration deletion removes from storage
*For any* saved configuration deleted by the user, the system should remove it from browser local storage.
**Validates: Requirements 6.5**

### Property 21: Concurrent requests are independent
*For any* multiple generation requests submitted simultaneously, the system should process each request independently without interference.
**Validates: Requirements 7.1**

### Property 22: Operations don't block endpoints
*For any* generation or augmentation in progress, other API endpoints should remain responsive and not be blocked.
**Validates: Requirements 7.2**

### Property 23: Resource limits trigger queueing
*For any* situation where system resources are limited, the system should queue requests and process them sequentially.
**Validates: Requirements 7.3**

### Property 24: Timeouts return error responses
*For any* request that times out, the system should return an appropriate error response.
**Validates: Requirements 7.4**

### Property 25: Ollama unavailability shows clear error
*For any* situation where the Ollama service is unavailable, the system should return a clear error message.
**Validates: Requirements 7.5**

### Property 26: Browser language detection works
*For any* browser language setting, the system should detect it and display the appropriate UI language.
**Validates: Requirements 8.1**

### Property 27: Language switching updates UI
*For any* language switch action, the system should update all interface text immediately.
**Validates: Requirements 8.2**

### Property 28: Error messages are localized
*For any* error message displayed, the system should show it in the current UI language.
**Validates: Requirements 8.3**

### Property 29: Language preference persists
*For any* UI language selected and saved, the system should remember it across sessions.
**Validates: Requirements 8.5**

### Property 30: API uses JSON format
*For any* API request or response, the system should use JSON format for the body.
**Validates: Requirements 9.2**

### Property 31: CORS headers are present
*For any* API endpoint accessed, the system should include proper CORS headers in the response.
**Validates: Requirements 9.3**

### Property 32: Authentication requires tokens
*For any* protected API endpoint, the system should require a valid authentication token.
**Validates: Requirements 9.4**

### Property 33: Model information includes size
*For any* Ollama model displayed when the service is running, the system should show the model name with size information.
**Validates: Requirements 10.2**

### Property 34: Model validation before generation
*For any* model selected by the user, the system should validate that the model exists before starting generation.
**Validates: Requirements 10.4**

## Error Handling

### Backend Error Handling

1. **Ollama Connection Errors**
   - Catch `OllamaConnectionError` from existing client
   - Return HTTP 503 with clear message
   - Suggest checking Ollama service status

2. **File Processing Errors**
   - Validate file format before processing
   - Return HTTP 400 for invalid formats
   - Provide list of supported formats

3. **Validation Errors**
   - Use Pydantic validation
   - Return HTTP 422 with detailed error messages
   - Include field-level error information

4. **Task Errors**
   - Store error state in task status
   - Emit error events via WebSocket
   - Preserve partial results when possible

5. **Resource Limits**
   - Implement request queueing
   - Return HTTP 429 for rate limiting
   - Provide retry-after information

### Frontend Error Handling

1. **API Errors**
   - Display user-friendly error messages
   - Show technical details in expandable section
   - Provide retry options where appropriate

2. **Network Errors**
   - Detect connection failures
   - Show offline indicator
   - Queue operations for retry

3. **Validation Errors**
   - Show inline form validation
   - Highlight invalid fields
   - Provide helpful error messages

4. **File Upload Errors**
   - Validate file size before upload
   - Check file format client-side
   - Show progress during upload

## Testing Strategy

### Backend Testing

**Unit Tests:**
- Test API route handlers with mock services
- Test service layer logic independently
- Test data model validation
- Test error handling paths

**Integration Tests:**
- Test API endpoints end-to-end
- Test WebSocket communication
- Test file upload and download
- Test integration with existing OllaForge core

**Property-Based Tests:**
- Use Hypothesis for property testing
- Test properties defined in Correctness Properties section
- Generate random valid inputs for API endpoints
- Verify invariants hold across all inputs

**Testing Framework:**
- pytest for backend tests
- pytest-asyncio for async tests
- httpx for API testing
- Hypothesis for property-based testing

**Configuration:**
- Minimum 100 iterations for each property test
- Tag each property test with: `# Feature: web-interface, Property X: [property text]`

### Frontend Testing

**Unit Tests:**
- Test individual components in isolation
- Test utility functions
- Test API client methods
- Test WebSocket client

**Integration Tests:**
- Test page components with mocked API
- Test form submission flows
- Test progress display updates
- Test configuration save/load

**E2E Tests:**
- Test complete user workflows
- Test generation flow from form to download
- Test augmentation flow with file upload
- Test configuration management

**Testing Framework:**
- Jest for unit tests
- React Testing Library for component tests
- Mock Service Worker (MSW) for API mocking
- Cypress or Playwright for E2E tests

### Manual Testing Checklist

1. **Generation Flow**
   - [ ] Form validation works correctly
   - [ ] Progress updates in real-time
   - [ ] Download works for all formats
   - [ ] Error messages are clear

2. **Augmentation Flow**
   - [ ] File upload accepts all formats
   - [ ] Field list displays correctly
   - [ ] Preview shows before/after
   - [ ] Download works correctly

3. **Configuration Management**
   - [ ] Save configuration works
   - [ ] Load configuration restores values
   - [ ] Delete configuration removes from storage
   - [ ] List shows all saved configs

4. **Internationalization**
   - [ ] Language detection works
   - [ ] Language switching updates all text
   - [ ] Error messages are localized
   - [ ] Preference persists across sessions

5. **Error Scenarios**
   - [ ] Ollama unavailable shows clear error
   - [ ] Invalid file format shows error
   - [ ] Network errors are handled gracefully
   - [ ] Timeout errors are handled

## Security Considerations

1. **File Upload Security**
   - Validate file types and sizes
   - Scan uploaded files for malicious content
   - Store uploaded files in temporary directory
   - Clean up temporary files after processing

2. **API Security**
   - Implement rate limiting
   - Validate all input parameters
   - Sanitize file paths
   - Use HTTPS in production

3. **CORS Configuration**
   - Restrict allowed origins in production
   - Validate origin headers
   - Use credentials only when necessary

4. **Data Privacy**
   - Don't log sensitive data
   - Clear temporary files promptly
   - Use secure WebSocket connections

## Performance Considerations

1. **Backend Performance**
   - Use async/await for I/O operations
   - Implement connection pooling
   - Cache model list responses
   - Stream large file downloads

2. **Frontend Performance**
   - Lazy load pages and components
   - Debounce form inputs
   - Virtualize large lists
   - Optimize bundle size

3. **WebSocket Optimization**
   - Batch progress updates
   - Throttle update frequency
   - Reconnect on connection loss
   - Clean up subscriptions

4. **File Handling**
   - Stream file uploads
   - Process files in chunks
   - Use worker threads for CPU-intensive tasks
   - Implement file size limits

## Deployment

### Development Setup

1. **Backend:**
   ```bash
   cd ollaforge
   pip install -e ".[web]"
   python -m ollaforge.web.server
   ```

2. **Frontend:**
   ```bash
   cd ollaforge-web
   npm install
   npm run dev
   ```

### Production Build

1. **Backend:**
   - Use uvicorn with multiple workers
   - Configure proper CORS origins
   - Set up logging and monitoring
   - Use environment variables for configuration

2. **Frontend:**
   - Build optimized production bundle
   - Configure API endpoint
   - Enable service worker for offline support
   - Set up CDN for static assets

### Docker Deployment

```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[web]"
CMD ["uvicorn", "ollaforge.web.server:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend Dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
```

### Docker Compose

```yaml
version: '3.8'
services:
  backend:
    build: ./ollaforge
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
    volumes:
      - ./data:/app/data
  
  frontend:
    build: ./ollaforge-web
    ports:
      - "80:80"
    depends_on:
      - backend
```

## Future Enhancements

1. **User Authentication**
   - User accounts and login
   - Personal dataset library
   - Sharing configurations

2. **Advanced Features**
   - Batch processing multiple files
   - Dataset versioning
   - Collaborative editing

3. **Analytics**
   - Usage statistics
   - Quality metrics
   - Performance monitoring

4. **Integration**
   - HuggingFace Hub integration
   - Direct upload to training platforms
   - API webhooks for automation
