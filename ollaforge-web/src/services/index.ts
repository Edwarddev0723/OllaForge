/**
 * Services module exports.
 *
 * This module re-exports all service modules for convenient imports.
 */

export {
  // API client
  apiClient,
  // API methods
  generationAPI,
  augmentationAPI,
  modelsAPI,
  // Utility functions
  downloadBlob,
  isApiError,
  // Types
  type GenerationConfig,
  type GenerationResponse,
  type TaskStatus,
  type AugmentUploadResponse,
  type AugmentPreviewRequest,
  type AugmentPreviewResponse,
  type AugmentationConfig,
  type ModelInfo,
  type ModelListResponse,
  type ApiError,
  // Enums
  DatasetType,
  OutputLanguage,
} from './api';

export {
  // WebSocket client
  wsClient,
  WebSocketClient,
  // Types
  type ProgressEvent,
  type CompletedEvent,
  type ErrorEvent,
  type FailedEvent,
  type ProgressCallback,
  type CompletedCallback,
  type ErrorCallback,
  type FailedCallback,
  type ConnectionState,
  type ConnectionStateCallback,
  type ReconnectionConfig,
} from './websocket';

export {
  // Operation queue
  operationQueue,
  // Types
  type QueuedOperation,
} from './operationQueue';
