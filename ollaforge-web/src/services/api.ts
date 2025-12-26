/**
 * API client for OllaForge web interface.
 *
 * This module provides a configured Axios instance and API methods
 * for communicating with the OllaForge backend server.
 *
 * Requirements satisfied:
 * - 9.2: JSON format for request and response bodies
 */

import axios, { type AxiosInstance, type AxiosError, type InternalAxiosRequestConfig, type AxiosResponse } from 'axios';

// ============================================================================
// Types
// ============================================================================

export const DatasetType = {
  SFT: 'sft',
  PRETRAIN: 'pretrain',
  SFT_CONV: 'sft_conv',
  DPO: 'dpo',
} as const;

export type DatasetType = (typeof DatasetType)[keyof typeof DatasetType];

export const OutputLanguage = {
  EN: 'en',
  ZH_TW: 'zh-tw',
} as const;

export type OutputLanguage = (typeof OutputLanguage)[keyof typeof OutputLanguage];

export interface GenerationConfig {
  topic: string;
  count: number;
  model: string;
  dataset_type: DatasetType;
  language: OutputLanguage;
  qc_enabled: boolean;
  qc_confidence: number;
}

export interface GenerationResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface TaskStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  total: number;
  result?: Record<string, unknown>;
  error?: string;
  duration?: number;
}

export interface AugmentUploadResponse {
  file_id: string;
  entry_count: number;
  fields: string[];
  preview: Record<string, unknown>[];
}

export interface AugmentPreviewRequest {
  file_id: string;
  target_field: string;
  instruction: string;
  model: string;
  create_new_field: boolean;
  context_fields: string[];
  preview_count: number;
}

export interface AugmentPreviewResponse {
  previews: Array<{
    original: Record<string, unknown>;
    augmented: Record<string, unknown>;
  }>;
}

export interface AugmentationConfig {
  file_id: string;
  target_field: string;
  instruction: string;
  model: string;
  language: OutputLanguage;
  create_new_field: boolean;
  context_fields: string[];
  concurrency: number;
}

export interface ModelInfo {
  name: string;
  size?: string;
  modified_at?: string;
}

export interface ModelListResponse {
  models: ModelInfo[];
}

export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, unknown>;
}

// ============================================================================
// API Client Configuration
// ============================================================================

/**
 * Determine the API base URL based on environment.
 * 
 * In Docker/production: Use relative path so nginx can proxy requests
 * In development: Use localhost:8000 directly
 */
const getApiBaseUrl = (): string => {
  // If explicitly set via environment variable, use that
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  // In production (served via nginx), use relative path for proxy
  if (import.meta.env.PROD) {
    return '';  // Empty string means relative to current origin
  }
  
  // In development, connect directly to backend
  return 'http://localhost:8000';
};

const API_BASE_URL = getApiBaseUrl();

/**
 * Create and configure the Axios instance.
 */
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_BASE_URL,
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: 30000, // 30 second timeout for regular requests
  });

  // Request interceptor for logging and adding headers
  client.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
      // Log request in development
      if (import.meta.env.DEV) {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
      }
      return config;
    },
    (error: AxiosError) => {
      console.error('[API] Request error:', error.message);
      return Promise.reject(error);
    }
  );

  // Response interceptor for error handling
  client.interceptors.response.use(
    (response: AxiosResponse) => {
      return response;
    },
    (error: AxiosError<ApiError>) => {
      // Extract error message from response
      const errorMessage = error.response?.data?.message || error.message;
      const errorType = error.response?.data?.error || 'NetworkError';

      // Log error in development
      if (import.meta.env.DEV) {
        console.error(`[API] Error: ${errorType} - ${errorMessage}`);
      }

      // Create enhanced error
      const enhancedError = new Error(errorMessage) as Error & {
        type: string;
        status?: number;
        details?: Record<string, unknown>;
      };
      enhancedError.type = errorType;
      enhancedError.status = error.response?.status;
      enhancedError.details = error.response?.data?.details;

      return Promise.reject(enhancedError);
    }
  );

  return client;
};

// Create the API client instance
export const apiClient = createApiClient();

// ============================================================================
// Generation API
// ============================================================================

/**
 * Generation API methods.
 *
 * Requirements satisfied:
 * - 1.2: Initiate dataset generation with valid parameters
 * - 1.3: Provide download link for generated dataset
 */
export const generationAPI = {
  /**
   * Start a new dataset generation task.
   *
   * @param config - Generation configuration
   * @returns Promise with task ID and status
   */
  startGeneration: async (config: GenerationConfig): Promise<GenerationResponse> => {
    const response = await apiClient.post<GenerationResponse>('/api/generate', config);
    return response.data;
  },

  /**
   * Get the status of a generation task.
   *
   * @param taskId - Task identifier
   * @returns Promise with task status
   */
  getStatus: async (taskId: string): Promise<TaskStatus> => {
    const response = await apiClient.get<TaskStatus>(`/api/generate/${taskId}`);
    return response.data;
  },

  /**
   * Download the generated dataset.
   *
   * @param taskId - Task identifier
   * @param format - Output format (jsonl, json, csv, tsv, parquet)
   * @returns Promise with Blob containing the dataset
   */
  downloadDataset: async (taskId: string, format: string = 'jsonl'): Promise<Blob> => {
    const response = await apiClient.get(`/api/generate/${taskId}/download`, {
      params: { format },
      responseType: 'blob',
      timeout: 60000, // 60 second timeout for downloads
    });
    return response.data;
  },
};

// ============================================================================
// Augmentation API
// ============================================================================

/**
 * Augmentation API methods.
 *
 * Requirements satisfied:
 * - 2.1: Upload and validate dataset files
 * - 2.2: Validate target field exists in dataset
 * - 2.3: Preview augmentation before full processing
 * - 2.4: Provide download link for augmented dataset
 */
export const augmentationAPI = {
  /**
   * Upload a dataset file for augmentation.
   *
   * @param file - File to upload
   * @returns Promise with file info and preview
   */
  uploadDataset: async (file: File): Promise<AugmentUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<AugmentUploadResponse>('/api/augment/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 60 second timeout for uploads
    });
    return response.data;
  },

  /**
   * Preview augmentation on sample entries.
   *
   * @param config - Preview configuration
   * @returns Promise with preview results
   */
  previewAugmentation: async (config: AugmentPreviewRequest): Promise<AugmentPreviewResponse> => {
    const response = await apiClient.post<AugmentPreviewResponse>('/api/augment/preview', config);
    return response.data;
  },

  /**
   * Start a full augmentation task.
   *
   * @param config - Augmentation configuration
   * @returns Promise with task ID and status
   */
  startAugmentation: async (config: AugmentationConfig): Promise<GenerationResponse> => {
    const response = await apiClient.post<GenerationResponse>('/api/augment', config);
    return response.data;
  },

  /**
   * Get the status of an augmentation task.
   *
   * @param taskId - Task identifier
   * @returns Promise with task status
   */
  getStatus: async (taskId: string): Promise<TaskStatus> => {
    const response = await apiClient.get<TaskStatus>(`/api/augment/${taskId}`);
    return response.data;
  },

  /**
   * Download the augmented dataset.
   *
   * @param taskId - Task identifier
   * @param format - Output format (jsonl, json, csv, tsv, parquet)
   * @returns Promise with Blob containing the dataset
   */
  downloadDataset: async (taskId: string, format: string = 'jsonl'): Promise<Blob> => {
    const response = await apiClient.get(`/api/augment/${taskId}/download`, {
      params: { format },
      responseType: 'blob',
      timeout: 60000, // 60 second timeout for downloads
    });
    return response.data;
  },
};

// ============================================================================
// Models API
// ============================================================================

/**
 * Models API methods.
 *
 * Requirements satisfied:
 * - 10.1: Fetch and display available Ollama models
 * - 10.2: Show model names with size information
 */
export const modelsAPI = {
  /**
   * List available Ollama models.
   *
   * @returns Promise with list of models
   */
  listModels: async (): Promise<ModelListResponse> => {
    const response = await apiClient.get<ModelListResponse>('/api/models');
    return response.data;
  },

  /**
   * Get information about a specific model.
   *
   * @param modelName - Name of the model
   * @returns Promise with model info
   */
  getModelInfo: async (modelName: string): Promise<ModelInfo> => {
    const response = await apiClient.get<ModelInfo>(`/api/models/${encodeURIComponent(modelName)}`);
    return response.data;
  },

  /**
   * Validate that a model exists and is available.
   *
   * @param modelName - Name of the model to validate
   * @returns Promise with validation result
   */
  validateModel: async (modelName: string): Promise<{ valid: boolean; model: string; message: string }> => {
    const response = await apiClient.get<{ valid: boolean; model: string; message: string }>(
      `/api/models/${encodeURIComponent(modelName)}/validate`
    );
    return response.data;
  },
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Helper function to trigger file download from a Blob.
 *
 * @param blob - Blob containing the file data
 * @param filename - Name for the downloaded file
 */
export const downloadBlob = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

/**
 * Check if an error is an API error with a specific type.
 *
 * @param error - Error to check
 * @param type - Error type to match
 * @returns True if error matches the type
 */
export const isApiError = (error: unknown, type?: string): error is Error & { type: string; status?: number } => {
  if (!(error instanceof Error)) return false;
  const apiError = error as Error & { type?: string };
  if (type) return apiError.type === type;
  return 'type' in apiError;
};

export default apiClient;
