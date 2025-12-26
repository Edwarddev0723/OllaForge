/**
 * WebSocket client for OllaForge web interface.
 *
 * This module provides a Socket.IO client wrapper for real-time progress tracking
 * during dataset generation and augmentation operations.
 *
 * Requirements satisfied:
 * - 3.1: Display progress bar showing completion percentage
 * - 3.2: Update progress in real-time during processing
 */

import { io, Socket } from 'socket.io-client';

// ============================================================================
// Types
// ============================================================================

/**
 * Progress event data received from the server.
 */
export interface ProgressEvent {
  task_id: string;
  progress: number;
  total: number;
  percentage: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message?: string;
  timestamp: string;
}

/**
 * Completion event data received from the server.
 */
export interface CompletedEvent {
  task_id: string;
  status: 'completed';
  total: number;
  success_count: number;
  failure_count: number;
  duration: number;
  message?: string;
  timestamp: string;
}

/**
 * Error event data received from the server.
 */
export interface ErrorEvent {
  task_id: string;
  error_type: 'error' | 'warning' | 'item_error';
  error: string;
  details?: Record<string, unknown>;
  timestamp: string;
}

/**
 * Failed event data received from the server.
 */
export interface FailedEvent {
  task_id: string;
  status: 'failed';
  error: string;
  details?: Record<string, unknown>;
  timestamp: string;
}

/**
 * Callback types for task events.
 */
export type ProgressCallback = (data: ProgressEvent) => void;
export type CompletedCallback = (data: CompletedEvent) => void;
export type ErrorCallback = (data: ErrorEvent) => void;
export type FailedCallback = (data: FailedEvent) => void;

/**
 * Connection state of the WebSocket client.
 */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

/**
 * Callback for connection state changes.
 */
export type ConnectionStateCallback = (state: ConnectionState) => void;

/**
 * Task subscription with callbacks.
 */
interface TaskSubscription {
  taskId: string;
  onProgress?: ProgressCallback;
  onCompleted?: CompletedCallback;
  onError?: ErrorCallback;
  onFailed?: FailedCallback;
}

// ============================================================================
// Configuration
// ============================================================================

/**
 * Determine the WebSocket base URL based on environment.
 * 
 * In Docker/production: Use current origin so nginx can proxy WebSocket
 * In development: Use localhost:8000 directly
 */
const getWsBaseUrl = (): string => {
  // If explicitly set via environment variable, use that
  if (import.meta.env.VITE_WS_BASE_URL) {
    return import.meta.env.VITE_WS_BASE_URL;
  }
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  // In production (served via nginx), use current origin for WebSocket proxy
  if (import.meta.env.PROD) {
    return window.location.origin;
  }
  
  // In development, connect directly to backend
  return 'http://localhost:8000';
};

const WS_BASE_URL = getWsBaseUrl();

/**
 * Default reconnection options.
 *
 * Requirements satisfied:
 * - 3.1: Maintain connection for progress updates
 */
const DEFAULT_RECONNECTION_OPTIONS = {
  reconnection: true,
  reconnectionAttempts: 10,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  randomizationFactor: 0.5,
  timeout: 20000,
};

/**
 * Reconnection configuration.
 */
export interface ReconnectionConfig {
  /** Maximum number of reconnection attempts */
  maxAttempts: number;
  /** Initial delay between reconnection attempts (ms) */
  initialDelay: number;
  /** Maximum delay between reconnection attempts (ms) */
  maxDelay: number;
  /** Whether to automatically reconnect on connection loss */
  autoReconnect: boolean;
}

// ============================================================================
// WebSocket Client Class
// ============================================================================

/**
 * WebSocket client for real-time task progress tracking.
 *
 * Provides:
 * - Connection management with automatic reconnection
 * - Task subscription for progress updates
 * - Event callbacks for progress, completion, and errors
 * - Manual reconnection capability
 *
 * Requirements satisfied:
 * - 3.1: Display progress bar showing completion percentage
 * - 3.2: Update progress in real-time during processing
 */
export class WebSocketClient {
  private socket: Socket | null = null;
  private subscriptions: Map<string, TaskSubscription> = new Map();
  private connectionState: ConnectionState = 'disconnected';
  private connectionStateCallbacks: Set<ConnectionStateCallback> = new Set();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = DEFAULT_RECONNECTION_OPTIONS.reconnectionAttempts;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private currentUrl: string = WS_BASE_URL;
  private autoReconnect: boolean = true;

  /**
   * Get the current connection state.
   */
  get state(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Check if the client is connected.
   */
  get isConnected(): boolean {
    return this.connectionState === 'connected';
  }

  /**
   * Get the number of reconnection attempts made.
   */
  get reconnectionAttempts(): number {
    return this.reconnectAttempts;
  }

  /**
   * Connect to the WebSocket server.
   *
   * @param url - Optional custom URL (defaults to WS_BASE_URL)
   * @param config - Optional reconnection configuration
   */
  connect(url?: string, config?: Partial<ReconnectionConfig>): void {
    if (this.socket?.connected) {
      return;
    }

    // Clear any pending reconnect timer
    this.clearReconnectTimer();

    this.currentUrl = url || WS_BASE_URL;
    
    if (config) {
      if (config.maxAttempts !== undefined) {
        this.maxReconnectAttempts = config.maxAttempts;
      }
      if (config.autoReconnect !== undefined) {
        this.autoReconnect = config.autoReconnect;
      }
    }

    this.setConnectionState('connecting');

    const socketOptions = {
      ...DEFAULT_RECONNECTION_OPTIONS,
      reconnection: this.autoReconnect,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: config?.initialDelay ?? DEFAULT_RECONNECTION_OPTIONS.reconnectionDelay,
      reconnectionDelayMax: config?.maxDelay ?? DEFAULT_RECONNECTION_OPTIONS.reconnectionDelayMax,
      transports: ['websocket', 'polling'],
    };

    this.socket = io(this.currentUrl, socketOptions);

    this.setupEventHandlers();
  }

  /**
   * Disconnect from the WebSocket server.
   */
  disconnect(): void {
    this.clearReconnectTimer();
    
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.subscriptions.clear();
    this.setConnectionState('disconnected');
    this.reconnectAttempts = 0;
  }

  /**
   * Manually trigger a reconnection attempt.
   *
   * Useful when automatic reconnection has been exhausted or disabled.
   *
   * Requirements satisfied:
   * - 3.1: Maintain connection for progress updates
   */
  reconnect(): void {
    if (this.connectionState === 'connected' || this.connectionState === 'connecting') {
      return;
    }

    // Reset reconnection attempts for manual reconnect
    this.reconnectAttempts = 0;
    
    // Disconnect existing socket if any
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    // Reconnect with stored URL
    this.connect(this.currentUrl);
  }

  /**
   * Subscribe to task progress updates.
   *
   * @param taskId - Task ID to subscribe to
   * @param callbacks - Event callbacks
   * @returns Unsubscribe function
   */
  subscribeToTask(
    taskId: string,
    callbacks: {
      onProgress?: ProgressCallback;
      onCompleted?: CompletedCallback;
      onError?: ErrorCallback;
      onFailed?: FailedCallback;
    }
  ): () => void {
    // Store subscription
    this.subscriptions.set(taskId, {
      taskId,
      ...callbacks,
    });

    // Send subscription request to server
    if (this.socket?.connected) {
      this.socket.emit('subscribe_task', { task_id: taskId });
    }

    // Return unsubscribe function
    return () => this.unsubscribeFromTask(taskId);
  }

  /**
   * Unsubscribe from task progress updates.
   *
   * @param taskId - Task ID to unsubscribe from
   */
  unsubscribeFromTask(taskId: string): void {
    this.subscriptions.delete(taskId);

    if (this.socket?.connected) {
      this.socket.emit('unsubscribe_task', { task_id: taskId });
    }
  }

  /**
   * Register a callback for connection state changes.
   *
   * @param callback - Callback function
   * @returns Unregister function
   */
  onConnectionStateChange(callback: ConnectionStateCallback): () => void {
    this.connectionStateCallbacks.add(callback);
    // Immediately call with current state
    callback(this.connectionState);
    return () => this.connectionStateCallbacks.delete(callback);
  }

  /**
   * Set up Socket.IO event handlers.
   */
  private setupEventHandlers(): void {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      this.setConnectionState('connected');
      this.reconnectAttempts = 0;
      this.resubscribeAll();
      
      if (import.meta.env.DEV) {
        console.log('[WebSocket] Connected');
      }
    });

    this.socket.on('disconnect', (reason) => {
      if (import.meta.env.DEV) {
        console.log(`[WebSocket] Disconnected: ${reason}`);
      }
      
      if (reason === 'io server disconnect') {
        // Server disconnected, don't auto-reconnect
        this.setConnectionState('disconnected');
      } else if (reason === 'io client disconnect') {
        // Client initiated disconnect
        this.setConnectionState('disconnected');
      } else {
        // Connection lost, will auto-reconnect if enabled
        if (this.autoReconnect) {
          this.setConnectionState('reconnecting');
        } else {
          this.setConnectionState('disconnected');
        }
      }
    });

    this.socket.on('connect_error', (error) => {
      if (import.meta.env.DEV) {
        console.log(`[WebSocket] Connection error: ${error.message}`);
      }
      
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        this.setConnectionState('disconnected');
        if (import.meta.env.DEV) {
          console.log('[WebSocket] Max reconnection attempts reached');
        }
      } else {
        this.setConnectionState('reconnecting');
      }
    });

    // Socket.IO built-in reconnection events
    this.socket.io.on('reconnect_attempt', (attempt) => {
      this.reconnectAttempts = attempt;
      this.setConnectionState('reconnecting');
      
      if (import.meta.env.DEV) {
        console.log(`[WebSocket] Reconnection attempt ${attempt}`);
      }
    });

    this.socket.io.on('reconnect', () => {
      if (import.meta.env.DEV) {
        console.log('[WebSocket] Reconnected');
      }
    });

    this.socket.io.on('reconnect_failed', () => {
      this.setConnectionState('disconnected');
      
      if (import.meta.env.DEV) {
        console.log('[WebSocket] Reconnection failed');
      }
    });

    // Task events
    this.socket.on('progress', (data: ProgressEvent) => {
      const subscription = this.subscriptions.get(data.task_id);
      if (subscription?.onProgress) {
        subscription.onProgress(data);
      }
    });

    this.socket.on('completed', (data: CompletedEvent) => {
      const subscription = this.subscriptions.get(data.task_id);
      if (subscription?.onCompleted) {
        subscription.onCompleted(data);
      }
    });

    this.socket.on('error', (data: ErrorEvent) => {
      const subscription = this.subscriptions.get(data.task_id);
      if (subscription?.onError) {
        subscription.onError(data);
      }
    });

    this.socket.on('failed', (data: FailedEvent) => {
      const subscription = this.subscriptions.get(data.task_id);
      if (subscription?.onFailed) {
        subscription.onFailed(data);
      }
    });
  }

  /**
   * Re-subscribe to all tasks after reconnection.
   */
  private resubscribeAll(): void {
    if (!this.socket?.connected) return;

    for (const taskId of this.subscriptions.keys()) {
      this.socket.emit('subscribe_task', { task_id: taskId });
    }
    
    if (import.meta.env.DEV && this.subscriptions.size > 0) {
      console.log(`[WebSocket] Re-subscribed to ${this.subscriptions.size} task(s)`);
    }
  }

  /**
   * Clear any pending reconnect timer.
   */
  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * Update connection state and notify callbacks.
   */
  private setConnectionState(state: ConnectionState): void {
    if (this.connectionState !== state) {
      this.connectionState = state;
      for (const callback of this.connectionStateCallbacks) {
        try {
          callback(state);
        } catch {
          // Ignore callback errors
        }
      }
    }
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

/**
 * Global WebSocket client instance.
 *
 * Use this singleton for all WebSocket operations in the application.
 */
export const wsClient = new WebSocketClient();

export default wsClient;
