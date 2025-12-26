/**
 * OperationQueue service for queuing operations during offline periods.
 *
 * Features:
 * - Queue operations when offline
 * - Retry operations when back online
 * - Persist queue to localStorage
 *
 * Requirements: 7.5
 */

export interface QueuedOperation {
  id: string;
  type: 'generation' | 'augmentation';
  config: Record<string, unknown>;
  timestamp: number;
  retryCount: number;
}

const QUEUE_STORAGE_KEY = 'ollaforge_operation_queue';
const MAX_RETRIES = 3;

/**
 * OperationQueue manages operations that need to be retried.
 */
class OperationQueue {
  private queue: QueuedOperation[] = [];
  private listeners: Set<(queue: QueuedOperation[]) => void> = new Set();

  constructor() {
    this.loadFromStorage();
    
    // Listen for online events to process queue
    if (typeof window !== 'undefined') {
      window.addEventListener('online', () => this.processQueue());
    }
  }

  /**
   * Load queue from localStorage.
   */
  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem(QUEUE_STORAGE_KEY);
      if (stored) {
        this.queue = JSON.parse(stored);
      }
    } catch {
      this.queue = [];
    }
  }

  /**
   * Save queue to localStorage.
   */
  private saveToStorage(): void {
    try {
      localStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify(this.queue));
    } catch {
      // Storage might be full or unavailable
    }
  }

  /**
   * Notify all listeners of queue changes.
   */
  private notifyListeners(): void {
    this.listeners.forEach((listener) => listener([...this.queue]));
  }

  /**
   * Add an operation to the queue.
   */
  add(type: QueuedOperation['type'], config: Record<string, unknown>): string {
    const operation: QueuedOperation = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      config,
      timestamp: Date.now(),
      retryCount: 0,
    };

    this.queue.push(operation);
    this.saveToStorage();
    this.notifyListeners();

    return operation.id;
  }

  /**
   * Remove an operation from the queue.
   */
  remove(id: string): void {
    this.queue = this.queue.filter((op) => op.id !== id);
    this.saveToStorage();
    this.notifyListeners();
  }

  /**
   * Get all queued operations.
   */
  getAll(): QueuedOperation[] {
    return [...this.queue];
  }

  /**
   * Get the number of queued operations.
   */
  getCount(): number {
    return this.queue.length;
  }

  /**
   * Clear all queued operations.
   */
  clear(): void {
    this.queue = [];
    this.saveToStorage();
    this.notifyListeners();
  }

  /**
   * Subscribe to queue changes.
   */
  subscribe(listener: (queue: QueuedOperation[]) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Process the queue when back online.
   */
  async processQueue(): Promise<void> {
    if (!navigator.onLine || this.queue.length === 0) {
      return;
    }

    const operations = [...this.queue];
    
    for (const operation of operations) {
      if (operation.retryCount >= MAX_RETRIES) {
        // Remove operations that have exceeded max retries
        this.remove(operation.id);
        continue;
      }

      try {
        // Increment retry count
        operation.retryCount++;
        this.saveToStorage();

        // The actual retry logic would be implemented by the consumer
        // This is just the queue management
        
        // For now, we'll emit an event that consumers can listen to
        const event = new CustomEvent('ollaforge:retry-operation', {
          detail: operation,
        });
        window.dispatchEvent(event);

        // Remove from queue after successful dispatch
        // The consumer should handle the actual API call
        this.remove(operation.id);
      } catch {
        // Keep in queue for next retry
      }
    }
  }
}

// Export singleton instance
export const operationQueue = new OperationQueue();

export default operationQueue;
