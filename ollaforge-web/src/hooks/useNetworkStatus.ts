/**
 * useNetworkStatus hook for detecting network connectivity.
 *
 * Features:
 * - Detects online/offline status
 * - Provides callbacks for status changes
 * - Tracks connection quality
 *
 * Requirements: 7.5
 */

import { useState, useEffect, useCallback } from 'react';

export interface NetworkStatus {
  /** Whether the browser is online */
  isOnline: boolean;
  /** Whether we've detected the status (false during initial load) */
  isDetected: boolean;
  /** Timestamp of last status change */
  lastChanged: Date | null;
  /** Connection type if available */
  connectionType?: string;
}

export interface UseNetworkStatusOptions {
  /** Callback when going online */
  onOnline?: () => void;
  /** Callback when going offline */
  onOffline?: () => void;
}

/**
 * Hook for monitoring network connectivity status.
 */
export const useNetworkStatus = (options: UseNetworkStatusOptions = {}): NetworkStatus => {
  const { onOnline, onOffline } = options;

  // Get connection type if available
  const getConnectionType = useCallback((): string | undefined => {
    if (typeof navigator !== 'undefined' && 'connection' in navigator) {
      const connection = (navigator as Navigator & { connection?: { effectiveType?: string } }).connection;
      return connection?.effectiveType;
    }
    return undefined;
  }, []);

  // Initialize state with detected values
  const [status, setStatus] = useState<NetworkStatus>(() => ({
    isOnline: typeof navigator !== 'undefined' ? navigator.onLine : true,
    isDetected: true,
    lastChanged: null,
    connectionType: typeof navigator !== 'undefined' ? getConnectionType() : undefined,
  }));

  // Handle online event
  const handleOnline = useCallback(() => {
    setStatus((prev) => ({
      ...prev,
      isOnline: true,
      isDetected: true,
      lastChanged: new Date(),
      connectionType: getConnectionType(),
    }));
    onOnline?.();
  }, [onOnline, getConnectionType]);

  // Handle offline event
  const handleOffline = useCallback(() => {
    setStatus((prev) => ({
      ...prev,
      isOnline: false,
      isDetected: true,
      lastChanged: new Date(),
      connectionType: undefined,
    }));
    onOffline?.();
  }, [onOffline]);

  useEffect(() => {
    // Add event listeners
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Listen for connection changes if available
    if ('connection' in navigator) {
      const connection = (navigator as Navigator & { connection?: EventTarget }).connection;
      const handleConnectionChange = () => {
        setStatus((prev) => ({
          ...prev,
          connectionType: getConnectionType(),
        }));
      };
      connection?.addEventListener('change', handleConnectionChange);
      
      return () => {
        window.removeEventListener('online', handleOnline);
        window.removeEventListener('offline', handleOffline);
        connection?.removeEventListener('change', handleConnectionChange);
      };
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [handleOnline, handleOffline, getConnectionType]);

  return status;
};

export default useNetworkStatus;
