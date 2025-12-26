/**
 * OfflineIndicator component for showing network connectivity status.
 *
 * Features:
 * - Shows banner when offline
 * - Auto-hides when back online
 * - Shows queued operations count
 *
 * Requirements: 7.5
 */

import React from 'react';
import { Alert, Badge, Space, Typography } from 'antd';
import { WifiOutlined, DisconnectOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNetworkStatus } from '../hooks/useNetworkStatus';

const { Text } = Typography;

export interface OfflineIndicatorProps {
  /** Number of queued operations */
  queuedOperations?: number;
  /** Callback when connection is restored */
  onConnectionRestored?: () => void;
  /** Whether to show as a floating banner */
  floating?: boolean;
  /** Additional CSS class */
  className?: string;
  /** Additional inline styles */
  style?: React.CSSProperties;
}

/**
 * OfflineIndicator shows a banner when the user is offline.
 */
export const OfflineIndicator: React.FC<OfflineIndicatorProps> = ({
  queuedOperations = 0,
  onConnectionRestored,
  floating = true,
  className,
  style,
}) => {
  const { t } = useTranslation();
  const { isOnline, isDetected } = useNetworkStatus({
    onOnline: onConnectionRestored,
  });

  // Don't render anything if online or not yet detected
  if (!isDetected || isOnline) {
    return null;
  }

  const floatingStyle: React.CSSProperties = floating
    ? {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1001,
        borderRadius: 0,
      }
    : {};

  return (
    <Alert
      message={
        <Space>
          <DisconnectOutlined />
          <Text strong>{t('errors.offline')}</Text>
          {queuedOperations > 0 && (
            <Badge
              count={queuedOperations}
              style={{ backgroundColor: '#faad14' }}
              title={t('errors.operationQueued')}
            />
          )}
        </Space>
      }
      description={t('errors.offlineDescription')}
      type="warning"
      showIcon={false}
      banner
      className={className}
      style={{ ...floatingStyle, ...style }}
    />
  );
};

/**
 * OnlineIndicator shows a brief notification when connection is restored.
 */
export const OnlineIndicator: React.FC<{
  visible: boolean;
  onClose: () => void;
}> = ({ visible, onClose }) => {
  const { t } = useTranslation();

  if (!visible) {
    return null;
  }

  return (
    <Alert
      message={
        <Space>
          <WifiOutlined style={{ color: '#52c41a' }} />
          <Text>{t('errors.connectionRestored')}</Text>
        </Space>
      }
      type="success"
      showIcon={false}
      banner
      closable
      onClose={onClose}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1001,
        borderRadius: 0,
      }}
    />
  );
};

export default OfflineIndicator;
