/**
 * LoadingOverlay component for showing loading states during API calls.
 *
 * Features:
 * - Full-screen or container-based overlay
 * - Customizable loading message
 * - Spin indicator with optional text
 *
 * Requirements: 3.1
 */

import React from 'react';
import { Spin, Typography } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

const { Text } = Typography;

export interface LoadingOverlayProps {
  /** Whether the loading overlay is visible */
  loading: boolean;
  /** Loading message to display */
  message?: string;
  /** Size of the spinner */
  size?: 'small' | 'default' | 'large';
  /** Whether to cover the full screen */
  fullScreen?: boolean;
  /** Children to render behind the overlay */
  children?: React.ReactNode;
  /** Custom tip text for the spinner */
  tip?: string;
}

/**
 * LoadingOverlay shows a loading indicator over content during API calls.
 */
export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  loading,
  message,
  size = 'large',
  fullScreen = false,
  children,
  tip,
}) => {
  const spinIcon = <LoadingOutlined style={{ fontSize: size === 'large' ? 48 : size === 'default' ? 32 : 24 }} spin />;

  const overlayStyle: React.CSSProperties = fullScreen
    ? {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(255, 255, 255, 0.85)',
        zIndex: 1000,
      }
    : {
        position: 'relative',
      };

  if (fullScreen && loading) {
    return (
      <div style={overlayStyle}>
        <Spin indicator={spinIcon} size={size} />
        {(message || tip) && (
          <Text style={{ marginTop: 16, color: '#666' }}>{message || tip}</Text>
        )}
      </div>
    );
  }

  return (
    <Spin
      spinning={loading}
      indicator={spinIcon}
      size={size}
      tip={tip || message}
    >
      {children}
    </Spin>
  );
};

export default LoadingOverlay;
