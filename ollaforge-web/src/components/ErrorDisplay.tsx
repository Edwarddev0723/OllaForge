/**
 * ErrorDisplay component for showing error messages with retry functionality.
 *
 * Features:
 * - Clear error message display
 * - Expandable technical details section
 * - Retry button with callback
 * - Localized error messages
 *
 * Requirements: 1.4, 2.5, 3.4
 */

import React, { useState } from 'react';
import { Alert, Button, Space, Typography, Collapse } from 'antd';
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Text, Paragraph } = Typography;

export type ErrorSeverity = 'error' | 'warning' | 'info';

export interface ErrorDisplayProps {
  /** Error message to display */
  message: string;
  /** Optional detailed description */
  description?: string;
  /** Technical details (shown in expandable section) */
  technicalDetails?: string | Record<string, unknown>;
  /** Error severity level */
  severity?: ErrorSeverity;
  /** Callback for retry action */
  onRetry?: () => void;
  /** Callback for dismiss action */
  onDismiss?: () => void;
  /** Whether retry is in progress */
  retrying?: boolean;
  /** Custom retry button text */
  retryText?: string;
  /** Whether to show the error */
  visible?: boolean;
  /** Additional CSS class */
  className?: string;
  /** Additional inline styles */
  style?: React.CSSProperties;
}

/**
 * ErrorDisplay component shows error messages with optional retry functionality
 * and expandable technical details.
 */
export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  message,
  description,
  technicalDetails,
  severity = 'error',
  onRetry,
  onDismiss,
  retrying = false,
  retryText,
  visible = true,
  className,
  style,
}) => {
  const { t } = useTranslation();
  const [detailsExpanded, setDetailsExpanded] = useState(false);

  if (!visible) {
    return null;
  }

  // Get icon based on severity
  const getIcon = () => {
    switch (severity) {
      case 'warning':
        return <WarningOutlined />;
      case 'info':
        return <InfoCircleOutlined />;
      case 'error':
      default:
        return <ExclamationCircleOutlined />;
    }
  };

  // Format technical details for display
  const formatTechnicalDetails = (): string => {
    if (!technicalDetails) return '';
    if (typeof technicalDetails === 'string') return technicalDetails;
    try {
      return JSON.stringify(technicalDetails, null, 2);
    } catch {
      return String(technicalDetails);
    }
  };

  // Build action buttons
  const actionButtons = (
    <Space>
      {onRetry && (
        <Button
          size="small"
          icon={<ReloadOutlined spin={retrying} />}
          onClick={onRetry}
          loading={retrying}
        >
          {retryText || t('common.retry')}
        </Button>
      )}
      {onDismiss && (
        <Button size="small" onClick={onDismiss}>
          {t('common.close')}
        </Button>
      )}
    </Space>
  );

  // Build description with optional technical details
  const fullDescription = (
    <div>
      {description && <Paragraph style={{ marginBottom: technicalDetails ? 8 : 0 }}>{description}</Paragraph>}
      {technicalDetails && (
        <Collapse
          ghost
          activeKey={detailsExpanded ? ['details'] : []}
          onChange={(keys) => setDetailsExpanded(keys.includes('details'))}
          items={[
            {
              key: 'details',
              label: (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {t('errors.technicalDetails', 'Technical Details')}
                </Text>
              ),
              children: (
                <pre
                  style={{
                    background: '#f5f5f5',
                    padding: 8,
                    borderRadius: 4,
                    fontSize: 12,
                    overflow: 'auto',
                    maxHeight: 200,
                    margin: 0,
                  }}
                >
                  {formatTechnicalDetails()}
                </pre>
              ),
            },
          ]}
        />
      )}
    </div>
  );

  return (
    <Alert
      message={message}
      description={fullDescription}
      type={severity}
      showIcon
      icon={getIcon()}
      action={actionButtons}
      className={className}
      style={style}
      closable={!!onDismiss}
      onClose={onDismiss}
    />
  );
};

export default ErrorDisplay;
