import React from 'react';
import { Progress, Typography, Space, Tag } from 'antd';
import { useTranslation } from 'react-i18next';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

export type ProgressStatus = 'pending' | 'active' | 'success' | 'exception';

export interface ProgressDisplayProps {
  /** Current progress count */
  current: number;
  /** Total count for progress calculation */
  total: number;
  /** Status of the progress */
  status: ProgressStatus;
  /** Optional status message to display */
  message?: string;
  /** Optional duration in seconds */
  duration?: number;
  /** Optional success count for statistics */
  successCount?: number;
  /** Optional failure count for statistics */
  failureCount?: number;
  /** Whether to show statistics */
  showStats?: boolean;
}

/**
 * ProgressDisplay component shows a progress bar with percentage,
 * status messages, and optional statistics.
 * 
 * Requirements: 3.1, 3.2, 3.3
 */
export const ProgressDisplay: React.FC<ProgressDisplayProps> = ({
  current,
  total,
  status,
  message,
  duration,
  successCount,
  failureCount,
  showStats = false,
}) => {
  const { t } = useTranslation();

  // Calculate percentage, handle edge cases
  const percent = total > 0 ? Math.round((current / total) * 100) : 0;

  // Map status to Ant Design Progress status
  const getProgressStatus = (): 'active' | 'success' | 'exception' | 'normal' => {
    switch (status) {
      case 'active':
        return 'active';
      case 'success':
        return 'success';
      case 'exception':
        return 'exception';
      case 'pending':
      default:
        return 'normal';
    }
  };

  // Get status icon
  const getStatusIcon = () => {
    switch (status) {
      case 'pending':
        return <ClockCircleOutlined />;
      case 'active':
        return <SyncOutlined spin />;
      case 'success':
        return <CheckCircleOutlined />;
      case 'exception':
        return <CloseCircleOutlined />;
      default:
        return null;
    }
  };

  // Get status color
  const getStatusColor = () => {
    switch (status) {
      case 'pending':
        return 'default';
      case 'active':
        return 'processing';
      case 'success':
        return 'success';
      case 'exception':
        return 'error';
      default:
        return 'default';
    }
  };

  // Get localized status text
  const getStatusText = () => {
    switch (status) {
      case 'pending':
        return t('common.pending');
      case 'active':
        return t('common.running');
      case 'success':
        return t('common.completed');
      case 'exception':
        return t('common.failed');
      default:
        return '';
    }
  };

  // Format duration
  const formatDuration = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
  };

  return (
    <div className="progress-display">
      <Space direction="vertical" style={{ width: '100%' }} size="small">
        {/* Status tag */}
        <Space>
          <Tag icon={getStatusIcon()} color={getStatusColor()}>
            {getStatusText()}
          </Tag>
          <Text type="secondary">
            {current} / {total}
          </Text>
        </Space>

        {/* Progress bar */}
        <Progress
          percent={percent}
          status={getProgressStatus()}
          strokeColor={status === 'active' ? { from: '#108ee9', to: '#87d068' } : undefined}
        />

        {/* Status message */}
        {message && (
          <Text type={status === 'exception' ? 'danger' : 'secondary'}>
            {message}
          </Text>
        )}

        {/* Statistics (shown on completion) */}
        {showStats && (status === 'success' || status === 'exception') && (
          <Space split={<Text type="secondary">|</Text>}>
            {duration !== undefined && (
              <Text type="secondary">
                {t('common.progress')}: {formatDuration(duration)}
              </Text>
            )}
            {successCount !== undefined && (
              <Text type="success">
                {t('common.success')}: {successCount}
              </Text>
            )}
            {failureCount !== undefined && failureCount > 0 && (
              <Text type="danger">
                {t('common.failed')}: {failureCount}
              </Text>
            )}
          </Space>
        )}
      </Space>
    </div>
  );
};

export default ProgressDisplay;
