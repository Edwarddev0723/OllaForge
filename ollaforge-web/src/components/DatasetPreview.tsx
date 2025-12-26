import React, { useState, useMemo } from 'react';
import { Table, Typography, Button, Modal, Tag, Space, Tooltip } from 'antd';
import { ExpandOutlined, CopyOutlined, CheckOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import type { ColumnsType } from 'antd/es/table';

const { Text, Paragraph } = Typography;

/** Maximum characters to display before truncation */
const MAX_TEXT_LENGTH = 100;

/** Maximum JSON depth to display inline */
const MAX_JSON_DEPTH = 2;

export interface DatasetPreviewProps {
  /** Array of dataset entries to display */
  entries: Array<Record<string, unknown>>;
  /** Maximum number of entries to show (default: 5) */
  maxEntries?: number;
  /** Optional title for the preview */
  title?: string;
  /** Whether to show row numbers */
  showRowNumbers?: boolean;
  /** Comparison mode: show before/after columns */
  comparisonMode?: boolean;
  /** Original entries for comparison (when comparisonMode is true) */
  originalEntries?: Array<Record<string, unknown>>;
}

interface ExpandedCellProps {
  value: unknown;
  fieldName: string;
}

/**
 * Component to display a cell value with truncation and expand option
 */
const ExpandableCell: React.FC<ExpandedCellProps> = ({ value, fieldName }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const { t } = useTranslation();

  const formattedValue = useMemo(() => formatValue(value, 0), [value]);
  const fullFormattedValue = useMemo(() => formatValueFull(value), [value]);
  const needsTruncation = typeof formattedValue === 'string' && formattedValue.length > MAX_TEXT_LENGTH;

  const displayValue = needsTruncation
    ? formattedValue.substring(0, MAX_TEXT_LENGTH) + '...'
    : formattedValue;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(
        typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)
      );
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available
    }
  };

  return (
    <>
      <Space>
        <Text style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {displayValue}
        </Text>
        {needsTruncation && (
          <Tooltip title={t('preview.expand')}>
            <Button
              type="text"
              size="small"
              icon={<ExpandOutlined />}
              onClick={() => setIsModalOpen(true)}
            />
          </Tooltip>
        )}
      </Space>

      <Modal
        title={fieldName}
        open={isModalOpen}
        onCancel={() => setIsModalOpen(false)}
        footer={[
          <Button
            key="copy"
            icon={copied ? <CheckOutlined /> : <CopyOutlined />}
            onClick={handleCopy}
          >
            {copied ? t('preview.copied') : t('preview.copy')}
          </Button>,
          <Button key="close" type="primary" onClick={() => setIsModalOpen(false)}>
            {t('common.cancel')}
          </Button>,
        ]}
        width={700}
      >
        <Paragraph
          style={{
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            maxHeight: '60vh',
            overflow: 'auto',
            backgroundColor: '#f5f5f5',
            padding: '12px',
            borderRadius: '4px',
            fontFamily: 'monospace',
          }}
        >
          {fullFormattedValue}
        </Paragraph>
      </Modal>
    </>
  );
};

/**
 * Format a value for display with depth limit
 */
function formatValue(value: unknown, depth: number): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';

  if (typeof value === 'string') {
    return value;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  if (Array.isArray(value)) {
    if (depth >= MAX_JSON_DEPTH) {
      return `[Array(${value.length})]`;
    }
    const items = value.slice(0, 3).map((item) => formatValue(item, depth + 1));
    const suffix = value.length > 3 ? `, ... +${value.length - 3} more` : '';
    return `[${items.join(', ')}${suffix}]`;
  }

  if (typeof value === 'object') {
    if (depth >= MAX_JSON_DEPTH) {
      return '[Object]';
    }
    const keys = Object.keys(value as object);
    const items = keys.slice(0, 3).map((key) => {
      const val = (value as Record<string, unknown>)[key];
      return `${key}: ${formatValue(val, depth + 1)}`;
    });
    const suffix = keys.length > 3 ? `, ... +${keys.length - 3} more` : '';
    return `{${items.join(', ')}${suffix}}`;
  }

  return String(value);
}

/**
 * Format a value for full display (in modal)
 */
function formatValueFull(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';

  if (typeof value === 'string') {
    return value;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  if (typeof value === 'object') {
    return JSON.stringify(value, null, 2);
  }

  return String(value);
}

/**
 * Get the type tag for a value
 */
function getTypeTag(value: unknown): React.ReactNode {
  if (value === null) return <Tag color="default">null</Tag>;
  if (value === undefined) return <Tag color="default">undefined</Tag>;
  if (typeof value === 'string') return <Tag color="green">string</Tag>;
  if (typeof value === 'number') return <Tag color="blue">number</Tag>;
  if (typeof value === 'boolean') return <Tag color="purple">boolean</Tag>;
  if (Array.isArray(value)) return <Tag color="orange">array</Tag>;
  if (typeof value === 'object') return <Tag color="cyan">object</Tag>;
  return <Tag>{typeof value}</Tag>;
}

/**
 * DatasetPreview component displays dataset entries in a formatted table
 * with JSON formatting and text truncation with expand option.
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
 */
export const DatasetPreview: React.FC<DatasetPreviewProps> = ({
  entries,
  maxEntries = 5,
  title,
  showRowNumbers = true,
  comparisonMode = false,
  originalEntries,
}) => {
  const { t } = useTranslation();

  // Get entries to display (limited by maxEntries)
  const displayEntries = useMemo(
    () => entries.slice(0, maxEntries),
    [entries, maxEntries]
  );

  // Extract all unique field names from entries
  const fieldNames = useMemo(() => {
    const fields = new Set<string>();
    displayEntries.forEach((entry) => {
      Object.keys(entry).forEach((key) => fields.add(key));
    });
    return Array.from(fields);
  }, [displayEntries]);

  // Build table columns
  const columns: ColumnsType<Record<string, unknown>> = useMemo(() => {
    const cols: ColumnsType<Record<string, unknown>> = [];

    // Row number column
    if (showRowNumbers) {
      cols.push({
        title: '#',
        key: 'rowNumber',
        width: 50,
        fixed: 'left',
        render: (_, __, index) => (
          <Text type="secondary">{index + 1}</Text>
        ),
      });
    }

    // Field columns
    fieldNames.forEach((fieldName) => {
      if (comparisonMode && originalEntries) {
        // Comparison mode: show original and augmented side by side
        cols.push({
          title: (
            <Space direction="vertical" size={0}>
              <Text strong>{fieldName}</Text>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {t('preview.original')}
              </Text>
            </Space>
          ),
          key: `${fieldName}_original`,
          width: 250,
          render: (_, __, index) => {
            const originalEntry = originalEntries[index];
            const value = originalEntry?.[fieldName];
            return (
              <Space direction="vertical" size={2}>
                {getTypeTag(value)}
                <ExpandableCell value={value} fieldName={`${fieldName} (${t('preview.original')})`} />
              </Space>
            );
          },
        });
        cols.push({
          title: (
            <Space direction="vertical" size={0}>
              <Text strong>{fieldName}</Text>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {t('preview.augmented')}
              </Text>
            </Space>
          ),
          key: `${fieldName}_augmented`,
          width: 250,
          render: (_, record) => {
            const value = record[fieldName];
            return (
              <Space direction="vertical" size={2}>
                {getTypeTag(value)}
                <ExpandableCell value={value} fieldName={`${fieldName} (${t('preview.augmented')})`} />
              </Space>
            );
          },
        });
      } else {
        // Normal mode: single column per field
        cols.push({
          title: fieldName,
          key: fieldName,
          width: 200,
          render: (_, record) => {
            const value = record[fieldName];
            return (
              <Space direction="vertical" size={2}>
                {getTypeTag(value)}
                <ExpandableCell value={value} fieldName={fieldName} />
              </Space>
            );
          },
        });
      }
    });

    return cols;
  }, [fieldNames, showRowNumbers, comparisonMode, originalEntries, t]);

  // Add row keys
  const dataWithKeys = useMemo(
    () =>
      displayEntries.map((entry, index) => ({
        ...entry,
        _key: index,
      })),
    [displayEntries]
  );

  return (
    <div className="dataset-preview">
      {title && (
        <Typography.Title level={5} style={{ marginBottom: 16 }}>
          {title}
        </Typography.Title>
      )}

      <Table
        columns={columns}
        dataSource={dataWithKeys}
        rowKey="_key"
        pagination={false}
        scroll={{ x: 'max-content' }}
        size="small"
        bordered
      />

      {entries.length > maxEntries && (
        <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
          {t('preview.showingEntries', {
            shown: maxEntries,
            total: entries.length,
          })}
        </Text>
      )}

      {entries.length === 0 && (
        <Text type="secondary">{t('preview.noEntries')}</Text>
      )}
    </div>
  );
};

export default DatasetPreview;
