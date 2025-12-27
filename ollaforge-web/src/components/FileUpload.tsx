/**
 * FileUpload component for dataset file upload.
 *
 * Provides drag-and-drop file upload functionality with:
 * - Support for JSONL, JSON, CSV, TSV, Parquet formats
 * - Support for HuggingFace dataset loading
 * - File information display after upload
 * - Field list display from uploaded dataset
 * - Preview of first few entries
 *
 * Requirements: 2.1, 4.1
 */

import React, { useState } from 'react';
import {
  Upload,
  Card,
  Typography,
  Space,
  Tag,
  Alert,
  Spin,
  Divider,
  Input,
  Button,
  Tabs,
  Form,
  InputNumber,
  Select,
} from 'antd';
import {
  InboxOutlined,
  FileOutlined,
  CheckCircleOutlined,
  DeleteOutlined,
  CloudDownloadOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import type { UploadProps } from 'antd';
import { augmentationAPI, type AugmentUploadResponse, type HuggingFaceLoadRequest } from '../services/api';
import { DatasetPreview } from './DatasetPreview';

const { Dragger } = Upload;
const { Text, Title } = Typography;
const { TabPane } = Tabs;

/** Supported file extensions */
const SUPPORTED_EXTENSIONS = ['.jsonl', '.json', '.csv', '.tsv', '.parquet'];

export interface FileUploadProps {
  /** Callback when file is successfully uploaded */
  onUploadSuccess: (response: AugmentUploadResponse) => void;
  /** Callback when upload is cleared */
  onClear: () => void;
  /** Whether the component is disabled */
  disabled?: boolean;
  /** Current upload response (for controlled mode) */
  uploadResponse?: AugmentUploadResponse | null;
}

/**
 * FileUpload component for uploading dataset files.
 */
export const FileUpload: React.FC<FileUploadProps> = ({
  onUploadSuccess,
  onClear,
  disabled = false,
  uploadResponse,
}) => {
  const { t } = useTranslation();
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>('file');
  const [hfForm] = Form.useForm();

  /**
   * Validate file before upload.
   */
  const validateFile = (file: File): boolean => {
    // Check file extension
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(extension)) {
      setError(t('errors.invalidFile'));
      return false;
    }

    // Check file size (max 100MB)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      setError(t('augment.fileTooLarge', 'File is too large. Maximum size is 100MB.'));
      return false;
    }

    return true;
  };

  /**
   * Custom upload handler.
   */
  const handleUpload = async (file: File): Promise<void> => {
    if (!validateFile(file)) {
      return;
    }

    setUploading(true);
    setError(null);
    setFileName(file.name);

    try {
      const response = await augmentationAPI.uploadDataset(file);
      onUploadSuccess(response);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
      setFileName(null);
    } finally {
      setUploading(false);
    }
  };

  /**
   * Handle HuggingFace dataset loading.
   */
  const handleHuggingFaceLoad = async (values: HuggingFaceLoadRequest): Promise<void> => {
    setUploading(true);
    setError(null);
    setFileName(values.dataset_name);

    try {
      const response = await augmentationAPI.loadHuggingFaceDataset(values);
      onUploadSuccess(response);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load HuggingFace dataset';
      setError(errorMessage);
      setFileName(null);
    } finally {
      setUploading(false);
    }
  };

  /**
   * Handle file removal/clear.
   */
  const handleClear = () => {
    setFileName(null);
    setError(null);
    hfForm.resetFields();
    onClear();
  };

  /**
   * Upload props for Ant Design Dragger.
   */
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: false,
    accept: SUPPORTED_EXTENSIONS.join(','),
    showUploadList: false,
    disabled: disabled || uploading || !!uploadResponse,
    beforeUpload: (file) => {
      handleUpload(file);
      return false; // Prevent default upload behavior
    },
  };

  /**
   * Get file type tag color.
   */
  const getFileTypeColor = (filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'jsonl':
        return 'blue';
      case 'json':
        return 'cyan';
      case 'csv':
        return 'green';
      case 'tsv':
        return 'orange';
      case 'parquet':
        return 'purple';
      default:
        return 'default';
    }
  };

  // Show upload result if we have a response
  if (uploadResponse) {
    const isHuggingFace = uploadResponse.source_type === 'huggingface';
    
    return (
      <Card
        title={
          <Space>
            <CheckCircleOutlined style={{ color: '#52c41a' }} />
            <span>{t('augment.uploadSuccess', 'File Uploaded Successfully')}</span>
          </Space>
        }
        extra={
          !disabled && (
            <a onClick={handleClear}>
              <DeleteOutlined /> {t('augment.clearFile', 'Clear')}
            </a>
          )
        }
      >
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          {/* File Info */}
          <div>
            <Space>
              {isHuggingFace ? <CloudDownloadOutlined /> : <FileOutlined />}
              <Text strong>{fileName}</Text>
              <Tag color={isHuggingFace ? 'gold' : getFileTypeColor(fileName || '')}>
                {isHuggingFace ? '🤗 HuggingFace' : fileName?.split('.').pop()?.toUpperCase()}
              </Tag>
            </Space>
          </div>

          {/* Entry Count */}
          <Text>
            {t('augment.entryCount', 'Entries')}: <Text strong>{uploadResponse.entry_count}</Text>
          </Text>

          <Divider style={{ margin: '12px 0' }} />

          {/* Field List */}
          <div>
            <Title level={5} style={{ marginBottom: 8 }}>
              {t('augment.availableFields', 'Available Fields')}
            </Title>
            <Space wrap>
              {uploadResponse.fields.map((field) => (
                <Tag key={field} color="blue">
                  {field}
                </Tag>
              ))}
            </Space>
          </div>

          <Divider style={{ margin: '12px 0' }} />

          {/* Preview */}
          {uploadResponse.preview && uploadResponse.preview.length > 0 && (
            <div>
              <Title level={5} style={{ marginBottom: 8 }}>
                {t('augment.dataPreview', 'Data Preview')}
              </Title>
              <DatasetPreview
                entries={uploadResponse.preview}
                maxEntries={3}
                showRowNumbers={true}
              />
            </div>
          )}
        </Space>
      </Card>
    );
  }

  return (
    <div className="file-upload">
      <Spin spinning={uploading} tip={t('common.loading')}>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab={t('augment.uploadTab', '📁 Upload File')} key="file">
            <Dragger {...uploadProps}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">{t('augment.upload')}</p>
              <p className="ant-upload-hint">{t('augment.uploadHint')}</p>
            </Dragger>
          </TabPane>
          
          <TabPane tab={t('augment.huggingfaceTab', '🤗 HuggingFace')} key="huggingface">
            <Form
              form={hfForm}
              layout="vertical"
              onFinish={handleHuggingFaceLoad}
              initialValues={{ split: 'train' }}
            >
              <Form.Item
                name="dataset_name"
                label={t('augment.hfDatasetName', 'Dataset Name')}
                rules={[
                  { required: true, message: t('augment.hfDatasetNameRequired', 'Dataset name is required') },
                  { pattern: /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9._-]+$/, message: t('augment.hfDatasetNameInvalid', 'Invalid format. Use: username/dataset-name') }
                ]}
                extra={t('augment.hfDatasetNameHint', 'e.g., renhehuang/govQA-database-zhtw')}
              >
                <Input 
                  placeholder="username/dataset-name" 
                  prefix={<CloudDownloadOutlined />}
                />
              </Form.Item>
              
              <Form.Item
                name="split"
                label={t('augment.hfSplit', 'Split')}
              >
                <Select>
                  <Select.Option value="train">train</Select.Option>
                  <Select.Option value="test">test</Select.Option>
                  <Select.Option value="validation">validation</Select.Option>
                </Select>
              </Form.Item>
              
              <Form.Item
                name="config_name"
                label={t('augment.hfConfig', 'Configuration (Optional)')}
              >
                <Input placeholder={t('augment.hfConfigPlaceholder', 'Leave empty for default')} />
              </Form.Item>
              
              <Form.Item
                name="max_entries"
                label={t('augment.hfMaxEntries', 'Max Entries (Optional)')}
                extra={t('augment.hfMaxEntriesHint', 'Limit the number of entries to load')}
              >
                <InputNumber min={1} max={100000} style={{ width: '100%' }} />
              </Form.Item>
              
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={uploading} block>
                  {t('augment.hfLoad', 'Load from HuggingFace')}
                </Button>
              </Form.Item>
            </Form>
          </TabPane>
        </Tabs>
      </Spin>

      {error && (
        <Alert
          message={t('common.error')}
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginTop: 16 }}
        />
      )}
    </div>
  );
};

export default FileUpload;
