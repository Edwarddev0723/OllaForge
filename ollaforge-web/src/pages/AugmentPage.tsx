/**
 * AugmentPage component for dataset augmentation workflow.
 * Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 5.2
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Typography, Card, Space, Button, Select, message, Modal, Input, Alert, Divider,
} from 'antd';
import {
  DownloadOutlined, SaveOutlined, ReloadOutlined, FolderOpenOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { FileUpload } from '../components/FileUpload';
import { AugmentationForm, type AugmentationFormValues } from '../components/AugmentationForm';
import { ProgressDisplay, type ProgressStatus } from '../components/ProgressDisplay';
import { DatasetPreview } from '../components/DatasetPreview';
import {
  augmentationAPI, downloadBlob, type AugmentationConfig, type AugmentUploadResponse,
  type TaskStatus, type AugmentPreviewResponse,
} from '../services/api';
import {
  wsClient, type ProgressEvent, type CompletedEvent, type ErrorEvent, type FailedEvent,
} from '../services/websocket';

const { Text } = Typography;
const { Option } = Select;
const DOWNLOAD_FORMATS = ['jsonl', 'json', 'csv', 'tsv', 'parquet'];
const SAVED_CONFIGS_KEY = 'ollaforge_augmentation_configs';

interface SavedConfig {
  id: string;
  name: string;
  type: 'augmentation';
  config: Omit<AugmentationFormValues, 'file_id'>;
  createdAt: string;
}

type PageState = 'idle' | 'previewing' | 'augmenting' | 'completed' | 'failed';

const AugmentPage: React.FC = () => {
  const { t } = useTranslation();
  const [pageState, setPageState] = useState<PageState>('idle');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [currentConfig, setCurrentConfig] = useState<AugmentationFormValues | null>(null);
  const [uploadResponse, setUploadResponse] = useState<AugmentUploadResponse | null>(null);
  const [previewData, setPreviewData] = useState<AugmentPreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0);
  const [progressStatus, setProgressStatus] = useState<ProgressStatus>('pending');
  const [progressMessage, setProgressMessage] = useState<string>('');
  const [duration, setDuration] = useState<number | undefined>(undefined);
  const [successCount, setSuccessCount] = useState<number | undefined>(undefined);
  const [failureCount, setFailureCount] = useState<number | undefined>(undefined);
  const [previewEntries, setPreviewEntries] = useState<Record<string, unknown>[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [downloadFormat, setDownloadFormat] = useState('jsonl');
  const [downloading, setDownloading] = useState(false);
  const [saveModalVisible, setSaveModalVisible] = useState(false);
  const [configName, setConfigName] = useState('');
  const [savedConfigs, setSavedConfigs] = useState<SavedConfig[]>([]);
  const [loadModalVisible, setLoadModalVisible] = useState(false);

  useEffect(() => {
    const configsStr = localStorage.getItem(SAVED_CONFIGS_KEY);
    if (configsStr) {
      try { setSavedConfigs(JSON.parse(configsStr) as SavedConfig[]); } catch { setSavedConfigs([]); }
    }
  }, []);

  useEffect(() => { wsClient.connect(); return () => {}; }, []);

  const handleProgress = useCallback((data: ProgressEvent) => {
    setProgress(data.progress);
    setTotal(data.total);
    setProgressStatus('active');
    if (data.message) setProgressMessage(data.message);
  }, []);

  const handleCompleted = useCallback((data: CompletedEvent) => {
    setProgress(data.total);
    setTotal(data.total);
    setProgressStatus('success');
    setDuration(data.duration);
    setSuccessCount(data.success_count);
    setFailureCount(data.failure_count);
    setPageState('completed');
    if (data.message) setProgressMessage(data.message);
    message.success(t('augment.augmentationComplete'));
  }, [t]);

  const handleError = useCallback((data: ErrorEvent) => {
    if (data.error_type === 'item_error') message.warning(data.error);
    else setProgressMessage(data.error);
  }, []);

  const handleFailed = useCallback((data: FailedEvent) => {
    setProgressStatus('exception');
    setErrorMessage(data.error);
    setPageState('failed');
    message.error(t('errors.augmentationFailed'));
  }, [t]);

  useEffect(() => {
    if (!taskId) return;
    const unsubscribe = wsClient.subscribeToTask(taskId, {
      onProgress: handleProgress, onCompleted: handleCompleted, onError: handleError, onFailed: handleFailed,
    });
    return () => { unsubscribe(); };
  }, [taskId, handleProgress, handleCompleted, handleError, handleFailed]);

  useEffect(() => {
    if (pageState !== 'completed' || !taskId) return;
    const fetchResult = async () => {
      try {
        const status: TaskStatus = await augmentationAPI.getStatus(taskId);
        if (status.result && Array.isArray(status.result.entries)) {
          setPreviewEntries(status.result.entries.slice(0, 5));
        }
      } catch (error) { console.error('Failed to fetch result:', error); }
    };
    fetchResult();
  }, [pageState, taskId]);

  const handleUploadSuccess = (response: AugmentUploadResponse) => {
    setUploadResponse(response);
    setPreviewData(null);
    setErrorMessage(null);
  };

  const handleClearFile = () => {
    setUploadResponse(null);
    setPreviewData(null);
    setCurrentConfig(null);
    setErrorMessage(null);
  };

  const handlePreview = async (config: AugmentationConfig) => {
    if (!uploadResponse) return;
    setPreviewLoading(true);
    setPreviewData(null);
    setErrorMessage(null);
    try {
      const response = await augmentationAPI.previewAugmentation({
        file_id: config.file_id, target_field: config.target_field, instruction: config.instruction,
        model: config.model, create_new_field: config.create_new_field, context_fields: config.context_fields,
        preview_count: 3,
      });
      setPreviewData(response);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Preview failed';
      setErrorMessage(errorMsg);
      message.error(errorMsg);
    } finally { setPreviewLoading(false); }
  };

  const handleSubmit = async (config: AugmentationConfig) => {
    if (!uploadResponse) { message.error(t('augment.uploadFirst')); return; }
    setCurrentConfig({
      target_field: config.target_field, instruction: config.instruction, model: config.model,
      language: config.language, create_new_field: config.create_new_field,
      context_fields: config.context_fields, concurrency: config.concurrency,
    });
    setPageState('augmenting');
    setProgress(0);
    setTotal(uploadResponse.entry_count);
    setProgressStatus('pending');
    setProgressMessage('');
    setErrorMessage(null);
    setPreviewEntries([]);
    setDuration(undefined);
    setSuccessCount(undefined);
    setFailureCount(undefined);
    try {
      const response = await augmentationAPI.startAugmentation(config);
      setTaskId(response.task_id);
      setProgressStatus('active');
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      setErrorMessage(errorMsg);
      setProgressStatus('exception');
      setPageState('failed');
      message.error(t('errors.augmentationFailed'));
    }
  };

  const handleDownload = async () => {
    if (!taskId) return;
    setDownloading(true);
    try {
      const blob = await augmentationAPI.downloadDataset(taskId, downloadFormat);
      downloadBlob(blob, `augmented_${taskId}.${downloadFormat}`);
      message.success(t('common.success'));
    } catch (error) {
      message.error(error instanceof Error ? error.message : 'Download failed');
    } finally { setDownloading(false); }
  };

  const handleSaveConfig = () => {
    if (!currentConfig || !configName.trim()) return;
    const savedConfig: SavedConfig = {
      id: Date.now().toString(), name: configName.trim(), type: 'augmentation',
      config: currentConfig, createdAt: new Date().toISOString(),
    };
    const existingConfigsStr = localStorage.getItem(SAVED_CONFIGS_KEY);
    const existingConfigs: SavedConfig[] = existingConfigsStr ? JSON.parse(existingConfigsStr) : [];
    existingConfigs.push(savedConfig);
    localStorage.setItem(SAVED_CONFIGS_KEY, JSON.stringify(existingConfigs));
    message.success(t('common.success'));
    setSaveModalVisible(false);
    setConfigName('');
    setSavedConfigs([...existingConfigs]);
  };

  const handleLoadConfig = (config: SavedConfig) => {
    setCurrentConfig(config.config);
    setLoadModalVisible(false);
    message.success(t('config.loadSuccess'));
  };

  const handleReset = () => {
    setPageState('idle');
    setTaskId(null);
    setProgress(0);
    setTotal(0);
    setProgressStatus('pending');
    setProgressMessage('');
    setErrorMessage(null);
    setPreviewEntries([]);
    setPreviewData(null);
    setDuration(undefined);
    setSuccessCount(undefined);
    setFailureCount(undefined);
  };

  const handleFullReset = () => { handleReset(); setUploadResponse(null); setCurrentConfig(null); };

  return (
    <div className="augment-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Card
          title={t('augment.upload')}
          extra={savedConfigs.length > 0 && pageState === 'idle' && (
            <Button icon={<FolderOpenOutlined />} onClick={() => setLoadModalVisible(true)}>
              {t('config.load')}
            </Button>
          )}
        >
          <FileUpload
            onUploadSuccess={handleUploadSuccess}
            onClear={handleClearFile}
            uploadResponse={uploadResponse}
            disabled={pageState === 'augmenting'}
          />
        </Card>

        {uploadResponse && pageState !== 'completed' && (
          <Card title={t('augment.title')}>
            <AugmentationForm
              availableFields={uploadResponse.fields}
              fileId={uploadResponse.file_id}
              onSubmit={handleSubmit}
              onPreview={handlePreview}
              initialValues={currentConfig || undefined}
              loading={pageState === 'augmenting'}
              previewLoading={previewLoading}
              disabled={pageState === 'augmenting'}
            />
          </Card>
        )}

        {previewData && previewData.previews.length > 0 && pageState === 'idle' && (
          <Card title={t('augment.previewTitle')}>
            <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
              {t('augment.previewDescription')}
            </Text>
            <DatasetPreview
              entries={previewData.previews.map(p => p.augmented)}
              originalEntries={previewData.previews.map(p => p.original)}
              maxEntries={3}
              comparisonMode={true}
            />
          </Card>
        )}

        {(pageState === 'augmenting' || pageState === 'completed' || pageState === 'failed') && (
          <Card title={t('common.progress')}>
            <ProgressDisplay
              current={progress} total={total} status={progressStatus} message={progressMessage}
              duration={duration} successCount={successCount} failureCount={failureCount}
              showStats={pageState === 'completed' || pageState === 'failed'}
            />
            {errorMessage && (
              <Alert
                message={t('common.error')} description={errorMessage} type="error" showIcon
                style={{ marginTop: 16 }}
                action={<Button size="small" onClick={handleReset}>{t('common.retry')}</Button>}
              />
            )}
          </Card>
        )}

        {pageState === 'completed' && (
          <>
            {previewEntries.length > 0 && (
              <Card title={t('augment.previewTitle')}>
                <DatasetPreview entries={previewEntries} maxEntries={5} />
              </Card>
            )}
            <Card title={t('augment.downloadReady')}>
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Text>{t('augment.entriesAugmented', { count: successCount || total })}</Text>
                {failureCount !== undefined && failureCount > 0 && (
                  <Alert
                    message={t('augment.partialFailure', 'Some entries failed to augment')}
                    description={t('augment.failureCount', '{{count}} entries failed', { count: failureCount })}
                    type="warning" showIcon
                  />
                )}
                <Space>
                  <Select value={downloadFormat} onChange={setDownloadFormat} style={{ width: 120 }}>
                    {DOWNLOAD_FORMATS.map((format) => (
                      <Option key={format} value={format}>{format.toUpperCase()}</Option>
                    ))}
                  </Select>
                  <Button type="primary" icon={<DownloadOutlined />} onClick={handleDownload} loading={downloading}>
                    {t('augment.download')}
                  </Button>
                </Space>
                <Divider />
                <Space>
                  <Text>{t('augment.saveConfigPrompt')}</Text>
                  <Button icon={<SaveOutlined />} onClick={() => setSaveModalVisible(true)}>
                    {t('common.saveConfig')}
                  </Button>
                </Space>
                <Divider />
                <Space>
                  <Button icon={<ReloadOutlined />} onClick={handleReset}>{t('augment.submit')}</Button>
                  <Button onClick={handleFullReset}>{t('augment.uploadNewFile', 'Upload New File')}</Button>
                </Space>
              </Space>
            </Card>
          </>
        )}
      </Space>

      <Modal
        title={t('common.saveConfig')} open={saveModalVisible} onOk={handleSaveConfig}
        onCancel={() => { setSaveModalVisible(false); setConfigName(''); }}
        okButtonProps={{ disabled: !configName.trim() }}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Text>{t('common.configName')}</Text>
          <Input
            placeholder={t('common.configNamePlaceholder')} value={configName}
            onChange={(e) => setConfigName(e.target.value)} onPressEnter={handleSaveConfig}
          />
        </Space>
      </Modal>

      <Modal
        title={t('config.loadConfig')} open={loadModalVisible}
        onCancel={() => setLoadModalVisible(false)} footer={null}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          {savedConfigs.length === 0 ? (
            <Text type="secondary">{t('config.noConfigs')}</Text>
          ) : (
            savedConfigs.map((config) => (
              <Card key={config.id} size="small" hoverable onClick={() => handleLoadConfig(config)} style={{ cursor: 'pointer' }}>
                <Space direction="vertical" size={0}>
                  <Text strong>{config.name}</Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {t('augment.instruction')}: {config.config.instruction.substring(0, 50)}
                    {config.config.instruction.length > 50 ? '...' : ''}
                  </Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>{new Date(config.createdAt).toLocaleString()}</Text>
                </Space>
              </Card>
            ))
          )}
        </Space>
      </Modal>
    </div>
  );
};

export default AugmentPage;
