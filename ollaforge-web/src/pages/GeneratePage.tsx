/**
 * GeneratePage component for dataset generation workflow.
 *
 * Provides the complete generation workflow:
 * - Form for generation parameters
 * - Real-time progress display via WebSocket
 * - Dataset preview after completion
 * - Download functionality with format selection
 * - Configuration save option
 *
 * Requirements: 1.1, 1.2, 1.3, 3.1, 5.1
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Typography,
  Card,
  Space,
  Button,
  Select,
  message,
  Modal,
  Input,
  Alert,
  Divider,
} from 'antd';
import {
  DownloadOutlined,
  SaveOutlined,
  ReloadOutlined,
  FolderOpenOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { GenerationForm } from '../components/GenerationForm';
import { ProgressDisplay, type ProgressStatus } from '../components/ProgressDisplay';
import { DatasetPreview } from '../components/DatasetPreview';
import {
  generationAPI,
  downloadBlob,
  type GenerationConfig,
  type TaskStatus,
} from '../services/api';
import {
  wsClient,
  type ProgressEvent,
  type CompletedEvent,
  type ErrorEvent,
  type FailedEvent,
} from '../services/websocket';

const { Text } = Typography;
const { Option } = Select;

/** Supported download formats */
const DOWNLOAD_FORMATS = ['jsonl', 'json', 'csv', 'tsv', 'parquet'];

/** Local storage key for saved configurations */
const SAVED_CONFIGS_KEY = 'ollaforge_generation_configs';

interface SavedConfig {
  id: string;
  name: string;
  type: 'generation';
  config: GenerationConfig;
  createdAt: string;
}

type PageState = 'idle' | 'generating' | 'completed' | 'failed';

const GeneratePage: React.FC = () => {
  const { t } = useTranslation();

  // Page state
  const [pageState, setPageState] = useState<PageState>('idle');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [currentConfig, setCurrentConfig] = useState<GenerationConfig | null>(null);

  // Progress state
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0);
  const [progressStatus, setProgressStatus] = useState<ProgressStatus>('pending');
  const [progressMessage, setProgressMessage] = useState<string>('');
  const [duration, setDuration] = useState<number | undefined>(undefined);
  const [successCount, setSuccessCount] = useState<number | undefined>(undefined);
  const [failureCount, setFailureCount] = useState<number | undefined>(undefined);

  // Result state
  const [previewEntries, setPreviewEntries] = useState<Record<string, unknown>[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Download state
  const [downloadFormat, setDownloadFormat] = useState('jsonl');
  const [downloading, setDownloading] = useState(false);

  // Save config modal state
  const [saveModalVisible, setSaveModalVisible] = useState(false);
  const [configName, setConfigName] = useState('');

  // Load config state
  const [savedConfigs, setSavedConfigs] = useState<SavedConfig[]>([]);
  const [loadModalVisible, setLoadModalVisible] = useState(false);

  // Load saved configurations from localStorage
  useEffect(() => {
    const loadSavedConfigs = () => {
      const configsStr = localStorage.getItem(SAVED_CONFIGS_KEY);
      if (configsStr) {
        try {
          const configs = JSON.parse(configsStr) as SavedConfig[];
          setSavedConfigs(configs);
        } catch {
          setSavedConfigs([]);
        }
      }
    };
    loadSavedConfigs();
  }, []);

  // Connect to WebSocket on mount
  useEffect(() => {
    wsClient.connect();
    return () => {
      // Don't disconnect on unmount as other pages might use it
    };
  }, []);

  // Handle progress events
  const handleProgress = useCallback((data: ProgressEvent) => {
    setProgress(data.progress);
    setTotal(data.total);
    setProgressStatus('active');
    if (data.message) {
      setProgressMessage(data.message);
    }
  }, []);

  // Handle completion events
  const handleCompleted = useCallback((data: CompletedEvent) => {
    setProgress(data.total);
    setTotal(data.total);
    setProgressStatus('success');
    setDuration(data.duration);
    setSuccessCount(data.success_count);
    setFailureCount(data.failure_count);
    setPageState('completed');
    if (data.message) {
      setProgressMessage(data.message);
    }
    message.success(t('generate.generationComplete'));
  }, [t]);

  // Handle error events
  const handleError = useCallback((data: ErrorEvent) => {
    // Don't stop progress for item errors, just show warning
    if (data.error_type === 'item_error') {
      message.warning(data.error);
    } else {
      setProgressMessage(data.error);
    }
  }, []);

  // Handle failed events
  const handleFailed = useCallback((data: FailedEvent) => {
    setProgressStatus('exception');
    setErrorMessage(data.error);
    setPageState('failed');
    message.error(t('errors.generationFailed'));
  }, [t]);

  // Subscribe to task updates
  useEffect(() => {
    if (!taskId) return;

    const unsubscribe = wsClient.subscribeToTask(taskId, {
      onProgress: handleProgress,
      onCompleted: handleCompleted,
      onError: handleError,
      onFailed: handleFailed,
    });

    return () => {
      unsubscribe();
    };
  }, [taskId, handleProgress, handleCompleted, handleError, handleFailed]);

  // Fetch task status and preview when completed
  useEffect(() => {
    if (pageState !== 'completed' || !taskId) return;

    const fetchResult = async () => {
      try {
        const status: TaskStatus = await generationAPI.getStatus(taskId);
        if (status.result && Array.isArray(status.result.entries)) {
          setPreviewEntries(status.result.entries.slice(0, 5));
        }
      } catch (error) {
        console.error('Failed to fetch result:', error);
      }
    };

    fetchResult();
  }, [pageState, taskId]);

  // Handle form submission
  const handleSubmit = async (config: GenerationConfig) => {
    setCurrentConfig(config);
    setPageState('generating');
    setProgress(0);
    setTotal(config.count);
    setProgressStatus('pending');
    setProgressMessage('');
    setErrorMessage(null);
    setPreviewEntries([]);
    setDuration(undefined);
    setSuccessCount(undefined);
    setFailureCount(undefined);

    try {
      const response = await generationAPI.startGeneration(config);
      setTaskId(response.task_id);
      setProgressStatus('active');
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      setErrorMessage(errorMsg);
      setProgressStatus('exception');
      setPageState('failed');
      message.error(t('errors.generationFailed'));
    }
  };

  // Handle download
  const handleDownload = async () => {
    if (!taskId) return;

    setDownloading(true);
    try {
      const blob = await generationAPI.downloadDataset(taskId, downloadFormat);
      const filename = `dataset_${taskId}.${downloadFormat}`;
      downloadBlob(blob, filename);
      message.success(t('common.success'));
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Download failed';
      message.error(errorMsg);
    } finally {
      setDownloading(false);
    }
  };

  // Handle save configuration
  const handleSaveConfig = () => {
    if (!currentConfig || !configName.trim()) return;

    const savedConfig: SavedConfig = {
      id: Date.now().toString(),
      name: configName.trim(),
      type: 'generation',
      config: currentConfig,
      createdAt: new Date().toISOString(),
    };

    // Get existing configs
    const existingConfigsStr = localStorage.getItem(SAVED_CONFIGS_KEY);
    const existingConfigs: SavedConfig[] = existingConfigsStr
      ? JSON.parse(existingConfigsStr)
      : [];

    // Add new config
    existingConfigs.push(savedConfig);
    localStorage.setItem(SAVED_CONFIGS_KEY, JSON.stringify(existingConfigs));

    message.success(t('common.success'));
    setSaveModalVisible(false);
    setConfigName('');

    // Refresh saved configs list
    setSavedConfigs([...existingConfigs]);
  };

  // Handle load configuration
  const handleLoadConfig = (config: SavedConfig) => {
    setCurrentConfig(config.config);
    setLoadModalVisible(false);
    message.success(t('config.loadSuccess', 'Configuration loaded'));
  };

  // Handle reset
  const handleReset = () => {
    setPageState('idle');
    setTaskId(null);
    setCurrentConfig(null);
    setProgress(0);
    setTotal(0);
    setProgressStatus('pending');
    setProgressMessage('');
    setErrorMessage(null);
    setPreviewEntries([]);
    setDuration(undefined);
    setSuccessCount(undefined);
    setFailureCount(undefined);
  };

  return (
    <div className="generate-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Generation Form */}
        <Card
          title={t('generate.title')}
          extra={
            savedConfigs.length > 0 && pageState === 'idle' && (
              <Button
                icon={<FolderOpenOutlined />}
                onClick={() => setLoadModalVisible(true)}
              >
                {t('config.load')}
              </Button>
            )
          }
        >
          <GenerationForm
            key={currentConfig ? JSON.stringify(currentConfig) : 'default'}
            onSubmit={handleSubmit}
            initialValues={currentConfig || undefined}
            loading={pageState === 'generating'}
            disabled={pageState === 'generating'}
          />
        </Card>

        {/* Progress Display */}
        {pageState !== 'idle' && (
          <Card title={t('common.progress')}>
            <ProgressDisplay
              current={progress}
              total={total}
              status={progressStatus}
              message={progressMessage}
              duration={duration}
              successCount={successCount}
              failureCount={failureCount}
              showStats={pageState === 'completed' || pageState === 'failed'}
            />

            {/* Error Message */}
            {errorMessage && (
              <Alert
                message={t('common.error')}
                description={errorMessage}
                type="error"
                showIcon
                style={{ marginTop: 16 }}
                action={
                  <Button size="small" onClick={handleReset}>
                    {t('common.retry')}
                  </Button>
                }
              />
            )}
          </Card>
        )}

        {/* Results Section */}
        {pageState === 'completed' && (
          <>
            {/* Preview */}
            {previewEntries.length > 0 && (
              <Card title={t('generate.previewTitle')}>
                <DatasetPreview
                  entries={previewEntries}
                  maxEntries={5}
                />
              </Card>
            )}

            {/* Download Section */}
            <Card title={t('generate.downloadReady')}>
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Text>
                  {t('generate.entriesGenerated', { count: successCount || total })}
                </Text>

                <Space>
                  <Select
                    value={downloadFormat}
                    onChange={setDownloadFormat}
                    style={{ width: 120 }}
                  >
                    {DOWNLOAD_FORMATS.map((format) => (
                      <Option key={format} value={format}>
                        {format.toUpperCase()}
                      </Option>
                    ))}
                  </Select>

                  <Button
                    type="primary"
                    icon={<DownloadOutlined />}
                    onClick={handleDownload}
                    loading={downloading}
                  >
                    {t('generate.download')}
                  </Button>
                </Space>

                <Divider />

                {/* Save Configuration Prompt */}
                <Space>
                  <Text>{t('generate.saveConfigPrompt')}</Text>
                  <Button
                    icon={<SaveOutlined />}
                    onClick={() => setSaveModalVisible(true)}
                  >
                    {t('common.saveConfig')}
                  </Button>
                </Space>

                <Divider />

                {/* Start New Generation */}
                <Button
                  icon={<ReloadOutlined />}
                  onClick={handleReset}
                >
                  {t('generate.submit')}
                </Button>
              </Space>
            </Card>
          </>
        )}
      </Space>

      {/* Save Configuration Modal */}
      <Modal
        title={t('common.saveConfig')}
        open={saveModalVisible}
        onOk={handleSaveConfig}
        onCancel={() => {
          setSaveModalVisible(false);
          setConfigName('');
        }}
        okButtonProps={{ disabled: !configName.trim() }}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Text>{t('common.configName')}</Text>
          <Input
            placeholder={t('common.configNamePlaceholder')}
            value={configName}
            onChange={(e) => setConfigName(e.target.value)}
            onPressEnter={handleSaveConfig}
          />
        </Space>
      </Modal>

      {/* Load Configuration Modal */}
      <Modal
        title={t('config.loadConfig', 'Load Configuration')}
        open={loadModalVisible}
        onCancel={() => setLoadModalVisible(false)}
        footer={null}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          {savedConfigs.length === 0 ? (
            <Text type="secondary">{t('config.noConfigs')}</Text>
          ) : (
            savedConfigs.map((config) => (
              <Card
                key={config.id}
                size="small"
                hoverable
                onClick={() => handleLoadConfig(config)}
                style={{ cursor: 'pointer' }}
              >
                <Space direction="vertical" size={0}>
                  <Text strong>{config.name}</Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {t('generate.topic')}: {config.config.topic.substring(0, 50)}
                    {config.config.topic.length > 50 ? '...' : ''}
                  </Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {new Date(config.createdAt).toLocaleString()}
                  </Text>
                </Space>
              </Card>
            ))
          )}
        </Space>
      </Modal>
    </div>
  );
};

export default GeneratePage;
