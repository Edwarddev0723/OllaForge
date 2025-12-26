/**
 * ConfigPage component for managing saved configurations.
 *
 * Provides functionality to:
 * - Display list of saved configurations (both generation and augmentation)
 * - Show configuration details (name, type, timestamp)
 * - Load configuration (navigate to appropriate page)
 * - Delete configuration
 *
 * Requirements: 6.3, 6.4, 6.5
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Typography,
  Card,
  Table,
  Button,
  Space,
  Tag,
  Empty,
  Modal,
  message,
  Popconfirm,
  Tooltip,
} from 'antd';
import {
  DeleteOutlined,
  PlayCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;

/** Local storage keys for saved configurations */
const GENERATION_CONFIGS_KEY = 'ollaforge_generation_configs';
const AUGMENTATION_CONFIGS_KEY = 'ollaforge_augmentation_configs';

/** Base interface for saved configurations */
interface BaseSavedConfig {
  id: string;
  name: string;
  type: 'generation' | 'augmentation';
  createdAt: string;
}

/** Generation configuration */
interface GenerationSavedConfig extends BaseSavedConfig {
  type: 'generation';
  config: {
    topic: string;
    count: number;
    model: string;
    dataset_type: string;
    language: string;
    qc_enabled: boolean;
    qc_confidence: number;
  };
}

/** Augmentation configuration */
interface AugmentationSavedConfig extends BaseSavedConfig {
  type: 'augmentation';
  config: {
    target_field: string;
    instruction: string;
    model: string;
    language: string;
    create_new_field: boolean;
    context_fields: string[];
    concurrency: number;
  };
}

type SavedConfig = GenerationSavedConfig | AugmentationSavedConfig;

/** Table row data structure */
interface ConfigTableRow {
  key: string;
  id: string;
  name: string;
  type: 'generation' | 'augmentation';
  createdAt: string;
  config: SavedConfig;
  summary: string;
}

const ConfigPage: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();

  const [configs, setConfigs] = useState<ConfigTableRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState<SavedConfig | null>(null);

  /** Load all saved configurations from localStorage */
  const loadConfigs = useCallback(() => {
    setLoading(true);
    try {
      const allConfigs: ConfigTableRow[] = [];

      // Load generation configs
      const generationStr = localStorage.getItem(GENERATION_CONFIGS_KEY);
      if (generationStr) {
        const generationConfigs = JSON.parse(generationStr) as GenerationSavedConfig[];
        generationConfigs.forEach((config) => {
          allConfigs.push({
            key: config.id,
            id: config.id,
            name: config.name,
            type: 'generation',
            createdAt: config.createdAt,
            config: config,
            summary: config.config.topic?.substring(0, 50) + (config.config.topic?.length > 50 ? '...' : '') || '',
          });
        });
      }

      // Load augmentation configs
      const augmentationStr = localStorage.getItem(AUGMENTATION_CONFIGS_KEY);
      if (augmentationStr) {
        const augmentationConfigs = JSON.parse(augmentationStr) as AugmentationSavedConfig[];
        augmentationConfigs.forEach((config) => {
          allConfigs.push({
            key: config.id,
            id: config.id,
            name: config.name,
            type: 'augmentation',
            createdAt: config.createdAt,
            config: config,
            summary: config.config.instruction?.substring(0, 50) + (config.config.instruction?.length > 50 ? '...' : '') || '',
          });
        });
      }

      // Sort by creation date (newest first)
      allConfigs.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

      setConfigs(allConfigs);
    } catch (error) {
      console.error('Failed to load configurations:', error);
      message.error(t('common.error'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  // Load configs on mount
  useEffect(() => {
    loadConfigs();
  }, [loadConfigs]);

  /** Handle loading a configuration */
  const handleLoad = (record: ConfigTableRow) => {
    // Navigate to the appropriate page
    // The page will need to read from localStorage to get the config
    if (record.type === 'generation') {
      navigate('/', { state: { loadConfigId: record.id } });
    } else {
      navigate('/augment', { state: { loadConfigId: record.id } });
    }
    message.success(t('config.loadSuccess'));
  };

  /** Handle deleting a configuration */
  const handleDelete = (record: ConfigTableRow) => {
    try {
      const storageKey = record.type === 'generation' 
        ? GENERATION_CONFIGS_KEY 
        : AUGMENTATION_CONFIGS_KEY;

      const configsStr = localStorage.getItem(storageKey);
      if (configsStr) {
        const existingConfigs = JSON.parse(configsStr) as SavedConfig[];
        const updatedConfigs = existingConfigs.filter((c) => c.id !== record.id);
        localStorage.setItem(storageKey, JSON.stringify(updatedConfigs));
      }

      // Refresh the list
      loadConfigs();
      message.success(t('common.success'));
    } catch (error) {
      console.error('Failed to delete configuration:', error);
      message.error(t('common.error'));
    }
  };

  /** Show configuration details */
  const handleShowDetails = (record: ConfigTableRow) => {
    setSelectedConfig(record.config);
    setDetailModalVisible(true);
  };

  /** Format date for display */
  const formatDate = (dateStr: string) => {
    try {
      return new Date(dateStr).toLocaleString();
    } catch {
      return dateStr;
    }
  };

  /** Render configuration details in modal */
  const renderConfigDetails = () => {
    if (!selectedConfig) return null;

    if (selectedConfig.type === 'generation') {
      const config = selectedConfig.config;
      return (
        <Space direction="vertical" style={{ width: '100%' }}>
          <div><Text strong>{t('generate.topic')}:</Text> <Text>{config.topic}</Text></div>
          <div><Text strong>{t('generate.count')}:</Text> <Text>{config.count}</Text></div>
          <div><Text strong>{t('generate.model')}:</Text> <Text>{config.model}</Text></div>
          <div><Text strong>{t('generate.datasetType')}:</Text> <Text>{config.dataset_type}</Text></div>
          <div><Text strong>{t('generate.language')}:</Text> <Text>{config.language}</Text></div>
          <div><Text strong>{t('generate.qcEnabled')}:</Text> <Text>{config.qc_enabled ? 'Yes' : 'No'}</Text></div>
          {config.qc_enabled && (
            <div><Text strong>{t('generate.qcConfidence')}:</Text> <Text>{config.qc_confidence}</Text></div>
          )}
        </Space>
      );
    } else {
      const config = selectedConfig.config;
      return (
        <Space direction="vertical" style={{ width: '100%' }}>
          <div><Text strong>{t('augment.targetField')}:</Text> <Text>{config.target_field}</Text></div>
          <div><Text strong>{t('augment.instruction')}:</Text> <Text>{config.instruction}</Text></div>
          <div><Text strong>{t('generate.model')}:</Text> <Text>{config.model}</Text></div>
          <div><Text strong>{t('generate.language')}:</Text> <Text>{config.language}</Text></div>
          <div><Text strong>{t('augment.createNewField')}:</Text> <Text>{config.create_new_field ? 'Yes' : 'No'}</Text></div>
          {config.context_fields && config.context_fields.length > 0 && (
            <div><Text strong>{t('augment.contextFields')}:</Text> <Text>{config.context_fields.join(', ')}</Text></div>
          )}
          <div><Text strong>{t('augment.concurrency')}:</Text> <Text>{config.concurrency}</Text></div>
        </Space>
      );
    }
  };

  /** Table columns definition */
  const columns: ColumnsType<ConfigTableRow> = [
    {
      title: t('config.name'),
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ConfigTableRow) => (
        <Tooltip title={record.summary}>
          <Button type="link" onClick={() => handleShowDetails(record)} style={{ padding: 0 }}>
            {text}
          </Button>
        </Tooltip>
      ),
    },
    {
      title: t('config.type'),
      dataIndex: 'type',
      key: 'type',
      width: 120,
      render: (type: 'generation' | 'augmentation') => (
        <Tag color={type === 'generation' ? 'blue' : 'green'}>
          {t(`config.${type}`)}
        </Tag>
      ),
      filters: [
        { text: t('config.generation'), value: 'generation' },
        { text: t('config.augmentation'), value: 'augmentation' },
      ],
      onFilter: (value, record) => record.type === value,
    },
    {
      title: t('config.createdAt'),
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 180,
      render: (date: string) => formatDate(date),
      sorter: (a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
      defaultSortOrder: 'descend',
    },
    {
      title: t('config.actions'),
      key: 'actions',
      width: 150,
      render: (_: unknown, record: ConfigTableRow) => (
        <Space>
          <Tooltip title={t('config.load')}>
            <Button
              type="primary"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleLoad(record)}
            >
              {t('config.load')}
            </Button>
          </Tooltip>
          <Popconfirm
            title={t('config.delete')}
            description={`${t('config.delete')} "${record.name}"?`}
            onConfirm={() => handleDelete(record)}
            okText={t('common.save')}
            cancelText={t('common.cancel')}
            icon={<ExclamationCircleOutlined style={{ color: 'red' }} />}
          >
            <Tooltip title={t('config.delete')}>
              <Button
                danger
                size="small"
                icon={<DeleteOutlined />}
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div className="config-page">
      <Card>
        <Title level={2}>{t('config.title')}</Title>
        
        {configs.length === 0 && !loading ? (
          <Empty
            description={t('config.noConfigs')}
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <Table
            columns={columns}
            dataSource={configs}
            loading={loading}
            pagination={{
              pageSize: 10,
              showSizeChanger: true,
              showTotal: (total) => `${total} ${t('config.title').toLowerCase()}`,
            }}
          />
        )}
      </Card>

      {/* Configuration Details Modal */}
      <Modal
        title={selectedConfig?.name || t('config.title')}
        open={detailModalVisible}
        onCancel={() => {
          setDetailModalVisible(false);
          setSelectedConfig(null);
        }}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            {t('common.cancel')}
          </Button>,
          <Button
            key="load"
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={() => {
              if (selectedConfig) {
                const record = configs.find((c) => c.id === selectedConfig.id);
                if (record) {
                  handleLoad(record);
                  setDetailModalVisible(false);
                }
              }
            }}
          >
            {t('config.load')}
          </Button>,
        ]}
      >
        {selectedConfig && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Tag color={selectedConfig.type === 'generation' ? 'blue' : 'green'}>
                {t(`config.${selectedConfig.type}`)}
              </Tag>
              <Text type="secondary" style={{ marginLeft: 8 }}>
                {formatDate(selectedConfig.createdAt)}
              </Text>
            </div>
            <div style={{ marginTop: 16 }}>
              {renderConfigDetails()}
            </div>
          </Space>
        )}
      </Modal>
    </div>
  );
};

export default ConfigPage;
