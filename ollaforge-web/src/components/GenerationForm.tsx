/**
 * GenerationForm component for dataset generation.
 *
 * Provides a form with all generation parameters including:
 * - Topic input
 * - Entry count
 * - Model selection dropdown
 * - Dataset type selection
 * - Language selection
 * - QC options (for Traditional Chinese)
 *
 * Requirements: 1.1, 1.5, 10.1
 */

import React, { useEffect, useState } from 'react';
import {
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  Slider,
  Button,
  Space,
  Alert,
  Spin,
} from 'antd';
import { useTranslation } from 'react-i18next';
import {
  modelsAPI,
  DatasetType,
  OutputLanguage,
  type GenerationConfig,
  type ModelInfo,
} from '../services/api';

const { TextArea } = Input;
const { Option } = Select;

export interface GenerationFormProps {
  /** Callback when form is submitted */
  onSubmit: (config: GenerationConfig) => void;
  /** Initial form values */
  initialValues?: Partial<GenerationConfig>;
  /** Whether the form is in loading/submitting state */
  loading?: boolean;
  /** Whether to disable the form */
  disabled?: boolean;
}

/** Default form values */
const DEFAULT_VALUES: GenerationConfig = {
  topic: '',
  count: 10,
  model: 'llama3.2',
  dataset_type: DatasetType.SFT,
  language: OutputLanguage.EN,
  qc_enabled: true,
  qc_confidence: 0.9,
};

/**
 * GenerationForm component for configuring dataset generation.
 */
export const GenerationForm: React.FC<GenerationFormProps> = ({
  onSubmit,
  initialValues,
  loading = false,
  disabled = false,
}) => {
  const { t } = useTranslation();
  const [form] = Form.useForm<GenerationConfig>();

  // State for models
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // Watch language field for QC options visibility
  const language = Form.useWatch('language', form);
  const showQcOptions = language === OutputLanguage.ZH_TW;

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      setModelsLoading(true);
      setModelsError(null);
      try {
        const response = await modelsAPI.listModels();
        setModels(response.models || []);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Failed to fetch models';
        setModelsError(errorMessage);
        // Set default model if fetch fails
        setModels([{ name: 'llama3.2' }]);
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
  }, []);

  // Handle form submission
  const handleFinish = (values: GenerationConfig) => {
    // If QC is not shown (non-Chinese), disable it
    if (!showQcOptions) {
      values.qc_enabled = false;
    }
    onSubmit(values);
  };

  // Merge initial values with defaults
  const mergedInitialValues = {
    ...DEFAULT_VALUES,
    ...initialValues,
  };

  return (
    <Form
      form={form}
      layout="vertical"
      initialValues={mergedInitialValues}
      onFinish={handleFinish}
      disabled={disabled || loading}
    >
      {/* Topic */}
      <Form.Item
        name="topic"
        label={t('generate.topic')}
        rules={[
          { required: true, message: t('generate.topicRequired', 'Topic is required') },
          { min: 3, message: t('generate.topicMinLength', 'Topic must be at least 3 characters') },
        ]}
      >
        <TextArea
          placeholder={t('generate.topicPlaceholder')}
          rows={3}
          maxLength={1000}
          showCount
        />
      </Form.Item>

      {/* Count */}
      <Form.Item
        name="count"
        label={t('generate.count')}
        rules={[
          { required: true, message: t('generate.countRequired', 'Count is required') },
        ]}
      >
        <InputNumber
          min={1}
          max={10000}
          style={{ width: '100%' }}
        />
      </Form.Item>

      {/* Model Selection */}
      <Form.Item
        name="model"
        label={t('generate.model')}
        rules={[
          { required: true, message: t('generate.modelRequired', 'Model is required') },
        ]}
      >
        <Select
          placeholder={t('generate.modelPlaceholder')}
          loading={modelsLoading}
          notFoundContent={modelsLoading ? <Spin size="small" /> : null}
        >
          {models.map((model) => (
            <Option key={model.name} value={model.name}>
              {model.name}
              {model.size && <span style={{ color: '#888', marginLeft: 8 }}>({model.size})</span>}
            </Option>
          ))}
        </Select>
      </Form.Item>

      {modelsError && (
        <Alert
          message={t('errors.ollamaUnavailable')}
          description={modelsError}
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Dataset Type */}
      <Form.Item
        name="dataset_type"
        label={t('generate.datasetType')}
        rules={[{ required: true }]}
      >
        <Select>
          <Option value={DatasetType.SFT}>SFT (Supervised Fine-tuning)</Option>
          <Option value={DatasetType.PRETRAIN}>Pretrain</Option>
          <Option value={DatasetType.SFT_CONV}>SFT Conversation</Option>
          <Option value={DatasetType.DPO}>DPO (Direct Preference Optimization)</Option>
        </Select>
      </Form.Item>

      {/* Language */}
      <Form.Item
        name="language"
        label={t('generate.language')}
        rules={[{ required: true }]}
      >
        <Select>
          <Option value={OutputLanguage.EN}>{t('language.en')}</Option>
          <Option value={OutputLanguage.ZH_TW}>{t('language.zh-tw')}</Option>
        </Select>
      </Form.Item>

      {/* QC Options - Only shown for Traditional Chinese */}
      {showQcOptions && (
        <>
          <Form.Item
            name="qc_enabled"
            label={t('generate.qcEnabled')}
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            name="qc_confidence"
            label={t('generate.qcConfidence')}
            tooltip={t('generate.qcConfidenceTooltip', 'Minimum confidence threshold for quality control (0.0 - 1.0)')}
          >
            <Slider
              min={0}
              max={1}
              step={0.05}
              marks={{
                0: '0',
                0.5: '0.5',
                1: '1',
              }}
            />
          </Form.Item>
        </>
      )}

      {/* Submit Button */}
      <Form.Item>
        <Space>
          <Button
            type="primary"
            htmlType="submit"
            loading={loading}
            disabled={disabled}
          >
            {loading ? t('generate.generating') : t('generate.submit')}
          </Button>
        </Space>
      </Form.Item>
    </Form>
  );
};

export default GenerationForm;
