/**
 * AugmentationForm component for dataset augmentation.
 *
 * Provides a form with all augmentation parameters including:
 * - Target field selection from uploaded file
 * - Augmentation instruction input
 * - Option to create new field
 * - Context fields selection
 * - Model selection
 * - Language selection
 * - Concurrency setting
 *
 * Requirements: 2.2, 2.3
 */

import React, { useEffect, useState } from 'react';
import {
  Form,
  Input,
  Select,
  Switch,
  InputNumber,
  Button,
  Space,
  Alert,
  Tooltip,
  AutoComplete,
} from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import {
  modelsAPI,
  OutputLanguage,
  type AugmentationConfig,
  type ModelInfo,
} from '../services/api';

const { TextArea } = Input;

export interface AugmentationFormValues {
  target_field: string;
  instruction: string;
  model: string;
  language: OutputLanguage;
  create_new_field: boolean;
  new_field_name?: string;
  context_fields: string[];
  concurrency: number;
}

export interface AugmentationFormProps {
  /** Available fields from uploaded file */
  availableFields: string[];
  /** File ID from upload response */
  fileId: string;
  /** Callback when form is submitted */
  onSubmit: (config: AugmentationConfig) => void;
  /** Callback when preview is requested */
  onPreview: (config: AugmentationConfig) => void;
  /** Initial form values */
  initialValues?: Partial<AugmentationFormValues>;
  /** Whether the form is in loading/submitting state */
  loading?: boolean;
  /** Whether preview is loading */
  previewLoading?: boolean;
  /** Whether to disable the form */
  disabled?: boolean;
}

/** Default form values */
const DEFAULT_VALUES: AugmentationFormValues = {
  target_field: '',
  instruction: '',
  model: 'llama3.2',
  language: OutputLanguage.EN,
  create_new_field: false,
  new_field_name: '',
  context_fields: [],
  concurrency: 5,
};

/** Common/recommended models as fallback when Ollama is unavailable */
const FALLBACK_MODELS: ModelInfo[] = [
  { name: 'llama3.2', size: '3B' },
  { name: 'llama3.2:1b', size: '1B' },
  { name: 'llama3.1', size: '8B' },
  { name: 'llama3.1:70b', size: '70B' },
  { name: 'llama3.3', size: '70B' },
  { name: 'qwen2.5', size: '7B' },
  { name: 'qwen2.5:14b', size: '14B' },
  { name: 'qwen2.5:32b', size: '32B' },
  { name: 'qwen2.5:72b', size: '72B' },
  { name: 'qwen2.5-coder', size: '7B' },
  { name: 'deepseek-r1:7b', size: '7B' },
  { name: 'deepseek-r1:14b', size: '14B' },
  { name: 'deepseek-r1:32b', size: '32B' },
  { name: 'deepseek-r1:70b', size: '70B' },
  { name: 'gemma2', size: '9B' },
  { name: 'gemma2:27b', size: '27B' },
  { name: 'mistral', size: '7B' },
  { name: 'mixtral', size: '8x7B' },
  { name: 'phi3', size: '3.8B' },
  { name: 'phi3:14b', size: '14B' },
  { name: 'codellama', size: '7B' },
  { name: 'codellama:34b', size: '34B' },
];

/**
 * AugmentationForm component for configuring dataset augmentation.
 */
export const AugmentationForm: React.FC<AugmentationFormProps> = ({
  availableFields,
  fileId,
  onSubmit,
  onPreview,
  initialValues,
  loading = false,
  previewLoading = false,
  disabled = false,
}) => {
  const { t } = useTranslation();
  const [form] = Form.useForm<AugmentationFormValues>();

  // State for models
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // Watch create_new_field for conditional rendering
  const createNewField = Form.useWatch('create_new_field', form);
  const targetField = Form.useWatch('target_field', form);

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      setModelsError(null);
      try {
        const response = await modelsAPI.listModels();
        if (response.models && response.models.length > 0) {
          setModels(response.models);
        } else {
          // No models from Ollama, use fallback list
          setModels(FALLBACK_MODELS);
          setModelsError('No models found in Ollama. Showing common models - make sure to install them first.');
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Failed to fetch models';
        setModelsError(errorMessage);
        // Use fallback models when Ollama is unavailable
        setModels(FALLBACK_MODELS);
      }
    };

    fetchModels();
  }, []);

  /**
   * Build AugmentationConfig from form values.
   */
  const buildConfig = (values: AugmentationFormValues): AugmentationConfig => {
    return {
      file_id: fileId,
      target_field: values.create_new_field && values.new_field_name
        ? values.new_field_name
        : values.target_field,
      instruction: values.instruction,
      model: values.model,
      language: values.language,
      create_new_field: values.create_new_field,
      context_fields: values.context_fields,
      concurrency: values.concurrency,
    };
  };

  /**
   * Handle form submission.
   */
  const handleFinish = (values: AugmentationFormValues) => {
    const config = buildConfig(values);
    onSubmit(config);
  };

  /**
   * Handle preview request.
   */
  const handlePreview = async () => {
    try {
      const values = await form.validateFields();
      const config = buildConfig(values);
      onPreview(config);
    } catch {
      // Validation failed, form will show errors
    }
  };

  // Merge initial values with defaults
  const mergedInitialValues = {
    ...DEFAULT_VALUES,
    ...initialValues,
  };

  // Filter context fields to exclude target field
  const contextFieldOptions = availableFields.filter(
    (field) => field !== targetField
  );

  return (
    <Form
      form={form}
      layout="vertical"
      initialValues={mergedInitialValues}
      onFinish={handleFinish}
      disabled={disabled || loading}
    >
      {/* Target Field */}
      <Form.Item
        name="target_field"
        label={t('augment.targetField')}
        rules={[
          { required: !createNewField, message: t('augment.targetFieldRequired') },
        ]}
      >
        <Select
          placeholder={t('augment.targetFieldPlaceholder')}
          allowClear
          showSearch
          disabled={createNewField}
          options={availableFields.map((field) => ({ value: field, label: field }))}
        />
      </Form.Item>

      {/* Create New Field Toggle */}
      <Form.Item
        name="create_new_field"
        label={t('augment.createNewField')}
        valuePropName="checked"
      >
        <Switch />
      </Form.Item>

      {/* New Field Name (shown when create_new_field is true) */}
      {createNewField && (
        <Form.Item
          name="new_field_name"
          label={t('augment.newFieldName')}
          rules={[
            { required: true, message: t('augment.newFieldNameRequired') },
            {
              pattern: /^[a-zA-Z_][a-zA-Z0-9_]*$/,
              message: t('augment.invalidFieldName', 'Field name must start with a letter or underscore and contain only alphanumeric characters and underscores'),
            },
          ]}
        >
          <Input placeholder={t('augment.newFieldNamePlaceholder')} />
        </Form.Item>
      )}

      {/* Instruction */}
      <Form.Item
        name="instruction"
        label={t('augment.instruction')}
        rules={[
          { required: true, message: t('augment.instructionRequired') },
          { min: 5, message: t('augment.instructionMinLength', 'Instruction must be at least 5 characters') },
        ]}
      >
        <TextArea
          placeholder={t('augment.instructionPlaceholder')}
          rows={4}
          maxLength={2000}
          showCount
        />
      </Form.Item>

      {/* Context Fields */}
      <Form.Item
        name="context_fields"
        label={t('augment.contextFields')}
      >
        <Select
          mode="multiple"
          placeholder={t('augment.contextFieldsPlaceholder')}
          allowClear
          options={contextFieldOptions.map((field) => ({ value: field, label: field }))}
        />
      </Form.Item>

      {/* Model Selection */}
      <Form.Item
        name="model"
        label={t('generate.model')}
        rules={[
          { required: true, message: t('generate.modelRequired') },
        ]}
        tooltip={t('generate.modelTooltip', 'Select a model or type a custom model name')}
      >
        <AutoComplete
          placeholder={t('generate.modelPlaceholder')}
          options={models.map((model) => ({
            value: model.name,
            label: (
              <span>
                {model.name}
                {model.size && <span style={{ color: '#888', marginLeft: 8 }}>({model.size})</span>}
              </span>
            ),
          }))}
          filterOption={(inputValue, option) =>
            option?.value.toLowerCase().includes(inputValue.toLowerCase()) ?? false
          }
          allowClear
        />
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

      {/* Language */}
      <Form.Item
        name="language"
        label={t('generate.language')}
        rules={[{ required: true }]}
      >
        <Select
          options={[
            { value: OutputLanguage.EN, label: t('language.en') },
            { value: OutputLanguage.ZH_TW, label: t('language.zh-tw') },
            { value: OutputLanguage.ZH_CN, label: t('language.zh-cn') },
          ]}
        />
      </Form.Item>

      {/* Concurrency */}
      <Form.Item
        name="concurrency"
        label={
          <Space>
            {t('augment.concurrency')}
            <Tooltip title={t('augment.concurrencyTooltip')}>
              <QuestionCircleOutlined />
            </Tooltip>
          </Space>
        }
        rules={[{ required: true }]}
      >
        <InputNumber min={1} max={10} style={{ width: '100%' }} />
      </Form.Item>

      {/* Action Buttons */}
      <Form.Item>
        <Space>
          <Button
            onClick={handlePreview}
            loading={previewLoading}
            disabled={disabled || loading}
          >
            {previewLoading ? t('augment.previewLoading') : t('augment.preview')}
          </Button>
          <Button
            type="primary"
            htmlType="submit"
            loading={loading}
            disabled={disabled || previewLoading}
          >
            {loading ? t('augment.augmenting') : t('augment.submit')}
          </Button>
        </Space>
      </Form.Item>
    </Form>
  );
};

export default AugmentationForm;
