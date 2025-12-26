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
  Spin,
  Tooltip,
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
const { Option } = Select;

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
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // Watch create_new_field for conditional rendering
  const createNewField = Form.useWatch('create_new_field', form);
  const targetField = Form.useWatch('target_field', form);

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
        >
          {availableFields.map((field) => (
            <Option key={field} value={field}>
              {field}
            </Option>
          ))}
        </Select>
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
        >
          {contextFieldOptions.map((field) => (
            <Option key={field} value={field}>
              {field}
            </Option>
          ))}
        </Select>
      </Form.Item>

      {/* Model Selection */}
      <Form.Item
        name="model"
        label={t('generate.model')}
        rules={[
          { required: true, message: t('generate.modelRequired') },
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
