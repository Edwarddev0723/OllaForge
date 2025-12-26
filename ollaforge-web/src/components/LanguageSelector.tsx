import React from 'react';
import { Select } from 'antd';
import { GlobalOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Option } = Select;

interface LanguageOption {
  code: string;
  label: string;
}

const languages: LanguageOption[] = [
  { code: 'en', label: 'English' },
  { code: 'zh-TW', label: '繁體中文' },
];

const LanguageSelector: React.FC = () => {
  const { i18n } = useTranslation();

  const handleLanguageChange = (value: string) => {
    i18n.changeLanguage(value);
    // Language preference is automatically saved to localStorage by i18next
    // via the detection config with caches: ['localStorage']
  };

  return (
    <Select
      value={i18n.language}
      onChange={handleLanguageChange}
      style={{ width: 140 }}
      suffixIcon={<GlobalOutlined style={{ color: 'rgba(255, 255, 255, 0.65)' }} />}
      dropdownStyle={{ minWidth: 140 }}
      bordered={false}
      className="language-selector"
    >
      {languages.map((lang) => (
        <Option key={lang.code} value={lang.code}>
          {lang.label}
        </Option>
      ))}
    </Select>
  );
};

export default LanguageSelector;
