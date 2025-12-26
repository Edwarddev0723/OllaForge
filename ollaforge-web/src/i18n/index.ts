import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

import en from './locales/en.json';
import zhTW from './locales/zh-TW.json';

const resources = {
  en: {
    translation: en,
  },
  'zh-TW': {
    translation: zhTW,
  },
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    supportedLngs: ['en', 'zh-TW'],
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
      lookupLocalStorage: 'ollaforge-language',
    },
    interpolation: {
      escapeValue: false,
    },
  });

export default i18n;
