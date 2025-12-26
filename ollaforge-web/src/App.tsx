import { useState, useCallback } from 'react';
import { ConfigProvider, message } from 'antd';
import { AppRouter } from './router';
import { theme } from './theme';
import { ErrorBoundary, OfflineIndicator, OnlineIndicator } from './components';
import { useNetworkStatus } from './hooks/useNetworkStatus';
import { useTranslation } from 'react-i18next';
import './i18n';
import './App.css';

function App() {
  const { t } = useTranslation();
  const [showOnlineIndicator, setShowOnlineIndicator] = useState(false);

  const handleOnline = useCallback(() => {
    setShowOnlineIndicator(true);
    message.success(t('errors.connectionRestored'));
    // Auto-hide after 3 seconds
    setTimeout(() => setShowOnlineIndicator(false), 3000);
  }, [t]);

  const handleOffline = useCallback(() => {
    message.warning(t('errors.offline'));
  }, [t]);

  useNetworkStatus({
    onOnline: handleOnline,
    onOffline: handleOffline,
  });

  return (
    <ConfigProvider theme={theme}>
      <ErrorBoundary>
        <OfflineIndicator floating={true} />
        <OnlineIndicator
          visible={showOnlineIndicator}
          onClose={() => setShowOnlineIndicator(false)}
        />
        <AppRouter />
      </ErrorBoundary>
    </ConfigProvider>
  );
}

export default App;
