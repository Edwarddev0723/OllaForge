import type { ThemeConfig } from 'antd';

export const theme: ThemeConfig = {
  token: {
    colorPrimary: '#1890ff',
    borderRadius: 6,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  components: {
    Layout: {
      headerBg: '#001529',
      headerHeight: 64,
    },
    Menu: {
      darkItemBg: '#001529',
    },
    Button: {
      primaryShadow: '0 2px 0 rgba(24, 144, 255, 0.1)',
    },
    Card: {
      borderRadiusLG: 8,
    },
    Form: {
      itemMarginBottom: 24,
    },
  },
};

export default theme;
