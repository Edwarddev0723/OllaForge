import React from 'react';
import { Layout, Menu } from 'antd';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  FileAddOutlined,
  EditOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { LanguageSelector } from '../components';

const { Header, Content, Footer } = Layout;

const MainLayout: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <FileAddOutlined />,
      label: t('nav.generate'),
    },
    {
      key: '/augment',
      icon: <EditOutlined />,
      label: t('nav.augment'),
    },
    {
      key: '/config',
      icon: <SettingOutlined />,
      label: t('nav.config'),
    },
  ];

  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key);
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center' }}>
        <div style={{ color: 'white', fontSize: '20px', fontWeight: 'bold', marginRight: '40px' }}>
          OllaForge
        </div>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ flex: 1, minWidth: 0 }}
        />
        <LanguageSelector />
      </Header>
      <Content style={{ padding: '24px 48px' }}>
        <div style={{ background: '#fff', padding: 24, minHeight: 360, borderRadius: 8 }}>
          <Outlet />
        </div>
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        OllaForge ©{new Date().getFullYear()} - {t('footer.copyright')}
      </Footer>
    </Layout>
  );
};

export default MainLayout;
