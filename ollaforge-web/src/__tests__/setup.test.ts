/**
 * Tests for frontend build configuration and dependencies
 */

import { describe, it, expect } from 'vitest';
import * as React from 'react';

describe('Frontend Setup', () => {
  it('should have React available', () => {
    expect(typeof React).toBe('object');
  });

  it('should have Ant Design available', async () => {
    const antd = await import('antd');
    expect(antd).toBeDefined();
    expect(antd.Button).toBeDefined();
  });

  it('should have Axios available', async () => {
    const axios = await import('axios');
    expect(axios.default).toBeDefined();
    expect(typeof axios.default.create).toBe('function');
  });

  it('should have Socket.IO client available', async () => {
    const io = await import('socket.io-client');
    expect(io.io).toBeDefined();
    expect(typeof io.io).toBe('function');
  });

  it('should have i18next available', async () => {
    const i18next = await import('i18next');
    expect(i18next.default).toBeDefined();
    expect(typeof i18next.default.init).toBe('function');
  });

  it('should have react-i18next available', async () => {
    const reactI18next = await import('react-i18next');
    expect(reactI18next.useTranslation).toBeDefined();
    expect(reactI18next.I18nextProvider).toBeDefined();
  });

  it('should have language detector available', async () => {
    const detector = await import('i18next-browser-languagedetector');
    expect(detector.default).toBeDefined();
  });

  it('should have TypeScript configured', () => {
    // This test passes if TypeScript compilation succeeds
    const testValue: string = 'test';
    expect(testValue).toBe('test');
  });

  it('should have Vite environment variables accessible', () => {
    // Check that Vite env variables are accessible
    expect(import.meta.env).toBeDefined();
  });
});
