import { test as base, expect } from '@playwright/test';

/**
 * Custom test fixtures for OllaForge E2E tests
 * Provides common setup and utilities for all E2E tests
 */

// Extend base test with custom fixtures
export const test = base.extend({
  // Auto-wait for page to be ready
  page: async ({ page }, use) => {
    // Wait for the app to be fully loaded
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await use(page);
  },
});

export { expect };

/**
 * Test data generators for E2E tests
 */
export const testData = {
  generation: {
    validConfig: {
      topic: 'Test topic for E2E testing',
      count: 5,
      model: 'llama3.2',
      datasetType: 'sft',
      language: 'en',
    },
  },
  augmentation: {
    sampleDataset: [
      { instruction: 'Test instruction 1', input: 'Test input 1', output: 'Test output 1' },
      { instruction: 'Test instruction 2', input: 'Test input 2', output: 'Test output 2' },
    ],
  },
};

/**
 * Helper functions for E2E tests
 */
export const helpers = {
  /**
   * Wait for API response with timeout
   */
  async waitForApiResponse(page: typeof base.prototype.page, urlPattern: string | RegExp, timeout = 30000) {
    return page.waitForResponse(
      (response) => {
        const url = response.url();
        if (typeof urlPattern === 'string') {
          return url.includes(urlPattern);
        }
        return urlPattern.test(url);
      },
      { timeout }
    );
  },

  /**
   * Create a test JSONL file content
   */
  createTestJsonlContent(entries: Record<string, unknown>[]): string {
    return entries.map((entry) => JSON.stringify(entry)).join('\n');
  },

  /**
   * Wait for progress to complete
   */
  async waitForProgressComplete(page: typeof base.prototype.page, timeout = 60000) {
    await page.waitForSelector('[data-testid="progress-complete"], [data-testid="progress-error"]', {
      timeout,
    });
  },
};
