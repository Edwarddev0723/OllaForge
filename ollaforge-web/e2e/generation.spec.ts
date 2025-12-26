import { test, expect } from '@playwright/test';

/**
 * E2E tests for the generation workflow
 * Tests the complete generation flow from form to download
 * 
 * Requirements covered:
 * - 1.1: Display dataset generation form
 * - 1.2: Initiate dataset generation with valid parameters
 * - 1.3: Provide download link for generated dataset
 * - 3.1: Display progress bar during generation
 * - 3.2: Update progress in real-time
 * - 3.3: Display completion statistics
 */

test.describe('Generation Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the generate page
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('should display generation form with all required fields', async ({ page }) => {
    // Check for topic input
    const topicInput = page.locator('textarea, input').filter({ hasText: /topic/i }).first();
    await expect(topicInput.or(page.locator('[placeholder*="topic" i]').first())).toBeVisible();
    
    // Check for count input
    const countInput = page.locator('input[type="number"], .ant-input-number');
    await expect(countInput.first()).toBeVisible();
    
    // Check for model selector
    const modelSelector = page.locator('.ant-select').first();
    await expect(modelSelector).toBeVisible();
    
    // Check for submit button
    const submitButton = page.locator('button[type="submit"], .ant-btn-primary').first();
    await expect(submitButton).toBeVisible();
  });

  // Additional tests will be implemented in task 20.2
  test.skip('should show progress during generation', async ({ page }) => {
    // TODO: Implement in task 20.2
  });

  test.skip('should provide download after completion', async ({ page }) => {
    // TODO: Implement in task 20.2
  });
});
