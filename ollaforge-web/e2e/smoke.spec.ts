import { test, expect } from '@playwright/test';

/**
 * Smoke tests to verify basic application functionality
 * These tests ensure the app loads and basic navigation works
 */

test.describe('Smoke Tests', () => {
  test('should load the application', async ({ page }) => {
    await page.goto('/');
    
    // Wait for the app to load
    await page.waitForLoadState('domcontentloaded');
    
    // Verify the page title
    await expect(page).toHaveTitle(/ollaforge/i);
  });

  test('should display navigation menu', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Check for the OllaForge header text
    const header = page.locator('text=OllaForge').first();
    await expect(header).toBeVisible();
    
    // Check for navigation menu
    const menu = page.locator('.ant-menu');
    await expect(menu).toBeVisible();
  });

  test('should display generate page by default', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // The generate page should be the default (index route)
    // Check for generate-related content
    const generateMenuItem = page.locator('.ant-menu-item-selected');
    await expect(generateMenuItem).toBeVisible();
  });

  test('should navigate to augment page', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Click on augment menu item
    const augmentMenuItem = page.locator('.ant-menu-item').filter({ hasText: /augment/i });
    await augmentMenuItem.click();
    
    // Verify URL changed
    await expect(page).toHaveURL(/.*augment.*/i);
  });

  test('should navigate to config page', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Click on config menu item
    const configMenuItem = page.locator('.ant-menu-item').filter({ hasText: /config/i });
    await configMenuItem.click();
    
    // Verify URL changed
    await expect(page).toHaveURL(/.*config.*/i);
  });

  test('should have language selector', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Check for language selector in header
    const languageSelector = page.locator('.ant-select, [data-testid="language-selector"]');
    await expect(languageSelector.first()).toBeVisible();
  });
});
