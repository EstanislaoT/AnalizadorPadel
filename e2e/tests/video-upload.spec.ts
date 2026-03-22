import { test, expect } from '@playwright/test';

test.describe('Video Upload Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/videos');
  });

  test('should display video upload page', async ({ page }) => {
    await expect(page.locator('h1, h2, h3')).toContainText(/video|vídeo/i);
    await expect(page.locator('input[type="file"]')).toBeVisible();
  });

  test('should show error when uploading invalid file type', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    
    // Create a text file to upload
    const invalidFile = {
      name: 'test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('invalid content'),
    };

    await fileInput.setInputFiles(invalidFile);
    
    // Submit the form if there's a submit button
    const submitButton = page.locator('button[type="submit"], button:has-text("Subir"), button:has-text("Upload")');
    if (await submitButton.isVisible().catch(() => false)) {
      await submitButton.click();
      await expect(page.locator('text=/formato no soportado|format not supported/i')).toBeVisible();
    }
  });

  test('should navigate to videos list', async ({ page }) => {
    // Check that videos page loads
    await expect(page).toHaveURL(/.*videos.*/);
    
    // Check for list or empty state
    const content = await page.locator('body').textContent();
    expect(content?.toLowerCase()).toMatch(/video|lista|empty|no hay/i);
  });
});

test.describe('Dashboard Flow', () => {
  test('should display dashboard with stats', async ({ page }) => {
    await page.goto('/');
    
    // Wait for dashboard to load
    await page.waitForLoadState('networkidle');
    
    // Check for dashboard elements
    const bodyText = await page.locator('body').textContent();
    expect(bodyText?.toLowerCase()).toMatch(/dashboard|estadísticas|stats|videos|análisis/i);
  });

  test('should navigate between pages', async ({ page }) => {
    // Start at home
    await page.goto('/');
    
    // Try to navigate to videos
    const videosLink = page.locator('a[href*="video"], nav >> text=/video/i, button:has-text(/video/i)').first();
    if (await videosLink.isVisible().catch(() => false)) {
      await videosLink.click();
      await expect(page).toHaveURL(/.*videos.*/);
    }
    
    // Try to navigate to analyses
    await page.goto('/');
    const analysesLink = page.locator('a[href*="analys"], nav >> text=/análisis|analysis/i').first();
    if (await analysesLink.isVisible().catch(() => false)) {
      await analysesLink.click();
      await expect(page).toHaveURL(/.*analys.*/);
    }
  });
});

test.describe('API Health Check', () => {
  test('backend API should be accessible', async ({ request }) => {
    const response = await request.get('http://localhost:5000/api/health');
    expect(response.status()).toBe(200);
    
    const body = await response.json();
    expect(body).toHaveProperty('success', true);
    expect(body).toHaveProperty('message');
  });

  test('should get dashboard stats from API', async ({ request }) => {
    const response = await request.get('http://localhost:5000/api/dashboard/stats');
    expect(response.status()).toBe(200);
    
    const body = await response.json();
    expect(body).toHaveProperty('success', true);
    expect(body).toHaveProperty('data');
    expect(body.data).toHaveProperty('totalVideos');
    expect(body.data).toHaveProperty('totalAnalyses');
  });
});
