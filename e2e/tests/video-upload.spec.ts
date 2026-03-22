import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { test, expect, type APIRequestContext } from '@playwright/test';

const apiBaseUrl = process.env.API_BASE_URL || 'http://localhost:5050';
const sampleVideoPath = path.resolve(__dirname, '../../test-videos/FinalPadelPrueba1.mp4');

type CreatedVideo = {
  id: number;
  name: string;
};

async function createVideoFixture(request: APIRequestContext): Promise<CreatedVideo> {
  const buffer = await fs.readFile(sampleVideoPath);
  const uniqueName = `e2e-${crypto.randomUUID()}.mp4`;

  const response = await request.post(`${apiBaseUrl}/api/videos`, {
    multipart: {
      file: {
        name: uniqueName,
        mimeType: 'video/mp4',
        buffer,
      },
    },
  });

  expect(response.status()).toBe(201);

  const body = await response.json();
  expect(body.success).toBe(true);

  return {
    id: body.data.id,
    name: body.data.name,
  };
}

test.describe('Dashboard Flow', () => {
  test('should display dashboard stats from the real backend', async ({ page, request }) => {
    const createdVideo = await createVideoFixture(request);

    try {
      await page.goto('/');
      await expect(page.getByTestId('dashboard-title')).toHaveText('Dashboard');
      await expect(page.getByText('Videos Recientes')).toBeVisible();
      await expect(page.getByTestId('recent-videos').getByText(createdVideo.name).first()).toBeVisible();
    } finally {
      await request.delete(`${apiBaseUrl}/api/videos/${createdVideo.id}`);
    }
  });

  test('should navigate between dashboard, videos and analyses', async ({ page }) => {
    await page.goto('/');

    await page.getByRole('button', { name: 'Videos' }).click();
    await expect(page).toHaveURL(/\/videos$/);
    await expect(page.getByRole('heading', { name: 'Videos', exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Análisis' }).click();
    await expect(page).toHaveURL(/\/analyses$/);
    await expect(page.getByRole('heading', { name: 'Análisis' })).toBeVisible();

    await page.getByRole('button', { name: 'Dashboard' }).click();
    await expect(page).toHaveURL(/\/$/);
    await expect(page.getByTestId('dashboard-title')).toBeVisible();
  });
});

test.describe('Videos Flow', () => {
  test('should list and stream a freshly uploaded video', async ({ page, request }) => {
    const createdVideo = await createVideoFixture(request);

    try {
      await page.goto('/videos');
      await expect(page.getByRole('heading', { name: 'Videos', exact: true })).toBeVisible();

      const videoEntry = page.getByText(createdVideo.name).first();
      await expect(videoEntry).toBeVisible();

      const streamResponsePromise = page.waitForResponse((response) =>
        response.url().includes(`/api/videos/${createdVideo.id}/stream`) &&
        [200, 206].includes(response.status())
      );

      await videoEntry.click();

      await expect(page.getByRole('heading', { name: createdVideo.name, exact: true })).toBeVisible();
      await expect(page.locator('video')).toBeVisible();

      const streamResponse = await streamResponsePromise;
      expect(streamResponse.ok()).toBe(true);
    } finally {
      await request.delete(`${apiBaseUrl}/api/videos/${createdVideo.id}`);
    }
  });
});

test.describe('API Health Check', () => {
  test('backend API should be accessible', async ({ request }) => {
    const response = await request.get(`${apiBaseUrl}/api/health`);
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('success', true);
    expect(body).toHaveProperty('message');
  });

  test('should get dashboard stats from API', async ({ request }) => {
    const response = await request.get(`${apiBaseUrl}/api/dashboard/stats`);
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('success', true);
    expect(body).toHaveProperty('data');
    expect(body.data).toHaveProperty('totalVideos');
    expect(body.data).toHaveProperty('totalAnalyses');
  });
});
