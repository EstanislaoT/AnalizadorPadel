import { defineConfig, devices } from '@playwright/test';

const backendUrl = process.env.API_BASE_URL || 'http://localhost:5050';
const frontendUrl = process.env.BASE_URL || 'http://localhost:3000';
const runAllBrowsers = process.env.PLAYWRIGHT_ALL_BROWSERS === '1';
const projects = [
  {
    name: 'chromium',
    use: { ...devices['Desktop Chrome'] },
  },
];

if (runAllBrowsers) {
  projects.push(
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    }
  );
}

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'playwright-report/results.json' }],
    ['list']
  ],
  use: {
    baseURL: frontendUrl,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects,
  webServer: [
    {
      command: `cd .. && dotnet run --project backend/src/AnalizadorPadel.Api/AnalizadorPadel.Api.csproj --urls ${backendUrl}`,
      url: `${backendUrl}/api/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
    },
    {
      command: `cd ../frontend && VITE_API_URL=${backendUrl} npm run dev -- --host localhost --port 3000`,
      url: frontendUrl,
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
    },
  ],
});
