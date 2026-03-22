const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  
  const consoleMessages = [];
  const errors = [];
  
  page.on('console', msg => {
    consoleMessages.push({ type: msg.type(), text: msg.text() });
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });
  
  page.on('pageerror', error => {
    errors.push(`Page Error: ${error.message}`);
  });
  
  try {
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);
    
    console.log('=== Console Messages ===');
    consoleMessages.forEach(m => console.log(`[${m.type}] ${m.text}`));
    
    console.log('\n=== Errors ===');
    if (errors.length === 0) {
      console.log('No errors found!');
    } else {
      errors.forEach(e => console.log(`ERROR: ${e}`));
    }
  } catch (e) {
    console.log(`Navigation error: ${e.message}`);
  }
  
  await browser.close();
})();
