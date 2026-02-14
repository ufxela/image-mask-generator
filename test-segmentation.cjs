const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
  });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('[Segmentation]') || text.includes('Error') || text.includes('marker') || text.includes('region')) {
      console.log('>', text);
    }
  });
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));

  console.log('Loading app...');
  await page.goto('http://localhost:5173/image-mask-generator/', { waitUntil: 'networkidle', timeout: 30000 });

  console.log('Waiting for OpenCV...');
  await page.waitForFunction(() => window.cv && window.cv.Mat, { timeout: 60000 });
  console.log('OpenCV loaded');

  const imagePath = path.resolve('/Users/ufxela/image-mask-generator/test-collage.jpg');
  await page.locator('input[type="file"]').setInputFiles(imagePath);
  await page.waitForTimeout(3000);
  console.log('Image uploaded');

  console.log('Segmenting...');
  await page.locator('button:has-text("Segment")').first().click();

  await page.waitForFunction(
    () => {
      const el = document.querySelector('.status');
      return el && (el.textContent.includes('region') || el.textContent.includes('Error') || el.textContent.includes('error'));
    },
    { timeout: 300000 }
  );

  console.log('RESULT:', await page.locator('.status').first().textContent());

  // Take screenshot of just the segmentation canvas area
  await page.screenshot({ path: '/Users/ufxela/image-mask-generator/segmentation-result.png', fullPage: true });
  console.log('Full page screenshot saved');

  await browser.close();
})().catch(err => { console.error('FATAL:', err.message); process.exit(1); });
