import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Image Segmentation Integration', () => {
  test('should load image and segment successfully', async ({ page }) => {
    // Array to collect console messages
    const consoleMessages = [];
    const errors = [];

    // Listen to console events
    page.on('console', msg => {
      const text = msg.text();
      consoleMessages.push(text);
      console.log(`[Browser Console] ${text}`);
    });

    // Listen to page errors
    page.on('pageerror', error => {
      errors.push(error.message);
      console.error(`[Browser Error] ${error.message}`);
    });

    // Navigate to the app
    await page.goto('http://localhost:5173/image-mask-generator/');

    // Wait for OpenCV to load (loading div should disappear and main content appears)
    await page.waitForFunction(() => {
      const loading = document.querySelector('.loading');
      const fileInput = document.querySelector('#imageUpload');
      return !loading && fileInput !== null;
    }, { timeout: 35000 });

    console.log('✓ OpenCV.js loaded successfully');

    // Upload the test image
    const testImagePath = path.join(__dirname, '../../public/test-images/test_image_collage.jpg');
    const fileInput = await page.locator('#imageUpload');
    await fileInput.setInputFiles(testImagePath);

    // Wait for image to load - status should show success message
    await page.waitForFunction(() => {
      const status = document.querySelector('.status.success');
      return status && status.textContent.includes('Image loaded successfully');
    }, { timeout: 10000 });

    console.log('✓ Test image loaded');

    // Click the segment button
    await page.click('button:has-text("Segment Image")');

    console.log('✓ Clicked segment button');

    // Wait for segmentation to complete (with generous timeout for large image)
    // Success message should contain "Found X regions"
    await page.waitForFunction(() => {
      const status = document.querySelector('.status');
      const text = status?.textContent || '';
      return text.includes('Found') && text.includes('regions');
    }, { timeout: 60000 }); // 60 second timeout for segmentation

    console.log('✓ Segmentation completed');

    // Get the status message
    const statusText = await page.locator('.status').textContent();
    console.log(`Status: ${statusText}`);

    // Extract number of regions from status
    const match = statusText.match(/Found (\d+) regions/);
    expect(match).not.toBeNull();
    const regionCount = parseInt(match[1]);

    console.log(`✓ Found ${regionCount} regions`);

    // Verify region count is reasonable
    expect(regionCount).toBeGreaterThan(0);
    expect(regionCount).toBeLessThan(1000); // Sanity check

    // Check that segmentation logs are present
    const segmentationLogs = consoleMessages.filter(msg => msg.includes('[Segmentation]'));
    expect(segmentationLogs.length).toBeGreaterThan(0);

    // Verify critical steps were logged
    expect(segmentationLogs.some(msg => msg.includes('Converting to grayscale'))).toBe(true);
    expect(segmentationLogs.some(msg => msg.includes('Applying Gaussian blur'))).toBe(true);
    expect(segmentationLogs.some(msg => msg.includes('Computing gradient'))).toBe(true);
    expect(segmentationLogs.some(msg => msg.includes('Applying watershed'))).toBe(true);
    expect(segmentationLogs.some(msg => msg.includes('Extracting regions'))).toBe(true);

    // Check for downscaling (since test image is 5006x3534)
    expect(segmentationLogs.some(msg => msg.includes('downscaling'))).toBe(true);

    // Verify no errors occurred
    expect(errors.length).toBe(0);

    console.log('✓ All assertions passed');

    // Try to select a region
    const canvas = await page.locator('#originalCanvas');
    await canvas.click({ position: { x: 500, y: 400 } });

    console.log('✓ Clicked on canvas to select region');

    // Create mask
    await page.click('button:has-text("Create Mask")');

    await page.waitForFunction(() => {
      const status = document.querySelector('.status');
      return status && status.textContent.includes('Mask created');
    }, { timeout: 10000 });

    console.log('✓ Mask created successfully');

    // Print summary
    console.log('\n=== Test Summary ===');
    console.log(`Regions found: ${regionCount}`);
    console.log(`Console messages: ${consoleMessages.length}`);
    console.log(`Errors: ${errors.length}`);
    console.log(`Segmentation steps logged: ${segmentationLogs.length}`);
  });

  test('should handle different sensitivity values', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => {
      errors.push(error.message);
      console.error(`[Browser Error] ${error.message}`);
    });

    await page.goto('http://localhost:5173/image-mask-generator/');

    // Wait for OpenCV to load (loading div should disappear)
    await page.waitForFunction(() => {
      const loading = document.querySelector('.loading');
      const fileInput = document.querySelector('#imageUpload');
      return !loading && fileInput !== null;
    }, { timeout: 35000 });

    // Upload image
    const testImagePath = path.join(__dirname, '../../public/test-images/test_image_collage.jpg');
    await page.locator('#imageUpload').setInputFiles(testImagePath);

    await page.waitForFunction(() => {
      const status = document.querySelector('.status.success');
      return status && status.textContent.includes('Image loaded successfully');
    }, { timeout: 10000 });

    // Test with sensitivity = 1 (fewer, larger regions)
    await page.locator('#sensitivitySlider').fill('1');
    await page.click('button:has-text("Segment Image")');

    await page.waitForFunction(() => {
      const status = document.querySelector('.status');
      const text = status?.textContent || '';
      return text.includes('Found') && text.includes('regions');
    }, { timeout: 60000 });

    const statusLow = await page.locator('.status').textContent();
    const matchLow = statusLow.match(/Found (\d+) regions/);
    const regionsLow = parseInt(matchLow[1]);

    console.log(`Sensitivity 1: ${regionsLow} regions`);

    // Test with sensitivity = 10 (more, smaller regions)
    await page.locator('#sensitivitySlider').fill('10');
    await page.click('button:has-text("Segment Image")');

    await page.waitForFunction(() => {
      const status = document.querySelector('.status');
      const text = status?.textContent || '';
      return text.includes('Found') && text.includes('regions');
    }, { timeout: 60000 });

    const statusHigh = await page.locator('.status').textContent();
    const matchHigh = statusHigh.match(/Found (\d+) regions/);
    const regionsHigh = parseInt(matchHigh[1]);

    console.log(`Sensitivity 10: ${regionsHigh} regions`);

    // Higher sensitivity should produce more regions
    expect(regionsHigh).toBeGreaterThan(regionsLow);

    // Verify no errors
    expect(errors.length).toBe(0);
  });
});
