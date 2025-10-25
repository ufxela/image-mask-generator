# Testing Guide

## Overview

This project uses **Vitest** for unit testing. Tests are located in `src/tests/`.

## Running Tests

```bash
# Run tests in watch mode
npm run test

# Run tests once
npm run test:run

# Run tests with UI
npm run test:ui
```

## Test Structure

### Unit Tests (`src/tests/segmentation.test.js`)

Unit tests cover:
- **cleanupMats()** - Memory management for OpenCV matrices
- **findRegionAtPoint()** - Hit testing for region selection
- **getCanvasMousePosition()** - Mouse coordinate transformation
- **Algorithm properties** - Grid spacing calculations and constraints

**19 tests** cover edge cases including:
- Null/undefined handling
- Boundary conditions
- Coordinate scaling
- Region overlap handling

### Test Coverage

Current test coverage focuses on:
- ✅ Utility functions (100%)
- ✅ Mathematical calculations
- ✅ Edge case handling
- ⚠️ Full segmentation pipeline (requires browser environment)

## Integration Testing

### Manual Browser Testing

The segmentation algorithm uses OpenCV.js which requires a real browser environment. For comprehensive testing:

1. **Start the dev server:**
   ```bash
   npm run dev
   ```

2. **Open browser:** Navigate to `http://localhost:5173/image-mask-generator/`

3. **Test with provided image:**
   - Click "Choose Image"
   - Upload `/public/test-images/test_image_collage.jpg`
   - Verify OpenCV.js loads successfully
   - Click "Segment Image"

4. **Verify segmentation properties:**
   - ✅ **Complete Coverage**: Every pixel belongs to a region (no black gaps)
   - ✅ **Blob-like Regions**: Segments are compact and roughly circular
   - ✅ **Size Constraints**: Regions are < 1/10th of image dimensions
   - ✅ **Edge Awareness**: Boundaries follow visible edges
   - ✅ **Sensitivity Control**: Slider affects region count/size

5. **Test interaction:**
   - Hover over regions (should highlight in yellow)
   - Click regions (should select in green)
   - Click "Create Mask"
   - Verify mask shows white (selected) and black (unselected)
   - Click "Download Mask" and check output

6. **Browser console checks:**
   - Open DevTools console
   - Look for `[Segmentation]` log messages
   - Verify each step completes
   - Check for error messages

### Expected Console Output

When segmenting an image, you should see:
```
[Segmentation] Step 1: Converting to grayscale
[Segmentation] Step 2: Applying Gaussian blur
[Segmentation] Step 3: Computing gradient
[Segmentation] Using cv.magnitude (or Computing magnitude manually)
[Segmentation] Step 4: Normalizing gradient
[Segmentation] Step 5: Creating grid markers
[Segmentation] Grid spacing: 80px, Max region size: 100px
[Segmentation] Created 156 markers
[Segmentation] Step 6: Applying watershed
[Segmentation] Step 7: Extracting regions
[Segmentation] Found 156 regions from watershed
[Segmentation] Returning 156 valid regions
```

## Automated E2E Testing (Future)

For automated browser testing, consider:

### Playwright

```javascript
import { test, expect } from '@playwright/test';

test('should segment image and create mask', async ({ page }) => {
  await page.goto('http://localhost:5173/image-mask-generator/');

  // Wait for OpenCV to load
  await page.waitForSelector('.loading', { state: 'hidden' });

  // Upload image
  const fileInput = page.locator('#imageUpload');
  await fileInput.setInputFiles('./public/test-images/test_image_collage.jpg');

  // Wait for image to load
  await expect(page.locator('.status.success')).toContainText('Image loaded');

  // Segment
  await page.click('button:has-text("Segment Image")');
  await expect(page.locator('.status.success')).toContainText('Found');

  // Select a region
  const canvas = page.locator('#originalCanvas');
  await canvas.click({ position: { x: 100, y: 100 } });

  // Create mask
  await page.click('button:has-text("Create Mask")');
  await expect(page.locator('.status.success')).toContainText('Mask created');
});
```

### Cypress

```javascript
describe('Image Segmentation', () => {
  it('should create mask from selected regions', () => {
    cy.visit('http://localhost:5173/image-mask-generator/');

    // Upload image
    cy.get('#imageUpload').selectFile('./public/test-images/test_image_collage.jpg');
    cy.contains('Image loaded').should('be.visible');

    // Segment
    cy.contains('button', 'Segment Image').click();
    cy.contains('Found').should('be.visible');

    // Select regions and create mask
    cy.get('#originalCanvas').click(100, 100);
    cy.contains('button', 'Create Mask').click();
    cy.contains('Mask created').should('be.visible');
  });
});
```

## Visual Regression Testing

To ensure segmentation results remain consistent:

1. **Capture baseline images:**
   - Segment the test image with known sensitivity values
   - Save screenshots of segmentation results
   - Commit to version control

2. **Use visual testing tools:**
   - **Percy**: `percy snapshot` during CI
   - **Chromatic**: Storybook + automated visual testing
   - **Applitools**: AI-powered visual testing

3. **Compare programmatically:**
   ```javascript
   import { toMatchImageSnapshot } from 'jest-image-snapshot';

   expect.extend({ toMatchImageSnapshot });

   test('segmentation visual regression', async () => {
     const screenshot = await page.screenshot();
     expect(screenshot).toMatchImageSnapshot();
   });
   ```

## Performance Testing

Monitor segmentation performance:

```javascript
test('segmentation performance', async () => {
  const start = performance.now();
  const regions = await segmentImage(testImage, 5, cv);
  const duration = performance.now() - start;

  expect(duration).toBeLessThan(5000); // Should complete in < 5s
  expect(regions.length).toBeGreaterThan(0);
});
```

## Test Data

Test images are stored in `/public/test-images/`:
- `test_image_collage.jpg` - Primary test image for manual testing

## Debugging Tests

1. **Enable verbose logging:**
   ```bash
   DEBUG=* npm run test
   ```

2. **Run single test:**
   ```bash
   npm run test -- segmentation.test.js
   ```

3. **Use test UI:**
   ```bash
   npm run test:ui
   ```
   Opens interactive test interface at `http://localhost:51204/__vitest__/`

## Contributing Tests

When adding new features:
1. Add unit tests for pure functions
2. Update integration test checklist
3. Add visual regression baselines if needed
4. Document expected behavior
5. Ensure all tests pass before committing
