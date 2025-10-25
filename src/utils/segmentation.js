/**
 * Segmentation utilities using OpenCV.js
 * Handles image segmentation, region detection, and mask generation
 */

/**
 * Clean up OpenCV matrices to prevent memory leaks
 * OpenCV.js runs in WebAssembly and requires manual memory management
 */
export function cleanupMats(...mats) {
  mats.forEach(mat => {
    if (mat && !mat.isDeleted()) {
      mat.delete();
    }
  });
}

/**
 * Segment an image into distinct regions using edge detection and contour finding
 *
 * @param {cv.Mat} originalImage - The source image as OpenCV Mat
 * @param {number} sensitivity - Segmentation sensitivity (1-10, lower = fewer regions)
 * @param {object} cv - OpenCV.js instance
 * @returns {Array} Array of region objects with contour, mask, bounds, and selection state
 */
export function segmentImage(originalImage, sensitivity, cv) {
  if (!originalImage || !cv) {
    throw new Error('Invalid image or OpenCV instance');
  }

  const regions = [];

  // Step 1: Convert to grayscale for edge detection
  const gray = new cv.Mat();
  cv.cvtColor(originalImage, gray, cv.COLOR_RGBA2GRAY);

  // Step 2: Apply Gaussian blur to reduce noise
  // This helps create cleaner edges and more coherent regions
  const blurred = new cv.Mat();
  const ksize = new cv.Size(5, 5);
  cv.GaussianBlur(gray, blurred, ksize, 0);

  // Step 3: Detect edges using Canny edge detector
  // Lower sensitivity = higher thresholds = fewer edges = larger regions
  const edges = new cv.Mat();
  const threshold1 = sensitivity * 10;
  const threshold2 = sensitivity * 20;
  cv.Canny(blurred, edges, threshold1, threshold2);

  // Step 4: Dilate edges to close gaps
  // This connects nearby edges to form complete region boundaries
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const dilated = new cv.Mat();
  cv.dilate(edges, dilated, kernel, new cv.Point(-1, -1), 2);

  // Step 5: Find contours (boundaries of regions)
  // RETR_EXTERNAL gets only outer contours (no nested regions)
  // CHAIN_APPROX_SIMPLE compresses contours to save memory
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(dilated, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  // Step 6: Filter and store significant contours as selectable regions
  // Minimum area = 0.1% of image to filter out noise
  const minArea = (originalImage.cols * originalImage.rows) * 0.001;

  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    const area = cv.contourArea(contour);

    if (area > minArea) {
      // Create a binary mask for this region
      const mask = cv.Mat.zeros(originalImage.rows, originalImage.cols, cv.CV_8UC1);
      const contourVec = new cv.MatVector();
      contourVec.push_back(contour);
      cv.drawContours(mask, contourVec, 0, new cv.Scalar(255), -1); // Fill with white
      contourVec.delete();

      // Get bounding rectangle for efficient hit testing
      const bounds = cv.boundingRect(contour);

      regions.push({
        contour: contour.clone(),
        mask: mask,
        bounds: bounds,
        selected: false
      });
    }
  }

  // Cleanup temporary matrices
  cleanupMats(gray, blurred, edges, kernel, dilated, hierarchy);
  contours.delete();

  return regions;
}

/**
 * Draw segmentation visualization on a canvas
 * Shows region boundaries with different colors based on state
 *
 * @param {cv.Mat} originalImage - The source image
 * @param {Array} regions - Array of region objects
 * @param {HTMLCanvasElement} canvas - Target canvas element
 * @param {number} highlightIndex - Index of region to highlight (or -1 for none)
 * @param {object} cv - OpenCV.js instance
 */
export function drawSegmentation(originalImage, regions, canvas, highlightIndex, cv) {
  if (!originalImage || !canvas || !cv) return;

  const display = originalImage.clone();

  // Draw all region boundaries
  for (let i = 0; i < regions.length; i++) {
    const region = regions[i];
    const contourVec = new cv.MatVector();
    contourVec.push_back(region.contour);

    if (i === highlightIndex) {
      // Highlighted region (on hover) - Yellow, thick
      cv.drawContours(display, contourVec, 0, new cv.Scalar(255, 255, 0, 255), 3);
    } else if (region.selected) {
      // Selected region - Green, medium
      cv.drawContours(display, contourVec, 0, new cv.Scalar(0, 255, 0, 255), 2);
    } else {
      // Unselected region - Red, thin
      cv.drawContours(display, contourVec, 0, new cv.Scalar(255, 0, 0, 255), 1);
    }

    contourVec.delete();
  }

  cv.imshow(canvas, display);
  cleanupMats(display);
}

/**
 * Find which region (if any) contains the given point
 * Uses bounding box optimization followed by pixel-perfect mask checking
 *
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 * @param {Array} regions - Array of region objects
 * @returns {number} Index of region containing the point, or -1 if none
 */
export function findRegionAtPoint(x, y, regions) {
  // Check regions in reverse order (top to bottom in z-order)
  for (let i = regions.length - 1; i >= 0; i--) {
    const region = regions[i];
    const bounds = region.bounds;

    // Quick bounding box check first
    if (x < bounds.x || x >= bounds.x + bounds.width ||
        y < bounds.y || y >= bounds.y + bounds.height) {
      continue;
    }

    // Precise check using the pixel mask
    const maskValue = region.mask.ucharAt(Math.floor(y), Math.floor(x));
    if (maskValue > 0) {
      return i;
    }
  }
  return -1;
}

/**
 * Generate a black-and-white mask from selected regions
 * White (255) = selected regions, Black (0) = background
 *
 * @param {cv.Mat} originalImage - The source image (for dimensions)
 * @param {Array} regions - Array of region objects
 * @param {HTMLCanvasElement} canvas - Target canvas for mask display
 * @param {object} cv - OpenCV.js instance
 * @returns {boolean} True if mask was created successfully
 */
export function createMask(originalImage, regions, canvas, cv) {
  if (!originalImage || !canvas || !cv) return false;

  const selectedRegions = regions.filter(r => r.selected);
  if (selectedRegions.length === 0) {
    return false;
  }

  // Create black mask
  const mask = cv.Mat.zeros(originalImage.rows, originalImage.cols, cv.CV_8UC1);

  // Fill selected regions with white
  for (let region of selectedRegions) {
    const contourVec = new cv.MatVector();
    contourVec.push_back(region.contour);
    cv.drawContours(mask, contourVec, 0, new cv.Scalar(255), -1);
    contourVec.delete();
  }

  // Display on canvas
  canvas.width = mask.cols;
  canvas.height = mask.rows;
  cv.imshow(canvas, mask);

  cleanupMats(mask);
  return true;
}

/**
 * Get mouse position relative to canvas, accounting for canvas scaling
 * Canvas may be displayed smaller than its actual size in the DOM
 *
 * @param {HTMLCanvasElement} canvas - The canvas element
 * @param {MouseEvent} event - Mouse event
 * @returns {object} Object with x and y coordinates in canvas space
 */
export function getCanvasMousePosition(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY
  };
}
