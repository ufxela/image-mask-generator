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
 * Segment an image into distinct regions using watershed segmentation
 * This ensures complete coverage (every pixel belongs to exactly one region)
 * and creates compact, blob-like regions that respect edges
 *
 * @param {cv.Mat} originalImage - The source image as OpenCV Mat
 * @param {number} sensitivity - Segmentation sensitivity (1-10, lower = fewer/larger regions)
 * @param {object} cv - OpenCV.js instance
 * @returns {Array} Array of region objects with contour, mask, bounds, and selection state
 */
export function segmentImage(originalImage, sensitivity, cv) {
  if (!originalImage || !cv) {
    throw new Error('Invalid image or OpenCV instance');
  }

  const regions = [];
  let gray, blurred, gradX, gradY, gradient, gradient3C, markers;
  let workingImage = originalImage;
  let scaleFactor = 1;

  try {
    // Check if image is too large and needs downscaling
    const maxDimension = 2000; // Max 2000px on longest side
    const maxPixels = originalImage.rows * originalImage.cols;
    const longSide = Math.max(originalImage.rows, originalImage.cols);

    console.log('[Segmentation] Original image:', originalImage.rows, 'x', originalImage.cols, `(${(maxPixels / 1000000).toFixed(1)}M pixels)`);

    if (longSide > maxDimension) {
      scaleFactor = maxDimension / longSide;
      const newWidth = Math.floor(originalImage.cols * scaleFactor);
      const newHeight = Math.floor(originalImage.rows * scaleFactor);

      console.log(`[Segmentation] Image too large, downscaling by ${scaleFactor.toFixed(2)}x to ${newHeight}x${newWidth}`);

      workingImage = new cv.Mat();
      const dsize = new cv.Size(newWidth, newHeight);
      cv.resize(originalImage, workingImage, dsize, 0, 0, cv.INTER_LINEAR);

      console.log('[Segmentation] Downscaled to:', workingImage.rows, 'x', workingImage.cols, `(${(workingImage.rows * workingImage.cols / 1000000).toFixed(1)}M pixels)`);
    }

    // Step 1: Convert to grayscale
    console.log('[Segmentation] Step 1: Converting to grayscale');
    gray = new cv.Mat();

    try {
      cv.cvtColor(workingImage, gray, cv.COLOR_RGBA2GRAY);
    } catch (e) {
      console.error('[Segmentation] Error in cvtColor:', e);
      throw new Error(`cvtColor failed: ${e.message}`);
    }

    // Step 2: Apply slight blur to reduce noise
    console.log('[Segmentation] Step 2: Applying Gaussian blur');
    blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(3, 3), 0);

    // Step 3: Compute gradient magnitude (edge strength)
    // This will guide watershed to follow edges
    console.log('[Segmentation] Step 3: Computing gradient');
    gradX = new cv.Mat();
    gradY = new cv.Mat();
    gradient = new cv.Mat();

    try {
      console.log('[Segmentation] Computing Sobel X');
      cv.Sobel(blurred, gradX, cv.CV_32F, 1, 0, 3);
      console.log('[Segmentation] Computing Sobel Y');
      cv.Sobel(blurred, gradY, cv.CV_32F, 0, 1, 3);
    } catch (e) {
      console.error('[Segmentation] Error in Sobel:', e);
      throw new Error(`Sobel failed: ${e.message}`);
    }

    // Compute magnitude: sqrt(gradX^2 + gradY^2)
    // Use cv.magnitude if available, otherwise compute manually
    if (typeof cv.magnitude === 'function') {
      console.log('[Segmentation] Using cv.magnitude');
      cv.magnitude(gradX, gradY, gradient);
    } else {
      console.log('[Segmentation] Computing magnitude manually');
      // Manual computation: magnitude = sqrt(x^2 + y^2)
      const gradXSquared = new cv.Mat();
      const gradYSquared = new cv.Mat();
      cv.multiply(gradX, gradX, gradXSquared);
      cv.multiply(gradY, gradY, gradYSquared);
      cv.add(gradXSquared, gradYSquared, gradient);
      cv.sqrt(gradient, gradient);
      cleanupMats(gradXSquared, gradYSquared);
    }

    // Normalize gradient to 0-255 range
    console.log('[Segmentation] Step 4: Normalizing gradient');
    cv.normalize(gradient, gradient, 0, 255, cv.NORM_MINMAX);
    gradient.convertTo(gradient, cv.CV_8U);

    // Step 5: Create markers on a regular grid
    // Sensitivity controls marker spacing (lower = larger spacing = larger/fewer regions)
    // Target: regions should be < 1/10th image dimensions
    console.log('[Segmentation] Step 5: Creating grid markers');
    const maxRegionSize = Math.min(workingImage.cols, workingImage.rows) / 10;
    const baseSpacing = maxRegionSize * 0.8; // Slightly smaller to ensure size constraint

    // Sensitivity adjusts spacing: 1 (low) = larger spacing, 10 (high) = smaller spacing
    const spacingMultiplier = 1.5 - (sensitivity / 10); // Range: 0.5 to 1.4
    const spacing = Math.max(15, Math.floor(baseSpacing * spacingMultiplier));

    console.log(`[Segmentation] Grid spacing: ${spacing}px, Max region size: ${maxRegionSize}px`);

    markers = cv.Mat.zeros(workingImage.rows, workingImage.cols, cv.CV_32S);
    let markerLabel = 1;

    // Place markers in a grid pattern
    for (let y = Math.floor(spacing / 2); y < workingImage.rows; y += spacing) {
      for (let x = Math.floor(spacing / 2); x < workingImage.cols; x += spacing) {
        markers.intPtr(Math.floor(y), Math.floor(x))[0] = markerLabel;
        markerLabel++;
      }
    }

    console.log(`[Segmentation] Created ${markerLabel - 1} markers`);

    // Step 6: Apply watershed algorithm
    // Watershed needs a 3-channel image
    console.log('[Segmentation] Step 6: Applying watershed');

    // Check if watershed is available
    if (typeof cv.watershed !== 'function') {
      throw new Error('cv.watershed is not available in this OpenCV.js build. Please check OpenCV.js version.');
    }

    gradient3C = new cv.Mat();
    console.log('[Segmentation] Converting gradient to BGR');
    cv.cvtColor(gradient, gradient3C, cv.COLOR_GRAY2BGR);

    console.log('[Segmentation] Calling watershed with markers');
    console.log('[Segmentation] gradient3C:', gradient3C.rows, 'x', gradient3C.cols, 'channels:', gradient3C.channels());
    console.log('[Segmentation] markers:', markers.rows, 'x', markers.cols, 'type:', markers.type());

    cv.watershed(gradient3C, markers);

    console.log('[Segmentation] Watershed completed');

    // Step 7: Extract regions from watershed result
    // Watershed labels: -1 = boundary, 0 = background, >0 = region labels
    console.log('[Segmentation] Step 7: Extracting regions');
    const regionMasks = new Map();

    // Build a mask for each region
    // Note: This is a pixel-by-pixel scan which can be slow for large images
    console.log(`[Segmentation] Scanning ${markers.rows * markers.cols} pixels...`);

    try {
      // Process in chunks to avoid memory issues with very large images
      const chunkSize = 100; // Process 100 rows at a time
      for (let startY = 0; startY < markers.rows; startY += chunkSize) {
        const endY = Math.min(startY + chunkSize, markers.rows);

        for (let y = startY; y < endY; y++) {
          for (let x = 0; x < markers.cols; x++) {
            let label;
            try {
              label = markers.intPtr(y, x)[0];
            } catch (e) {
              console.error(`[Segmentation] Error accessing marker at (${y}, ${x}):`, e);
              throw new Error(`Failed to read marker at (${y}, ${x}): ${e.message}`);
            }

            if (label > 0) { // Valid region (not boundary or background)
              if (!regionMasks.has(label)) {
                regionMasks.set(label, cv.Mat.zeros(workingImage.rows, workingImage.cols, cv.CV_8U));
              }
              try {
                regionMasks.get(label).ucharPtr(y, x)[0] = 255;
              } catch (e) {
                console.error(`[Segmentation] Error setting mask pixel at (${y}, ${x}):`, e);
                throw new Error(`Failed to set mask pixel at (${y}, ${x}): ${e.message}`);
              }
            }
          }
        }

        // Log progress for large images
        if (markers.rows > 1000 && (startY % 500 === 0)) {
          console.log(`[Segmentation] Progress: ${Math.floor((startY / markers.rows) * 100)}%`);
        }
      }
    } catch (e) {
      console.error('[Segmentation] Error during region extraction:', e);
      // Clean up any partially created masks
      regionMasks.forEach(mask => mask.delete());
      throw e;
    }

    console.log(`[Segmentation] Found ${regionMasks.size} regions from watershed`);

    // Step 8: Convert each mask to contour and create region objects
    console.log('[Segmentation] Step 8: Converting masks to contours');
    const minArea = (workingImage.cols * workingImage.rows) * 0.0001; // 0.01% minimum

    let processedCount = 0;
    for (const [label, mask] of regionMasks.entries()) {
      try {
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();

        // Find contours for this region
        cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        if (contours.size() > 0) {
          const contour = contours.get(0);
          const area = cv.contourArea(contour);

          if (area > minArea) {
            const bounds = cv.boundingRect(contour);

            // Extract only the bounding box region of the mask to save memory
            const rect = new cv.Rect(bounds.x, bounds.y, bounds.width, bounds.height);
            const croppedMask = mask.roi(rect);

            // Store contour in downscaled space - we'll scale during drawing
            regions.push({
              contour: contour.clone(),
              mask: croppedMask.clone(),
              bounds: bounds, // In downscaled space
              scaleFactor: scaleFactor, // Store for later scaling
              selected: false
            });

            croppedMask.delete();
          }
        }

        // Clean up immediately
        hierarchy.delete();
        contours.delete();
        mask.delete(); // Delete mask immediately after processing

        processedCount++;
        if (processedCount % 50 === 0) {
          console.log(`[Segmentation] Processed ${processedCount}/${regionMasks.size} regions`);
        }
      } catch (e) {
        console.error(`[Segmentation] Error processing region ${label}:`, e);
        // Clean up this region's mask
        mask.delete();
        throw new Error(`Failed to process region ${label}: ${e.message}`);
      }
    }

    // Masks already deleted in loop above
    console.log(`[Segmentation] Returning ${regions.length} valid regions`);

  } catch (error) {
    console.error('[Segmentation] Error:', error);
    // Clean up on error
    cleanupMats(gray, blurred, gradX, gradY, gradient, markers, gradient3C);
    if (scaleFactor !== 1 && workingImage) {
      cleanupMats(workingImage);
    }
    throw error;
  }

  // Cleanup temporary matrices
  cleanupMats(gray, blurred, gradX, gradY, gradient, markers, gradient3C);
  if (scaleFactor !== 1) {
    cleanupMats(workingImage);
  }

  return regions;
}

/**
 * Scale a contour by a given factor
 * @param {cv.Mat} contour - The contour to scale
 * @param {number} scale - Scale factor
 * @param {object} cv - OpenCV instance
 * @returns {cv.Mat} Scaled contour
 */
function scaleContour(contour, scale, cv) {
  if (scale === 1) return contour.clone();

  // Create a new contour with scaled coordinates
  const scaledContour = new cv.Mat(contour.rows, 1, cv.CV_32SC2);

  for (let i = 0; i < contour.rows; i++) {
    const x = contour.data32S[i * 2];
    const y = contour.data32S[i * 2 + 1];
    scaledContour.data32S[i * 2] = Math.round(x * scale);
    scaledContour.data32S[i * 2 + 1] = Math.round(y * scale);
  }

  return scaledContour;
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

    // Scale the contour to match the original image size
    const scaleFactor = region.scaleFactor || 1;
    const scale = 1 / scaleFactor; // Inverse scale
    const scaledContour = scaleContour(region.contour, scale, cv);

    const contourVec = new cv.MatVector();
    contourVec.push_back(scaledContour);

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
    scaledContour.delete();
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
    const scaleFactor = region.scaleFactor || 1;

    // Scale the click coordinates to match the downscaled space
    const scaledX = Math.floor(x * scaleFactor);
    const scaledY = Math.floor(y * scaleFactor);

    // Quick bounding box check first (bounds are in downscaled space)
    if (scaledX < bounds.x || scaledX >= bounds.x + bounds.width ||
        scaledY < bounds.y || scaledY >= bounds.y + bounds.height) {
      continue;
    }

    // Precise check using the pixel mask (also in downscaled space)
    const localX = Math.floor(scaledX - bounds.x);
    const localY = Math.floor(scaledY - bounds.y);

    // Safety check to ensure we're within mask bounds
    if (localX >= 0 && localX < region.mask.cols &&
        localY >= 0 && localY < region.mask.rows) {
      const maskValue = region.mask.ucharAt(localY, localX);
      if (maskValue > 0) {
        return i;
      }
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

  // Create black mask at original image size
  const mask = cv.Mat.zeros(originalImage.rows, originalImage.cols, cv.CV_8UC1);

  // Fill selected regions with white
  for (let region of selectedRegions) {
    // Scale the contour to match original image size
    const scaleFactor = region.scaleFactor || 1;
    const scale = 1 / scaleFactor;
    const scaledContour = scaleContour(region.contour, scale, cv);

    const contourVec = new cv.MatVector();
    contourVec.push_back(scaledContour);
    cv.drawContours(mask, contourVec, 0, new cv.Scalar(255), -1);
    contourVec.delete();
    scaledContour.delete();
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
