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
 * @param {number} regionSize - Target region size divisor (10-40, higher = smaller regions)
 * @param {object} cv - OpenCV.js instance
 * @returns {Array} Array of region objects with contour, mask, bounds, and selection state
 */
export function segmentImage(originalImage, sensitivity, regionSize, cv) {
  if (!originalImage || !cv) {
    throw new Error('Invalid image or OpenCV instance');
  }

  // Default regionSize to 20 if not provided (for backward compatibility)
  if (regionSize === undefined || regionSize === null) {
    regionSize = 20;
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

    // Step 2: Apply bilateral filter to reduce noise while preserving edges
    // Bilateral filtering is better than Gaussian for preserving sharp edges
    console.log('[Segmentation] Step 2: Applying bilateral filter to preserve edges');
    blurred = new cv.Mat();
    cv.bilateralFilter(gray, blurred, 5, 75, 75);

    // Step 3: Compute gradient magnitude (edge strength) with larger kernel for better edge detection
    // This will guide watershed to follow edges more accurately
    console.log('[Segmentation] Step 3: Computing gradient with enhanced edge detection');
    gradX = new cv.Mat();
    gradY = new cv.Mat();
    gradient = new cv.Mat();

    try {
      console.log('[Segmentation] Computing Sobel X with kernel size 5');
      cv.Sobel(blurred, gradX, cv.CV_32F, 1, 0, 5); // Larger kernel (5 instead of 3) for better edges
      console.log('[Segmentation] Computing Sobel Y with kernel size 5');
      cv.Sobel(blurred, gradY, cv.CV_32F, 0, 1, 5);
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
    // regionSize parameter controls target region size (higher = smaller regions)
    console.log('[Segmentation] Step 5: Creating grid markers');
    const maxRegionSize = Math.min(workingImage.cols, workingImage.rows) / regionSize;
    const baseSpacing = maxRegionSize * 0.7; // Base spacing for medium-small regions

    // Sensitivity adjusts spacing: 1 (low) = larger spacing, 10 (high) = much smaller spacing
    // Use exponential curve to bias heavily toward smaller regions
    const spacingMultiplier = 1.3 - (sensitivity / 10) * 1.1; // Range: 0.2 to 1.2
    const spacing = Math.max(10, Math.floor(baseSpacing * spacingMultiplier));

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

    const markerCount = markerLabel - 1;
    console.log(`[Segmentation] Created ${markerCount} markers`);

    // Safety check: too many markers can cause memory issues
    // For a 2000x2000 image with small spacing, we might need 3000-5000 markers
    if (markerCount > 5000) {
      throw new Error(`Too many markers (${markerCount}). Try reducing sensitivity.`);
    }

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

    // MEMORY-EFFICIENT APPROACH: Collect pixel coordinates first, then create masks only for valid regions
    // This avoids creating 2000+ full-sized masks which would consume ~5-10GB of memory
    const regionPoints = new Map(); // Map<label, Array<{x, y}>>
    const minArea = (workingImage.cols * workingImage.rows) * 0.0001; // 0.01% minimum

    console.log(`[Segmentation] Scanning ${markers.rows * markers.cols} pixels...`);

    try {
      // First pass: collect pixel coordinates for each region
      // Also assign boundary pixels (-1) and background (0) to nearest region for complete coverage
      const chunkSize = 100; // Process 100 rows at a time
      const boundaryPixels = []; // Store boundary pixels to assign later

      for (let startY = 0; startY < markers.rows; startY += chunkSize) {
        const endY = Math.min(startY + chunkSize, markers.rows);

        for (let y = startY; y < endY; y++) {
          for (let x = 0; x < markers.cols; x++) {
            let label;
            try {
              label = markers.intPtr(y, x)[0];
            } catch (e) {
              console.error(`[Segmentation] Error accessing marker at (${y}, ${x}):`, e);
              // OpenCV.js may throw numeric error codes instead of Error objects
              const errorMsg = typeof e === 'object' && e.message ? e.message : String(e);
              throw new Error(`Failed to read marker at (${y}, ${x}): ${errorMsg}`);
            }

            // Validate label value
            if (label === undefined || label === null || isNaN(label)) {
              throw new Error(`Invalid marker label at (${y}, ${x}): ${label}`);
            }

            if (label > 0) {
              // Valid region (not boundary or background)
              if (!regionPoints.has(label)) {
                regionPoints.set(label, []);
              }
              regionPoints.get(label).push({ x, y });
            } else if (label === -1 || label === 0) {
              // Boundary or background - save to assign to nearest region later
              boundaryPixels.push({ x, y });
            }
          }
        }

        // Log progress for large images
        if (markers.rows > 1000 && (startY % 500 === 0)) {
          console.log(`[Segmentation] Progress: ${Math.floor((startY / markers.rows) * 100)}%`);
        }
      }

      // Second pass: assign boundary/background pixels to nearest region
      console.log(`[Segmentation] Assigning ${boundaryPixels.length} boundary/background pixels to nearest regions...`);
      for (const pixel of boundaryPixels) {
        let nearestLabel = null;
        let minDist = Infinity;

        // Check neighboring pixels (8-connected) to find nearest region
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;

            const nx = pixel.x + dx;
            const ny = pixel.y + dy;

            if (nx >= 0 && nx < markers.cols && ny >= 0 && ny < markers.rows) {
              const neighborLabel = markers.intPtr(ny, nx)[0];
              if (neighborLabel > 0) {
                // Found a neighbor with a valid region
                const dist = Math.abs(dx) + Math.abs(dy); // Manhattan distance
                if (dist < minDist) {
                  minDist = dist;
                  nearestLabel = neighborLabel;
                }
              }
            }
          }
        }

        // Assign to nearest region (or first available region if no neighbors)
        if (nearestLabel !== null) {
          if (!regionPoints.has(nearestLabel)) {
            regionPoints.set(nearestLabel, []);
          }
          regionPoints.get(nearestLabel).push({ x: pixel.x, y: pixel.y });
        } else if (regionPoints.size > 0) {
          // No neighbors found, assign to first region as fallback
          const firstLabel = regionPoints.keys().next().value;
          regionPoints.get(firstLabel).push({ x: pixel.x, y: pixel.y });
        }
      }
    } catch (e) {
      console.error('[Segmentation] Error during region extraction:', e);
      // Ensure we throw a proper Error object, not a numeric code
      if (e instanceof Error) {
        throw e;
      } else {
        throw new Error(`Region extraction failed with code: ${e}`);
      }
    }

    console.log(`[Segmentation] Found ${regionPoints.size} regions from watershed`);

    // Step 8: Create masks and contours only for regions that meet minimum area
    console.log('[Segmentation] Step 8: Creating masks and contours for valid regions');

    let processedCount = 0;
    for (const [label, points] of regionPoints.entries()) {
      try {
        // Skip regions that are too small
        if (points.length < minArea) {
          processedCount++;
          continue;
        }

        // Calculate bounding box from points
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const pt of points) {
          if (pt.x < minX) minX = pt.x;
          if (pt.x > maxX) maxX = pt.x;
          if (pt.y < minY) minY = pt.y;
          if (pt.y > maxY) maxY = pt.y;
        }

        const bounds = {
          x: minX,
          y: minY,
          width: maxX - minX + 1,
          height: maxY - minY + 1
        };

        // Validate bounds
        if (bounds.width <= 0 || bounds.height <= 0) {
          console.warn(`[Segmentation] Skipping invalid bounds for region ${label}`);
          processedCount++;
          continue;
        }

        // Create a CROPPED mask (only as large as the bounding box)
        const mask = cv.Mat.zeros(bounds.height, bounds.width, cv.CV_8U);

        // Fill the mask with points (translated to local coordinates)
        for (const pt of points) {
          const localX = pt.x - minX;
          const localY = pt.y - minY;
          if (localX >= 0 && localX < bounds.width && localY >= 0 && localY < bounds.height) {
            mask.ucharPtr(localY, localX)[0] = 255;
          }
        }

        // Find contours from this mask
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        if (contours.size() > 0) {
          const contour = contours.get(0);

          // Validate contour
          if (!contour || contour.rows === 0) {
            console.warn(`[Segmentation] Skipping empty contour for region ${label}`);
            hierarchy.delete();
            contours.delete();
            mask.delete();
            processedCount++;
            continue;
          }

          const area = cv.contourArea(contour);

          if (area > minArea) {
            // Translate contour coordinates back to image space
            const translatedContour = new cv.Mat(contour.rows, 1, cv.CV_32SC2);
            for (let i = 0; i < contour.rows; i++) {
              translatedContour.data32S[i * 2] = contour.data32S[i * 2] + minX;
              translatedContour.data32S[i * 2 + 1] = contour.data32S[i * 2 + 1] + minY;
            }

            // Store region with CROPPED mask (memory efficient)
            // NOTE: We store translatedContour directly (don't delete it, it's owned by the region now)
            regions.push({
              contour: translatedContour, // We OWN this now, don't delete
              mask: mask.clone(),
              bounds: bounds, // In downscaled space
              scaleFactor: scaleFactor, // Store for later scaling
              selected: false
            });

            // Don't delete translatedContour - it's now owned by the region!
          }
        }

        // Clean up
        hierarchy.delete();
        contours.delete();
        mask.delete();

        processedCount++;
        if (processedCount % 100 === 0) {
          console.log(`[Segmentation] Processed ${processedCount}/${regionPoints.size} regions`);
        }
      } catch (e) {
        console.error(`[Segmentation] Error processing region ${label}:`, e);
        throw new Error(`Failed to process region ${label}: ${e.message || e}`);
      }
    }

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
  if (!contour || contour.rows === 0) {
    throw new Error('Invalid contour: contour is empty or undefined');
  }

  if (scale === 1) return contour.clone();

  if (!contour.data32S) {
    throw new Error('Invalid contour: data32S is undefined');
  }

  // Create a new contour with scaled coordinates
  const scaledContour = new cv.Mat(contour.rows, 1, cv.CV_32SC2);

  try {
    for (let i = 0; i < contour.rows; i++) {
      const x = contour.data32S[i * 2];
      const y = contour.data32S[i * 2 + 1];

      if (x === undefined || y === undefined) {
        throw new Error(`Invalid contour data at index ${i}: x=${x}, y=${y}`);
      }

      scaledContour.data32S[i * 2] = Math.round(x * scale);
      scaledContour.data32S[i * 2 + 1] = Math.round(y * scale);
    }
  } catch (error) {
    scaledContour.delete();
    throw error;
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
 * Find all regions within a given radius of a point
 * Useful for drag-selection with a "brush" radius
 *
 * @param {number} x - X coordinate (center)
 * @param {number} y - Y coordinate (center)
 * @param {number} radius - Radius in pixels
 * @param {Array} regions - Array of region objects
 * @returns {Array<number>} Array of region indices within the radius
 */
export function findRegionsInRadius(x, y, radius, regions) {
  const foundRegions = [];

  for (let i = 0; i < regions.length; i++) {
    const region = regions[i];
    const bounds = region.bounds;
    const scaleFactor = region.scaleFactor || 1;

    // Scale the coordinates to match the downscaled space
    const scaledX = x * scaleFactor;
    const scaledY = y * scaleFactor;
    const scaledRadius = radius * scaleFactor;

    // Check if the region's bounding box intersects with the circle
    // Find the closest point on the bounding box to the circle center
    const closestX = Math.max(bounds.x, Math.min(scaledX, bounds.x + bounds.width));
    const closestY = Math.max(bounds.y, Math.min(scaledY, bounds.y + bounds.height));

    // Calculate distance from circle center to this closest point
    const distanceX = scaledX - closestX;
    const distanceY = scaledY - closestY;
    const distanceSquared = distanceX * distanceX + distanceY * distanceY;

    // If the bounding box intersects the circle, check mask pixels
    if (distanceSquared <= scaledRadius * scaledRadius) {
      // Sample points around the center within the region to see if any are in the mask
      let found = false;

      // Check center point first
      const localX = Math.floor(scaledX - bounds.x);
      const localY = Math.floor(scaledY - bounds.y);

      if (localX >= 0 && localX < region.mask.cols &&
          localY >= 0 && localY < region.mask.rows) {
        if (region.mask.ucharAt(localY, localX) > 0) {
          found = true;
        }
      }

      // If center isn't in mask, sample around the perimeter of the radius
      if (!found) {
        const samples = 8; // Sample 8 points around the circle
        for (let angle = 0; angle < Math.PI * 2; angle += (Math.PI * 2) / samples) {
          const sampleX = Math.floor(scaledX + Math.cos(angle) * scaledRadius * 0.5);
          const sampleY = Math.floor(scaledY + Math.sin(angle) * scaledRadius * 0.5);

          const sLocalX = Math.floor(sampleX - bounds.x);
          const sLocalY = Math.floor(sampleY - bounds.y);

          if (sLocalX >= 0 && sLocalX < region.mask.cols &&
              sLocalY >= 0 && sLocalY < region.mask.rows) {
            if (region.mask.ucharAt(sLocalY, sLocalX) > 0) {
              found = true;
              break;
            }
          }
        }
      }

      if (found) {
        foundRegions.push(i);
      }
    }
  }

  return foundRegions;
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

  // Fill selected regions with white (no outlines, just filled regions)
  for (let region of selectedRegions) {
    // Scale the contour to match original image size
    const scaleFactor = region.scaleFactor || 1;
    const scale = 1 / scaleFactor;
    const scaledContour = scaleContour(region.contour, scale, cv);

    const contourVec = new cv.MatVector();
    contourVec.push_back(scaledContour);
    // Use -1 to fill the contour completely (no outline)
    cv.drawContours(mask, contourVec, 0, new cv.Scalar(255), -1);
    contourVec.delete();
    scaledContour.delete();
  }

  // Apply binary threshold to ensure pure black (0) and white (255) with no gray values
  // This removes any anti-aliasing or interpolation artifacts
  cv.threshold(mask, mask, 127, 255, cv.THRESH_BINARY);

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
