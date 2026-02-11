/**
 * Segmentation utilities using OpenCV.js
 * Handles image segmentation, region detection, and mask generation
 */

/**
 * Union-Find (Disjoint Set) data structure for efficient region merging
 */
class UnionFind {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }
  find(x) {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]);
    }
    return this.parent[x];
  }
  union(a, b) {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return false;
    if (this.rank[ra] < this.rank[rb]) { this.parent[ra] = rb; }
    else if (this.rank[ra] > this.rank[rb]) { this.parent[rb] = ra; }
    else { this.parent[rb] = ra; this.rank[ra]++; }
    return true;
  }
}

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
export function segmentImage(originalImage, sensitivity, regionSize, cv, mergeThreshold = 10) {
  if (!originalImage || !cv) {
    throw new Error('Invalid image or OpenCV instance');
  }

  // Default regionSize to 20 if not provided (for backward compatibility)
  if (regionSize === undefined || regionSize === null) {
    regionSize = 20;
  }

  const regions = [];
  let gray, blurred, gradient, gradient3C, markers;
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

    // Step 1: Edge detection for boundary-aware watershed
    // Use both Sobel gradient and Canny for robust edge detection
    console.log('[Segmentation] Step 1: Computing edges');

    gray = new cv.Mat();
    cv.cvtColor(workingImage, gray, cv.COLOR_RGBA2GRAY);

    // Bilateral filter preserves edges while smoothing flat areas
    blurred = new cv.Mat();
    cv.bilateralFilter(gray, blurred, 5, 50, 50);

    // Sobel gradient for broad edge detection
    const gradX = new cv.Mat();
    const gradY = new cv.Mat();
    gradient = new cv.Mat();
    cv.Sobel(blurred, gradX, cv.CV_32F, 1, 0, 3);
    cv.Sobel(blurred, gradY, cv.CV_32F, 0, 1, 3);
    if (typeof cv.magnitude === 'function') {
      cv.magnitude(gradX, gradY, gradient);
    } else {
      const gx2 = new cv.Mat();
      const gy2 = new cv.Mat();
      cv.multiply(gradX, gradX, gx2);
      cv.multiply(gradY, gradY, gy2);
      cv.add(gx2, gy2, gradient);
      cv.sqrt(gradient, gradient);
      cleanupMats(gx2, gy2);
    }
    cleanupMats(gradX, gradY);
    cv.normalize(gradient, gradient, 0, 255, cv.NORM_MINMAX);
    gradient.convertTo(gradient, cv.CV_8U);

    // Canny edge detection for thin/subtle lines (lower thresholds to catch low-contrast edges)
    const canny = new cv.Mat();
    cv.Canny(blurred, canny, 30, 90);

    // Dilate Canny edges to make them thicker barriers
    const dilateKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    const cannyDilated = new cv.Mat();
    cv.dilate(canny, cannyDilated, dilateKernel);
    dilateKernel.delete();
    canny.delete();

    // Combine: take maximum of Sobel gradient and dilated Canny edges
    // This creates strong barriers at both broad gradients AND thin lines
    const combined = new cv.Mat();
    cv.max(gradient, cannyDilated, combined);
    cannyDilated.delete();

    console.log('[Segmentation] Edge detection complete');

    // Step 2: Create markers on a regular grid
    console.log('[Segmentation] Step 2: Creating grid markers');
    const maxRegionSize = Math.min(workingImage.cols, workingImage.rows) / regionSize;
    const baseSpacing = maxRegionSize * 0.7;

    const spacingMultiplier = 1.3 - (sensitivity / 10) * 1.1;
    const spacing = Math.max(10, Math.floor(baseSpacing * spacingMultiplier));

    console.log(`[Segmentation] Grid spacing: ${spacing}px, Max region size: ${maxRegionSize}px`);

    markers = cv.Mat.zeros(workingImage.rows, workingImage.cols, cv.CV_32S);
    let markerLabel = 1;

    for (let y = Math.floor(spacing / 2); y < workingImage.rows; y += spacing) {
      for (let x = Math.floor(spacing / 2); x < workingImage.cols; x += spacing) {
        markers.intPtr(Math.floor(y), Math.floor(x))[0] = markerLabel;
        markerLabel++;
      }
    }

    const markerCount = markerLabel - 1;
    console.log(`[Segmentation] Created ${markerCount} markers`);

    if (markerCount > 5000) {
      throw new Error(`Too many markers (${markerCount}). Try reducing detail level.`);
    }

    // Step 3: Apply watershed on the edge/gradient image
    // Using the gradient creates natural barriers at edges, so watershed
    // boundaries follow image edges even for low-contrast lines
    console.log('[Segmentation] Step 3: Applying watershed on edge-enhanced image');

    if (typeof cv.watershed !== 'function') {
      throw new Error('cv.watershed is not available in this OpenCV.js build.');
    }

    // Watershed requires 3-channel 8-bit image
    gradient3C = new cv.Mat();
    cv.cvtColor(combined, gradient3C, cv.COLOR_GRAY2BGR);
    combined.delete();

    cv.watershed(gradient3C, markers);

    console.log('[Segmentation] Watershed completed');

    // Step 4: Extract regions from watershed result
    // Watershed labels: -1 = boundary, 0 = background, >0 = region labels
    console.log('[Segmentation] Step 4: Extracting regions');

    // MEMORY-EFFICIENT APPROACH: Collect pixel coordinates first, then create masks only for valid regions
    // This avoids creating 2000+ full-sized masks which would consume ~5-10GB of memory
    const regionPoints = new Map(); // Map<label, Array<{x, y}>>
    const minArea = (workingImage.cols * workingImage.rows) * 0.0001; // 0.01% minimum
    const adjacencySet = new Set(); // Stores "min_max" strings for unique region adjacency pairs

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

      // Second pass: assign boundary/background pixels to nearest region and build adjacency graph
      console.log(`[Segmentation] Assigning ${boundaryPixels.length} boundary/background pixels to nearest regions...`);
      for (const pixel of boundaryPixels) {
        let nearestLabel = null;
        let minDist = Infinity;
        const neighborLabels = [];

        // Check neighboring pixels (8-connected) to find nearest region
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;

            const nx = pixel.x + dx;
            const ny = pixel.y + dy;

            if (nx >= 0 && nx < markers.cols && ny >= 0 && ny < markers.rows) {
              const neighborLabel = markers.intPtr(ny, nx)[0];
              if (neighborLabel > 0) {
                neighborLabels.push(neighborLabel);
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

        // Record adjacency: all distinct region labels around this boundary pixel are adjacent
        const uniqueNeighborLabels = [...new Set(neighborLabels)];
        for (let i = 0; i < uniqueNeighborLabels.length; i++) {
          for (let j = i + 1; j < uniqueNeighborLabels.length; j++) {
            const a = Math.min(uniqueNeighborLabels[i], uniqueNeighborLabels[j]);
            const b = Math.max(uniqueNeighborLabels[i], uniqueNeighborLabels[j]);
            adjacencySet.add(`${a}_${b}`);
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

    // Step 3.5: Redistribute pixels from small regions to ensure complete coverage
    console.log('[Segmentation] Step 3.5: Redistributing pixels from small regions to ensure 100% coverage');

    // Identify small regions that will be filtered out
    const smallRegions = [];
    const largeRegions = [];

    for (const [label, points] of regionPoints.entries()) {
      if (points.length < minArea) {
        smallRegions.push({ label, points });
      } else {
        largeRegions.push({ label, points });
      }
    }

    console.log(`[Segmentation] Found ${smallRegions.length} small regions to redistribute (${largeRegions.length} large regions)`);

    // Redistribute pixels from small regions to nearest large region
    for (const smallRegion of smallRegions) {
      for (const pixel of smallRegion.points) {
        // Find nearest large region by checking neighbors
        let nearestLabel = null;
        let searchRadius = 1;
        const maxSearchRadius = 50; // Maximum search distance

        // Expand search radius until we find a large region
        while (nearestLabel === null && searchRadius <= maxSearchRadius) {
          for (let dy = -searchRadius; dy <= searchRadius; dy++) {
            for (let dx = -searchRadius; dx <= searchRadius; dx++) {
              if (dx === 0 && dy === 0) continue;

              const nx = pixel.x + dx;
              const ny = pixel.y + dy;

              if (nx >= 0 && nx < markers.cols && ny >= 0 && ny < markers.rows) {
                const neighborLabel = markers.intPtr(ny, nx)[0];

                // Check if this neighbor belongs to a large region
                if (neighborLabel > 0) {
                  const neighborRegion = regionPoints.get(neighborLabel);
                  if (neighborRegion && neighborRegion.length >= minArea) {
                    nearestLabel = neighborLabel;
                    break;
                  }
                }
              }
            }
            if (nearestLabel !== null) break;
          }
          searchRadius++;
        }

        // Add pixel to nearest large region
        if (nearestLabel !== null) {
          regionPoints.get(nearestLabel).push({ x: pixel.x, y: pixel.y });
        } else if (largeRegions.length > 0) {
          // Fallback: add to first large region if no neighbor found
          const firstLargeLabel = largeRegions[0].label;
          regionPoints.get(firstLargeLabel).push({ x: pixel.x, y: pixel.y });
        }
      }

      // Remove the small region from the map
      regionPoints.delete(smallRegion.label);
    }

    console.log(`[Segmentation] After redistribution: ${regionPoints.size} regions remaining`);

    // Step 4: Compute average color for each region (for merging and Select Similar)
    console.log('[Segmentation] Step 4: Computing average colors for region merging');
    const regionColors = new Map();

    for (const [label, points] of regionPoints.entries()) {
      let totalR = 0, totalG = 0, totalB = 0;
      const sampleStep = Math.max(1, Math.floor(points.length / 200));
      let sampleCount = 0;
      for (let i = 0; i < points.length; i += sampleStep) {
        const pt = points[i];
        const pixelIdx = (pt.y * workingImage.cols + pt.x) * 4;
        totalR += workingImage.data[pixelIdx];
        totalG += workingImage.data[pixelIdx + 1];
        totalB += workingImage.data[pixelIdx + 2];
        sampleCount++;
      }
      if (sampleCount > 0) {
        regionColors.set(label, {
          r: totalR / sampleCount,
          g: totalG / sampleCount,
          b: totalB / sampleCount
        });
      }
    }

    // Step 5: Merge adjacent regions with similar colors
    console.log(`[Segmentation] Step 5: Merging similar regions (threshold: ${mergeThreshold})`);

    if (mergeThreshold > 0 && adjacencySet.size > 0) {
      const labels = Array.from(regionPoints.keys());
      const labelToIndex = new Map();
      labels.forEach((label, idx) => labelToIndex.set(label, idx));

      const uf = new UnionFind(labels.length);
      // Track pixel count per union-find group to prevent oversized merges
      const groupSize = labels.map(label => regionPoints.get(label).length);
      const maxMergedArea = workingImage.cols * workingImage.rows * 0.03; // 3% of image max

      for (const pairKey of adjacencySet) {
        const sepIdx = pairKey.indexOf('_');
        const a = parseInt(pairKey.substring(0, sepIdx));
        const b = parseInt(pairKey.substring(sepIdx + 1));

        if (!regionPoints.has(a) || !regionPoints.has(b)) continue;

        const colorA = regionColors.get(a);
        const colorB = regionColors.get(b);
        if (!colorA || !colorB) continue;

        const dist = Math.sqrt(
          (colorA.r - colorB.r) ** 2 +
          (colorA.g - colorB.g) ** 2 +
          (colorA.b - colorB.b) ** 2
        );

        if (dist < mergeThreshold) {
          const idxA = labelToIndex.get(a);
          const idxB = labelToIndex.get(b);
          if (idxA !== undefined && idxB !== undefined) {
            const rootA = uf.find(idxA);
            const rootB = uf.find(idxB);
            if (rootA !== rootB) {
              // Don't merge if combined size would be too large
              if (groupSize[rootA] + groupSize[rootB] > maxMergedArea) continue;
              uf.union(idxA, idxB);
              // Update group size on the new root
              const newRoot = uf.find(idxA);
              groupSize[newRoot] = groupSize[rootA] + groupSize[rootB];
            }
          }
        }
      }

      // Group regions by their union-find root
      const rootToLabels = new Map();
      for (let i = 0; i < labels.length; i++) {
        const root = uf.find(i);
        if (!rootToLabels.has(root)) {
          rootToLabels.set(root, []);
        }
        rootToLabels.get(root).push(labels[i]);
      }

      // Merge pixel arrays and recompute average colors
      const mergedRegionPoints = new Map();
      for (const [root, memberLabels] of rootToLabels.entries()) {
        const primaryLabel = memberLabels[0];
        const allPoints = [];
        let totalR = 0, totalG = 0, totalB = 0, totalWeight = 0;
        for (const label of memberLabels) {
          const pts = regionPoints.get(label);
          allPoints.push(...pts);
          const color = regionColors.get(label);
          if (color) {
            totalR += color.r * pts.length;
            totalG += color.g * pts.length;
            totalB += color.b * pts.length;
            totalWeight += pts.length;
          }
        }
        mergedRegionPoints.set(primaryLabel, allPoints);
        if (totalWeight > 0) {
          regionColors.set(primaryLabel, {
            r: totalR / totalWeight,
            g: totalG / totalWeight,
            b: totalB / totalWeight
          });
        }
      }

      const premergeCount = regionPoints.size;
      regionPoints.clear();
      for (const [label, points] of mergedRegionPoints.entries()) {
        regionPoints.set(label, points);
      }

      console.log(`[Segmentation] Merged ${premergeCount} regions down to ${regionPoints.size}`);
    }

    // Build post-merge adjacency map for Select Similar feature
    const mergedAdjacency = new Map();
    const survivingLabels = new Set(regionPoints.keys());
    for (const pairKey of adjacencySet) {
      const sepIdx = pairKey.indexOf('_');
      const a = parseInt(pairKey.substring(0, sepIdx));
      const b = parseInt(pairKey.substring(sepIdx + 1));
      if (survivingLabels.has(a) && survivingLabels.has(b) && a !== b) {
        if (!mergedAdjacency.has(a)) mergedAdjacency.set(a, new Set());
        if (!mergedAdjacency.has(b)) mergedAdjacency.set(b, new Set());
        mergedAdjacency.get(a).add(b);
        mergedAdjacency.get(b).add(a);
      }
    }

    // Step 6: Create masks and contours only for regions that meet minimum area
    console.log('[Segmentation] Step 6: Creating masks and contours for valid regions');

    let processedCount = 0;
    for (const [label, points] of regionPoints.entries()) {
      try {
        // At this point, all regions should be large enough (small ones were redistributed)
        if (points.length < minArea) {
          console.warn(`[Segmentation] WARNING: Found small region after redistribution (${points.length} < ${minArea})`);
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

          // Since we already redistributed small regions, we should keep this one
          // Only skip if contour extraction genuinely failed
          if (area > 0) {
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
              selected: false,
              label: label,
              avgColor: regionColors.get(label) || { r: 128, g: 128, b: 128 },
              adjacentIndices: [] // Will be populated after all regions are created
            });

            // Don't delete translatedContour - it's now owned by the region!
          } else {
            console.warn(`[Segmentation] Skipping region ${label} with zero contour area`);
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

    // Populate adjacentIndices: map labels to final region indices
    const labelToRegionIndex = new Map();
    for (let i = 0; i < regions.length; i++) {
      labelToRegionIndex.set(regions[i].label, i);
    }
    for (let i = 0; i < regions.length; i++) {
      const adj = mergedAdjacency.get(regions[i].label);
      if (adj) {
        for (const neighborLabel of adj) {
          const neighborIdx = labelToRegionIndex.get(neighborLabel);
          if (neighborIdx !== undefined) {
            regions[i].adjacentIndices.push(neighborIdx);
          }
        }
      }
    }

  } catch (error) {
    console.error('[Segmentation] Error:', error);
    // Clean up on error
    cleanupMats(gray, blurred, gradient, markers, gradient3C);
    if (scaleFactor !== 1 && workingImage) {
      cleanupMats(workingImage);
    }
    throw error;
  }

  // Cleanup temporary matrices
  cleanupMats(gray, blurred, gradient, markers, gradient3C);
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

  // Draw semi-transparent fills for selected and hovered regions
  const hasHighlightOrSelected = highlightIndex >= 0 || regions.some(r => r.selected);
  if (hasHighlightOrSelected) {
    const overlay = display.clone();

    for (let i = 0; i < regions.length; i++) {
      const region = regions[i];
      if (!region.selected && i !== highlightIndex) continue;

      const scaleFactor = region.scaleFactor || 1;
      const scale = 1 / scaleFactor;
      const scaledContour = scaleContour(region.contour, scale, cv);
      const contourVec = new cv.MatVector();
      contourVec.push_back(scaledContour);

      if (i === highlightIndex) {
        cv.drawContours(overlay, contourVec, 0, new cv.Scalar(255, 255, 0, 255), -1);
      } else if (region.selected) {
        cv.drawContours(overlay, contourVec, 0, new cv.Scalar(0, 200, 0, 255), -1);
      }

      contourVec.delete();
      scaledContour.delete();
    }

    cv.addWeighted(overlay, 0.25, display, 0.75, 0, display);
    overlay.delete();
  }

  // Draw contour outlines on top of fills
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

/**
 * Select all regions connected to a start region that have similar colors
 * Uses BFS through the adjacency graph, comparing against the start region's color
 *
 * @param {number} startRegionIndex - Index of the region to start from
 * @param {Array} regions - Array of region objects with avgColor and adjacentIndices
 * @param {number} threshold - Maximum RGB distance to consider "similar"
 * @returns {Array<number>} Array of region indices to select
 */
export function selectSimilarRegions(startRegionIndex, regions, threshold) {
  const startRegion = regions[startRegionIndex];
  if (!startRegion || !startRegion.avgColor) return [startRegionIndex];

  const startColor = startRegion.avgColor;
  const visited = new Set();
  const toSelect = [];
  const queue = [startRegionIndex];
  visited.add(startRegionIndex);

  while (queue.length > 0) {
    const current = queue.shift();
    toSelect.push(current);

    const adjacentIndices = regions[current].adjacentIndices || [];
    for (const neighborIdx of adjacentIndices) {
      if (visited.has(neighborIdx)) continue;
      visited.add(neighborIdx);

      const neighbor = regions[neighborIdx];
      if (!neighbor || !neighbor.avgColor) continue;

      const dist = Math.sqrt(
        (neighbor.avgColor.r - startColor.r) ** 2 +
        (neighbor.avgColor.g - startColor.g) ** 2 +
        (neighbor.avgColor.b - startColor.b) ** 2
      );

      if (dist < threshold) {
        queue.push(neighborIdx);
      }
    }
  }

  return toSelect;
}
