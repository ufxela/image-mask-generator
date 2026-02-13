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
  let gray, blurred, gradient, gradient3C, markers, edgeMap;
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
    console.log('[Segmentation] Step 1: Computing edges');

    gray = new cv.Mat();
    cv.cvtColor(workingImage, gray, cv.COLOR_RGBA2GRAY);

    // Bilateral filter preserves edges while smoothing flat areas
    blurred = new cv.Mat();
    cv.bilateralFilter(gray, blurred, 5, 50, 50);

    // Sobel gradient — smooth, clean edges for the binary edge mask
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

    // Single Canny pass on bilateral-filtered image for the Step 1 combined map
    const cannyStep1 = new cv.Mat();
    cv.Canny(blurred, cannyStep1, 50, 150);

    // This is the basic combined map — will be enhanced in Step 1.5
    const combinedBasic = new cv.Mat();
    cv.max(gradient, cannyStep1, combinedBasic);
    cannyStep1.delete();

    console.log('[Segmentation] Edge detection complete');

    // Save edge map for other features
    edgeMap = combinedBasic.clone();
    combinedBasic.delete();

    const imgCols = workingImage.cols;
    const imgRows = workingImage.rows;

    // Step 1.5: Create binary edge mask
    // Use gradient + Canny on grayscale AND individual color channels for text detection
    console.log('[Segmentation] Step 1.5: Creating binary edge mask');

    // Canny on raw grayscale
    const cannyGray = new cv.Mat();
    cv.Canny(gray, cannyGray, 15, 60);

    // Canny on individual color channels to catch colored text/features
    const channels = new cv.MatVector();
    cv.split(workingImage, channels);
    const cannyR = new cv.Mat();
    const cannyG = new cv.Mat();
    const cannyB = new cv.Mat();
    cv.Canny(channels.get(0), cannyR, 15, 50); // Lower thresholds for text
    cv.Canny(channels.get(1), cannyG, 15, 50);
    cv.Canny(channels.get(2), cannyB, 15, 50);
    channels.delete();

    // Combine all Canny edges first, then close gaps
    const cannyAll = new cv.Mat();
    cv.max(cannyGray, cannyR, cannyAll);
    cv.max(cannyAll, cannyG, cannyAll);
    cv.max(cannyAll, cannyB, cannyAll);
    cannyGray.delete();
    cannyR.delete();
    cannyG.delete();
    cannyB.delete();

    // Morphological close: seal gaps in edge chains for the WATERSHED BARRIER
    // (Not used in the edge mask — we want thin edges for precise distance transform)
    const closeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    const cannyClosed = new cv.Mat();
    cv.morphologyEx(cannyAll, cannyClosed, cv.MORPH_CLOSE, closeKernel);
    closeKernel.delete();

    // Edge mask input: gradient + RAW Canny (thin, precise for distance transform)
    const edgeMaskInput = new cv.Mat();
    cv.max(gradient, cannyAll, edgeMaskInput);
    cannyAll.delete();

    // Watershed barrier: gradient + CLOSED Canny (thick, leak-proof for watershed)
    const combined = new cv.Mat();
    cv.max(gradient, cannyClosed, combined);
    cannyClosed.delete();

    const edgeMask = new cv.Mat();
    cv.threshold(edgeMaskInput, edgeMask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    edgeMaskInput.delete();

    let edgeCount = 0;
    for (let i = 0; i < imgCols * imgRows; i++) {
      if (edgeMask.data[i] > 0) edgeCount++;
    }
    let edgePct = edgeCount / (imgCols * imgRows);
    console.log(`[Segmentation] Edge mask: ${(edgePct * 100).toFixed(1)}% edge pixels`);

    // Step 2: Distance transform and marker generation
    console.log('[Segmentation] Step 2: Distance transform for marker placement');

    // Invert edge mask: interior regions (non-edge) become foreground
    const interior = new cv.Mat();
    cv.bitwise_not(edgeMask, interior);
    edgeMask.delete();

    // Distance transform: each pixel gets its distance to nearest edge
    const dist = new cv.Mat();
    cv.distanceTransform(interior, dist, cv.DIST_L2, 5);
    interior.delete();

    // Map sensitivity (1-20) to distance threshold in pixels
    // Sweet spot: high enough that blobs don't merge across thin edges,
    // low enough that small features (text characters) get markers
    // sensitivity=1 → threshold=2.0
    // sensitivity=5 (default) → threshold=1.5
    // sensitivity=10 → threshold=1.1
    // sensitivity=20 → threshold=0.8
    const rawThreshold = Math.max(0.8, 1.5 - (sensitivity - 1) * (0.7 / 19));
    console.log(`[Segmentation] Distance threshold: ${rawThreshold.toFixed(2)}px (sensitivity=${sensitivity})`);

    // Threshold distance transform to find region interiors
    const foreground = new cv.Mat();
    cv.threshold(dist, foreground, rawThreshold, 255, cv.THRESH_BINARY);
    foreground.convertTo(foreground, cv.CV_8U);
    dist.delete();

    // Find contours of each foreground blob — each becomes a marker
    const fgContours = new cv.MatVector();
    const fgHierarchy = new cv.Mat();
    cv.findContours(foreground, fgContours, fgHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    foreground.delete();
    fgHierarchy.delete();

    // Create marker matrix: place a single point at each blob's centroid
    // (Not filled contours — filling would create huge single-label regions in flat areas)
    markers = cv.Mat.zeros(imgRows, imgCols, cv.CV_32S);
    let markerCount = fgContours.size();
    console.log(`[Segmentation] Found ${markerCount} marker regions from distance transform`);

    if (markerCount > 100000) {
      fgContours.delete();
      throw new Error(`Too many markers (${markerCount}). Try reducing detail level.`);
    }

    if (markerCount === 0) {
      fgContours.delete();
      throw new Error('No markers generated. Try increasing detail level or using a different image.');
    }

    const markerGrid = markers.data32S;
    for (let i = 0; i < markerCount; i++) {
      const contour = fgContours.get(i);
      // Compute centroid of the contour
      const moments = cv.moments(contour);
      if (moments.m00 > 0) {
        const cx = Math.round(moments.m10 / moments.m00);
        const cy = Math.round(moments.m01 / moments.m00);
        if (cx >= 0 && cx < imgCols && cy >= 0 && cy < imgRows) {
          markerGrid[cy * imgCols + cx] = i + 1;
        }
      } else if (contour.rows > 0) {
        // Fallback: use first point of contour
        const px = contour.data32S[0];
        const py = contour.data32S[1];
        if (px >= 0 && px < imgCols && py >= 0 && py < imgRows) {
          markerGrid[py * imgCols + px] = i + 1;
        }
      }
    }
    fgContours.delete();

    // Add supplementary grid markers to subdivide large uniform areas
    // Only place markers where there are NO edges nearby (to avoid straddling feature boundaries)
    let nextLabel = markerCount + 1;
    const maxGap = Math.max(8, Math.round(Math.min(imgCols, imgRows) / 60));
    const combinedEdgeData = combined.data;
    let supplementary = 0;
    for (let y = Math.floor(maxGap / 2); y < imgRows; y += maxGap) {
      for (let x = Math.floor(maxGap / 2); x < imgCols; x += maxGap) {
        if (markerGrid[y * imgCols + x] !== 0) continue;

        // Check if any edge exists within 2px radius
        let nearEdge = false;
        for (let dy = -2; dy <= 2 && !nearEdge; dy++) {
          for (let dx = -2; dx <= 2 && !nearEdge; dx++) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < imgCols && ny >= 0 && ny < imgRows) {
              if (combinedEdgeData[ny * imgCols + nx] > 30) nearEdge = true;
            }
          }
        }

        if (!nearEdge) {
          markerGrid[y * imgCols + x] = nextLabel++;
          supplementary++;
        }
      }
    }
    markerCount = nextLabel - 1;
    console.log(`[Segmentation] Added ${supplementary} supplementary grid markers (total: ${markerCount})`);

    if (markerCount > 100000) {
      throw new Error(`Too many markers (${markerCount}). Try reducing detail level.`);
    }

    // Step 2.5: Boost edge barriers for watershed
    const boosted = new cv.Mat();
    combined.convertTo(boosted, cv.CV_8U, 2.0, 0);
    combined.delete();

    // Step 3: Apply watershed
    console.log('[Segmentation] Step 3: Applying watershed');

    if (typeof cv.watershed !== 'function') {
      throw new Error('cv.watershed is not available in this OpenCV.js build.');
    }

    gradient3C = new cv.Mat();
    cv.cvtColor(boosted, gradient3C, cv.COLOR_GRAY2BGR);
    boosted.delete();

    cv.watershed(gradient3C, markers);

    console.log('[Segmentation] Watershed completed');

    // Step 4: Extract regions from watershed result using direct typed array access
    console.log('[Segmentation] Step 4: Extracting regions');

    const markerResult = markers.data32S;
    const totalPixels = imgCols * imgRows;
    const minArea = Math.max(10, totalPixels * 0.00002); // ~15px for 736x1042, keeps text/fine features

    // Use a label map: for each pixel, store which label it belongs to (after boundary assignment)
    // This avoids storing {x,y} objects and uses flat arrays
    const pixelLabels = new Int32Array(totalPixels); // Final label for each pixel
    const boundaryIndices = []; // Flat indices of boundary/background pixels

    // First pass: classify pixels
    for (let i = 0; i < totalPixels; i++) {
      const label = markerResult[i];
      if (label > 0) {
        pixelLabels[i] = label;
      } else {
        pixelLabels[i] = 0; // boundary/background, to be assigned
        boundaryIndices.push(i);
      }
    }

    // Build adjacency using numeric encoding
    const adjacencyPairs = new Set(); // Stores a*65536+b (a < b)

    // Second pass: assign boundary pixels to nearest region and build adjacency
    console.log(`[Segmentation] Assigning ${boundaryIndices.length} boundary pixels...`);
    for (const idx of boundaryIndices) {
      const px = idx % imgCols;
      const py = (idx - px) / imgCols;
      let nearestLabel = 0;
      let minDist = Infinity;
      let label1 = 0, label2 = 0; // Track up to 2 distinct neighbor labels for adjacency

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = px + dx;
          const ny = py + dy;
          if (nx >= 0 && nx < imgCols && ny >= 0 && ny < imgRows) {
            const neighborLabel = markerResult[ny * imgCols + nx];
            if (neighborLabel > 0) {
              const dist = Math.abs(dx) + Math.abs(dy);
              if (dist < minDist) {
                minDist = dist;
                nearestLabel = neighborLabel;
              }
              // Track adjacency
              if (label1 === 0) label1 = neighborLabel;
              else if (neighborLabel !== label1 && label2 === 0) label2 = neighborLabel;
              else if (neighborLabel !== label1 && neighborLabel !== label2) {
                // More than 2 labels: add pairs for all combinations
                const a = Math.min(label1, neighborLabel);
                const b = Math.max(label1, neighborLabel);
                adjacencyPairs.add(a * 65536 + b);
                const a2 = Math.min(label2, neighborLabel);
                const b2 = Math.max(label2, neighborLabel);
                adjacencyPairs.add(a2 * 65536 + b2);
              }
            }
          }
        }
      }

      if (label1 > 0 && label2 > 0) {
        const a = Math.min(label1, label2);
        const b = Math.max(label1, label2);
        adjacencyPairs.add(a * 65536 + b);
      }

      pixelLabels[idx] = nearestLabel > 0 ? nearestLabel : 1; // fallback to label 1
    }

    // Count pixels per label
    const labelCounts = new Map();
    for (let i = 0; i < totalPixels; i++) {
      const l = pixelLabels[i];
      if (l > 0) {
        labelCounts.set(l, (labelCounts.get(l) || 0) + 1);
      }
    }

    console.log(`[Segmentation] Found ${labelCounts.size} regions from watershed`);

    // Step 3.5: Redistribute small regions using label map (fast: no expanding search)
    console.log('[Segmentation] Step 3.5: Redistributing small regions');
    const largeLabels = new Set();
    const smallLabels = new Set();
    for (const [label, count] of labelCounts) {
      if (count >= minArea) largeLabels.add(label);
      else smallLabels.add(label);
    }

    console.log(`[Segmentation] ${smallLabels.size} small regions to redistribute`);

    if (smallLabels.size > 0) {
      // For each small region pixel, find the nearest large-region pixel via BFS-like scan
      // Simple approach: scan neighbors of each small pixel and reassign to adjacent large region
      let reassigned = 0;
      for (let pass = 0; pass < 5 && smallLabels.size > 0; pass++) {
        for (let i = 0; i < totalPixels; i++) {
          const l = pixelLabels[i];
          if (!smallLabels.has(l)) continue;

          const px = i % imgCols;
          const py = (i - px) / imgCols;

          // Check 8-connected neighbors for a large region
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              const nx = px + dx;
              const ny = py + dy;
              if (nx >= 0 && nx < imgCols && ny >= 0 && ny < imgRows) {
                const nl = pixelLabels[ny * imgCols + nx];
                if (largeLabels.has(nl)) {
                  pixelLabels[i] = nl;
                  labelCounts.set(nl, (labelCounts.get(nl) || 0) + 1);
                  const oldCount = labelCounts.get(l) - 1;
                  labelCounts.set(l, oldCount);
                  reassigned++;
                  dy = 2; // break outer
                  break;
                }
              }
            }
          }
        }
        // After each pass, check if any small labels became empty
        for (const sl of [...smallLabels]) {
          if ((labelCounts.get(sl) || 0) <= 0) {
            smallLabels.delete(sl);
            labelCounts.delete(sl);
          }
        }
      }
      // Any remaining small region pixels: assign to nearest large region (fallback)
      if (smallLabels.size > 0) {
        const fallbackLabel = largeLabels.values().next().value || 1;
        for (let i = 0; i < totalPixels; i++) {
          if (smallLabels.has(pixelLabels[i])) {
            pixelLabels[i] = fallbackLabel;
          }
        }
        for (const sl of smallLabels) labelCounts.delete(sl);
      }
      console.log(`[Segmentation] Reassigned ${reassigned} pixels from small regions`);
    }

    // Step 4b: Compute average color for each region
    console.log('[Segmentation] Computing average colors');
    const regionColors = new Map();
    const colorAccum = new Map(); // label -> {r, g, b, count}
    const imgData = workingImage.data;

    for (let i = 0; i < totalPixels; i++) {
      const l = pixelLabels[i];
      if (l <= 0 || !labelCounts.has(l)) continue;
      let acc = colorAccum.get(l);
      if (!acc) {
        acc = { r: 0, g: 0, b: 0, count: 0 };
        colorAccum.set(l, acc);
      }
      // Sample at most ~200 pixels per region
      if (acc.count < 200 || Math.random() < 200 / labelCounts.get(l)) {
        const pi = i * 4;
        acc.r += imgData[pi];
        acc.g += imgData[pi + 1];
        acc.b += imgData[pi + 2];
        acc.count++;
      }
    }
    for (const [label, acc] of colorAccum) {
      if (acc.count > 0) {
        regionColors.set(label, { r: acc.r / acc.count, g: acc.g / acc.count, b: acc.b / acc.count });
      }
    }

    // Step 5: Merge adjacent regions with similar colors
    console.log(`[Segmentation] Step 5: Merging similar regions (threshold: ${mergeThreshold})`);

    // Convert adjacencyPairs to a label-based adjacency map
    const adjacencyMap = new Map();
    for (const pair of adjacencyPairs) {
      const a = Math.floor(pair / 65536);
      const b = pair % 65536;
      if (!labelCounts.has(a) || !labelCounts.has(b)) continue;
      if (!adjacencyMap.has(a)) adjacencyMap.set(a, []);
      if (!adjacencyMap.has(b)) adjacencyMap.set(b, []);
      adjacencyMap.get(a).push(b);
      adjacencyMap.get(b).push(a);
    }

    // Merge using union-find
    const mergeLabels = Array.from(labelCounts.keys());
    const labelToMergeIdx = new Map();
    mergeLabels.forEach((l, i) => labelToMergeIdx.set(l, i));

    if (mergeThreshold > 0) {
      const uf = new UnionFind(mergeLabels.length);
      const groupSize = mergeLabels.map(l => labelCounts.get(l));
      const maxMergedArea = totalPixels * 0.005; // 0.5% of image (was 3%)
      const maxMergedWidth = Math.floor(imgCols / 80); // ~9px for 736px wide (was /40)
      const maxMergedHeight = Math.floor(imgRows / 80); // ~13px for 1042px tall (was /40)

      // Compute bounding boxes per label by scanning pixelLabels once
      const labelBounds = new Map();
      for (let i = 0; i < totalPixels; i++) {
        const l = pixelLabels[i];
        if (!labelCounts.has(l)) continue;
        const px = i % imgCols;
        const py = (i - px) / imgCols;
        let b = labelBounds.get(l);
        if (!b) {
          b = { minX: px, minY: py, maxX: px, maxY: py };
          labelBounds.set(l, b);
        } else {
          if (px < b.minX) b.minX = px;
          if (px > b.maxX) b.maxX = px;
          if (py < b.minY) b.minY = py;
          if (py > b.maxY) b.maxY = py;
        }
      }

      const groupBounds = mergeLabels.map(l => {
        const b = labelBounds.get(l);
        return b || { minX: 0, minY: 0, maxX: 0, maxY: 0 };
      });

      for (const pair of adjacencyPairs) {
        const a = Math.floor(pair / 65536);
        const b = pair % 65536;
        const idxA = labelToMergeIdx.get(a);
        const idxB = labelToMergeIdx.get(b);
        if (idxA === undefined || idxB === undefined) continue;

        const colorA = regionColors.get(a);
        const colorB = regionColors.get(b);
        if (!colorA || !colorB) continue;

        const dist = Math.sqrt((colorA.r - colorB.r) ** 2 + (colorA.g - colorB.g) ** 2 + (colorA.b - colorB.b) ** 2);
        if (dist < mergeThreshold) {
          const rootA = uf.find(idxA);
          const rootB = uf.find(idxB);
          if (rootA !== rootB) {
            if (groupSize[rootA] + groupSize[rootB] > maxMergedArea) continue;
            const cb = {
              minX: Math.min(groupBounds[rootA].minX, groupBounds[rootB].minX),
              minY: Math.min(groupBounds[rootA].minY, groupBounds[rootB].minY),
              maxX: Math.max(groupBounds[rootA].maxX, groupBounds[rootB].maxX),
              maxY: Math.max(groupBounds[rootA].maxY, groupBounds[rootB].maxY),
            };
            if (cb.maxX - cb.minX > maxMergedWidth) continue;
            if (cb.maxY - cb.minY > maxMergedHeight) continue;
            uf.union(idxA, idxB);
            const newRoot = uf.find(idxA);
            groupSize[newRoot] = groupSize[rootA] + groupSize[rootB];
            groupBounds[newRoot] = cb;
          }
        }
      }

      // Remap pixelLabels to use root labels
      const indexToRoot = mergeLabels.map((l, i) => mergeLabels[uf.find(i)]);
      const labelRemap = new Map();
      mergeLabels.forEach((l, i) => labelRemap.set(l, indexToRoot[i]));

      const preMergeCount = labelCounts.size;
      for (let i = 0; i < totalPixels; i++) {
        const remapped = labelRemap.get(pixelLabels[i]);
        if (remapped !== undefined) pixelLabels[i] = remapped;
      }

      // Recompute label counts and colors after merge
      labelCounts.clear();
      for (let i = 0; i < totalPixels; i++) {
        const l = pixelLabels[i];
        labelCounts.set(l, (labelCounts.get(l) || 0) + 1);
      }

      // Recompute average colors for merged regions
      colorAccum.clear();
      for (let i = 0; i < totalPixels; i++) {
        const l = pixelLabels[i];
        if (!labelCounts.has(l)) continue;
        let acc = colorAccum.get(l);
        if (!acc) { acc = { r: 0, g: 0, b: 0, count: 0 }; colorAccum.set(l, acc); }
        if (acc.count < 200) {
          const pi = i * 4;
          acc.r += imgData[pi]; acc.g += imgData[pi + 1]; acc.b += imgData[pi + 2];
          acc.count++;
        }
      }
      regionColors.clear();
      for (const [label, acc] of colorAccum) {
        if (acc.count > 0) {
          regionColors.set(label, { r: acc.r / acc.count, g: acc.g / acc.count, b: acc.b / acc.count });
        }
      }

      console.log(`[Segmentation] Merged ${preMergeCount} regions down to ${labelCounts.size}`);
    }

    // Rebuild adjacency after merge
    const mergedAdjacency = new Map();
    for (const pair of adjacencyPairs) {
      const a = Math.floor(pair / 65536);
      const b = pair % 65536;
      // Map to post-merge labels
      const remappedA = pixelLabels.length > 0 ? a : a; // already remapped in pixelLabels
      const remappedB = pixelLabels.length > 0 ? b : b;
      if (labelCounts.has(a) && labelCounts.has(b) && a !== b) {
        if (!mergedAdjacency.has(a)) mergedAdjacency.set(a, new Set());
        if (!mergedAdjacency.has(b)) mergedAdjacency.set(b, new Set());
        mergedAdjacency.get(a).add(b);
        mergedAdjacency.get(b).add(a);
      }
    }

    // Step 6: Create masks and contours using direct typed array access
    console.log('[Segmentation] Step 6: Creating masks and contours');

    let processedCount = 0;
    const finalLabels = Array.from(labelCounts.keys()).filter(l => labelCounts.get(l) >= minArea);

    for (const label of finalLabels) {
      // Compute bounding box by scanning pixelLabels
      let minX = imgCols, minY = imgRows, maxX = 0, maxY = 0;
      for (let i = 0; i < totalPixels; i++) {
        if (pixelLabels[i] !== label) continue;
        const px = i % imgCols;
        const py = (i - px) / imgCols;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      }

      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      if (bw <= 0 || bh <= 0) continue;

      const bounds = { x: minX, y: minY, width: bw, height: bh };

      // Create cropped mask using direct data access
      const mask = cv.Mat.zeros(bh, bw, cv.CV_8U);
      const maskData = mask.data;
      for (let i = 0; i < totalPixels; i++) {
        if (pixelLabels[i] !== label) continue;
        const px = i % imgCols;
        const py = (i - px) / imgCols;
        maskData[(py - minY) * bw + (px - minX)] = 255;
      }

      // Find contours
      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      if (contours.size() > 0) {
        const contour = contours.get(0);
        if (contour && contour.rows > 0) {
          const area = cv.contourArea(contour);
          if (area > 0) {
            // Translate contour to image space
            const translatedContour = new cv.Mat(contour.rows, 1, cv.CV_32SC2);
            for (let i = 0; i < contour.rows; i++) {
              translatedContour.data32S[i * 2] = contour.data32S[i * 2] + minX;
              translatedContour.data32S[i * 2 + 1] = contour.data32S[i * 2 + 1] + minY;
            }

            regions.push({
              contour: translatedContour,
              mask: mask, // Transfer ownership (no clone needed)
              bounds: bounds,
              scaleFactor: scaleFactor,
              selected: false,
              label: label,
              avgColor: regionColors.get(label) || { r: 128, g: 128, b: 128 },
              adjacentIndices: []
            });

            hierarchy.delete();
            contours.delete();
            processedCount++;
            if (processedCount % 200 === 0) {
              console.log(`[Segmentation] Processed ${processedCount}/${finalLabels.length} regions`);
            }
            continue; // Skip the cleanup below since mask is transferred
          }
        }
      }

      // Cleanup if region wasn't added
      hierarchy.delete();
      contours.delete();
      mask.delete();
      processedCount++;
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

  return { regions, edgeMap };
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

    // Always include the region directly under the cursor
    const localCursorX = Math.floor(scaledX - bounds.x);
    const localCursorY = Math.floor(scaledY - bounds.y);
    if (localCursorX >= 0 && localCursorX < region.mask.cols &&
        localCursorY >= 0 && localCursorY < region.mask.rows &&
        region.mask.ucharAt(localCursorY, localCursorX) > 0) {
      foundRegions.push(i);
      continue;
    }

    // Check if the region's bounding box intersects with the circle
    // Find the closest point on the bounding box to the circle center
    const closestX = Math.max(bounds.x, Math.min(scaledX, bounds.x + bounds.width));
    const closestY = Math.max(bounds.y, Math.min(scaledY, bounds.y + bounds.height));

    // Calculate distance from circle center to this closest point
    const distanceX = scaledX - closestX;
    const distanceY = scaledY - closestY;
    const distanceSquared = distanceX * distanceX + distanceY * distanceY;

    // If the bounding box intersects the circle, check what fraction of the region is inside
    if (distanceSquared <= scaledRadius * scaledRadius) {
      let insideCount = 0;
      let totalCount = 0;

      // Sample the region's mask to estimate coverage
      const step = Math.max(1, Math.floor(Math.max(bounds.width, bounds.height) / 20));

      for (let ly = 0; ly < region.mask.rows; ly += step) {
        for (let lx = 0; lx < region.mask.cols; lx += step) {
          if (region.mask.ucharAt(ly, lx) > 0) {
            totalCount++;
            // Check if this mask pixel is within the circle
            const px = bounds.x + lx;
            const py = bounds.y + ly;
            const dx = px - scaledX;
            const dy = py - scaledY;
            if (dx * dx + dy * dy <= scaledRadius * scaledRadius) {
              insideCount++;
            }
          }
        }
      }

      // Include region if >= 90% of its area is inside the circle
      if (totalCount > 0 && (insideCount / totalCount) >= 0.9) {
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

/**
 * Densify a contour by inserting interpolated points between vertices
 * that are too far apart. Needed for smooth boundary editing.
 *
 * @param {cv.Mat} contour - Contour in CV_32SC2 format
 * @param {number} maxSpacing - Maximum distance between consecutive points
 * @param {object} cv - OpenCV instance
 * @returns {cv.Mat} New densified contour (caller must delete)
 */
export function densifyContour(contour, maxSpacing, cv) {
  if (!contour || contour.rows < 2) return contour.clone();

  const newPoints = [];
  for (let i = 0; i < contour.rows; i++) {
    const x1 = contour.data32S[i * 2];
    const y1 = contour.data32S[i * 2 + 1];
    newPoints.push({ x: x1, y: y1 });

    const nextI = (i + 1) % contour.rows;
    const x2 = contour.data32S[nextI * 2];
    const y2 = contour.data32S[nextI * 2 + 1];

    const dist = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    if (dist > maxSpacing) {
      const steps = Math.ceil(dist / maxSpacing);
      for (let s = 1; s < steps; s++) {
        const t = s / steps;
        newPoints.push({
          x: Math.round(x1 + (x2 - x1) * t),
          y: Math.round(y1 + (y2 - y1) * t)
        });
      }
    }
  }

  const result = new cv.Mat(newPoints.length, 1, cv.CV_32SC2);
  for (let i = 0; i < newPoints.length; i++) {
    result.data32S[i * 2] = newPoints[i].x;
    result.data32S[i * 2 + 1] = newPoints[i].y;
  }
  return result;
}

/**
 * Rebuild a region's mask from its contour.
 * Call after modifying contour points.
 *
 * @param {object} region - Region object with contour, bounds, mask
 * @param {object} cv - OpenCV instance
 */
export function rebuildRegionMask(region, cv) {
  // Recompute bounds from contour
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < region.contour.rows; i++) {
    const x = region.contour.data32S[i * 2];
    const y = region.contour.data32S[i * 2 + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const newBounds = {
    x: minX,
    y: minY,
    width: maxX - minX + 1,
    height: maxY - minY + 1
  };

  // Create new mask from contour
  if (region.mask && !region.mask.isDeleted()) {
    region.mask.delete();
  }
  const newMask = cv.Mat.zeros(newBounds.height, newBounds.width, cv.CV_8U);

  // Offset contour to local coordinates
  const localContour = new cv.Mat(region.contour.rows, 1, cv.CV_32SC2);
  for (let i = 0; i < region.contour.rows; i++) {
    localContour.data32S[i * 2] = region.contour.data32S[i * 2] - newBounds.x;
    localContour.data32S[i * 2 + 1] = region.contour.data32S[i * 2 + 1] - newBounds.y;
  }

  const contourVec = new cv.MatVector();
  contourVec.push_back(localContour);
  cv.drawContours(newMask, contourVec, 0, new cv.Scalar(255), -1);
  contourVec.delete();
  localContour.delete();

  region.bounds = newBounds;
  region.mask = newMask;
}

/**
 * Split selected regions into finer sub-segments by re-running segmentation
 * on the cropped area containing those regions.
 *
 * @param {cv.Mat} originalImage - The full original image
 * @param {Array} regions - All current regions
 * @param {Array<number>} selectedIndices - Indices of regions to split
 * @param {Object} cv - OpenCV instance
 * @param {number} sensitivity - Sensitivity for sub-segmentation
 * @param {number} regionSize - Region size for sub-segmentation
 * @param {number} mergeThreshold - Merge threshold for sub-segmentation
 * @returns {Array} Updated regions array with selected regions replaced by finer sub-regions
 */
export function splitRegions(originalImage, regions, selectedIndices, cv, sensitivity, regionSize, mergeThreshold) {
  if (!selectedIndices || selectedIndices.length === 0) return regions;

  const selectedRegions = selectedIndices.map(i => regions[i]);
  const existingScaleFactor = selectedRegions[0].scaleFactor || 1;

  // Step 1: Compute combined bounding box in original image coords
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const scale = 1 / existingScaleFactor;
  for (const region of selectedRegions) {
    const b = region.bounds;
    minX = Math.min(minX, Math.round(b.x * scale));
    minY = Math.min(minY, Math.round(b.y * scale));
    maxX = Math.max(maxX, Math.round((b.x + b.width) * scale));
    maxY = Math.max(maxY, Math.round((b.y + b.height) * scale));
  }

  // Add small padding and clamp to image bounds
  const pad = 5;
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(originalImage.cols, maxX + pad);
  maxY = Math.min(originalImage.rows, maxY + pad);

  const cropWidth = maxX - minX;
  const cropHeight = maxY - minY;
  if (cropWidth <= 0 || cropHeight <= 0) return regions;

  console.log(`[Split] Cropping to (${minX},${minY}) ${cropWidth}x${cropHeight} from ${originalImage.cols}x${originalImage.rows}`);

  // Step 2: Crop the original image
  const cropRect = new cv.Rect(minX, minY, cropWidth, cropHeight);
  const croppedImage = originalImage.roi(cropRect).clone();

  // Step 3: Create a combined mask of selected regions in the crop's coordinate space
  // We'll use this to filter out new regions that fall outside the original selection
  const combinedMask = cv.Mat.zeros(cropHeight, cropWidth, cv.CV_8UC1);
  for (const region of selectedRegions) {
    const scaledContour = new cv.Mat(region.contour.rows, 1, cv.CV_32SC2);
    for (let i = 0; i < region.contour.rows; i++) {
      scaledContour.data32S[i * 2] = Math.round(region.contour.data32S[i * 2] * scale) - minX;
      scaledContour.data32S[i * 2 + 1] = Math.round(region.contour.data32S[i * 2 + 1] * scale) - minY;
    }
    const contourVec = new cv.MatVector();
    contourVec.push_back(scaledContour);
    cv.drawContours(combinedMask, contourVec, 0, new cv.Scalar(255), -1);
    contourVec.delete();
    scaledContour.delete();
  }

  // Step 4: Run segmentation on the crop
  let subResult;
  try {
    subResult = segmentImage(croppedImage, sensitivity, regionSize, cv, mergeThreshold);
  } catch (error) {
    console.error('[Split] Sub-segmentation failed:', error);
    croppedImage.delete();
    combinedMask.delete();
    return regions;
  }

  const subRegions = subResult.regions;
  const subScaleFactor = subRegions.length > 0 ? (subRegions[0].scaleFactor || 1) : 1;
  // subScaleFactor is relative to the crop. The overall scaleFactor from original is:
  // overallScaleFactor = subScaleFactor (crop was at full res, segmentImage may have downscaled it)

  console.log(`[Split] Got ${subRegions.length} sub-regions (subScaleFactor=${subScaleFactor.toFixed(3)})`);

  // Step 5: Filter and remap new regions
  const newSubRegions = [];
  const subScale = 1 / subScaleFactor; // to go from sub-downscaled to crop coords
  const cropToOrigScale = existingScaleFactor; // original -> working scale

  for (const subRegion of subRegions) {
    // Check overlap with combined mask: sample contour points
    let insideCount = 0;
    let totalCount = 0;
    for (let i = 0; i < subRegion.contour.rows; i++) {
      const cx = Math.round(subRegion.contour.data32S[i * 2] * subScale);
      const cy = Math.round(subRegion.contour.data32S[i * 2 + 1] * subScale);
      if (cx >= 0 && cx < cropWidth && cy >= 0 && cy < cropHeight) {
        totalCount++;
        if (combinedMask.ucharAt(cy, cx) > 0) {
          insideCount++;
        }
      }
    }

    // Discard regions mostly outside the original selection
    if (totalCount === 0 || insideCount / totalCount < 0.5) {
      cleanupMats(subRegion.contour, subRegion.mask);
      continue;
    }

    // Remap contour coordinates from sub-downscaled-crop space to global working space
    // sub-downscaled-crop -> crop (original res) -> original -> working
    // x_working = (x_sub / subScaleFactor + minX) * existingScaleFactor
    const remappedContour = new cv.Mat(subRegion.contour.rows, 1, cv.CV_32SC2);
    for (let i = 0; i < subRegion.contour.rows; i++) {
      const cropX = subRegion.contour.data32S[i * 2] * subScale + minX;
      const cropY = subRegion.contour.data32S[i * 2 + 1] * subScale + minY;
      remappedContour.data32S[i * 2] = Math.round(cropX * cropToOrigScale);
      remappedContour.data32S[i * 2 + 1] = Math.round(cropY * cropToOrigScale);
    }

    // Remap bounds
    const remappedBounds = {
      x: Math.round((subRegion.bounds.x * subScale + minX) * cropToOrigScale),
      y: Math.round((subRegion.bounds.y * subScale + minY) * cropToOrigScale),
      width: Math.round(subRegion.bounds.width * subScale * cropToOrigScale),
      height: Math.round(subRegion.bounds.height * subScale * cropToOrigScale)
    };

    // Rebuild the mask for the remapped contour
    const newMask = cv.Mat.zeros(remappedBounds.height, remappedBounds.width, cv.CV_8UC1);
    const localContour = new cv.Mat(remappedContour.rows, 1, cv.CV_32SC2);
    for (let i = 0; i < remappedContour.rows; i++) {
      localContour.data32S[i * 2] = remappedContour.data32S[i * 2] - remappedBounds.x;
      localContour.data32S[i * 2 + 1] = remappedContour.data32S[i * 2 + 1] - remappedBounds.y;
    }
    const contourVec = new cv.MatVector();
    contourVec.push_back(localContour);
    cv.drawContours(newMask, contourVec, 0, new cv.Scalar(255), -1);
    contourVec.delete();
    localContour.delete();

    // Clean up old sub-region mask and contour
    cleanupMats(subRegion.contour, subRegion.mask);

    newSubRegions.push({
      contour: remappedContour,
      mask: newMask,
      bounds: remappedBounds,
      scaleFactor: existingScaleFactor,
      selected: false,
      label: -1, // Will be reassigned
      avgColor: subRegion.avgColor,
      adjacentIndices: []
    });
  }

  // Clean up sub-regions that were filtered out (edgeMap)
  if (subResult.edgeMap) subResult.edgeMap.delete();
  croppedImage.delete();
  combinedMask.delete();

  console.log(`[Split] Kept ${newSubRegions.length} sub-regions after filtering`);

  // Step 6: Build updated regions array
  const selectedSet = new Set(selectedIndices);
  const updatedRegions = [];

  // Keep non-selected regions
  for (let i = 0; i < regions.length; i++) {
    if (!selectedSet.has(i)) {
      updatedRegions.push(regions[i]);
    } else {
      // Clean up old selected region's OpenCV objects
      cleanupMats(regions[i].contour, regions[i].mask);
    }
  }

  // Add new sub-regions
  for (const subRegion of newSubRegions) {
    updatedRegions.push(subRegion);
  }

  // Step 7: Rebuild simple adjacency (clear all, not critical for basic functionality)
  for (const region of updatedRegions) {
    region.adjacentIndices = [];
  }

  return updatedRegions;
}
