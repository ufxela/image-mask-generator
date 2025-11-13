/**
 * Homography computation and transformation utilities
 * Implements Direct Linear Transform (DLT) algorithm for computing planar homographies
 */

/**
 * Compute homography matrix from 4 point correspondences using DLT
 * @param {Array} sourcePoints - Array of 4 source points [{x, y}, ...]
 * @param {Array} destPoints - Array of 4 destination points [{x, y}, ...]
 * @returns {Array<Array<number>>} - 3x3 homography matrix
 */
export function computeHomography(sourcePoints, destPoints) {
  if (sourcePoints.length !== 4 || destPoints.length !== 4) {
    throw new Error('Need exactly 4 point correspondences');
  }

  // Normalize points for numerical stability
  const [srcNorm, srcT] = normalizePoints(sourcePoints);
  const [dstNorm, dstT] = normalizePoints(destPoints);

  // Build matrix A for DLT (Direct Linear Transform)
  // For each point correspondence (x, y) -> (x', y'), we have 2 equations:
  // -x  -y  -1   0   0   0  x'x  x'y  x'  | 0
  //  0   0   0  -x  -y  -1  y'x  y'y  y'  | 0
  const A = [];

  for (let i = 0; i < 4; i++) {
    const src = srcNorm[i];
    const dst = dstNorm[i];

    A.push([
      -src.x, -src.y, -1, 0, 0, 0, dst.x * src.x, dst.x * src.y, dst.x
    ]);
    A.push([
      0, 0, 0, -src.x, -src.y, -1, dst.y * src.x, dst.y * src.y, dst.y
    ]);
  }

  // Solve using SVD: A * h = 0
  // h is the eigenvector corresponding to the smallest eigenvalue
  const h = solveHomogeneousDLT(A);

  // Reshape h into 3x3 matrix
  const Hnorm = [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], h[8]]
  ];

  // Denormalize: H = inv(dstT) * Hnorm * srcT
  const H = multiplyMatrices(
    multiplyMatrices(invertNormalizationMatrix(dstT), Hnorm),
    srcT
  );

  return H;
}

/**
 * Normalize points to improve numerical stability
 * Centers points at origin and scales to average distance sqrt(2) from origin
 * @param {Array} points - Array of points [{x, y}, ...]
 * @returns {Array} - [normalizedPoints, transformMatrix]
 */
function normalizePoints(points) {
  // Compute centroid
  const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
  const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;

  // Compute average distance from centroid
  const avgDist = points.reduce((sum, p) => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    return sum + Math.sqrt(dx * dx + dy * dy);
  }, 0) / points.length;

  // Scale factor to make average distance = sqrt(2)
  const scale = Math.sqrt(2) / (avgDist + 1e-10);

  // Transformation matrix: scale and translate
  const T = [
    [scale, 0, -scale * cx],
    [0, scale, -scale * cy],
    [0, 0, 1]
  ];

  // Apply transformation to points
  const normalized = points.map(p => ({
    x: scale * (p.x - cx),
    y: scale * (p.y - cy)
  }));

  return [normalized, T];
}

/**
 * Invert a normalization matrix (simple 2D affine)
 */
function invertNormalizationMatrix(T) {
  const s = T[0][0]; // scale
  const tx = T[0][2]; // x translation
  const ty = T[1][2]; // y translation

  return [
    [1/s, 0, -tx/s],
    [0, 1/s, -ty/s],
    [0, 0, 1]
  ];
}

/**
 * Solve homogeneous DLT using eigenvalue decomposition
 * Finds the eigenvector corresponding to the smallest eigenvalue of A^T * A
 * @param {Array<Array<number>>} A - 8x9 matrix
 * @returns {Array<number>} - 9-element solution vector
 */
function solveHomogeneousDLT(A) {
  // Compute A^T * A (9x9 symmetric matrix)
  const ATA = multiplyMatrices(transpose(A), A);

  // Find eigenvalues and eigenvectors using power iteration
  // We'll find all eigenvectors, then select the one with smallest eigenvalue
  const n = 9;
  const eigenvectors = [];
  const eigenvalues = [];

  // Find the largest eigenvalue/eigenvector first
  for (let k = 0; k < n; k++) {
    // Start with random vector
    let v = Array(n).fill(0).map(() => Math.random() - 0.5);

    // Make v orthogonal to all previously found eigenvectors
    for (let i = 0; i < k; i++) {
      const dot = v.reduce((sum, val, j) => sum + val * eigenvectors[i][j], 0);
      v = v.map((val, j) => val - dot * eigenvectors[i][j]);
    }

    // Normalize
    let norm = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    if (norm < 1e-10) continue; // Skip if degenerate
    v = v.map(val => val / norm);

    // Power iteration
    for (let iter = 0; iter < 100; iter++) {
      // v_new = ATA * v
      const v_new = multiplyMatrixVector(ATA, v);

      // Make orthogonal to previous eigenvectors
      for (let i = 0; i < k; i++) {
        const dot = v_new.reduce((sum, val, j) => sum + val * eigenvectors[i][j], 0);
        v_new.forEach((val, j) => { v_new[j] -= dot * eigenvectors[i][j]; });
      }

      // Normalize
      norm = Math.sqrt(v_new.reduce((sum, val) => sum + val * val, 0));
      if (norm < 1e-10) break;

      const v_norm = v_new.map(val => val / norm);

      // Check convergence
      const diff = v_norm.reduce((sum, val, i) => sum + Math.abs(val - v[i]), 0);
      v = v_norm;

      if (diff < 1e-10) break;
    }

    // Compute eigenvalue: lambda = v^T * ATA * v
    const ATAv = multiplyMatrixVector(ATA, v);
    const lambda = v.reduce((sum, val, i) => sum + val * ATAv[i], 0);

    eigenvectors.push(v);
    eigenvalues.push(lambda);
  }

  // Find index of smallest eigenvalue
  let minIndex = 0;
  let minEigenvalue = eigenvalues[0];
  for (let i = 1; i < eigenvalues.length; i++) {
    if (eigenvalues[i] < minEigenvalue) {
      minEigenvalue = eigenvalues[i];
      minIndex = i;
    }
  }

  return eigenvectors[minIndex];
}

/**
 * Matrix multiplication (3x3 or general)
 */
function multiplyMatrices(A, B) {
  const rowsA = A.length;
  const colsA = A[0].length;
  const rowsB = B.length;
  const colsB = B[0].length;

  if (colsA !== rowsB) {
    throw new Error('Matrix dimensions incompatible for multiplication');
  }

  const result = Array(rowsA).fill(0).map(() => Array(colsB).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result;
}

/**
 * Matrix-vector multiplication
 */
function multiplyMatrixVector(A, v) {
  return A.map(row => row.reduce((sum, val, i) => sum + val * v[i], 0));
}

/**
 * Matrix transpose
 */
function transpose(A) {
  const rows = A.length;
  const cols = A[0].length;
  const result = Array(cols).fill(0).map(() => Array(rows).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = A[i][j];
    }
  }

  return result;
}

/**
 * Invert a 3x3 homography matrix
 */
export function invertHomography(H) {
  const a = H[0][0], b = H[0][1], c = H[0][2];
  const d = H[1][0], e = H[1][1], f = H[1][2];
  const g = H[2][0], h = H[2][1], i = H[2][2];

  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

  if (Math.abs(det) < 1e-10) {
    throw new Error('Matrix is singular, cannot invert');
  }

  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]
  ];
}

/**
 * Apply homography to a point
 * @param {Array<Array<number>>} H - 3x3 homography matrix
 * @param {Object} point - Point {x, y}
 * @returns {Object} - Transformed point {x, y}
 */
export function applyHomography(H, point) {
  const x = point.x;
  const y = point.y;

  const w = H[2][0] * x + H[2][1] * y + H[2][2];

  if (Math.abs(w) < 1e-10) {
    throw new Error('Point at infinity');
  }

  return {
    x: (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    y: (H[1][0] * x + H[1][1] * y + H[1][2]) / w
  };
}

/**
 * Warp an image using a homography matrix
 * @param {HTMLCanvasElement} sourceCanvas - Source image canvas
 * @param {Array<Array<number>>} H - 3x3 homography matrix
 * @param {number} width - Output width
 * @param {number} height - Output height
 * @returns {HTMLCanvasElement} - Warped image canvas
 */
export function warpImage(sourceCanvas, H, width, height) {
  const output = document.createElement('canvas');
  output.width = width;
  output.height = height;

  const srcCtx = sourceCanvas.getContext('2d');
  const dstCtx = output.getContext('2d');

  const srcData = srcCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  const dstData = dstCtx.createImageData(width, height);

  // Use inverse homography to map from destination to source (backward warping)
  const Hinv = invertHomography(H);

  // For each pixel in output, find corresponding source pixel
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      try {
        const srcPt = applyHomography(Hinv, { x, y });

        // Bilinear interpolation
        const sx = srcPt.x;
        const sy = srcPt.y;

        if (sx >= 0 && sx < sourceCanvas.width - 1 && sy >= 0 && sy < sourceCanvas.height - 1) {
          const x0 = Math.floor(sx);
          const y0 = Math.floor(sy);
          const x1 = x0 + 1;
          const y1 = y0 + 1;

          const fx = sx - x0;
          const fy = sy - y0;

          const dstIdx = (y * width + x) * 4;

          // Interpolate each channel
          for (let c = 0; c < 4; c++) {
            const v00 = srcData.data[(y0 * sourceCanvas.width + x0) * 4 + c];
            const v10 = srcData.data[(y0 * sourceCanvas.width + x1) * 4 + c];
            const v01 = srcData.data[(y1 * sourceCanvas.width + x0) * 4 + c];
            const v11 = srcData.data[(y1 * sourceCanvas.width + x1) * 4 + c];

            const v0 = v00 * (1 - fx) + v10 * fx;
            const v1 = v01 * (1 - fx) + v11 * fx;
            const v = v0 * (1 - fy) + v1 * fy;

            dstData.data[dstIdx + c] = Math.round(v);
          }
        }
      } catch (e) {
        // Point at infinity or outside bounds, leave black
      }
    }
  }

  dstCtx.putImageData(dstData, 0, 0);
  return output;
}
