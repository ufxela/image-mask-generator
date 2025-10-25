# Segmentation Algorithm

## Overview

The Image Mask Generator uses **marker-based watershed segmentation** to divide images into selectable regions. This document details the algorithm, its properties, and implementation.

## Algorithm: Watershed Segmentation

Watershed segmentation treats the image as a topographic surface where pixel intensities represent elevation. The algorithm "floods" this surface from markers (seeds), creating regions that meet at watershed boundaries.

### Why Watershed?

Traditional edge-based segmentation (Canny + contours) has limitations:
- ❌ **Gaps**: Weak edges create disconnected regions
- ❌ **Irregular sizes**: No control over region dimensions
- ❌ **Coverage**: Not all pixels belong to a region

Watershed solves these problems:
- ✅ **Complete coverage**: Every pixel belongs to exactly one region
- ✅ **Size control**: Grid-based markers ensure predictable sizing
- ✅ **Blob-like**: Regions grow uniformly from markers
- ✅ **Edge-aware**: Uses gradient to guide boundaries

## Pipeline Stages

### 1. Grayscale Conversion

```javascript
const gray = new cv.Mat();
cv.cvtColor(originalImage, gray, cv.COLOR_RGBA2GRAY);
```

Converts RGB to grayscale for simpler processing.

### 2. Noise Reduction

```javascript
cv.GaussianBlur(gray, blurred, new cv.Size(3, 3), 0);
```

Applies 3x3 Gaussian blur to reduce noise while preserving edges.

### 3. Gradient Computation

```javascript
cv.Sobel(blurred, gradX, cv.CV_32F, 1, 0, 3);  // X gradient
cv.Sobel(blurred, gradY, cv.CV_32F, 0, 1, 3);  // Y gradient
cv.magnitude(gradX, gradY, gradient);           // Magnitude
```

Computes gradient magnitude (edge strength) using Sobel operator:
- **gradX**: Horizontal edges
- **gradY**: Vertical edges
- **gradient**: √(gradX² + gradY²)

The gradient guides watershed boundaries to follow strong edges.

**Fallback**: If `cv.magnitude` is unavailable:
```javascript
cv.multiply(gradX, gradX, gradXSquared);
cv.multiply(gradY, gradY, gradYSquared);
cv.add(gradXSquared, gradYSquared, gradient);
cv.sqrt(gradient, gradient);
```

### 4. Gradient Normalization

```javascript
cv.normalize(gradient, gradient, 0, 255, cv.NORM_MINMAX);
gradient.convertTo(gradient, cv.CV_8U);
```

Normalizes gradient to 0-255 range for watershed processing.

### 5. Marker Placement

```javascript
const maxRegionSize = Math.min(image.cols, image.rows) / 10;
const baseSpacing = maxRegionSize * 0.8;
const spacingMultiplier = 1.5 - (sensitivity / 10);
const spacing = Math.max(15, Math.floor(baseSpacing * spacingMultiplier));
```

**Grid-based markers** ensure:
- Uniform distribution across image
- Predictable region sizes
- Complete coverage

**Sensitivity mapping:**
| Sensitivity | Multiplier | Spacing (1000px image) | Approx. Regions |
|-------------|------------|------------------------|-----------------|
| 1 (Low)     | 1.4        | 112px                  | ~80             |
| 5 (Medium)  | 1.0        | 80px                   | ~156            |
| 10 (High)   | 0.5        | 40px                   | ~625            |

**Marker creation:**
```javascript
for (let y = spacing/2; y < rows; y += spacing) {
  for (let x = spacing/2; x < cols; x += spacing) {
    markers.intPtr(y, x)[0] = markerLabel++;
  }
}
```

Markers start at `spacing/2` to center the grid.

### 6. Watershed Execution

```javascript
cv.cvtColor(gradient, gradient3C, cv.COLOR_GRAY2BGR);
cv.watershed(gradient3C, markers);
```

Watershed requires 3-channel image. The `markers` matrix is modified in-place:
- `-1`: Boundary pixels (watershed lines)
- `0`: Background (shouldn't occur with grid markers)
- `>0`: Region labels (1, 2, 3, ...)

### 7. Region Extraction

```javascript
for (let y = 0; y < rows; y++) {
  for (let x = 0; x < cols; x++) {
    const label = markers.intPtr(y, x)[0];
    if (label > 0) {
      regionMasks.get(label).ucharPtr(y, x)[0] = 255;
    }
  }
}
```

Builds binary mask for each region by scanning the markers matrix.

### 8. Contour Generation

```javascript
cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
```

Converts each mask to a contour for:
- Boundary visualization
- Hit-testing (findRegionAtPoint)
- Mask generation

**Filtering:**
```javascript
const minArea = (cols * rows) * 0.0001; // 0.01% of image
if (cv.contourArea(contour) > minArea) {
  regions.push({ contour, mask, bounds, selected: false });
}
```

Removes tiny noise regions.

## Mathematical Properties

### Region Size Constraint

**Goal**: Regions < 1/10th of image dimensions

**Calculation**:
```
maxRegionSize = min(width, height) / 10
baseSpacing = maxRegionSize * 0.8  (slightly smaller to ensure constraint)
spacing = baseSpacing * (1.5 - sensitivity/10)
```

For a 1000x800 image:
```
maxRegionSize = min(1000, 800) / 10 = 80px
baseSpacing = 80 * 0.8 = 64px
```

At sensitivity = 5:
```
spacing = 64 * (1.5 - 0.5) = 64px
region_count ≈ (1000 / 64) * (800 / 64) ≈ 195 regions
```

### Coverage Guarantee

Grid markers ensure:
```
coverage = (rows / spacing) * (cols / spacing) regions
```

No pixels are left unassigned because:
1. Grid markers cover entire image
2. Watershed fills space between markers
3. Boundary pixels (`-1`) are minimal (1-2px wide)

### Edge Alignment

Regions follow edges because:
- Watershed uses gradient as "elevation"
- Strong edges = high gradient = high barriers
- Water flows around barriers
- Boundaries form at gradient ridges

## Performance Characteristics

### Time Complexity

| Stage                | Complexity      | Notes                          |
|----------------------|-----------------|--------------------------------|
| Grayscale conversion | O(n)            | n = total pixels               |
| Blur                 | O(n)            | 3x3 kernel                     |
| Sobel                | O(n)            | 3x3 kernel, 2 passes           |
| Normalize            | O(n)            | Single pass                    |
| Marker placement     | O(m)            | m = marker count               |
| Watershed            | O(n log m)      | Priority queue-based flooding  |
| Region extraction    | O(n * r)        | r = region count, worst case   |
| Contour finding      | O(n * r)        | Per-region operation           |
| **Total**            | **O(n log m)**  | Watershed dominates            |

For a 1000x1000 image with 200 markers:
- n = 1,000,000 pixels
- m = 200 markers
- Time ≈ 1-5 seconds (depends on hardware/browser)

### Space Complexity

| Data Structure       | Size            | Notes                          |
|----------------------|-----------------|--------------------------------|
| Original image       | O(n * 4)        | RGBA                           |
| Grayscale            | O(n)            | Single channel                 |
| Gradient             | O(n)            | Single channel                 |
| Markers              | O(n * 4)        | 32-bit integers                |
| Region masks         | O(n * r)        | One mask per region            |
| Contours             | O(p * r)        | p = perimeter pixels           |
| **Total**            | **O(n * r)**    | Dominated by region masks      |

**Memory usage** for 1000x1000 image with 200 regions:
- ~1 million pixels * 200 regions = ~200MB
- Optimization: Could store contours only, regenerate masks on demand

## Limitations & Tradeoffs

### Current Limitations

1. **Memory Usage**: O(n * r) for region masks
   - Large images with many regions use significant RAM
   - Browser may struggle with 4000x3000+ images

2. **Processing Time**: Watershed is O(n log m)
   - Larger images take longer
   - Higher sensitivity (more markers) increases time
   - Consider showing progress bar for large images

3. **Edge Accuracy**: Grid-based markers
   - May not align perfectly with object boundaries
   - Users must select multiple regions for complex objects
   - Gradient-based boundaries help but aren't perfect

4. **Browser Dependency**: OpenCV.js in WebAssembly
   - Requires modern browser with WASM support
   - ~8MB download on first load
   - CPU-bound (no GPU acceleration)

### Design Tradeoffs

| Choice                    | Benefit                          | Cost                            |
|---------------------------|----------------------------------|---------------------------------|
| Grid markers              | Uniform sizes, predictable       | May not align with objects      |
| Store full masks          | Fast hit-testing                 | High memory usage               |
| Watershed vs. SLIC        | Better edge alignment            | Slower than SLIC                |
| Client-side only          | No backend needed                | Limited by browser resources    |
| OpenCV.js vs. TensorFlow  | Proven CV algorithms             | Large library size              |

## Future Improvements

1. **Adaptive Markers**
   - Place more markers in high-gradient areas
   - Fewer markers in uniform regions
   - Better object alignment

2. **Memory Optimization**
   - Store contours only
   - Regenerate masks on-demand
   - Use sparse data structures

3. **Progressive Segmentation**
   - Show progress bar
   - Process in chunks
   - Allow cancellation

4. **GPU Acceleration**
   - Use WebGL for Sobel/blur
   - Parallel watershed implementation
   - 10-100x speedup potential

5. **Smart Region Merging**
   - Merge similar adjacent regions
   - Detect object boundaries
   - Suggest grouped selections

6. **Alternative Algorithms**
   - **SLIC superpixels**: Faster, more uniform
   - **Graph-based**: Better object boundaries
   - **Deep learning**: Semantic segmentation

## References

- [OpenCV Watershed Tutorial](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
- [Watershed Transform (Wikipedia)](https://en.wikipedia.org/wiki/Watershed_(image_processing))
- [Sobel Operator](https://en.wikipedia.org/wiki/Sobel_operator)
- [Image Gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html)
