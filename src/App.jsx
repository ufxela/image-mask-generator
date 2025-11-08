import { useState, useRef, useCallback, useEffect } from 'react';
import { useOpenCV } from './hooks/useOpenCV';
import {
  segmentImage,
  drawSegmentation,
  findRegionAtPoint,
  findRegionsInRadius,
  createMask,
  getCanvasMousePosition,
  cleanupMats
} from './utils/segmentation';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import './App.css';

/**
 * Main application component
 * Manages image upload, segmentation, region selection, and mask generation
 */
function App() {
  const { cv, loading: cvLoading, error: cvError } = useOpenCV();

  // State management
  const [originalImage, setOriginalImage] = useState(null);
  const [regions, setRegions] = useState([]);
  const [sensitivity, setSensitivity] = useState(5);
  const [regionSize, setRegionSize] = useState(20); // Default 20 = 1/20th of image
  const [selectionRadius, setSelectionRadius] = useState(30); // Radius in pixels for drag selection
  const [status, setStatus] = useState({ message: '', type: 'info' });
  const [highlightedRegion, setHighlightedRegion] = useState(-1);
  const [showAISegment, setShowAISegment] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [cocoModel, setCocoModel] = useState(null);
  const [loadingAI, setLoadingAI] = useState(false);

  // Canvas refs
  const segmentationCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);

  // Refs for cleanup to avoid stale closures
  const originalImageRef = useRef(null);
  const regionsRef = useRef([]);

  /**
   * Load image into OpenCV and set up canvases
   */
  const loadImage = useCallback((imgElement) => {
    if (!cv) return;

    try {
      // Clean up previous image using ref
      if (originalImageRef.current) {
        cleanupMats(originalImageRef.current);
      }
      // Clean up previous regions using ref
      regionsRef.current.forEach(region => {
        cleanupMats(region.mask);
        if (region.contour && !region.contour.isDeleted()) {
          region.contour.delete();
        }
      });

      // Load new image
      const mat = cv.imread(imgElement);
      const clonedMat = mat.clone();
      setOriginalImage(clonedMat);
      originalImageRef.current = clonedMat;

      // Set up canvases
      const segCanvas = segmentationCanvasRef.current;
      const maskCanvas = maskCanvasRef.current;

      segCanvas.width = mat.cols;
      segCanvas.height = mat.rows;
      cv.imshow(segCanvas, mat);

      // Clear mask canvas
      maskCanvas.width = mat.cols;
      maskCanvas.height = mat.rows;
      const ctx = maskCanvas.getContext('2d');
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

      cleanupMats(mat);

      setRegions([]);
      regionsRef.current = [];
      setStatus({
        message: 'Image loaded successfully! Adjust sensitivity and click "Segment Image".',
        type: 'success'
      });
    } catch (error) {
      console.error('Error loading image:', error);
      setStatus({ message: 'Error loading image: ' + error.message, type: 'error' });
    }
  }, [cv]);

  /**
   * Handle image file upload
   */
  const handleImageUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.match('image/(jpeg|png|jpg)')) {
      setStatus({ message: 'Please upload a JPEG or PNG image.', type: 'error' });
      return;
    }

    setStatus({ message: 'Loading image...', type: 'info' });

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        loadImage(img);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }, [loadImage]);

  /**
   * Perform image segmentation
   */
  const handleSegment = useCallback(() => {
    if (!originalImage || !cv) {
      setStatus({ message: 'Please upload an image first.', type: 'error' });
      return;
    }

    setStatus({ message: 'Segmenting image... This may take a few seconds.', type: 'info' });

    // Save current selection mask if there are selected regions
    let savedSelectionMask = null;
    const selectedRegions = regions.filter(r => r.selected);

    if (selectedRegions.length > 0) {
      console.log('[Segmentation] Preserving selection from', selectedRegions.length, 'regions');
      // Create a mask of currently selected regions to preserve
      savedSelectionMask = cv.Mat.zeros(originalImage.rows, originalImage.cols, cv.CV_8U);
      createMask(originalImage, regions, {
        width: originalImage.cols,
        height: originalImage.rows,
        getContext: () => ({
          // Mock context - we'll use cv.imshow alternative
          fillStyle: '',
          fillRect: () => {}
        })
      }, cv);

      // Actually create the mask properly
      for (let region of selectedRegions) {
        const scaleFactor = region.scaleFactor || 1;
        const scale = 1 / scaleFactor;

        // Create scaled contour
        const scaledContour = new cv.Mat(region.contour.rows, 1, cv.CV_32SC2);
        for (let i = 0; i < region.contour.rows; i++) {
          scaledContour.data32S[i * 2] = Math.round(region.contour.data32S[i * 2] * scale);
          scaledContour.data32S[i * 2 + 1] = Math.round(region.contour.data32S[i * 2 + 1] * scale);
        }

        const contourVec = new cv.MatVector();
        contourVec.push_back(scaledContour);
        cv.drawContours(savedSelectionMask, contourVec, 0, new cv.Scalar(255), -1);
        contourVec.delete();
        scaledContour.delete();
      }
    }

    // Use setTimeout to allow UI to update
    setTimeout(() => {
      try {
        const newRegions = segmentImage(originalImage, sensitivity, regionSize, cv);

        // If we have a saved selection, mark overlapping regions as selected
        if (savedSelectionMask) {
          console.log('[Segmentation] Restoring selection to new regions...');
          let restoredCount = 0;

          for (let i = 0; i < newRegions.length; i++) {
            const region = newRegions[i];
            const scaleFactor = region.scaleFactor || 1;
            const scale = 1 / scaleFactor;

            // Check if this region overlaps with the saved selection
            // Sample several points in the region's mask
            let overlapCount = 0;
            let totalSamples = 0;

            // Sample points from the region's mask
            const sampleStep = Math.max(1, Math.floor(region.mask.rows / 10));
            for (let y = 0; y < region.mask.rows; y += sampleStep) {
              for (let x = 0; x < region.mask.cols; x += sampleStep) {
                if (region.mask.ucharAt(y, x) > 0) {
                  totalSamples++;

                  // Transform to original image coordinates
                  const origX = Math.floor((region.bounds.x + x) * scale);
                  const origY = Math.floor((region.bounds.y + y) * scale);

                  if (origX >= 0 && origX < savedSelectionMask.cols &&
                      origY >= 0 && origY < savedSelectionMask.rows) {
                    if (savedSelectionMask.ucharAt(origY, origX) > 0) {
                      overlapCount++;
                    }
                  }
                }
              }
            }

            // If more than 50% of sampled points overlap, mark as selected
            if (totalSamples > 0 && (overlapCount / totalSamples) > 0.5) {
              newRegions[i].selected = true;
              restoredCount++;
            }
          }

          console.log('[Segmentation] Restored selection to', restoredCount, 'new regions');
          savedSelectionMask.delete();
        }

        setRegions(newRegions);
        regionsRef.current = newRegions;

        drawSegmentation(
          originalImage,
          newRegions,
          segmentationCanvasRef.current,
          -1,
          cv
        );

        // Update mask with restored selection
        if (newRegions.some(r => r.selected)) {
          createMask(originalImage, newRegions, maskCanvasRef.current, cv);
        }

        const selectedCount = newRegions.filter(r => r.selected).length;
        setStatus({
          message: `Found ${newRegions.length} regions${selectedCount > 0 ? `, restored ${selectedCount} selected` : ''}. Click and drag to select them.`,
          type: 'success'
        });
      } catch (error) {
        console.error('Error during segmentation:', error);
        setStatus({ message: 'Error during segmentation: ' + error.message, type: 'error' });
      }
    }, 100);
  }, [originalImage, sensitivity, regionSize, cv, regions]);

  /**
   * Handle mouse movement over segmentation canvas
   */
  const handleCanvasMouseMove = useCallback((event) => {
    if (regions.length === 0 || !originalImage || !cv) return;

    const pos = getCanvasMousePosition(segmentationCanvasRef.current, event);

    // If dragging, select all regions within the selection radius
    if (isDragging) {
      const regionIndices = findRegionsInRadius(pos.x, pos.y, selectionRadius, regions);

      if (regionIndices.length > 0) {
        const newRegions = [...regions];
        let changed = false;

        for (const idx of regionIndices) {
          if (!newRegions[idx].selected) {
            newRegions[idx].selected = true;
            changed = true;
          }
        }

        if (changed) {
          setRegions(newRegions);
          regionsRef.current = newRegions;

          // Automatically update mask
          createMask(originalImage, newRegions, maskCanvasRef.current, cv);
        }
      }

      // For highlighting during drag, just show the center point
      const regionIndex = findRegionAtPoint(pos.x, pos.y, regions);
      setHighlightedRegion(regionIndex);
      drawSegmentation(
        originalImage,
        regions,
        segmentationCanvasRef.current,
        regionIndex,
        cv
      );
    } else {
      // When not dragging, just highlight the region under cursor
      const regionIndex = findRegionAtPoint(pos.x, pos.y, regions);
      setHighlightedRegion(regionIndex);
      drawSegmentation(
        originalImage,
        regions,
        segmentationCanvasRef.current,
        regionIndex,
        cv
      );
    }
  }, [regions, originalImage, cv, isDragging, selectionRadius]);

  /**
   * Handle mouse leaving the segmentation canvas
   */
  const handleCanvasMouseLeave = useCallback(() => {
    if (regions.length === 0 || !originalImage || !cv) return;

    setIsDragging(false);
    setHighlightedRegion(-1);
    drawSegmentation(
      originalImage,
      regions,
      segmentationCanvasRef.current,
      -1,
      cv
    );
  }, [regions, originalImage, cv]);

  /**
   * Handle mouse down on segmentation canvas to start drag selection
   */
  const handleCanvasMouseDown = useCallback((event) => {
    if (regions.length === 0 || !originalImage || !cv) return;

    setIsDragging(true);

    // Also select the region under the cursor immediately
    const pos = getCanvasMousePosition(segmentationCanvasRef.current, event);
    const regionIndex = findRegionAtPoint(pos.x, pos.y, regions);

    if (regionIndex !== -1) {
      const newRegions = [...regions];
      // Toggle selection on mouse down
      newRegions[regionIndex].selected = !newRegions[regionIndex].selected;
      setRegions(newRegions);
      regionsRef.current = newRegions;

      drawSegmentation(
        originalImage,
        newRegions,
        segmentationCanvasRef.current,
        regionIndex,
        cv
      );

      // Automatically update mask
      createMask(originalImage, newRegions, maskCanvasRef.current, cv);
    }
  }, [regions, originalImage, cv]);

  /**
   * Handle mouse up on segmentation canvas to end drag selection
   */
  const handleCanvasMouseUp = useCallback(() => {
    if (!isDragging) return;

    setIsDragging(false);

    const selectedCount = regions.filter(r => r.selected).length;
    setStatus({ message: `${selectedCount} region(s) selected`, type: 'info' });
  }, [isDragging, regions]);

  /**
   * Clear all region selections
   */
  const handleClearSelection = useCallback(() => {
    const newRegions = regions.map(r => ({ ...r, selected: false }));
    setRegions(newRegions);
    regionsRef.current = newRegions;
    drawSegmentation(
      originalImage,
      newRegions,
      segmentationCanvasRef.current,
      -1,
      cv
    );

    // Clear the mask canvas
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

    setStatus({ message: 'Selection cleared.', type: 'info' });
  }, [regions, originalImage, cv]);


  /**
   * Download the generated mask
   */
  const handleDownloadMask = useCallback(() => {
    try {
      maskCanvasRef.current.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'mask.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        setStatus({ message: 'Mask downloaded successfully!', type: 'success' });
      });
    } catch (error) {
      console.error('Error downloading mask:', error);
      setStatus({ message: 'Error downloading mask: ' + error.message, type: 'error' });
    }
  }, []);

  /**
   * AI-based segmentation using TensorFlow.js COCO-SSD
   */
  const handleAISegment = useCallback(async () => {
    if (!originalImage || !cv) {
      setStatus({ message: 'Please upload an image first', type: 'error' });
      return;
    }

    try {
      setLoadingAI(true);
      setStatus({ message: 'Loading AI model... (first time may take a moment)', type: 'info' });

      // Load COCO-SSD model if not already loaded
      let model = cocoModel;
      if (!model) {
        model = await cocoSsd.load();
        setCocoModel(model);
      }

      setStatus({ message: 'Detecting objects in image...', type: 'info' });

      // Get image from canvas
      const canvas = segmentationCanvasRef.current;

      // Run object detection
      const predictions = await model.detect(canvas);

      if (predictions.length === 0) {
        setStatus({ message: 'No objects detected. Try watershed segmentation instead.', type: 'warning' });
        setShowAISegment(false);
        setLoadingAI(false);
        return;
      }

      console.log('[AI Segmentation] Detected', predictions.length, 'objects:', predictions);

      // Convert predictions to our region format
      const newRegions = predictions.map((prediction, index) => {
        const [x, y, width, height] = prediction.bbox;

        // Create a rectangular contour
        const contour = cv.matFromArray(4, 1, cv.CV_32SC2, [
          Math.floor(x), Math.floor(y),
          Math.floor(x + width), Math.floor(y),
          Math.floor(x + width), Math.floor(y + height),
          Math.floor(x), Math.floor(y + height)
        ]);

        // Create a simple mask (filled rectangle)
        const mask = cv.Mat.zeros(Math.floor(height), Math.floor(width), cv.CV_8U);
        mask.setTo(new cv.Scalar(255));

        return {
          contour: contour,
          mask: mask,
          bounds: {
            x: Math.floor(x),
            y: Math.floor(y),
            width: Math.floor(width),
            height: Math.floor(height)
          },
          scaleFactor: 1, // No scaling for AI regions
          selected: false,
          name: `${prediction.class} (${Math.round(prediction.score * 100)}%)` // Store detected class and confidence
        };
      });

      setRegions(newRegions);
      regionsRef.current = newRegions;

      // Draw the regions
      drawSegmentation(
        originalImage,
        newRegions,
        segmentationCanvasRef.current,
        -1,
        cv
      );

      setStatus({
        message: `AI detected ${newRegions.length} objects! Click to select regions.`,
        type: 'success'
      });
      setShowAISegment(false);
      setLoadingAI(false);

    } catch (error) {
      console.error('AI Segmentation error:', error);
      setStatus({ message: 'AI Segmentation failed: ' + error.message, type: 'error' });
      setLoadingAI(false);
    }
  }, [originalImage, cv, cocoModel]);

  // Cleanup OpenCV resources on unmount
  useEffect(() => {
    return () => {
      if (originalImageRef.current) {
        cleanupMats(originalImageRef.current);
      }
      regionsRef.current.forEach(region => {
        cleanupMats(region.mask);
        if (region.contour && !region.contour.isDeleted()) {
          region.contour.delete();
        }
      });
    };
  }, []);

  // Show loading state while OpenCV loads
  if (cvLoading) {
    return (
      <div className="app">
        <div className="container">
          <div className="loading">
            Loading OpenCV.js... This may take a few seconds on first load.
          </div>
        </div>
      </div>
    );
  }

  // Show error if OpenCV failed to load
  if (cvError) {
    return (
      <div className="app">
        <div className="container">
          <div className="status error">{cvError}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="container">
        <h1>Image Mask Generator</h1>
        <p className="description">
          Upload a photo of a collage, automatically segment it into regions,
          select the pieces you want to highlight, and generate a black-and-white mask for projection.
        </p>

        <div className="instructions">
          <h3>How to Use:</h3>
          <ul>
            <li><strong>Upload Image:</strong> Click "Choose Image" to upload a photo of your wall/collage</li>
            <li><strong>Adjust Settings:</strong> Use the sliders to control segmentation:
              <ul>
                <li><strong>Sensitivity</strong> (1-10): Higher = more regions, better edge detection</li>
                <li><strong>Region Size</strong> (10-40): Higher = smaller regions, more detail</li>
                <li><strong>Brush Size</strong> (0-100px): Larger = select more regions at once when dragging</li>
              </ul>
            </li>
            <li><strong>Segment:</strong> Click "Segment Image" to divide the image into selectable regions</li>
            <li><strong>Select Regions:</strong> Click and drag to paint-select regions (mask updates live on the right!)</li>
            <li><strong>Download:</strong> Click "Download Mask" to save your projection mask</li>
          </ul>
        </div>

        <div className="controls">
          <div className="file-input-wrapper">
            <label htmlFor="imageUpload" className="btn btn-primary">Choose Image</label>
            <input
              type="file"
              id="imageUpload"
              accept="image/jpeg,image/png,image/jpg"
              onChange={handleImageUpload}
            />
          </div>

          <div className="slider-group">
            <label htmlFor="sensitivitySlider">Sensitivity:</label>
            <input
              type="range"
              id="sensitivitySlider"
              min="1"
              max="10"
              value={sensitivity}
              onChange={(e) => setSensitivity(parseInt(e.target.value))}
              disabled={!originalImage}
              title="Lower = fewer/larger regions, Higher = more/smaller regions"
            />
            <span className="value">{sensitivity}</span>
          </div>

          <div className="slider-group">
            <label htmlFor="regionSizeSlider">Region Size:</label>
            <input
              type="range"
              id="regionSizeSlider"
              min="10"
              max="40"
              value={regionSize}
              onChange={(e) => setRegionSize(parseInt(e.target.value))}
              disabled={!originalImage}
              title="Lower = larger regions, Higher = smaller regions"
            />
            <span className="value">{regionSize}</span>
          </div>

          <div className="slider-group">
            <label htmlFor="selectionRadiusSlider">Brush Size:</label>
            <input
              type="range"
              id="selectionRadiusSlider"
              min="0"
              max="100"
              value={selectionRadius}
              onChange={(e) => setSelectionRadius(parseInt(e.target.value))}
              disabled={regions.length === 0}
              title="Selection radius for drag painting (0 = single region, 100 = large brush)"
            />
            <span className="value">{selectionRadius}px</span>
          </div>

          <button
            className="btn btn-secondary"
            onClick={handleSegment}
            disabled={!originalImage}
          >
            Segment Image
          </button>

          <button
            className="btn btn-secondary"
            onClick={() => setShowAISegment(true)}
            disabled={!originalImage}
            title="Use AI to detect objects (free, runs in browser)"
          >
            AI Object Detection
          </button>

          <button
            className="btn btn-secondary"
            onClick={handleClearSelection}
            disabled={regions.length === 0}
          >
            Clear Selection
          </button>

          <button
            className="btn btn-success"
            onClick={handleDownloadMask}
            disabled={regions.length === 0}
          >
            Download Mask
          </button>
        </div>

        {status.message && (
          <div className={`status ${status.type}`}>
            {status.message}
          </div>
        )}

        <div className="workspace">
          <div className="canvas-container">
            <h3>Original Image with Segmentation</h3>
            <div className="canvas-wrapper">
              <canvas
                id="originalCanvas"
                ref={segmentationCanvasRef}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={handleCanvasMouseLeave}
                onMouseDown={handleCanvasMouseDown}
                onMouseUp={handleCanvasMouseUp}
              />
            </div>
          </div>

          <div className="canvas-container">
            <h3>Live Mask Preview (White = Selected)</h3>
            <div className="canvas-wrapper">
              <canvas id="maskCanvas" ref={maskCanvasRef} />
            </div>
          </div>
        </div>

        {/* AI Segmentation Modal */}
        {showAISegment && (
          <div className="modal-overlay" onClick={() => setShowAISegment(false)}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
              <h3>AI Object Detection</h3>
              <p>Use TensorFlow.js COCO-SSD to automatically detect common objects in your image.</p>

              <div className="modal-content">
                <p className="help-text">
                  <strong>How it works:</strong>
                </p>
                <ul style={{textAlign: 'left', marginLeft: '20px'}}>
                  <li>Completely free - no API key required</li>
                  <li>Runs entirely in your browser (client-side)</li>
                  <li>Works offline after first model load</li>
                  <li>Detects 80+ common object types (people, furniture, animals, vehicles, etc.)</li>
                  <li>First use may take a moment to download the model (~5MB)</li>
                </ul>
                <p className="help-text">
                  <strong>Note:</strong> AI detection works best on photos with clear, distinct objects.
                  For abstract patterns or precise edge-following, use watershed segmentation instead.
                </p>
              </div>

              <div className="modal-buttons">
                <button className="btn btn-secondary" onClick={() => setShowAISegment(false)}>
                  Cancel
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleAISegment}
                  disabled={loadingAI}
                >
                  {loadingAI ? 'Loading...' : 'Detect Objects'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
