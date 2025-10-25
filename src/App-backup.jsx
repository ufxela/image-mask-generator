import { useState, useRef, useCallback, useEffect } from 'react';
import { useOpenCV } from './hooks/useOpenCV';
import {
  segmentImage,
  drawSegmentation,
  findRegionAtPoint,
  createMask,
  getCanvasMousePosition,
  cleanupMats
} from './utils/segmentation';
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
  const [status, setStatus] = useState({ message: '', type: 'info' });
  const [highlightedRegion, setHighlightedRegion] = useState(-1);

  // Canvas refs
  const segmentationCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);

  // Refs for cleanup to avoid stale closures
  const originalImageRef = useRef(null);
  const regionsRef = useRef([]);

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
   * Perform image segmentation
   */
  const handleSegment = useCallback(() => {
    if (!originalImage || !cv) {
      setStatus({ message: 'Please upload an image first.', type: 'error' });
      return;
    }

    setStatus({ message: 'Segmenting image... This may take a few seconds.', type: 'info' });

    // Use setTimeout to allow UI to update
    setTimeout(() => {
      try {
        const newRegions = segmentImage(originalImage, sensitivity, cv);
        setRegions(newRegions);
        regionsRef.current = newRegions;

        drawSegmentation(
          originalImage,
          newRegions,
          segmentationCanvasRef.current,
          -1,
          cv
        );

        setStatus({
          message: `Found ${newRegions.length} regions. Click regions to select them.`,
          type: 'success'
        });
      } catch (error) {
        console.error('Error during segmentation:', error);
        setStatus({ message: 'Error during segmentation: ' + error.message, type: 'error' });
      }
    }, 100);
  }, [originalImage, sensitivity, cv]);

  /**
   * Handle mouse movement over segmentation canvas
   */
  const handleCanvasMouseMove = useCallback((event) => {
    if (regions.length === 0 || !originalImage || !cv) return;

    const pos = getCanvasMousePosition(segmentationCanvasRef.current, event);
    const regionIndex = findRegionAtPoint(pos.x, pos.y, regions);

    setHighlightedRegion(regionIndex);
    drawSegmentation(
      originalImage,
      regions,
      segmentationCanvasRef.current,
      regionIndex,
      cv
    );
  }, [regions, originalImage, cv]);

  /**
   * Handle mouse leaving the segmentation canvas
   */
  const handleCanvasMouseLeave = useCallback(() => {
    if (regions.length === 0 || !originalImage || !cv) return;

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
   * Handle click on segmentation canvas to select/deselect regions
   */
  const handleCanvasClick = useCallback((event) => {
    if (regions.length === 0 || !originalImage || !cv) return;

    const pos = getCanvasMousePosition(segmentationCanvasRef.current, event);
    const regionIndex = findRegionAtPoint(pos.x, pos.y, regions);

    if (regionIndex !== -1) {
      // Toggle selection
      const newRegions = [...regions];
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

      const selectedCount = newRegions.filter(r => r.selected).length;
      setStatus({ message: `${selectedCount} region(s) selected`, type: 'info' });
    }
  }, [regions, originalImage, cv]);

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
    setStatus({ message: 'Selection cleared.', type: 'info' });
  }, [regions, originalImage, cv]);

  /**
   * Generate mask from selected regions
   */
  const handleCreateMask = useCallback(() => {
    if (!originalImage || !cv) {
      setStatus({ message: 'Please upload an image first.', type: 'error' });
      return;
    }

    if (regions.length === 0) {
      setStatus({ message: 'Please segment the image first.', type: 'error' });
      return;
    }

    const selectedCount = regions.filter(r => r.selected).length;
    if (selectedCount === 0) {
      setStatus({ message: 'Please select at least one region.', type: 'warning' });
      return;
    }

    try {
      const success = createMask(originalImage, regions, maskCanvasRef.current, cv);
      if (success) {
        setStatus({
          message: `Mask created successfully with ${selectedCount} selected region(s)!`,
          type: 'success'
        });
      }
    } catch (error) {
      console.error('Error creating mask:', error);
      setStatus({ message: 'Error creating mask: ' + error.message, type: 'error' });
    }
  }, [originalImage, regions, cv]);

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
            <li><strong>Adjust Sensitivity:</strong> Use the slider to control segmentation detail (lower = fewer, larger regions)</li>
            <li><strong>Segment:</strong> Click "Segment Image" to divide the image into selectable regions</li>
            <li><strong>Select Regions:</strong> Hover over regions to highlight them, click to select/deselect</li>
            <li><strong>Generate Mask:</strong> Click "Create Mask" to generate the black-and-white projection mask</li>
            <li><strong>Download:</strong> Click "Download Mask" to save the mask image</li>
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
            />
            <span className="value">{sensitivity}</span>
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
            onClick={handleClearSelection}
            disabled={regions.length === 0}
          >
            Clear Selection
          </button>

          <button
            className="btn btn-success"
            onClick={handleCreateMask}
            disabled={regions.length === 0}
          >
            Create Mask
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
                ref={segmentationCanvasRef}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={handleCanvasMouseLeave}
                onClick={handleCanvasClick}
              />
            </div>
          </div>

          <div className="canvas-container">
            <h3>Generated Mask (White = Selected)</h3>
            <div className="canvas-wrapper">
              <canvas ref={maskCanvasRef} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
