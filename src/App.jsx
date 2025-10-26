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
  const [apiKey, setApiKey] = useState('');
  const [isDragging, setIsDragging] = useState(false);

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

    // Use setTimeout to allow UI to update
    setTimeout(() => {
      try {
        const newRegions = segmentImage(originalImage, sensitivity, regionSize, cv);
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
          message: `Found ${newRegions.length} regions. Click and drag to select them.`,
          type: 'success'
        });
      } catch (error) {
        console.error('Error during segmentation:', error);
        setStatus({ message: 'Error during segmentation: ' + error.message, type: 'error' });
      }
    }, 100);
  }, [originalImage, sensitivity, regionSize, cv]);

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
   * AI-based segmentation using Claude Vision API
   */
  const handleAISegment = useCallback(async () => {
    if (!originalImage || !cv || !apiKey) {
      setStatus({ message: 'Please provide an API key', type: 'error' });
      return;
    }

    try {
      setStatus({ message: 'Sending image to Claude API...', type: 'info' });

      // Convert canvas to base64
      const canvas = segmentationCanvasRef.current;
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      const base64Data = imageData.split(',')[1];

      // Call Anthropic API
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          model: 'claude-3-5-sonnet-20241022',
          max_tokens: 4096,
          messages: [{
            role: 'user',
            content: [
              {
                type: 'image',
                source: {
                  type: 'base64',
                  media_type: 'image/jpeg',
                  data: base64Data
                }
              },
              {
                type: 'text',
                text: `Analyze this image and identify distinct objects, text regions, and visual elements that someone might want to select. For each region, provide a bounding box.

Return your response as a JSON array where each object has:
- name: string (brief description)
- box: [x, y, width, height] (coordinates in pixels, relative to image size ${canvas.width}x${canvas.height})

Example format:
[
  {"name": "poster in top left", "box": [10, 20, 300, 400]},
  {"name": "text saying hello", "box": [350, 50, 200, 80]}
]

Return ONLY the JSON array, no other text.`
              }
            ]
          }]
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error?.message || 'API request failed');
      }

      const data = await response.json();
      const content = data.content[0].text;

      // Parse the JSON response
      const jsonMatch = content.match(/\[[\s\S]*\]/);
      if (!jsonMatch) {
        throw new Error('Could not parse AI response. Please try again.');
      }

      const aiRegions = JSON.parse(jsonMatch[0]);

      // Convert AI regions to our region format
      const newRegions = aiRegions.map((region, index) => {
        const [x, y, width, height] = region.box;

        // Create a rectangular contour
        const contour = cv.matFromArray(4, 1, cv.CV_32SC2, [
          x, y,
          x + width, y,
          x + width, y + height,
          x, y + height
        ]);

        // Create a simple mask (we'll just use the bounding box)
        const mask = cv.Mat.zeros(height, width, cv.CV_8U);
        mask.setTo(new cv.Scalar(255));

        return {
          contour: contour,
          mask: mask,
          bounds: { x, y, width, height },
          scaleFactor: 1, // No scaling for AI regions
          selected: false,
          name: region.name // Store the AI-provided name
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
        message: `AI found ${newRegions.length} regions! Hover to see names, click to select.`,
        type: 'success'
      });
      setShowAISegment(false);

    } catch (error) {
      console.error('AI Segmentation error:', error);
      setStatus({ message: 'AI Segmentation failed: ' + error.message, type: 'error' });
    }
  }, [originalImage, cv, apiKey]);

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
            title="Use AI (Claude) to segment the image"
          >
            AI Segment (Beta)
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
              <h3>AI Segmentation (Beta)</h3>
              <p>Use Claude's vision API to intelligently segment your image into objects and text regions.</p>

              <div className="modal-content">
                <label htmlFor="apiKey">Anthropic API Key:</label>
                <input
                  type="password"
                  id="apiKey"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-ant-..."
                  className="api-key-input"
                />
                <p className="help-text">
                  Get your API key from <a href="https://console.anthropic.com/" target="_blank" rel="noopener noreferrer">console.anthropic.com</a>
                </p>
                <p className="help-text">
                  Your API key is stored locally and never saved on any server.
                </p>
              </div>

              <div className="modal-buttons">
                <button className="btn btn-secondary" onClick={() => setShowAISegment(false)}>
                  Cancel
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleAISegment}
                  disabled={!apiKey}
                >
                  Segment with AI
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
