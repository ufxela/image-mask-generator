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

  // Presenter mode state
  const [presenterMode, setPresenterMode] = useState(false);
  const [presenterSubMode, setPresenterSubMode] = useState('segment'); // 'segment' | 'brush-white' | 'brush-black' | 'eraser'
  const [brushStrokes, setBrushStrokes] = useState([]); // Array of {type: 'white'|'black', points: [{x, y}], size: number}
  const [currentStroke, setCurrentStroke] = useState(null);
  const [brushSize, setBrushSize] = useState(20);
  const [presenterIsDragging, setPresenterIsDragging] = useState(false);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [finalPresenterImage, setFinalPresenterImage] = useState(null);

  // Canvas refs
  const segmentationCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const presenterCanvasRef = useRef(null);
  const presenterContainerRef = useRef(null);

  // Refs for cleanup to avoid stale closures
  const originalImageRef = useRef(null);
  const regionsRef = useRef([]);
  const brushStrokesRef = useRef([]);

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
   * Download the presenter mode image
   */
  const handleDownloadPresenterImage = useCallback(() => {
    if (!finalPresenterImage) return;

    try {
      const a = document.createElement('a');
      a.href = finalPresenterImage;
      a.download = 'presenter-mask.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      setStatus({ message: 'Presenter mask downloaded successfully!', type: 'success' });
      setShowDownloadModal(false);
      setFinalPresenterImage(null);
    } catch (error) {
      console.error('Error downloading presenter image:', error);
      setStatus({ message: 'Error downloading image: ' + error.message, type: 'error' });
    }
  }, [finalPresenterImage]);

  /**
   * Enter presenter mode (full screen)
   */
  const enterPresenterMode = useCallback(() => {
    if (regions.length === 0) {
      setStatus({ message: 'Please segment the image first.', type: 'error' });
      return;
    }

    setPresenterMode(true);
    setPresenterSubMode('segment');
    setBrushStrokes([]);
    brushStrokesRef.current = [];

    // Request fullscreen on the container
    setTimeout(() => {
      if (presenterContainerRef.current) {
        if (presenterContainerRef.current.requestFullscreen) {
          presenterContainerRef.current.requestFullscreen();
        } else if (presenterContainerRef.current.webkitRequestFullscreen) {
          presenterContainerRef.current.webkitRequestFullscreen();
        } else if (presenterContainerRef.current.mozRequestFullScreen) {
          presenterContainerRef.current.mozRequestFullScreen();
        }
      }
    }, 100);
  }, [regions]);

  /**
   * Exit presenter mode
   */
  const exitPresenterMode = useCallback(() => {
    // Capture the final canvas as an image before exiting
    if (presenterCanvasRef.current) {
      try {
        const dataURL = presenterCanvasRef.current.toDataURL('image/png');
        setFinalPresenterImage(dataURL);
        setShowDownloadModal(true);
      } catch (error) {
        console.error('Error capturing presenter canvas:', error);
      }
    }

    setPresenterMode(false);
    setPresenterSubMode('segment');
    setPresenterIsDragging(false);
    setCurrentStroke(null);

    // Exit fullscreen
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else if (document.webkitFullscreenElement) {
      document.webkitExitFullscreen();
    } else if (document.mozFullScreenElement) {
      document.mozCancelFullScreen();
    }
  }, []);

  /**
   * Render the presenter mode canvas with segments + brush overlay
   */
  const renderPresenterCanvas = useCallback(() => {
    if (!presenterCanvasRef.current || !originalImage || !cv) return;

    const canvas = presenterCanvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas to full window size
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // Fill with black background
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Calculate scaling to fit the mask in the canvas while maintaining aspect ratio
    const imageAspect = originalImage.cols / originalImage.rows;
    const canvasAspect = canvas.width / canvas.height;

    let displayWidth, displayHeight, offsetX, offsetY;

    if (imageAspect > canvasAspect) {
      // Image is wider than canvas
      displayWidth = canvas.width;
      displayHeight = canvas.width / imageAspect;
      offsetX = 0;
      offsetY = (canvas.height - displayHeight) / 2;
    } else {
      // Image is taller than canvas
      displayHeight = canvas.height;
      displayWidth = canvas.height * imageAspect;
      offsetX = (canvas.width - displayWidth) / 2;
      offsetY = 0;
    }

    // Create the mask from selected regions
    const mask = cv.Mat.zeros(originalImage.rows, originalImage.cols, cv.CV_8UC1);
    const selectedRegions = regions.filter(r => r.selected);

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
      cv.drawContours(mask, contourVec, 0, new cv.Scalar(255), -1);
      contourVec.delete();
      scaledContour.delete();
    }

    // Convert mask to ImageData
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = mask.cols;
    tempCanvas.height = mask.rows;
    cv.imshow(tempCanvas, mask);
    mask.delete();

    // Draw the mask scaled to fit
    ctx.drawImage(tempCanvas, offsetX, offsetY, displayWidth, displayHeight);

    // Apply brush strokes overlay
    if (brushStrokes.length > 0 || currentStroke) {
      const allStrokes = currentStroke ? [...brushStrokes, currentStroke] : brushStrokes;

      for (const stroke of allStrokes) {
        ctx.strokeStyle = stroke.type === 'white' ? 'white' : 'black';
        ctx.fillStyle = stroke.type === 'white' ? 'white' : 'black';
        const strokeSize = stroke.size || brushSize; // Use stroke's stored size, or current brushSize for in-progress strokes
        ctx.lineWidth = strokeSize;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        if (stroke.points.length === 1) {
          // Single point - draw a circle
          const point = stroke.points[0];
          ctx.beginPath();
          ctx.arc(point.x, point.y, strokeSize / 2, 0, Math.PI * 2);
          ctx.fill();
        } else if (stroke.points.length > 1) {
          // Multiple points - draw a path
          ctx.beginPath();
          ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
          for (let i = 1; i < stroke.points.length; i++) {
            ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
          }
          ctx.stroke();

          // Also draw circles at each point for smoother appearance
          for (const point of stroke.points) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, strokeSize / 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    }

    return { displayWidth, displayHeight, offsetX, offsetY };
  }, [originalImage, regions, cv, brushStrokes, currentStroke, brushSize]);

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

  // Presenter mode: Keyboard event handler
  useEffect(() => {
    if (!presenterMode) return;

    const handleKeyDown = (e) => {
      switch (e.key.toLowerCase()) {
        case 'escape':
          exitPresenterMode();
          break;
        case 's':
          setPresenterSubMode('segment');
          break;
        case 'b':
          setPresenterSubMode('brush-black');
          break;
        case 'w':
          setPresenterSubMode('brush-white');
          break;
        case 'e':
          setPresenterSubMode('eraser');
          break;
        case 'z':
          // Decrease brush size
          setBrushSize(prev => Math.max(5, prev - 5));
          break;
        case 'x':
          // Increase brush size
          setBrushSize(prev => Math.min(100, prev + 5));
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [presenterMode, exitPresenterMode]);

  // Presenter mode: Render canvas when state changes
  useEffect(() => {
    if (presenterMode) {
      renderPresenterCanvas();
    }
  }, [presenterMode, regions, brushStrokes, currentStroke, renderPresenterCanvas]);

  // Presenter mode: Handle window resize
  useEffect(() => {
    if (!presenterMode) return;

    const handleResize = () => {
      renderPresenterCanvas();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [presenterMode, renderPresenterCanvas]);

  /**
   * Presenter mode: Handle mouse down
   */
  const handlePresenterMouseDown = useCallback((event) => {
    if (!presenterMode || !originalImage || !cv) return;

    setPresenterIsDragging(true);

    const canvas = presenterCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (presenterSubMode === 'segment') {
      // Convert screen coordinates to image coordinates
      const imageAspect = originalImage.cols / originalImage.rows;
      const canvasAspect = canvas.width / canvas.height;

      let displayWidth, displayHeight, offsetX, offsetY;

      if (imageAspect > canvasAspect) {
        displayWidth = canvas.width;
        displayHeight = canvas.width / imageAspect;
        offsetX = 0;
        offsetY = (canvas.height - displayHeight) / 2;
      } else {
        displayHeight = canvas.height;
        displayWidth = canvas.height * imageAspect;
        offsetX = (canvas.width - displayWidth) / 2;
        offsetY = 0;
      }

      // Check if click is within the displayed image bounds
      if (x < offsetX || x > offsetX + displayWidth || y < offsetY || y > offsetY + displayHeight) {
        return;
      }

      // Convert to image coordinates
      const imgX = ((x - offsetX) / displayWidth) * originalImage.cols;
      const imgY = ((y - offsetY) / displayHeight) * originalImage.rows;

      // Find region at this point
      const regionIndex = findRegionAtPoint(imgX, imgY, regions);

      if (regionIndex !== -1) {
        const newRegions = [...regions];
        newRegions[regionIndex].selected = !newRegions[regionIndex].selected;
        setRegions(newRegions);
        regionsRef.current = newRegions;

        // Update the regular mask canvas too
        createMask(originalImage, newRegions, maskCanvasRef.current, cv);
      }
    } else if (presenterSubMode === 'brush-white' || presenterSubMode === 'brush-black') {
      // Start a new brush stroke with current brush size
      const strokeType = presenterSubMode === 'brush-white' ? 'white' : 'black';
      setCurrentStroke({ type: strokeType, points: [{ x, y }], size: brushSize });
    } else if (presenterSubMode === 'eraser') {
      // Start erasing - we'll remove strokes that intersect
      // Store current brush size for eraser radius
      setCurrentStroke({ type: 'erase', points: [{ x, y }], size: brushSize });
    }
  }, [presenterMode, presenterSubMode, originalImage, regions, cv, brushSize]);

  /**
   * Presenter mode: Handle mouse move
   */
  const handlePresenterMouseMove = useCallback((event) => {
    if (!presenterMode || !presenterIsDragging) return;

    const canvas = presenterCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (presenterSubMode === 'segment') {
      // Similar to mouse down - toggle regions as we drag
      const imageAspect = originalImage.cols / originalImage.rows;
      const canvasAspect = canvas.width / canvas.height;

      let displayWidth, displayHeight, offsetX, offsetY;

      if (imageAspect > canvasAspect) {
        displayWidth = canvas.width;
        displayHeight = canvas.width / imageAspect;
        offsetX = 0;
        offsetY = (canvas.height - displayHeight) / 2;
      } else {
        displayHeight = canvas.height;
        displayWidth = canvas.height * imageAspect;
        offsetX = (canvas.width - displayWidth) / 2;
        offsetY = 0;
      }

      if (x < offsetX || x > offsetX + displayWidth || y < offsetY || y > offsetY + displayHeight) {
        return;
      }

      const imgX = ((x - offsetX) / displayWidth) * originalImage.cols;
      const imgY = ((y - offsetY) / displayHeight) * originalImage.rows;

      const regionIndex = findRegionAtPoint(imgX, imgY, regions);

      if (regionIndex !== -1 && !regions[regionIndex].selected) {
        const newRegions = [...regions];
        newRegions[regionIndex].selected = true;
        setRegions(newRegions);
        regionsRef.current = newRegions;
        createMask(originalImage, newRegions, maskCanvasRef.current, cv);
      }
    } else if (presenterSubMode === 'brush-white' || presenterSubMode === 'brush-black') {
      // Continue the brush stroke
      if (currentStroke) {
        setCurrentStroke({
          ...currentStroke,
          points: [...currentStroke.points, { x, y }]
        });
      }
    } else if (presenterSubMode === 'eraser') {
      // Continue eraser path
      if (currentStroke) {
        setCurrentStroke({
          ...currentStroke,
          points: [...currentStroke.points, { x, y }]
        });
      }
    }
  }, [presenterMode, presenterIsDragging, presenterSubMode, currentStroke, originalImage, regions, cv]);

  /**
   * Presenter mode: Handle mouse up
   */
  const handlePresenterMouseUp = useCallback(() => {
    if (!presenterMode) return;

    setPresenterIsDragging(false);

    if (presenterSubMode === 'brush-white' || presenterSubMode === 'brush-black') {
      // Finalize the brush stroke
      if (currentStroke && currentStroke.points.length > 0) {
        const newStrokes = [...brushStrokes, currentStroke];
        setBrushStrokes(newStrokes);
        brushStrokesRef.current = newStrokes;
        setCurrentStroke(null);
      }
    } else if (presenterSubMode === 'eraser') {
      // Erase strokes that intersect with the eraser path
      if (currentStroke && currentStroke.points.length > 0) {
        const eraserPoints = currentStroke.points;
        const eraserRadius = currentStroke.size / 2; // Use the eraser's stored size

        // Filter out strokes that intersect with eraser
        const newStrokes = brushStrokes.filter(stroke => {
          const strokeRadius = stroke.size / 2;
          // Check if any point in the stroke intersects with any eraser point
          for (const strokePoint of stroke.points) {
            for (const eraserPoint of eraserPoints) {
              const dx = strokePoint.x - eraserPoint.x;
              const dy = strokePoint.y - eraserPoint.y;
              const distance = Math.sqrt(dx * dx + dy * dy);

              if (distance < eraserRadius + strokeRadius) {
                return false; // Remove this stroke
              }
            }
          }
          return true; // Keep this stroke
        });

        setBrushStrokes(newStrokes);
        brushStrokesRef.current = newStrokes;
        setCurrentStroke(null);
      }
    }
  }, [presenterMode, presenterSubMode, currentStroke, brushStrokes]);

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

          <button
            className="btn btn-primary"
            onClick={enterPresenterMode}
            disabled={regions.length === 0}
            title="Enter full-screen presenter mode (Esc to exit)"
          >
            Presenter Mode
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

        {/* Presenter Mode */}
        {presenterMode && (
          <div
            ref={presenterContainerRef}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              width: '100vw',
              height: '100vh',
              backgroundColor: 'black',
              zIndex: 9999,
              cursor: presenterSubMode === 'segment' ? 'pointer' : 'crosshair'
            }}
          >
            <canvas
              ref={presenterCanvasRef}
              onMouseDown={handlePresenterMouseDown}
              onMouseMove={handlePresenterMouseMove}
              onMouseUp={handlePresenterMouseUp}
              onMouseLeave={handlePresenterMouseUp}
              style={{
                display: 'block',
                width: '100%',
                height: '100%'
              }}
            />

            {/* Mode indicator */}
            <div style={{
              position: 'absolute',
              top: '20px',
              left: '20px',
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              color: 'white',
              padding: '15px 20px',
              borderRadius: '8px',
              fontFamily: 'monospace',
              fontSize: '14px',
              zIndex: 10000
            }}>
              <div style={{ marginBottom: '8px', fontWeight: 'bold', fontSize: '16px' }}>
                Presenter Mode
              </div>
              <div style={{ marginBottom: '4px' }}>
                <strong>Mode:</strong> {
                  presenterSubMode === 'segment' ? 'Segment Selection' :
                  presenterSubMode === 'brush-white' ? 'White Brush' :
                  presenterSubMode === 'brush-black' ? 'Black Brush' :
                  'Eraser'
                }
              </div>
              <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.3)' }}>
                <div><kbd>S</kbd> Segment Mode</div>
                <div><kbd>W</kbd> White Brush</div>
                <div><kbd>B</kbd> Black Brush</div>
                <div><kbd>E</kbd> Eraser</div>
                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.2)' }}>
                  <kbd>Z</kbd> Smaller Brush
                </div>
                <div><kbd>X</kbd> Larger Brush</div>
                <div style={{ marginTop: '8px' }}><kbd>ESC</kbd> Exit</div>
              </div>
            </div>

            {/* Brush size indicator */}
            {(presenterSubMode === 'brush-white' || presenterSubMode === 'brush-black' || presenterSubMode === 'eraser') && (
              <div style={{
                position: 'absolute',
                top: '20px',
                right: '20px',
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
                padding: '15px 20px',
                borderRadius: '8px',
                fontFamily: 'monospace',
                fontSize: '14px',
                zIndex: 10000
              }}>
                <div><strong>Brush Size:</strong> {brushSize}px</div>
                <div style={{ marginTop: '8px' }}>
                  <button
                    onClick={() => setBrushSize(Math.max(5, brushSize - 5))}
                    style={{
                      padding: '4px 8px',
                      marginRight: '4px',
                      backgroundColor: '#444',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                  >
                    -
                  </button>
                  <button
                    onClick={() => setBrushSize(Math.min(100, brushSize + 5))}
                    style={{
                      padding: '4px 8px',
                      backgroundColor: '#444',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                  >
                    +
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Download Modal after exiting presenter mode */}
        {showDownloadModal && finalPresenterImage && (
          <div className="modal-overlay" onClick={() => { setShowDownloadModal(false); setFinalPresenterImage(null); }}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
              <h3>Download Presenter Mask</h3>
              <p>Your presenter mask is ready! Preview below and download when ready.</p>

              <div className="modal-content">
                <div style={{
                  maxHeight: '400px',
                  overflow: 'auto',
                  backgroundColor: '#000',
                  borderRadius: '4px',
                  padding: '10px',
                  marginBottom: '15px'
                }}>
                  <img
                    src={finalPresenterImage}
                    alt="Presenter mask preview"
                    style={{
                      width: '100%',
                      height: 'auto',
                      display: 'block'
                    }}
                  />
                </div>
                <p className="help-text">
                  This image includes your selected segments plus any brush strokes you added in presenter mode.
                </p>
              </div>

              <div className="modal-buttons">
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    setShowDownloadModal(false);
                    setFinalPresenterImage(null);
                  }}
                >
                  Close
                </button>
                <button
                  className="btn btn-success"
                  onClick={handleDownloadPresenterImage}
                >
                  Download Image
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
