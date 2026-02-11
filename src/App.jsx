import { useState, useRef, useCallback, useEffect } from 'react';
import { useOpenCV } from './hooks/useOpenCV';
import {
  segmentImage,
  drawSegmentation,
  findRegionAtPoint,
  findRegionsInRadius,
  createMask,
  getCanvasMousePosition,
  cleanupMats,
  selectSimilarRegions
} from './utils/segmentation';
import {
  computeHomography,
  invertHomography,
  applyHomography,
  warpImage
} from './utils/homography';
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
  const [detailLevel, setDetailLevel] = useState(5);
  const [mergeStrength, setMergeStrength] = useState(10);
  const [selectionRadius, setSelectionRadius] = useState(30); // Radius in pixels for drag selection
  const [status, setStatus] = useState({ message: '', type: 'info' });
  const [highlightedRegion, setHighlightedRegion] = useState(-1);
  const [showAISegment, setShowAISegment] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [cocoModel, setCocoModel] = useState(null);
  const [loadingAI, setLoadingAI] = useState(false);

  // Undo/redo state
  const [selectionHistory, setSelectionHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Presenter mode state
  const [presenterMode, setPresenterMode] = useState(false);
  const [presenterSubMode, setPresenterSubMode] = useState('segment'); // 'segment' | 'brush-white' | 'brush-black' | 'eraser'
  const [brushStrokes, setBrushStrokes] = useState([]); // Array of {type: 'white'|'black', points: [{x, y}], size: number}
  const [currentStroke, setCurrentStroke] = useState(null);
  const [brushSize, setBrushSize] = useState(20);
  const [presenterIsDragging, setPresenterIsDragging] = useState(false);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [finalPresenterImage, setFinalPresenterImage] = useState(null);
  const [presenterMousePos, setPresenterMousePos] = useState(null); // {x, y, imgX, imgY} for hover preview
  const [presenterSelectionRadius, setPresenterSelectionRadius] = useState(30); // Radius for segment selection in presenter mode
  const [presenterDragMode, setPresenterDragMode] = useState(null); // 'select' or 'deselect' - set on mousedown, maintained during drag
  const [showImageOverlay, setShowImageOverlay] = useState(false); // Toggle original image overlay in presenter mode

  // Transform mode state
  const [transformMode, setTransformMode] = useState(false);
  const [transformPoints, setTransformPoints] = useState([]); // Array of {x, y, type: 'source'|'dest'}
  const [homographyMatrix, setHomographyMatrix] = useState(null); // 3x3 matrix

  // Canvas refs
  const segmentationCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const presenterCanvasRef = useRef(null);
  const presenterContainerRef = useRef(null);

  // Refs for cleanup to avoid stale closures
  const originalImageRef = useRef(null);
  const regionsRef = useRef([]);
  const brushStrokesRef = useRef([]);

  // Store base/untransformed state for reset
  const baseImageRef = useRef(null);
  const baseRegionsRef = useRef([]);
  const baseBrushStrokesRef = useRef([]);

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
        message: 'Image loaded successfully! Adjust settings and click "Segment Image".',
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
        const derivedSensitivity = detailLevel;
        const derivedRegionSize = Math.round(10 + (detailLevel - 1) * (30 / 9));
        const newRegions = segmentImage(originalImage, derivedSensitivity, derivedRegionSize, cv, mergeStrength);

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

        // Reset undo/redo history for new segmentation
        const initialSnapshot = {
          selections: newRegions.map(r => r.selected),
          brushStrokes: []
        };
        setSelectionHistory([initialSnapshot]);
        setHistoryIndex(0);

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
  }, [originalImage, detailLevel, mergeStrength, cv, regions]);

  /**
   * Push the current state onto the undo history
   * Tracks both region selections and brush strokes
   */
  const pushHistory = useCallback((currentRegions, currentBrushStrokes) => {
    const snapshot = {
      selections: currentRegions.map(r => r.selected),
      brushStrokes: JSON.parse(JSON.stringify(currentBrushStrokes))
    };
    setSelectionHistory(prev => {
      const truncated = prev.slice(0, historyIndex + 1);
      const newHistory = [...truncated, snapshot];
      if (newHistory.length > 50) newHistory.shift();
      return newHistory;
    });
    setHistoryIndex(prev => Math.min(prev + 1, 49));
  }, [historyIndex]);

  /**
   * Undo the last change (selection or brush stroke)
   */
  const undo = useCallback(() => {
    if (historyIndex <= 0 || selectionHistory.length === 0) return;
    const newIndex = historyIndex - 1;
    const snapshot = selectionHistory[newIndex];
    if (!snapshot) return;

    // Restore selections if they match
    if (snapshot.selections && snapshot.selections.length === regions.length) {
      const newRegions = regions.map((r, i) => ({ ...r, selected: snapshot.selections[i] || false }));
      setRegions(newRegions);
      regionsRef.current = newRegions;
      if (originalImage && cv) {
        drawSegmentation(originalImage, newRegions, segmentationCanvasRef.current, -1, cv);
        createMask(originalImage, newRegions, maskCanvasRef.current, cv);
      }
    }

    // Restore brush strokes
    if (snapshot.brushStrokes !== undefined) {
      const restoredStrokes = JSON.parse(JSON.stringify(snapshot.brushStrokes));
      setBrushStrokes(restoredStrokes);
      brushStrokesRef.current = restoredStrokes;
    }

    setHistoryIndex(newIndex);
    setStatus({ message: 'Undo', type: 'info' });
  }, [historyIndex, selectionHistory, regions, originalImage, cv]);

  /**
   * Redo a previously undone change
   */
  const redo = useCallback(() => {
    if (historyIndex >= selectionHistory.length - 1) return;
    const newIndex = historyIndex + 1;
    const snapshot = selectionHistory[newIndex];
    if (!snapshot) return;

    if (snapshot.selections && snapshot.selections.length === regions.length) {
      const newRegions = regions.map((r, i) => ({ ...r, selected: snapshot.selections[i] || false }));
      setRegions(newRegions);
      regionsRef.current = newRegions;
      if (originalImage && cv) {
        drawSegmentation(originalImage, newRegions, segmentationCanvasRef.current, -1, cv);
        createMask(originalImage, newRegions, maskCanvasRef.current, cv);
      }
    }

    if (snapshot.brushStrokes !== undefined) {
      const restoredStrokes = JSON.parse(JSON.stringify(snapshot.brushStrokes));
      setBrushStrokes(restoredStrokes);
      brushStrokesRef.current = restoredStrokes;
    }

    setHistoryIndex(newIndex);
    setStatus({ message: 'Redo', type: 'info' });
  }, [historyIndex, selectionHistory, regions, originalImage, cv]);

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

    const pos = getCanvasMousePosition(segmentationCanvasRef.current, event);
    const regionIndex = findRegionAtPoint(pos.x, pos.y, regions);

    // Shift+Click: Select Similar regions
    if (event.shiftKey && regionIndex !== -1) {
      const toSelect = selectSimilarRegions(regionIndex, regions, mergeStrength > 0 ? mergeStrength : 40);
      const newRegions = [...regions];
      for (const idx of toSelect) {
        newRegions[idx].selected = true;
      }
      setRegions(newRegions);
      regionsRef.current = newRegions;
      drawSegmentation(originalImage, newRegions, segmentationCanvasRef.current, regionIndex, cv);
      createMask(originalImage, newRegions, maskCanvasRef.current, cv);
      pushHistory(newRegions, brushStrokes);
      setStatus({ message: `Selected ${toSelect.length} similar connected regions`, type: 'info' });
      return;
    }

    setIsDragging(true);

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
  }, [regions, originalImage, cv, mergeStrength, pushHistory, brushStrokes]);

  /**
   * Handle mouse up on segmentation canvas to end drag selection
   */
  const handleCanvasMouseUp = useCallback(() => {
    if (!isDragging) return;

    setIsDragging(false);
    pushHistory(regions, brushStrokes);

    const selectedCount = regions.filter(r => r.selected).length;
    setStatus({ message: `${selectedCount} region(s) selected`, type: 'info' });
  }, [isDragging, regions, pushHistory, brushStrokes]);

  /**
   * Clear all region selections
   */
  const handleClearSelection = useCallback(() => {
    const newRegions = regions.map(r => ({ ...r, selected: false }));
    setRegions(newRegions);
    regionsRef.current = newRegions;
    pushHistory(newRegions, brushStrokes);
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
  }, [regions, originalImage, cv, pushHistory, brushStrokes]);


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
    setPresenterDragMode(null);
    setCurrentStroke(null);
    setPresenterMousePos(null);

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

    // If in transform mode, show the original color image instead of the mask
    if (transformMode) {
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

      // Convert OpenCV Mat to canvas image
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = originalImage.cols;
      tempCanvas.height = originalImage.rows;
      cv.imshow(tempCanvas, originalImage);

      // Draw the original image
      ctx.drawImage(tempCanvas, offsetX, offsetY, displayWidth, displayHeight);

      // Draw transform points
      for (let i = 0; i < transformPoints.length; i++) {
        const point = transformPoints[i];
        const isSource = point.type === 'source';

        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = isSource ? 'red' : 'lime';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Add number label
        ctx.fillStyle = 'white';
        ctx.font = 'bold 10px monospace';
        ctx.fillText(String(Math.floor(i / 2) + 1), point.x - 3, point.y + 3);
      }

      return { displayWidth, displayHeight, offsetX, offsetY };
    }

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

    // Optionally overlay the original image at low opacity (toggled with 'O' key)
    if (showImageOverlay) {
      const imgCanvas = document.createElement('canvas');
      imgCanvas.width = originalImage.cols;
      imgCanvas.height = originalImage.rows;
      cv.imshow(imgCanvas, originalImage);
      ctx.globalAlpha = 0.25;
      ctx.drawImage(imgCanvas, offsetX, offsetY, displayWidth, displayHeight);
      ctx.globalAlpha = 1.0;
    }

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

    // Draw hover preview
    if (presenterMousePos && !transformMode) {
      if (presenterSubMode === 'segment') {
        // Check if hovering over a selected region (for deselecting)
        const centerRegionIndex = findRegionAtPoint(presenterMousePos.imgX, presenterMousePos.imgY, regions);
        const isHoveringSelected = centerRegionIndex !== -1 && regions[centerRegionIndex].selected;

        if (isHoveringSelected) {
          // Show only the single region that would be deselected (in red)
          const region = regions[centerRegionIndex];
          const scaleFactor = region.scaleFactor || 1;
          const scale = 1 / scaleFactor;

          ctx.globalAlpha = 0.5;
          ctx.strokeStyle = 'red';
          ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
          ctx.lineWidth = 3;

          // Create scaled contour for drawing
          const scaledContour = new cv.Mat(region.contour.rows, 1, cv.CV_32SC2);
          for (let i = 0; i < region.contour.rows; i++) {
            const origX = region.contour.data32S[i * 2] * scale;
            const origY = region.contour.data32S[i * 2 + 1] * scale;

            // Convert from image coordinates to canvas coordinates
            scaledContour.data32S[i * 2] = Math.round((origX / originalImage.cols) * displayWidth + offsetX);
            scaledContour.data32S[i * 2 + 1] = Math.round((origY / originalImage.rows) * displayHeight + offsetY);
          }

          // Draw the contour as a path
          ctx.beginPath();
          for (let i = 0; i < scaledContour.rows; i++) {
            const x = scaledContour.data32S[i * 2];
            const y = scaledContour.data32S[i * 2 + 1];
            if (i === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          }
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
          ctx.globalAlpha = 1.0;

          scaledContour.delete();
        } else {
          // Show all segments within radius that would be selected (in yellow/green)
          const regionIndices = findRegionsInRadius(
            presenterMousePos.imgX,
            presenterMousePos.imgY,
            presenterSelectionRadius,
            regions
          );

          if (regionIndices.length > 0) {
            for (const regionIndex of regionIndices) {
              const region = regions[regionIndex];
              if (region.selected) continue; // Don't highlight already selected regions

              const scaleFactor = region.scaleFactor || 1;
              const scale = 1 / scaleFactor;

              // Draw the highlighted region with a yellow/green overlay
              ctx.globalAlpha = 0.4;
              ctx.strokeStyle = 'lime';
              ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
              ctx.lineWidth = 3;

              // Create scaled contour for drawing
              const scaledContour = new cv.Mat(region.contour.rows, 1, cv.CV_32SC2);
              for (let i = 0; i < region.contour.rows; i++) {
                const origX = region.contour.data32S[i * 2] * scale;
                const origY = region.contour.data32S[i * 2 + 1] * scale;

                // Convert from image coordinates to canvas coordinates
                scaledContour.data32S[i * 2] = Math.round((origX / originalImage.cols) * displayWidth + offsetX);
                scaledContour.data32S[i * 2 + 1] = Math.round((origY / originalImage.rows) * displayHeight + offsetY);
              }

              // Draw the contour as a path
              ctx.beginPath();
              for (let i = 0; i < scaledContour.rows; i++) {
                const x = scaledContour.data32S[i * 2];
                const y = scaledContour.data32S[i * 2 + 1];
                if (i === 0) {
                  ctx.moveTo(x, y);
                } else {
                  ctx.lineTo(x, y);
                }
              }
              ctx.closePath();
              ctx.fill();
              ctx.stroke();

              scaledContour.delete();
            }
            ctx.globalAlpha = 1.0;
          }

          // Draw selection radius circle (only when selecting, not deselecting)
          ctx.globalAlpha = 0.3;
          ctx.strokeStyle = 'white';
          ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]);

          ctx.beginPath();
          ctx.arc(presenterMousePos.x, presenterMousePos.y, presenterSelectionRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1.0;
        }
      } else if (presenterSubMode === 'brush-white' || presenterSubMode === 'brush-black' || presenterSubMode === 'eraser') {
        // Draw brush preview circle
        const brushColor = presenterSubMode === 'brush-white' ? 'rgba(255, 255, 255, 0.5)' :
                           presenterSubMode === 'brush-black' ? 'rgba(0, 0, 0, 0.8)' :
                           'rgba(255, 0, 0, 0.5)'; // Red for eraser

        ctx.globalAlpha = 0.7;
        ctx.strokeStyle = presenterSubMode === 'eraser' ? 'red' : 'white';
        ctx.fillStyle = brushColor;
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.arc(presenterMousePos.x, presenterMousePos.y, brushSize / 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }
    }

    return { displayWidth, displayHeight, offsetX, offsetY };
  }, [originalImage, regions, cv, brushStrokes, currentStroke, brushSize, transformMode, transformPoints, homographyMatrix, presenterMousePos, presenterSubMode, presenterSelectionRadius, showImageOverlay]);

  /**
   * Save the current state as the base (untransformed) state
   */
  const saveBaseState = useCallback(() => {
    if (!originalImage || !cv) return;

    console.log('[Transform] Saving base state for reset');

    // Save a clone of the original image
    if (baseImageRef.current) {
      cleanupMats(baseImageRef.current);
    }
    baseImageRef.current = originalImage.clone();

    // Save a deep copy of regions (clone contours and masks)
    const baseRegions = regions.map(region => {
      const clonedContour = region.contour.clone();
      const clonedMask = region.mask.clone();

      return {
        ...region,
        contour: clonedContour,
        mask: clonedMask
      };
    });

    // Clean up old base regions
    baseRegionsRef.current.forEach(region => {
      if (region.contour && !region.contour.isDeleted()) {
        region.contour.delete();
      }
      if (region.mask && !region.mask.isDeleted()) {
        region.mask.delete();
      }
    });

    baseRegionsRef.current = baseRegions;

    // Save brush strokes (deep copy)
    baseBrushStrokesRef.current = JSON.parse(JSON.stringify(brushStrokes));

  }, [originalImage, regions, brushStrokes, cv]);

  /**
   * Restore the base (untransformed) state
   */
  const restoreBaseState = useCallback(() => {
    if (!baseImageRef.current || !cv) {
      console.log('[Transform] No base state to restore');
      return;
    }

    try {
      console.log('[Transform] Restoring base state');

      // Clean up current image
      if (originalImageRef.current) {
        cleanupMats(originalImageRef.current);
      }

      // Restore image
      const restoredImage = baseImageRef.current.clone();
      setOriginalImage(restoredImage);
      originalImageRef.current = restoredImage;

      // Restore regions (clone from base)
      const restoredRegions = baseRegionsRef.current.map(region => {
        const clonedContour = region.contour.clone();
        const clonedMask = region.mask.clone();

        return {
          ...region,
          contour: clonedContour,
          mask: clonedMask
        };
      });

      // Clean up old regions
      regionsRef.current.forEach(region => {
        if (region.contour && !region.contour.isDeleted()) {
          region.contour.delete();
        }
        if (region.mask && !region.mask.isDeleted()) {
          region.mask.delete();
        }
      });

      setRegions(restoredRegions);
      regionsRef.current = restoredRegions;

      // Restore brush strokes
      const restoredStrokes = JSON.parse(JSON.stringify(baseBrushStrokesRef.current));
      setBrushStrokes(restoredStrokes);
      brushStrokesRef.current = restoredStrokes;

      // Update canvases
      const segCanvas = segmentationCanvasRef.current;
      cv.imshow(segCanvas, restoredImage);
      drawSegmentation(restoredImage, restoredRegions, segCanvas, -1, cv);
      createMask(restoredImage, restoredRegions, maskCanvasRef.current, cv);

      console.log('[Transform] Base state restored');

    } catch (error) {
      console.error('[Transform] Error restoring base state:', error);
    }
  }, [cv]);

  /**
   * Update the base state with new values (e.g., after applying a transformation)
   * This allows the user to make changes after a transformation and have those
   * changes become the new baseline for future transformations
   */
  const updateBaseState = useCallback((newImage, newRegions, newBrushStrokes) => {
    if (!cv) return;

    console.log('[Transform] Updating base state with new transformed values');

    try {
      // Clean up old base image
      if (baseImageRef.current) {
        cleanupMats(baseImageRef.current);
      }

      // Save new base image
      baseImageRef.current = newImage.clone();

      // Clean up old base regions
      baseRegionsRef.current.forEach(region => {
        if (region.contour && !region.contour.isDeleted()) {
          region.contour.delete();
        }
        if (region.mask && !region.mask.isDeleted()) {
          region.mask.delete();
        }
      });

      // Save new base regions (deep clone)
      baseRegionsRef.current = newRegions.map(region => {
        const clonedContour = region.contour.clone();
        const clonedMask = region.mask.clone();
        return { ...region, contour: clonedContour, mask: clonedMask };
      });

      // Save new base brush strokes
      baseBrushStrokesRef.current = JSON.parse(JSON.stringify(newBrushStrokes));

      console.log('[Transform] Base state updated');
    } catch (error) {
      console.error('[Transform] Error updating base state:', error);
    }
  }, [cv]);

  /**
   * Apply the computed homography transformation to the image and all layers
   * IMPORTANT: Always transform from the BASE state, not from current state
   */
  const applyTransformation = useCallback(() => {
    if (!homographyMatrix || !baseImageRef.current || !cv) {
      console.error('[Transform] Cannot apply transformation - missing data');
      return;
    }

    try {
      setStatus({ message: 'Applying perspective transformation...', type: 'info' });

      console.log('[Transform] Applying homography transformation FROM BASE STATE');

      // Step 1: Warp the BASE image (not the current possibly-already-transformed image)
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = baseImageRef.current.cols;
      tempCanvas.height = baseImageRef.current.rows;
      cv.imshow(tempCanvas, baseImageRef.current);

      // Warp the image
      const warpedCanvas = warpImage(
        tempCanvas,
        homographyMatrix,
        baseImageRef.current.cols,
        baseImageRef.current.rows
      );

      // Convert warped canvas back to OpenCV Mat
      const warpedMat = cv.imread(warpedCanvas);

      // Clean up old original image
      if (originalImageRef.current) {
        cleanupMats(originalImageRef.current);
      }

      // Update original image
      setOriginalImage(warpedMat);
      originalImageRef.current = warpedMat;

      // Step 2: Transform all regions FROM BASE regions (not current regions)
      const transformedRegions = baseRegionsRef.current.map(baseRegion => {
        // Transform the BASE contour points
        const newContour = new cv.Mat(baseRegion.contour.rows, 1, cv.CV_32SC2);

        for (let i = 0; i < baseRegion.contour.rows; i++) {
          const x = baseRegion.contour.data32S[i * 2];
          const y = baseRegion.contour.data32S[i * 2 + 1];

          try {
            const transformed = applyHomography(homographyMatrix, { x, y });
            newContour.data32S[i * 2] = Math.round(transformed.x);
            newContour.data32S[i * 2 + 1] = Math.round(transformed.y);
          } catch (e) {
            // If point goes to infinity, keep original
            newContour.data32S[i * 2] = x;
            newContour.data32S[i * 2 + 1] = y;
          }
        }

        // Recalculate bounds from transformed contour
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (let i = 0; i < newContour.rows; i++) {
          const x = newContour.data32S[i * 2];
          const y = newContour.data32S[i * 2 + 1];
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }

        const newBounds = {
          x: Math.max(0, minX),
          y: Math.max(0, minY),
          width: Math.min(warpedMat.cols, maxX) - Math.max(0, minX) + 1,
          height: Math.min(warpedMat.rows, maxY) - Math.max(0, minY) + 1
        };

        // Generate a new mask from the transformed contour
        // We can't just clone the old mask because pixel coordinates have changed
        const newMask = cv.Mat.zeros(newBounds.height, newBounds.width, cv.CV_8UC1);

        // Create a contour vector for drawContours
        const contourVec = new cv.MatVector();

        // Offset the contour to be relative to the bounds
        const relativeContour = new cv.Mat(newContour.rows, 1, cv.CV_32SC2);
        for (let i = 0; i < newContour.rows; i++) {
          relativeContour.data32S[i * 2] = newContour.data32S[i * 2] - newBounds.x;
          relativeContour.data32S[i * 2 + 1] = newContour.data32S[i * 2 + 1] - newBounds.y;
        }

        contourVec.push_back(relativeContour);
        cv.drawContours(newMask, contourVec, 0, new cv.Scalar(255), -1);

        // Clean up temporary objects
        relativeContour.delete();
        contourVec.delete();

        // Keep the selected state, but use new contour, bounds, and mask
        // Set scaleFactor to 1 since transformed contours are in full image coordinates
        return {
          ...baseRegion,
          contour: newContour,
          bounds: newBounds,
          mask: newMask,
          scaleFactor: 1
        };
      });

      // Clean up old regions
      regionsRef.current.forEach(region => {
        if (region.contour && !region.contour.isDeleted()) {
          region.contour.delete();
        }
        if (region.mask && !region.mask.isDeleted()) {
          region.mask.delete();
        }
      });

      setRegions(transformedRegions);
      regionsRef.current = transformedRegions;

      // Step 3: Transform brush strokes FROM BASE strokes (not current strokes)
      const transformedStrokes = baseBrushStrokesRef.current.map(baseStroke => {
        const transformedPoints = baseStroke.points.map(point => {
          try {
            return applyHomography(homographyMatrix, point);
          } catch (e) {
            return point; // Keep original if transformation fails
          }
        });

        return {
          ...baseStroke,
          points: transformedPoints
        };
      });

      setBrushStrokes(transformedStrokes);
      brushStrokesRef.current = transformedStrokes;

      // Update canvases
      const segCanvas = segmentationCanvasRef.current;
      cv.imshow(segCanvas, warpedMat);

      // Redraw segmentation
      drawSegmentation(warpedMat, transformedRegions, segCanvas, -1, cv);

      // Update mask
      createMask(warpedMat, transformedRegions, maskCanvasRef.current, cv);

      // Reset transform points
      setTransformPoints([]);
      setHomographyMatrix(null);

      // Force re-render of presenter canvas if in presenter mode
      if (presenterMode) {
        setTimeout(() => {
          renderPresenterCanvas();
        }, 100);
      }

      setStatus({ message: 'Transformation applied successfully!', type: 'success' });
      console.log('[Transform] Transformation complete');

      // NOTE: We do NOT update the base state here
      // The base state should always remain the original untransformed state
      // This ensures that pressing T again will show the original image
      // and any new transformation is applied from the original coordinates

    } catch (error) {
      console.error('[Transform] Error applying transformation:', error);
      setStatus({ message: 'Error applying transformation: ' + error.message, type: 'error' });
    }
  }, [homographyMatrix, cv, presenterMode, renderPresenterCanvas]);

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

  // Keyboard shortcuts for undo/redo (normal mode)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (presenterMode) return;
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [presenterMode, undo, redo]);

  // Presenter mode: Keyboard event handler
  useEffect(() => {
    if (!presenterMode) return;

    const handleKeyDown = (e) => {
      switch (e.key.toLowerCase()) {
        case 'escape':
          exitPresenterMode();
          break;
        case 's':
          if (!transformMode) {
            setPresenterSubMode('segment');
          }
          break;
        case 'b':
          if (!transformMode) {
            setPresenterSubMode('brush-black');
          }
          break;
        case 'w':
          if (!transformMode) {
            setPresenterSubMode('brush-white');
          }
          break;
        case 'e':
          if (!transformMode) {
            setPresenterSubMode('eraser');
          }
          break;
        case 'o':
          if (!transformMode) {
            setShowImageOverlay(prev => !prev);
          }
          break;
        case 'z':
          if ((e.ctrlKey || e.metaKey) && !transformMode) {
            e.preventDefault();
            undo();
          } else if (!transformMode) {
            // Decrease brush/selection radius size
            if (presenterSubMode === 'segment') {
              setPresenterSelectionRadius(prev => Math.max(5, Math.round(prev / 1.2)));
            } else {
              setBrushSize(prev => Math.max(5, Math.round(prev / 1.2)));
            }
          }
          break;
        case 'y':
          if ((e.ctrlKey || e.metaKey) && !transformMode) {
            e.preventDefault();
            redo();
          }
          break;
        case 'x':
          // Increase brush/selection radius size
          if (!transformMode) {
            if (presenterSubMode === 'segment') {
              // Geometric scaling for selection radius (20% increase), no upper limit
              setPresenterSelectionRadius(prev => Math.round(prev * 1.2));
            } else {
              // Geometric scaling for brush size (20% increase), no upper limit
              setBrushSize(prev => Math.round(prev * 1.2));
            }
          }
          break;
        case 't':
          // Enter transform mode and reset to original untransformed state
          if (!transformMode) {
            // On first entry: save the current (untransformed) state as base
            if (!baseImageRef.current) {
              console.log('[Transform] First T press - saving base state');
              saveBaseState();
            } else {
              // On subsequent entries: restore to original untransformed state
              console.log('[Transform] Subsequent T press - restoring to original base state');
              restoreBaseState();
            }

            // Clear any previous transform points
            setTransformPoints([]);
            setHomographyMatrix(null);

            setTransformMode(true);
          }
          break;
      }
    };

    const handleKeyUp = (e) => {
      if (e.key.toLowerCase() === 't') {
        // Apply transformation if we have a homography matrix
        if (homographyMatrix && transformPoints.length === 8) {
          applyTransformation();
        }
        // Exit transform mode when T is released
        setTransformMode(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [presenterMode, transformMode, presenterSubMode, exitPresenterMode, homographyMatrix, transformPoints, applyTransformation, saveBaseState, restoreBaseState, undo, redo]);

  // Presenter mode: Render canvas when state changes
  useEffect(() => {
    if (presenterMode) {
      renderPresenterCanvas();
    }
  }, [presenterMode, regions, brushStrokes, currentStroke, transformMode, transformPoints, renderPresenterCanvas, showImageOverlay]);

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

    const canvas = presenterCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Handle transform mode point collection
    if (transformMode) {
      if (transformPoints.length < 8) {
        // Alternate between source (red) and destination (green)
        const pointType = transformPoints.length % 2 === 0 ? 'source' : 'dest';
        const newPoints = [...transformPoints, { x, y, type: pointType }];
        setTransformPoints(newPoints);

        // If we have all 8 points (4 pairs), compute homography
        if (newPoints.length === 8) {
          try {
            // Extract source and destination points
            const sourcePoints = [];
            const destPoints = [];

            for (let i = 0; i < newPoints.length; i += 2) {
              sourcePoints.push({ x: newPoints[i].x, y: newPoints[i].y });
              destPoints.push({ x: newPoints[i + 1].x, y: newPoints[i + 1].y });
            }

            // Compute homography
            const H = computeHomography(sourcePoints, destPoints);
            setHomographyMatrix(H);

            console.log('[Transform] Computed homography:', H);
            setStatus({ message: 'Homography computed! Press T again to apply transformation.', type: 'success' });
          } catch (error) {
            console.error('[Transform] Error computing homography:', error);
            setStatus({ message: 'Error computing homography: ' + error.message, type: 'error' });
          }
        }
      }
      return; // Don't process other interactions in transform mode
    }

    setPresenterIsDragging(true);

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

      // Shift+Click: Select Similar regions
      if (event.shiftKey) {
        const regionIndex = findRegionAtPoint(imgX, imgY, regions);
        if (regionIndex !== -1) {
          const toSelect = selectSimilarRegions(regionIndex, regions, mergeStrength > 0 ? mergeStrength : 40);
          const newRegions = [...regions];
          for (const idx of toSelect) {
            newRegions[idx].selected = true;
          }
          setRegions(newRegions);
          regionsRef.current = newRegions;
          createMask(originalImage, newRegions, maskCanvasRef.current, cv);
          return;
        }
      }

      // Check if we're clicking on a selected region (to deselect)
      const centerRegionIndex = findRegionAtPoint(imgX, imgY, regions);

      if (centerRegionIndex !== -1 && regions[centerRegionIndex].selected) {
        // Start deselecting mode: only affect the single region under cursor
        setPresenterDragMode('deselect');
        const newRegions = [...regions];
        newRegions[centerRegionIndex].selected = false;
        setRegions(newRegions);
        regionsRef.current = newRegions;
        createMask(originalImage, newRegions, maskCanvasRef.current, cv);
      } else {
        // Start selecting mode: use radius to select all regions within radius
        setPresenterDragMode('select');
        const regionIndices = findRegionsInRadius(imgX, imgY, presenterSelectionRadius, regions);

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
            createMask(originalImage, newRegions, maskCanvasRef.current, cv);
          }
        }
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
  }, [presenterMode, presenterSubMode, originalImage, regions, cv, brushSize, transformMode, transformPoints, homographyMatrix, mergeStrength]);

  /**
   * Presenter mode: Handle mouse move
   */
  const handlePresenterMouseMove = useCallback((event) => {
    if (!presenterMode || transformMode) return;

    const canvas = presenterCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Calculate image coordinates for hover preview
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

    // Check if cursor is within the displayed image bounds
    let imgX, imgY;
    if (x >= offsetX && x <= offsetX + displayWidth && y >= offsetY && y <= offsetY + displayHeight) {
      imgX = ((x - offsetX) / displayWidth) * originalImage.cols;
      imgY = ((y - offsetY) / displayHeight) * originalImage.rows;
      setPresenterMousePos({ x, y, imgX, imgY });
    } else {
      setPresenterMousePos(null);
      if (!presenterIsDragging) return; // Exit early if not dragging and outside bounds
    }

    // Only process drag interactions if actually dragging
    if (!presenterIsDragging) return;

    if (presenterSubMode === 'segment' && imgX !== undefined && imgY !== undefined) {
      // Continue the drag in the mode we started with
      if (presenterDragMode === 'deselect') {
        // Deselecting: only affect single regions one at a time
        const regionIndex = findRegionAtPoint(imgX, imgY, regions);

        if (regionIndex !== -1 && regions[regionIndex].selected) {
          const newRegions = [...regions];
          newRegions[regionIndex].selected = false;
          setRegions(newRegions);
          regionsRef.current = newRegions;
          createMask(originalImage, newRegions, maskCanvasRef.current, cv);
        }
      } else if (presenterDragMode === 'select') {
        // Selecting: use radius to select all regions within radius
        const regionIndices = findRegionsInRadius(imgX, imgY, presenterSelectionRadius, regions);

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
            createMask(originalImage, newRegions, maskCanvasRef.current, cv);
          }
        }
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
  }, [presenterMode, presenterIsDragging, presenterSubMode, presenterDragMode, currentStroke, originalImage, regions, cv, presenterSelectionRadius]);

  /**
   * Presenter mode: Handle mouse leave
   */
  const handlePresenterMouseLeave = useCallback(() => {
    setPresenterMousePos(null);
  }, []);

  /**
   * Presenter mode: Handle mouse up
   */
  const handlePresenterMouseUp = useCallback(() => {
    if (!presenterMode) return;

    setPresenterIsDragging(false);
    setPresenterDragMode(null); // Reset drag mode

    // Push undo history for segment mode
    if (presenterSubMode === 'segment' && regions.length > 0) {
      pushHistory(regions, brushStrokes);
    }

    if (presenterSubMode === 'brush-white' || presenterSubMode === 'brush-black') {
      // Finalize the brush stroke
      if (currentStroke && currentStroke.points.length > 0) {
        const newStrokes = [...brushStrokes, currentStroke];
        setBrushStrokes(newStrokes);
        brushStrokesRef.current = newStrokes;
        setCurrentStroke(null);
        pushHistory(regions, newStrokes);
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
        pushHistory(regions, newStrokes);
      }
    }
  }, [presenterMode, presenterSubMode, currentStroke, brushStrokes, regions, pushHistory]);

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
                <li><strong>Detail Level</strong> (1-10): Lower = fewer, larger regions. Higher = more, smaller regions.</li>
                <li><strong>Merge</strong> (0-100): How aggressively to merge similar adjacent regions. Higher = fewer, more meaningful segments.</li>
                <li><strong>Brush Size</strong> (0-100px): Larger = select more regions at once when dragging</li>
              </ul>
            </li>
            <li><strong>Segment:</strong> Click "Segment Image" to divide the image into selectable regions</li>
            <li><strong>Select Regions:</strong> Click to toggle, drag to paint-select, Shift+Click to select all similar connected regions</li>
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
            <label htmlFor="detailLevelSlider">Detail Level:</label>
            <input
              type="range"
              id="detailLevelSlider"
              min="1"
              max="10"
              value={detailLevel}
              onChange={(e) => setDetailLevel(parseInt(e.target.value))}
              disabled={!originalImage}
              title="Lower = fewer/larger regions, Higher = more/smaller regions"
            />
            <span className="value">{detailLevel}</span>
          </div>

          <div className="slider-group">
            <label htmlFor="mergeStrengthSlider">Merge:</label>
            <input
              type="range"
              id="mergeStrengthSlider"
              min="0"
              max="100"
              value={mergeStrength}
              onChange={(e) => setMergeStrength(parseInt(e.target.value))}
              disabled={!originalImage}
              title="Higher = merge more similar adjacent regions (0 = no merging)"
            />
            <span className="value">{mergeStrength}</span>
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
              onMouseLeave={() => {
                handlePresenterMouseUp();
                handlePresenterMouseLeave();
              }}
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
              backgroundColor: transformMode ? 'rgba(255, 100, 0, 0.9)' : 'rgba(0, 0, 0, 0.7)',
              color: 'white',
              padding: '15px 20px',
              borderRadius: '8px',
              fontFamily: 'monospace',
              fontSize: '14px',
              zIndex: 10000,
              border: transformMode ? '3px solid yellow' : 'none'
            }}>
              <div style={{ marginBottom: '8px', fontWeight: 'bold', fontSize: '16px' }}>
                {transformMode ? 'TRANSFORM MODE' : 'Presenter Mode'}
              </div>

              {transformMode ? (
                <>
                  <div style={{ marginBottom: '4px' }}>
                    <strong>Points:</strong> {transformPoints.length} / 8
                  </div>
                  <div style={{ marginBottom: '4px' }}>
                    {transformPoints.length < 8 ? (
                      <span>Click to add {transformPoints.length % 2 === 0 ? 'RED (source)' : 'GREEN (dest)'} point</span>
                    ) : (
                      <span style={{ color: 'lime' }}>✓ Ready to apply!</span>
                    )}
                  </div>
                  <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.3)' }}>
                    <div><kbd>T</kbd> (hold) Transform Mode</div>
                    <div style={{ marginTop: '8px', fontSize: '12px', opacity: 0.8 }}>
                      Click 4 pairs of points:<br/>
                      Red = source (image)<br/>
                      Green = dest (wall)
                    </div>
                  </div>
                </>
              ) : (
                <>
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
                    <div><kbd>O</kbd> Image Overlay {showImageOverlay ? '(ON)' : '(off)'}</div>
                    <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.2)' }}>
                      <kbd>Z</kbd> Smaller {presenterSubMode === 'segment' ? 'Radius' : 'Brush'}
                    </div>
                    <div><kbd>X</kbd> Larger {presenterSubMode === 'segment' ? 'Radius' : 'Brush'}</div>
                    <div><kbd>T</kbd> Transform Mode</div>
                    <div style={{ marginTop: '8px' }}><kbd>ESC</kbd> Exit</div>
                  </div>
                </>
              )}
            </div>

            {/* Transform controls */}
            {transformMode && homographyMatrix && (
              <div style={{
                position: 'absolute',
                top: '20px',
                right: '20px',
                backgroundColor: 'rgba(0, 200, 0, 0.9)',
                color: 'white',
                padding: '15px 20px',
                borderRadius: '8px',
                fontFamily: 'monospace',
                fontSize: '14px',
                zIndex: 10000,
                border: '3px solid lime'
              }}>
                <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>
                  Homography Ready!
                </div>
                <button
                  onClick={() => {
                    setTransformPoints([]);
                    setHomographyMatrix(null);
                  }}
                  style={{
                    width: '100%',
                    padding: '8px',
                    marginBottom: '8px',
                    backgroundColor: '#ff4444',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: 'bold'
                  }}
                >
                  Reset Points
                </button>
                <div style={{ fontSize: '12px', opacity: 0.9 }}>
                  Release T to exit<br/>
                  Transform will be applied
                </div>
              </div>
            )}

            {/* Size indicator for brush/selection radius */}
            {!transformMode && (
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
                {presenterSubMode === 'segment' ? (
                  <>
                    <div><strong>Selection Radius:</strong> {presenterSelectionRadius}px</div>
                    <div style={{ marginTop: '8px' }}>
                      <button
                        onClick={() => setPresenterSelectionRadius(Math.max(5, Math.round(presenterSelectionRadius / 1.2)))}
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
                        onClick={() => setPresenterSelectionRadius(Math.round(presenterSelectionRadius * 1.2))}
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
                  </>
                ) : (
                  <>
                    <div><strong>Brush Size:</strong> {brushSize}px</div>
                    <div style={{ marginTop: '8px' }}>
                      <button
                        onClick={() => setBrushSize(Math.max(5, Math.round(brushSize / 1.2)))}
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
                        onClick={() => setBrushSize(Math.round(brushSize * 1.2))}
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
                  </>
                )}
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
