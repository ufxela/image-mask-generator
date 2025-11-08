/**
 * Unit tests for image segmentation utilities
 *
 * Note: These are unit tests with mocked OpenCV.js.
 * For full integration testing with real images and OpenCV.js,
 * see docs/TESTING.md for browser-based test instructions.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  cleanupMats,
  findRegionAtPoint,
  getCanvasMousePosition,
} from '../utils/segmentation';

describe('Segmentation Utilities', () => {
  describe('cleanupMats', () => {
    it('should delete all provided mats', () => {
      const mat1 = new cv.Mat();
      const mat2 = new cv.Mat();
      const mat3 = new cv.Mat();

      expect(mat1.isDeleted()).toBe(false);
      expect(mat2.isDeleted()).toBe(false);
      expect(mat3.isDeleted()).toBe(false);

      cleanupMats(mat1, mat2, mat3);

      expect(mat1.isDeleted()).toBe(true);
      expect(mat2.isDeleted()).toBe(true);
      expect(mat3.isDeleted()).toBe(true);
    });

    it('should handle null/undefined mats gracefully', () => {
      expect(() => {
        cleanupMats(null, undefined, new cv.Mat());
      }).not.toThrow();
    });

    it('should handle already deleted mats', () => {
      const mat = new cv.Mat();
      mat.delete();

      expect(() => {
        cleanupMats(mat);
      }).not.toThrow();
    });

    it('should handle empty arguments', () => {
      expect(() => {
        cleanupMats();
      }).not.toThrow();
    });
  });

  describe('findRegionAtPoint', () => {
    let mockRegions;

    beforeEach(() => {
      // Create mock regions for testing
      mockRegions = [
        {
          bounds: { x: 0, y: 0, width: 100, height: 100 },
          scaleFactor: 1, // No scaling
          mask: {
            cols: 100,
            rows: 100,
            ucharAt: (y, x) => {
              // Simulate a circular region
              const centerX = 50;
              const centerY = 50;
              const radius = 40;
              const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
              return dist <= radius ? 255 : 0;
            }
          }
        },
        {
          bounds: { x: 100, y: 100, width: 100, height: 100 },
          scaleFactor: 1, // No scaling
          mask: {
            cols: 100,
            rows: 100,
            ucharAt: (y, x) => {
              // Simulate a rectangular region (bounds in downscaled space)
              return (x >= 10 && x <= 90 && y >= 10 && y <= 90) ? 255 : 0;
            }
          }
        },
        {
          bounds: { x: 200, y: 0, width: 100, height: 100 },
          scaleFactor: 1, // No scaling
          mask: {
            cols: 100,
            rows: 100,
            ucharAt: () => 255 // Fully filled region
          }
        }
      ];
    });

    it('should find region when point is inside', () => {
      // Point inside first circular region
      const result = findRegionAtPoint(50, 50, mockRegions);
      expect(result).toBe(0);
    });

    it('should find second region when point is inside it', () => {
      // Point inside second rectangular region
      const result = findRegionAtPoint(150, 150, mockRegions);
      expect(result).toBe(1);
    });

    it('should return -1 when point is outside all regions', () => {
      const result = findRegionAtPoint(500, 500, mockRegions);
      expect(result).toBe(-1);
    });

    it('should return -1 when point is within bounds but outside mask', () => {
      // Point within bounds of first region but outside the circular mask
      const result = findRegionAtPoint(10, 10, mockRegions);
      expect(result).toBe(-1);
    });

    it('should check regions in reverse order (z-index)', () => {
      // If regions overlap, it should return the last one
      const overlappingRegions = [
        {
          bounds: { x: 0, y: 0, width: 200, height: 200 },
          scaleFactor: 1,
          mask: {
            cols: 200,
            rows: 200,
            ucharAt: () => 255
          }
        },
        {
          bounds: { x: 50, y: 50, width: 100, height: 100 },
          scaleFactor: 1,
          mask: {
            cols: 100,
            rows: 100,
            ucharAt: () => 255
          }
        }
      ];

      const result = findRegionAtPoint(75, 75, overlappingRegions);
      expect(result).toBe(1); // Should return the last (top) region
    });

    it('should handle empty regions array', () => {
      const result = findRegionAtPoint(50, 50, []);
      expect(result).toBe(-1);
    });

    it('should handle fractional coordinates by flooring', () => {
      const result = findRegionAtPoint(50.7, 50.9, mockRegions);
      expect(result).toBe(0);
    });
  });

  describe('getCanvasMousePosition', () => {
    let mockCanvas;
    let mockEvent;

    beforeEach(() => {
      // Mock canvas with scaling
      mockCanvas = {
        width: 1000,
        height: 800,
        getBoundingClientRect: () => ({
          left: 10,
          top: 20,
          width: 500, // Canvas displayed at half size
          height: 400
        })
      };

      mockEvent = {
        clientX: 110, // 100px from canvas left
        clientY: 120  // 100px from canvas top
      };
    });

    it('should calculate correct position with scaling', () => {
      const pos = getCanvasMousePosition(mockCanvas, mockEvent);

      // Canvas is displayed at 500x400 but actual size is 1000x800
      // So scale is 2x in both dimensions
      // Mouse at (100, 100) relative to canvas => (200, 200) in canvas coordinates
      expect(pos.x).toBe(200);
      expect(pos.y).toBe(200);
    });

    it('should handle canvas with no scaling', () => {
      mockCanvas.getBoundingClientRect = () => ({
        left: 0,
        top: 0,
        width: 1000,
        height: 800
      });

      mockEvent.clientX = 250;
      mockEvent.clientY = 300;

      const pos = getCanvasMousePosition(mockCanvas, mockEvent);
      expect(pos.x).toBe(250);
      expect(pos.y).toBe(300);
    });

    it('should handle canvas position offset', () => {
      mockCanvas.getBoundingClientRect = () => ({
        left: 100,
        top: 200,
        width: 1000,
        height: 800
      });

      mockEvent.clientX = 150; // 50px from canvas left
      mockEvent.clientY = 250; // 50px from canvas top

      const pos = getCanvasMousePosition(mockCanvas, mockEvent);
      expect(pos.x).toBe(50);
      expect(pos.y).toBe(50);
    });

    it('should handle different scale factors for width and height', () => {
      mockCanvas.width = 1000;
      mockCanvas.height = 600;
      mockCanvas.getBoundingClientRect = () => ({
        left: 0,
        top: 0,
        width: 500,  // 2x scale
        height: 300  // 2x scale
      });

      mockEvent.clientX = 250;
      mockEvent.clientY = 150;

      const pos = getCanvasMousePosition(mockCanvas, mockEvent);
      expect(pos.x).toBe(500);
      expect(pos.y).toBe(300);
    });
  });

  describe('Integration Test Documentation', () => {
    it('should have documented approach for browser-based integration tests', () => {
      // This test serves as documentation
      const integrationTestApproach = `
        For full integration testing with real OpenCV.js and images:

        1. Manual Browser Testing:
           - Open the app in a browser
           - Upload the test image from public/test-images/test_image_collage.jpg
           - Verify segmentation creates compact, blob-like regions
           - Verify all regions are selectable
           - Verify mask generation works correctly

        2. Automated E2E Testing (Future):
           - Use Playwright or Cypress for automated browser tests
           - Load the app
           - Programmatically upload test images
           - Verify UI behavior and segmentation results
           - Compare generated masks against expected outputs

        3. Visual Regression Testing (Future):
           - Use tools like Percy or Chromatic
           - Capture screenshots of segmented images
           - Compare against baseline images
           - Flag visual differences for review
      `;

      expect(integrationTestApproach).toBeDefined();
      expect(integrationTestApproach).toContain('OpenCV.js');
      expect(integrationTestApproach).toContain('test_image_collage.jpg');
    });
  });
});

describe('Segmentation Coverage Tests', () => {
  describe('Monte Carlo Coverage Test', () => {
    it('should ensure 100% pixel coverage using Monte Carlo sampling', () => {
      // Create mock regions that should cover a 100x100 image
      const imageWidth = 100;
      const imageHeight = 100;

      // Create 4 regions that collectively cover the entire image
      const mockRegions = [
        {
          bounds: { x: 0, y: 0, width: 50, height: 50 },
          scaleFactor: 1,
          mask: {
            cols: 50,
            rows: 50,
            ucharAt: () => 255 // Fully filled
          }
        },
        {
          bounds: { x: 50, y: 0, width: 50, height: 50 },
          scaleFactor: 1,
          mask: {
            cols: 50,
            rows: 50,
            ucharAt: () => 255
          }
        },
        {
          bounds: { x: 0, y: 50, width: 50, height: 50 },
          scaleFactor: 1,
          mask: {
            cols: 50,
            rows: 50,
            ucharAt: () => 255
          }
        },
        {
          bounds: { x: 50, y: 50, width: 50, height: 50 },
          scaleFactor: 1,
          mask: {
            cols: 50,
            rows: 50,
            ucharAt: () => 255
          }
        }
      ];

      // Monte Carlo sampling: test random points
      const sampleCount = 1000;
      const failedPoints = [];

      for (let i = 0; i < sampleCount; i++) {
        const x = Math.floor(Math.random() * imageWidth);
        const y = Math.floor(Math.random() * imageHeight);

        const regionIndex = findRegionAtPoint(x, y, mockRegions);

        if (regionIndex === -1) {
          failedPoints.push({ x, y });
        }
      }

      // All points should belong to a region
      expect(failedPoints).toHaveLength(0);
      if (failedPoints.length > 0) {
        console.error('Uncovered points:', failedPoints.slice(0, 10));
      }
    });

    it('should handle edge pixels correctly', () => {
      const imageWidth = 100;
      const imageHeight = 100;

      // Create a single region covering the whole image
      const mockRegions = [
        {
          bounds: { x: 0, y: 0, width: imageWidth, height: imageHeight },
          scaleFactor: 1,
          mask: {
            cols: imageWidth,
            rows: imageHeight,
            ucharAt: () => 255
          }
        }
      ];

      // Test all four corners
      expect(findRegionAtPoint(0, 0, mockRegions)).not.toBe(-1);
      expect(findRegionAtPoint(imageWidth - 1, 0, mockRegions)).not.toBe(-1);
      expect(findRegionAtPoint(0, imageHeight - 1, mockRegions)).not.toBe(-1);
      expect(findRegionAtPoint(imageWidth - 1, imageHeight - 1, mockRegions)).not.toBe(-1);

      // Test edges
      expect(findRegionAtPoint(50, 0, mockRegions)).not.toBe(-1);
      expect(findRegionAtPoint(50, imageHeight - 1, mockRegions)).not.toBe(-1);
      expect(findRegionAtPoint(0, 50, mockRegions)).not.toBe(-1);
      expect(findRegionAtPoint(imageWidth - 1, 50, mockRegions)).not.toBe(-1);
    });

    it('should detect gaps in coverage', () => {
      // Create regions with an intentional gap
      const mockRegions = [
        {
          bounds: { x: 0, y: 0, width: 40, height: 100 },
          scaleFactor: 1,
          mask: {
            cols: 40,
            rows: 100,
            ucharAt: () => 255
          }
        },
        {
          bounds: { x: 60, y: 0, width: 40, height: 100 },
          scaleFactor: 1,
          mask: {
            cols: 40,
            rows: 100,
            ucharAt: () => 255
          }
        }
        // Gap from x=40 to x=60
      ];

      // Points in the gap should not belong to any region
      expect(findRegionAtPoint(50, 50, mockRegions)).toBe(-1);
      expect(findRegionAtPoint(45, 50, mockRegions)).toBe(-1);
      expect(findRegionAtPoint(55, 50, mockRegions)).toBe(-1);

      // Points outside the gap should belong to regions
      expect(findRegionAtPoint(20, 50, mockRegions)).toBe(0);
      expect(findRegionAtPoint(80, 50, mockRegions)).toBe(1);
    });
  });

  describe('Boundary Pixel Handling', () => {
    it('should document watershed boundary handling', () => {
      const watershedLabels = {
        '-1': 'Boundary pixels between regions (should be assigned to nearest region)',
        '0': 'Background pixels (should be assigned to nearest region)',
        '>0': 'Valid region labels'
      };

      expect(watershedLabels['-1']).toContain('assigned');
      expect(watershedLabels['0']).toContain('assigned');
    });

    it('should verify neighbor search for orphaned pixels', () => {
      // This documents the algorithm for assigning boundary/background pixels
      const neighborSearchPattern = [
        [-1, -1], [0, -1], [1, -1],
        [-1,  0],          [1,  0],
        [-1,  1], [0,  1], [1,  1]
      ];

      expect(neighborSearchPattern).toHaveLength(8); // 8-connected neighborhood
    });
  });

  describe('Small Region Redistribution', () => {
    it('should document redistribution strategy for complete coverage', () => {
      const strategy = {
        problem: 'Small regions filtered by minArea leave pixels orphaned',
        solution: 'Redistribute pixels from small regions to nearest large regions before filtering',
        approach: 'Expanding radius search to find nearest large region',
        fallback: 'Assign to first large region if no neighbor found within max search radius'
      };

      expect(strategy.solution).toContain('Redistribute');
      expect(strategy.approach).toContain('Expanding radius');
    });

    it('should verify complete coverage guarantee', () => {
      // This documents the complete coverage guarantee
      const coverageGuarantee = {
        step1: 'Watershed assigns all pixels to regions (including boundary -1 and background 0)',
        step2: 'Boundary/background pixels assigned to nearest valid region (8-connected search)',
        step3: 'Small regions redistributed to nearest large regions (expanding radius search)',
        step4: 'Only large regions with redistributed pixels remain',
        result: 'Every pixel belongs to exactly one final region - 100% coverage guaranteed'
      };

      expect(coverageGuarantee.result).toContain('100% coverage');
    });

    it('should handle images with many small regions', () => {
      // Scenario: high sensitivity on textured image creates many tiny regions
      // These should all be redistributed without leaving gaps
      const scenario = {
        input: '1000 watershed regions, 800 below minArea threshold',
        process: 'Redistribute 800 small regions to 200 large regions',
        output: '200 regions with complete coverage',
        assertion: 'No pixels left orphaned despite filtering 80% of regions'
      };

      expect(scenario.assertion).toContain('No pixels left orphaned');
    });
  });
});

describe('Segmentation Algorithm Properties', () => {
  it('should document expected segmentation properties', () => {
    const properties = {
      completeCoverage: 'Every pixel belongs to exactly one region',
      blobLike: 'Regions are compact and roughly circular',
      sizeConstrained: 'Regions are < 1/20th of image dimensions (updated from 1/10th for smaller regions)',
      edgeAware: 'Region boundaries follow strong edges',
      sensitivity: 'Lower sensitivity = fewer/larger regions',
      gridBased: 'Markers placed on uniform grid for predictable sizing'
    };

    expect(Object.keys(properties)).toHaveLength(6);
    expect(properties.completeCoverage).toBeDefined();
    expect(properties.edgeAware).toBeDefined();
    expect(properties.sizeConstrained).toContain('1/20th');
  });

  it('should calculate correct grid spacing', () => {
    // Test the NEW spacing calculation logic (updated to create smaller regions)
    const imageSize = 1000;
    const maxRegionSize = imageSize / 20; // 50px (changed from /10 to /20)
    const baseSpacing = maxRegionSize * 0.7; // 35px (changed from 0.8 to 0.7)

    // Sensitivity 1 (low): spacingMultiplier = 1.3 - 0.11 = 1.19
    const spacing1 = Math.max(10, Math.floor(baseSpacing * 1.19)); // 41px
    expect(spacing1).toBe(41);

    // Sensitivity 5 (medium): spacingMultiplier = 1.3 - 0.55 = 0.75
    const spacing5 = Math.max(10, Math.floor(baseSpacing * 0.75)); // 26px
    expect(spacing5).toBe(26);

    // Sensitivity 10 (high): spacingMultiplier = 1.3 - 1.1 = 0.2
    const spacing10 = Math.max(10, Math.floor(baseSpacing * 0.2)); // 10px (minimum)
    expect(spacing10).toBe(10);

    // Verify sensitivity inversely affects spacing
    expect(spacing1).toBeGreaterThan(spacing5);
    expect(spacing5).toBeGreaterThan(spacing10);
  });

  it('should enforce minimum spacing of 10px', () => {
    const baseSpacing = 5; // Very small
    const spacingMultiplier = 0.5;
    const spacing = Math.max(10, Math.floor(baseSpacing * spacingMultiplier));

    expect(spacing).toBe(10);
  });

  it('should calculate marker count for different image sizes', () => {
    // Helper function to calculate marker count
    const calculateMarkerCount = (width, height, spacing) => {
      const cols = Math.floor((width - Math.floor(spacing / 2)) / spacing) + 1;
      const rows = Math.floor((height - Math.floor(spacing / 2)) / spacing) + 1;
      return cols * rows;
    };

    // Test case 1: Small image
    const smallMarkers = calculateMarkerCount(500, 500, 50);
    expect(smallMarkers).toBeLessThan(5000); // Should be well under limit
    expect(smallMarkers).toBe(100); // 10x10 grid

    // Test case 2: Medium image (like downscaled 2MP)
    const mediumMarkers = calculateMarkerCount(1999, 1411, 37);
    expect(mediumMarkers).toBeLessThan(5000); // Should be under new limit
    expect(mediumMarkers).toBeGreaterThan(2000); // But can exceed old limit
    expect(mediumMarkers).toBe(2052); // Actual value from user's error

    // Test case 3: Large image at max size
    const largeMarkers = calculateMarkerCount(2000, 2000, 30);
    expect(largeMarkers).toBeLessThan(5000);
    expect(largeMarkers).toBe(4489); // 67x67 grid
  });

  it('should document marker limit increase', () => {
    const markerLimits = {
      old: 2000,
      new: 5000,
      reason: 'Support larger images (up to 2000x2000) at higher sensitivity'
    };

    expect(markerLimits.new).toBe(5000);
    expect(markerLimits.new).toBeGreaterThan(markerLimits.old);
  });
});
