import { describe, it, expect } from 'vitest';
import {
  computeHomography,
  applyHomography,
  invertHomography
} from '../utils/homography.js';

/**
 * Test suite for homography computation and transformation
 */
describe('Homography Utilities', () => {

  describe('computeHomography', () => {

    it('should throw error if not exactly 4 point correspondences', () => {
      const src = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }];
      const dst = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }];

      expect(() => computeHomography(src, dst)).toThrow('Need exactly 4 point correspondences');
    });

    it('should compute identity homography for identical points', () => {
      const points = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 }
      ];

      const H = computeHomography(points, points);

      // Apply homography to test points - should remain unchanged
      const testPoint = { x: 50, y: 50 };
      const transformed = applyHomography(H, testPoint);

      expect(transformed.x).toBeCloseTo(testPoint.x, 0);
      expect(transformed.y).toBeCloseTo(testPoint.y, 0);
    });

    it('should compute translation homography correctly', () => {
      const src = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 }
      ];

      const dst = [
        { x: 50, y: 50 },
        { x: 150, y: 50 },
        { x: 150, y: 150 },
        { x: 50, y: 150 }
      ];

      const H = computeHomography(src, dst);

      // Test corner points
      const p1 = applyHomography(H, { x: 0, y: 0 });
      expect(p1.x).toBeCloseTo(50, 0);
      expect(p1.y).toBeCloseTo(50, 0);

      const p2 = applyHomography(H, { x: 100, y: 0 });
      expect(p2.x).toBeCloseTo(150, 0);
      expect(p2.y).toBeCloseTo(50, 0);

      const p3 = applyHomography(H, { x: 100, y: 100 });
      expect(p3.x).toBeCloseTo(150, 0);
      expect(p3.y).toBeCloseTo(150, 0);

      const p4 = applyHomography(H, { x: 0, y: 100 });
      expect(p4.x).toBeCloseTo(50, 0);
      expect(p4.y).toBeCloseTo(150, 0);
    });

    it('should compute scaling homography correctly', () => {
      const src = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 }
      ];

      const dst = [
        { x: 0, y: 0 },
        { x: 200, y: 0 },
        { x: 200, y: 200 },
        { x: 0, y: 200 }
      ];

      const H = computeHomography(src, dst);

      // Test midpoint - should be scaled 2x
      const midpoint = applyHomography(H, { x: 50, y: 50 });
      expect(midpoint.x).toBeCloseTo(100, 0);
      expect(midpoint.y).toBeCloseTo(100, 0);
    });

    it('should compute rotation homography correctly (90 degrees)', () => {
      const src = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 }
      ];

      // 90 degree counter-clockwise rotation
      const dst = [
        { x: 0, y: 0 },
        { x: 0, y: 100 },
        { x: -100, y: 100 },
        { x: -100, y: 0 }
      ];

      const H = computeHomography(src, dst);

      // Test corner points
      const p1 = applyHomography(H, { x: 100, y: 0 });
      expect(p1.x).toBeCloseTo(0, 0);
      expect(p1.y).toBeCloseTo(100, 0);
    });

    it('should handle perspective transformation (trapezoid)', () => {
      // Rectangle to trapezoid (perspective effect)
      const src = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 }
      ];

      const dst = [
        { x: 20, y: 0 },
        { x: 80, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 }
      ];

      const H = computeHomography(src, dst);

      // Test corner points map correctly
      const p1 = applyHomography(H, { x: 0, y: 0 });
      expect(p1.x).toBeCloseTo(20, 0);
      expect(p1.y).toBeCloseTo(0, 0);

      const p2 = applyHomography(H, { x: 100, y: 0 });
      expect(p2.x).toBeCloseTo(80, 0);
      expect(p2.y).toBeCloseTo(0, 0);

      const p3 = applyHomography(H, { x: 100, y: 100 });
      expect(p3.x).toBeCloseTo(100, 0);
      expect(p3.y).toBeCloseTo(100, 0);

      const p4 = applyHomography(H, { x: 0, y: 100 });
      expect(p4.x).toBeCloseTo(0, 0);
      expect(p4.y).toBeCloseTo(100, 0);
    });
  });

  describe('invertHomography', () => {

    it('should compute inverse correctly', () => {
      const src = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 }
      ];

      const dst = [
        { x: 50, y: 50 },
        { x: 150, y: 50 },
        { x: 150, y: 150 },
        { x: 50, y: 150 }
      ];

      const H = computeHomography(src, dst);
      const Hinv = invertHomography(H);

      // H * Hinv should be identity
      // Test: forward then inverse should give original point
      const original = { x: 25, y: 25 };
      const forward = applyHomography(H, original);
      const backward = applyHomography(Hinv, forward);

      expect(backward.x).toBeCloseTo(original.x, 0);
      expect(backward.y).toBeCloseTo(original.y, 0);
    });

    it('should throw error for singular matrix', () => {
      // Create a singular homography matrix (determinant = 0)
      const singular = [
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
      ];

      expect(() => invertHomography(singular)).toThrow('Matrix is singular');
    });
  });

  describe('applyHomography', () => {

    it('should throw error for point at infinity', () => {
      // Create a homography that sends a point to infinity
      const H = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0]  // w = x, so when x = 0, w = 0 (point at infinity)
      ];

      expect(() => applyHomography(H, { x: 0, y: 0 })).toThrow('Point at infinity');
    });

    it('should handle homogeneous coordinates correctly', () => {
      // Simple 2x scaling
      const H = [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
      ];

      const result = applyHomography(H, { x: 10, y: 20 });
      expect(result.x).toBe(20);
      expect(result.y).toBe(40);
    });
  });

  describe('Round-trip transformations', () => {

    it('should preserve points through forward and inverse transformation', () => {
      const src = [
        { x: 0, y: 0 },
        { x: 200, y: 0 },
        { x: 200, y: 200 },
        { x: 0, y: 200 }
      ];

      const dst = [
        { x: 50, y: 30 },
        { x: 180, y: 40 },
        { x: 190, y: 180 },
        { x: 40, y: 170 }
      ];

      const H = computeHomography(src, dst);
      const Hinv = invertHomography(H);

      // Test multiple points
      const testPoints = [
        { x: 50, y: 50 },
        { x: 100, y: 100 },
        { x: 150, y: 75 },
        { x: 25, y: 175 }
      ];

      for (const point of testPoints) {
        const forward = applyHomography(H, point);
        const backward = applyHomography(Hinv, forward);

        expect(backward.x).toBeCloseTo(point.x, 0);
        expect(backward.y).toBeCloseTo(point.y, 0);
      }
    });

    it('should handle perspective distortion round-trip', () => {
      // Realistic projector scenario: rectangle to trapezoid
      const src = [
        { x: 0, y: 0 },
        { x: 1920, y: 0 },
        { x: 1920, y: 1080 },
        { x: 0, y: 1080 }
      ];

      const dst = [
        { x: 100, y: 50 },
        { x: 1800, y: 80 },
        { x: 1850, y: 1000 },
        { x: 70, y: 1030 }
      ];

      const H = computeHomography(src, dst);
      const Hinv = invertHomography(H);

      // Test center point
      const center = { x: 960, y: 540 };
      const transformed = applyHomography(H, center);
      const restored = applyHomography(Hinv, transformed);

      expect(restored.x).toBeCloseTo(center.x, 0);
      expect(restored.y).toBeCloseTo(center.y, 0);
    });
  });
});
