/**
 * Test setup file
 * This runs before all tests
 */

// Mock OpenCV.js for unit tests
// Note: Full integration tests with actual OpenCV.js should be run in a real browser
global.cv = {
  Mat: class MockMat {
    constructor(rows, cols, type) {
      this._deleted = false;
      this.rows = rows || 0;
      this.cols = cols || 0;
      this.type_ = type;
      // Simulate data32S for CV_32SC2 contours
      if (rows && cols) {
        this.data32S = new Int32Array(rows * 2);
      }
    }
    delete() {
      this._deleted = true;
    }
    isDeleted() {
      return this._deleted;
    }
    clone() {
      const clone = new MockMat(this.rows, this.cols, this.type_);
      if (this.data32S) {
        clone.data32S = new Int32Array(this.data32S);
      }
      return clone;
    }
    ucharAt(y, x) {
      return 0;
    }
    ucharPtr(y, x) {
      return [0];
    }
    intPtr(y, x) {
      return [0];
    }
    static zeros(rows, cols, type) {
      return new MockMat(rows, cols, type);
    }
  },
  MatVector: class MockMatVector {
    constructor() {
      this.vectors = [];
    }
    size() {
      return this.vectors.length;
    }
    get(i) {
      return this.vectors[i];
    }
    push_back(mat) {
      this.vectors.push(mat);
    }
    delete() {}
  },
  Size: class MockSize {
    constructor(width, height) {
      this.width = width;
      this.height = height;
    }
  },
  Point: class MockPoint {
    constructor(x, y) {
      this.x = x;
      this.y = y;
    }
  },
  Scalar: class MockScalar {
    constructor(...values) {
      this.values = values;
    }
  },
  CV_8U: 0,
  CV_8UC1: 0,
  CV_32S: 4,
  CV_32SC2: 12,
  CV_32F: 5,
  COLOR_RGBA2GRAY: 7,
  COLOR_GRAY2BGR: 8,
  NORM_MINMAX: 32,
  MORPH_RECT: 0,
  RETR_EXTERNAL: 0,
  CHAIN_APPROX_SIMPLE: 2,
  drawContours: vi.fn(),
  findContours: vi.fn(),
};

// Mock console to reduce noise in tests
global.console = {
  ...console,
  log: vi.fn(),
  error: console.error, // Keep error for debugging
};
