# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image Mask Generator (IMG) is a tool for generating black and white image masks that can be projected onto walls to highlight physical objects. The workflow is:
1. User takes a photo from the projector's perspective
2. User uploads the photo to IMG
3. User creates an image mask using the UI
4. User projects the generated mask onto the wall

## Development Commands

**Local Development:**
```bash
# Install dependencies
npm install

# Start development server (http://localhost:5173)
npm run dev

# Build for production
npm run build

# Preview production build locally
npm run preview
```

**Testing:**
```bash
# Run unit tests (Vitest)
npm run test

# Run unit tests once
npm run test:run

# Run unit tests with UI
npm run test:ui

# Run browser integration tests (Playwright)
npm run test:integration

# Run integration tests with UI
npm run test:integration:ui

# Debug integration tests
npm run test:integration:debug
```

**Deployment:**
The app automatically deploys to GitHub Pages via GitHub Actions on every push to `main` branch.

Manual deployment (if needed):
```bash
npm run deploy
```

View the deployed app at: `https://ufxela.github.io/image-mask-generator/`

## Architecture

**React SPA with Vite:**
Modern React application built with Vite for fast development and optimized production builds.

**Project Structure:**
```
├── src/
│   ├── App.jsx                 # Main application component
│   ├── App.css                 # Application styles
│   ├── main.jsx               # React entry point
│   ├── hooks/
│   │   └── useOpenCV.js       # Custom hook for OpenCV.js loading
│   └── utils/
│       └── segmentation.js    # OpenCV segmentation utilities
├── .github/
│   └── workflows/
│       └── deploy.yml         # GitHub Actions deployment workflow
├── index.html                 # HTML entry point (loads OpenCV.js CDN)
├── vite.config.js            # Vite configuration for GitHub Pages
└── package.json              # Dependencies and scripts
```

**Core Modules:**

1. **App.jsx** - Main application component
   - Manages all application state (image, regions, selection)
   - Handles user interactions (upload, segment, select, create mask)
   - Coordinates between OpenCV utilities and UI rendering
   - Uses React hooks (useState, useCallback, useRef) for state management

2. **useOpenCV.js** - Custom React hook
   - Monitors OpenCV.js loading from CDN
   - Returns `{ cv, loading, error }` to components
   - Polls for OpenCV availability with timeout handling
   - Ensures OpenCV is ready before allowing image operations

3. **segmentation.js** - Computer vision utilities
   - `segmentImage()`: Watershed segmentation for complete coverage
     - Pipeline: Grayscale → Blur → Gradient (Sobel) → Grid markers → Watershed
     - Ensures every pixel belongs to exactly one region (no gaps)
     - Creates compact, blob-like regions (< 1/10th image dimensions)
     - Gradient guides watershed to follow edges in the image
     - Sensitivity controls marker spacing (lower = fewer/larger regions)
     - Grid-based markers ensure size constraints and uniform coverage
   - `drawSegmentation()`: Renders regions with color-coded boundaries
     - Red (unselected), Green (selected), Yellow (hover)
   - `findRegionAtPoint()`: Hit-testing for interactive selection
     - Bounding box optimization + pixel-perfect mask checking
   - `createMask()`: Generates binary black/white mask from selection
   - `cleanupMats()`: Memory management for OpenCV WebAssembly objects

4. **GitHub Actions Deployment** (.github/workflows/deploy.yml)
   - Triggers on push to `main` branch
   - Builds React app with Vite
   - Deploys to GitHub Pages automatically
   - Uses artifacts for build/deploy separation

## Testing

**Unit Tests (Vitest):**
Located in `src/tests/`, these test core utility functions:
- Memory management (`cleanupMats`)
- Hit-testing (`findRegionAtPoint`)
- Coordinate transformations (`getCanvasMousePosition`)
- Algorithm properties (grid spacing calculations)

19 unit tests covering edge cases, null handling, and boundary conditions.

**Integration Tests (Playwright):**
Located in `tests/integration/`, these test the full application with real OpenCV.js:

1. **Full Segmentation Pipeline Test:**
   - Loads the app and waits for OpenCV.js
   - Uploads test image (5006x3534 pixels)
   - Performs segmentation
   - Validates complete pipeline execution:
     - Image downscaling (17.7M → 2.8M pixels)
     - All 8 segmentation steps
     - Region extraction and validation
   - Tests region selection and mask creation

2. **Sensitivity Variation Test:**
   - Tests segmentation at different sensitivity levels
   - Validates that higher sensitivity produces more regions
   - Ensures algorithm behaves consistently across parameters

**Test Results:**
The integration tests verify that:
- ✅ Segmentation completes without errors
- ✅ Memory optimizations work (cropped masks, immediate cleanup)
- ✅ Watershed algorithm executes successfully
- ✅ Complete image coverage (every pixel in exactly one region)
- ✅ Region count and properties are reasonable

## Technology Stack

- **React 18**: Component-based UI framework
- **Vite**: Build tool and development server
  - Fast HMR (Hot Module Replacement)
  - Optimized production builds with code splitting
  - Configured for GitHub Pages deployment (base path)
- **OpenCV.js 4.5.2**: Computer vision algorithms (loaded from CDN)
  - Canny edge detection
  - Contour detection (findContours with RETR_EXTERNAL)
  - Morphological operations (dilate, blur)
  - Drawing primitives
  - Runs in WebAssembly for performance
- **HTML5 Canvas**: Rendering and interactive image display
- **File API**: Client-side image upload (no backend required)
- **GitHub Actions**: Automated CI/CD pipeline
- **GitHub Pages**: Free static hosting

## Performance Considerations

**Current Limitations:**

1. **Large Images**: Images > 2000x2000 may cause slowdown during segmentation
   - Consider downscaling large images before processing
   - Add image resize option in future iteration

2. **Memory Usage**: Each region stores a full-size mask (cv.Mat)
   - For images with 100+ regions, memory usage can be significant
   - Consider storing contours only and regenerating masks on-demand

3. **Segmentation Quality**:
   - Works best on images with clear edges and distinct objects
   - May struggle with gradients, shadows, or textured surfaces
   - Sensitivity slider helps but manual region merging could improve UX

4. **Browser Compatibility**:
   - Requires WebAssembly support (all modern browsers)
   - OpenCV.js loads ~8MB from CDN (one-time download)
   - Works offline after initial load (if CDN is cached)

**Potential Improvements:**

- Add image preprocessing options (contrast, brightness adjustment)
- Implement region merging/splitting tools
- Add "magic wand" selection tool using color similarity
- Consider using TensorFlow.js semantic segmentation for better object detection
- Add undo/redo functionality for selections
- Implement polygon drawing for manual region definition
- Add erosion/dilation controls to refine region boundaries
- Lazy load OpenCV.js only when image is uploaded (reduce initial bundle)
- Add image compression/resize before processing large images
- Implement worker threads for segmentation to keep UI responsive

## GitHub Pages Setup

**First-Time Setup:**

1. Go to repository Settings → Pages
2. Set Source to "GitHub Actions"
3. Push to `main` branch to trigger deployment
4. Check Actions tab for deployment progress
5. Once complete, site will be live at `https://ufxela.github.io/image-mask-generator/`

**Important Configuration:**

- `vite.config.js` has `base: '/image-mask-generator/'` for GitHub Pages routing
- GitHub Actions workflow in `.github/workflows/deploy.yml` handles build and deployment
- No secrets or environment variables required (fully client-side app)

**Troubleshooting:**

- If deployment fails, check Actions tab for error logs
- Ensure repository has Pages enabled in Settings
- Verify Node.js version compatibility (requires Node 18+)
- Check that `package-lock.json` is committed for reproducible builds
