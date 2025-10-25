# Image Mask Generator

🎨 A fully client-side web application for creating projection masks from photos of wall collages.

**[Live Demo](https://ufxela.github.io/image-mask-generator/)** ✨

## What is it?

Image Mask Generator helps you highlight specific items on your wall using a projector. Take a photo from your projector's perspective, automatically segment it into regions, select the pieces you want to highlight, and generate a black-and-white mask for projection.

### Use Case

1. You have a collage of posters/photos on your wall
2. You want to use a projector to highlight specific items
3. Take a photo from the projector's viewpoint
4. Upload to IMG and select regions to highlight
5. Project the generated mask to illuminate only your selected items

## Features

✅ **Fully Client-Side** - No backend, all processing happens in your browser
✅ **Automatic Segmentation** - Uses OpenCV.js edge detection to find distinct regions
✅ **Interactive Selection** - Hover to preview, click to select regions
✅ **Black & White Masks** - Generate projection-ready images
✅ **Adjustable Sensitivity** - Control how the image is segmented
✅ **Zero Installation** - Works directly in your browser

## Quick Start

**Option 1: Use the live demo**
Visit: [https://ufxela.github.io/image-mask-generator/](https://ufxela.github.io/image-mask-generator/)

**Option 2: Run locally**
```bash
npm install
npm run dev
# Visit http://localhost:5173
```

## How to Use

1. **Upload Image**: Click "Choose Image" and select a photo (JPEG/PNG)
2. **Adjust Sensitivity**: Use the slider (1-10) to control region detail
   - Lower values = fewer, larger regions
   - Higher values = more, smaller regions
3. **Segment**: Click "Segment Image" to divide into selectable regions
4. **Select Regions**: Hover over regions to highlight, click to select/deselect
5. **Create Mask**: Click "Create Mask" to generate the black-and-white image
6. **Download**: Click "Download Mask" to save your projection mask

## Technology

- **React** - UI framework
- **Vite** - Build tool and dev server
- **OpenCV.js** - Computer vision (edge detection, contours)
- **HTML5 Canvas** - Image rendering
- **GitHub Actions** - Automated deployment
- **GitHub Pages** - Free hosting

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Run unit tests
npm run test

# Run tests once
npm run test:run

# Run tests with UI
npm run test:ui

# Build for production
npm run build

# Preview production build
npm run preview

# Deploy to GitHub Pages
npm run deploy
```

## Documentation

- **[Testing Guide](docs/TESTING.md)** - Unit tests, integration tests, and manual testing procedures
- **[Segmentation Algorithm](docs/SEGMENTATION_ALGORITHM.md)** - Technical deep-dive into the watershed algorithm

## How It Works

The segmentation pipeline uses **watershed segmentation** for complete image coverage:
1. Convert image to grayscale
2. Apply Gaussian blur to reduce noise
3. Compute gradient magnitude (Sobel operator) to detect edge strength
4. Create markers on a regular grid (spacing based on sensitivity)
5. Apply watershed algorithm using gradient to guide region boundaries
6. Extract regions - every pixel belongs to exactly one region
7. Create compact, blob-like regions (< 1/10th image dimensions)
8. Allow interactive selection of regions
9. Generate binary mask from selected regions

**Key Features:**
- Complete coverage: No gaps between regions
- Edge-aware: Boundaries follow strong edges in the image
- Size-constrained: Regions are compact and uniformly sized
- Multiple segments per object: Select multiple regions to highlight complete objects

## License

MIT