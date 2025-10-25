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

# Build for production
npm run build

# Preview production build
npm run preview

# Deploy to GitHub Pages
npm run deploy
```

## How It Works

The segmentation pipeline:
1. Convert image to grayscale
2. Apply Gaussian blur to reduce noise
3. Detect edges using Canny edge detector
4. Dilate edges to close gaps
5. Find contours (region boundaries)
6. Filter by minimum area to remove noise
7. Allow interactive selection of regions
8. Generate binary mask from selected regions

## License

MIT