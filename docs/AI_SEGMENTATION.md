# AI Object Detection

## Overview

The AI Object Detection feature uses TensorFlow.js and the COCO-SSD model to automatically identify and segment common objects in your image. This provides a complementary approach to the watershed algorithm, with semantic understanding of what objects are in the image.

## How It Works

1. **Upload Your Image** - Load an image as usual
2. **Click "AI Object Detection"** - Opens the AI detection modal
3. **Detect Objects** - TensorFlow.js loads the model (first time only) and detects objects
4. **Select & Export** - Use the same UI to select detected regions and create masks

## No API Key Required!

Unlike the previous version that used Claude's API, this implementation is:
- ✅ **Completely Free** - No API key, no costs, no account needed
- ✅ **Client-Side** - Runs entirely in your browser using WebAssembly
- ✅ **Works Offline** - After the first model download, works without internet
- ✅ **Privacy-Preserving** - Your images never leave your device

## How COCO-SSD Works

COCO-SSD (Common Objects in Context - Single Shot MultiBox Detector) is a pre-trained deep learning model that:

1. **Analyzes Your Image** - Processes the canvas directly in your browser
2. **Detects Objects** - Identifies 80+ common object types from the COCO dataset
3. **Returns Bounding Boxes** - Provides coordinates `[x, y, width, height]` for each detected object
4. **Includes Confidence Scores** - Each detection has a confidence percentage (e.g., "person (95%)")

The model can detect: people, animals, vehicles, furniture, electronics, food items, sports equipment, and much more.

## Advantages vs Watershed

**AI Object Detection (COCO-SSD):**
- ✅ **Semantic Understanding** - Recognizes 80+ common object types
- ✅ **Object-Aware** - Identifies whole objects (people, chairs, cups, etc.)
- ✅ **Confidence Scores** - Shows how certain the model is about each detection
- ✅ **Fully Client-Side** - No API needed, runs in your browser
- ✅ **Free Forever** - No costs whatsoever
- ✅ **Works Offline** - After initial model download
- ✅ **Privacy** - Images never leave your device
- ❌ **Bounding Boxes Only** - Returns rectangles, not pixel-perfect masks
- ❌ **Limited to COCO Classes** - Only detects trained object types
- ❌ **Poor for Abstract Patterns** - Needs recognizable objects

**Watershed Segmentation:**
- ✅ **Fully Client-Side** - No API needed
- ✅ **Free** - No costs
- ✅ **Offline** - Works without internet
- ✅ **Pixel-Perfect** - Follows exact edges
- ✅ **Universal** - Works on any image (patterns, textures, etc.)
- ✅ **Complete Coverage** - Every pixel belongs to a region
- ❌ **No Semantic Understanding** - Doesn't know what objects are
- ❌ **Uniform Regions** - Creates blob-like shapes

## Current Limitations

1. **Bounding Boxes Only** - AI regions are rectangular, not contoured
   - Future: Could use COCO-SSD detections as seeds for watershed refinement

2. **COCO Dataset Only** - Limited to 80 object classes from the COCO dataset
   - Detects common objects but may miss domain-specific items
   - See [COCO classes](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd#what-objects-can-the-model-detect) for full list

3. **Model Size** - First load downloads ~5MB model
   - Subsequent loads use cached model (works offline)
   - May take 5-10 seconds on first use

4. **Performance** - Inference speed depends on your device
   - Modern computers: 1-3 seconds per image
   - Older devices may take longer
   - Uses WebGL for GPU acceleration when available

## Future Improvements

### Hybrid Approach
Combine both methods:
1. Use AI to identify semantic regions
2. Use watershed to refine boundaries
3. Best of both worlds!

### Polygon/Mask Support
- Use segmentation models instead of detection (e.g., DeepLab)
- Generate pixel-perfect masks for detected objects
- Combine with watershed for edge refinement

### Smart Region Merging
- Detect overlapping detections
- Merge or split as needed
- Handle nested objects (e.g., cup on table)

### Performance Optimization
- Model quantization for faster inference
- WebWorker support for non-blocking UI
- Progressive detection (show results as they come)
- Image downsampling for very large images

## Example Use Cases

**Best for AI Object Detection:**
- Photos with distinct, recognizable objects (books, chairs, people, animals)
- Collages where you want to select whole items quickly
- Images with common household items or furniture
- Photos where you want to select "all people" or "all cups"

**Best for Watershed:**
- Abstract patterns or textures
- Images where you want precise edge following
- Posters, artwork, or non-standard objects not in COCO dataset
- When you need complete coverage of all image regions

## Privacy & Security

**Complete Privacy:**
- Everything runs in your browser - no servers involved
- Your images NEVER leave your device
- No tracking, no analytics, no data collection
- Model is downloaded from TensorFlow.js CDN (only on first use)
- After first load, works completely offline

**Zero Cost:**
- No API keys required
- No subscriptions or accounts needed
- Completely free forever
- No hidden costs or usage limits

## Troubleshooting

**"No objects detected"**
- Image may not contain recognizable COCO objects
- Try watershed segmentation instead
- Common with abstract patterns, artwork, or unusual objects

**"Loading AI model... taking too long"**
- First load downloads ~5MB model - may take time on slow connections
- Check browser console for errors
- Ensure WebGL is enabled in your browser
- Try refreshing the page

**Model loads slowly or inference is slow**
- Older devices may take longer (10-20 seconds)
- First inference is slowest (model initialization)
- Subsequent detections on same page session are faster
- Close other tabs to free up memory

**Detections are inaccurate or missing objects**
- COCO-SSD is trained on specific object types
- May not detect custom/unusual items
- Overlapping objects may be detected as one bounding box
- Try adjusting image orientation or cropping
- Use watershed segmentation for non-standard objects

**WebGL errors**
- TensorFlow.js uses WebGL for acceleration
- Ensure WebGL is enabled in browser settings
- Update graphics drivers if needed
- May fall back to CPU (slower but works)

## Code Structure

**Model Loading** (`App.jsx:handleAISegment`)
```javascript
// Load COCO-SSD model (cached after first load)
let model = cocoModel;
if (!model) {
  model = await cocoSsd.load();
  setCocoModel(model);
}
```

**Object Detection**
```javascript
// Detect objects directly from canvas
const predictions = await model.detect(canvas);

// Each prediction has:
// - bbox: [x, y, width, height]
// - class: "person", "chair", etc.
// - score: confidence (0-1)
```

**Region Creation**
```javascript
// Convert predictions to region objects
const regions = predictions.map(prediction => ({
  contour: createRectangularContour(prediction.bbox),
  mask: createRectangularMask(prediction.bbox),
  bounds: { x, y, width, height },
  name: `${prediction.class} (${Math.round(prediction.score * 100)}%)`
}));
```

**Display**
- Uses same `drawSegmentation()` function as watershed
- Regions are compatible with selection/mask creation
- Region names show object class and confidence score

## Roadmap

- [ ] Display region names on hover (already stored, just need UI)
- [ ] Hybrid segmentation (use COCO-SSD regions as seeds for watershed refinement)
- [ ] Polygon support using TensorFlow.js segmentation models
- [ ] Support for additional models:
  - [ ] MobileNet for faster detection on mobile devices
  - [ ] DeepLab for semantic segmentation (pixel-perfect masks)
  - [ ] PoseNet for detecting people/poses
- [ ] Confidence threshold slider (filter low-confidence detections)
- [ ] Custom class filtering (e.g., "only detect people")

## Detected Object Classes

COCO-SSD can detect 80 object classes including:

**People:** person
**Vehicles:** bicycle, car, motorcycle, airplane, bus, train, truck, boat
**Animals:** bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
**Accessories:** backpack, umbrella, handbag, tie, suitcase
**Sports:** frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket
**Kitchen:** bottle, wine glass, cup, fork, knife, spoon, bowl
**Food:** banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
**Furniture:** chair, couch, potted plant, bed, dining table, toilet
**Electronics:** tv, laptop, mouse, remote, keyboard, cell phone
**Appliances:** microwave, oven, toaster, sink, refrigerator
**Indoor:** book, clock, vase, scissors, teddy bear, hair drier, toothbrush

See [full COCO dataset](http://cocodataset.org/#explore) for details.

## Feedback

Please report:
- Images where detection fails or is inaccurate
- Performance issues or slow loading
- Feature requests
- Browser compatibility issues

Open an issue at: https://github.com/ufxela/image-mask-generator/issues
