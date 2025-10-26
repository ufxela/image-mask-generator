# AI Segmentation (Beta)

## Overview

The AI Segmentation feature uses Claude's vision API to intelligently identify and segment objects, text, and visual elements in your image. This provides a complementary approach to the watershed algorithm, with better semantic understanding.

## How It Works

1. **Upload Your Image** - Load an image as usual
2. **Click "AI Segment (Beta)"** - Opens the AI segmentation modal
3. **Enter API Key** - Provide your Anthropic API key (required)
4. **Segment** - Claude analyzes the image and returns bounding boxes for detected regions
5. **Select & Export** - Use the same UI to select regions and create masks

## Getting an API Key

1. Visit [console.anthropic.com](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key
5. Copy and paste it into the modal

**Note:** Your API key is stored only in your browser's memory and is never sent to any server except Anthropic's API.

## How Claude Segments Images

The AI segmentation sends your image to Claude with this prompt:

```
Analyze this image and identify distinct objects, text regions,
and visual elements that someone might want to select.

For each region, provide a bounding box with:
- name: brief description
- box: [x, y, width, height] coordinates
```

Claude returns a JSON array of detected regions, which are then converted to the same format as watershed regions and displayed in the UI.

## Advantages vs Watershed

**AI Segmentation (Claude):**
- ✅ **Semantic Understanding** - Recognizes objects, text, people, etc.
- ✅ **Better for Text** - Detects entire words, sentences, or text blocks
- ✅ **Object-Aware** - Groups related elements (e.g., "poster with red border")
- ✅ **Hierarchical** - Can identify both large objects and small details
- ❌ **Requires API Key** - Not fully client-side
- ❌ **API Costs** - ~$0.003 per image (claude-3-5-sonnet)
- ❌ **Internet Required** - Must be online
- ❌ **Bounding Boxes** - Currently returns rectangles, not pixel-perfect masks

**Watershed Segmentation:**
- ✅ **Fully Client-Side** - No API needed
- ✅ **Free** - No costs
- ✅ **Offline** - Works without internet
- ✅ **Pixel-Perfect** - Follows exact edges
- ❌ **No Semantic Understanding** - Doesn't know what objects are
- ❌ **Poor for Text** - Splits letters/words into many regions
- ❌ **Uniform Regions** - Creates blob-like shapes

## Current Limitations

1. **Bounding Boxes Only** - AI regions are rectangular, not contoured
   - Future: Could use Claude's regions as seeds for watershed refinement

2. **API Rate Limits** - Anthropic has rate limits on API calls
   - If you hit limits, wait a moment and try again

3. **Image Size** - Large images may exceed API limits
   - Currently sends the full canvas (up to ~10MB after JPEG compression)
   - May need to downsample very large images

4. **Parse Errors** - Claude occasionally returns non-JSON responses
   - The code extracts JSON from the response, but edge cases may fail

## Future Improvements

### Hybrid Approach
Combine both methods:
1. Use AI to identify semantic regions
2. Use watershed to refine boundaries
3. Best of both worlds!

### Polygon Support
- Ask Claude to return polygon vertices instead of bounding boxes
- Would require more complex prompt engineering

### Smart Region Merging
- Detect overlapping AI regions
- Merge or split as needed

### Cost Optimization
- Downsample images before sending to API
- Cache results for repeated segmentation
- Batch multiple requests

## Example Use Cases

**Best for AI Segmentation:**
- Photos with distinct objects (books, posters, people)
- Images with text you want to select (signs, labels, captions)
- Collages where you want to select whole items
- Mixed media (photos + text + graphics)

**Best for Watershed:**
- Abstract patterns or textures
- Images where you want precise edge following
- When you don't have an API key
- When you need completely client-side processing

## Privacy & Security

**Your API Key:**
- Stored only in browser memory (state variable)
- Not persisted to localStorage or cookies
- Lost when you refresh the page
- Never sent anywhere except Anthropic's API

**Your Images:**
- Converted to JPEG and sent to Anthropic's API
- Subject to [Anthropic's Privacy Policy](https://www.anthropic.com/privacy)
- Not used for model training (as of their current policy)
- Consider sensitivity before uploading private images

## API Costs

Using Claude 3.5 Sonnet (as of October 2024):
- ~1,000 tokens per image (depending on size/quality)
- Input: $3 per million tokens
- **Estimated cost: ~$0.003 per image**

For a typical user segmenting 10-20 images: **$0.03-$0.06 total**

## Troubleshooting

**"API request failed"**
- Check your API key is correct
- Ensure you have credits in your Anthropic account
- Check console for detailed error message

**"Could not parse AI response"**
- Claude occasionally returns explanatory text instead of JSON
- Try again - usually works on retry
- Report persistent issues with example images

**"Too many requests"**
- You've hit Anthropic's rate limit
- Wait 1 minute and try again
- Consider upgrading your API tier

**Regions look wrong**
- AI segmentation is best-effort
- Different images work better/worse
- Try watershed segmentation as alternative
- Or manually select multiple AI regions to build desired shape

## Code Structure

**API Call** (`App.jsx:handleAISegment`)
```javascript
// Convert canvas to base64
const imageData = canvas.toDataURL('image/jpeg', 0.8);

// Send to Anthropic API
const response = await fetch('https://api.anthropic.com/v1/messages', {
  headers: { 'x-api-key': apiKey },
  body: JSON.stringify({
    model: 'claude-3-5-sonnet-20241022',
    messages: [{ role: 'user', content: [image, prompt] }]
  })
});
```

**Region Creation**
```javascript
// Convert AI bounding boxes to region objects
const regions = aiRegions.map(region => ({
  contour: createRectangularContour(region.box),
  mask: createRectangularMask(region.box),
  bounds: region.box,
  name: region.name
}));
```

**Display**
- Uses same `drawSegmentation()` function as watershed
- Regions are compatible with selection/mask creation
- Only difference: `name` property for hover tooltips (future feature)

## Roadmap

- [ ] Display region names on hover
- [ ] Hybrid segmentation (AI + watershed refinement)
- [ ] Polygon support (not just bounding boxes)
- [ ] Support for other LLMs (GPT-4V, Gemini Vision)
- [ ] Prompt customization (let users specify what to detect)
- [ ] Region confidence scores
- [ ] Hierarchical region tree (objects within objects)

## Feedback

This is a **beta feature**! Please report:
- Images that segment poorly
- API errors or edge cases
- Feature requests
- Cost concerns

Open an issue at: https://github.com/ufxela/image-mask-generator/issues
