const sharp = require('sharp');
const path = require('path');

async function main() {
  console.log('Starting...');

  // Load opencv-wasm
  const { cv } = require('opencv-wasm');
  console.log('OpenCV loaded. cv.Mat:', typeof cv.Mat);

  // Dynamically import segmentation.js (ES module)
  const seg = await import('./src/utils/segmentation.js');
  console.log('Segmentation module loaded');

  // Load image
  const inputPath = path.resolve(process.argv[2] || '/Users/ufxela/image-mask-generator/test-collage.jpg');
  const outputPath = path.resolve(process.argv[3] || '/Users/ufxela/image-mask-generator/segmentation-result.png');
  const sensitivity = parseInt(process.argv[4] || '5');
  const regionSize = Math.round(10 + (sensitivity - 1) * (70 / 19));
  const mergeStrength = parseInt(process.argv[5] || '10');

  const { data, info } = await sharp(inputPath).raw().ensureAlpha().toBuffer({ resolveWithObject: true });
  console.log(`Image: ${info.width}x${info.height}, ${data.length} bytes`);

  const mat = new cv.Mat(info.height, info.width, cv.CV_8UC4);
  mat.data.set(data);
  console.log(`Mat created: ${mat.rows}x${mat.cols}`);

  console.log(`Segmenting: sensitivity=${sensitivity}, regionSize=${regionSize}, merge=${mergeStrength}`);
  const result = seg.segmentImage(mat, sensitivity, regionSize, cv, mergeStrength);
  console.log(`Result: ${result.regions.length} regions`);

  // Build label map and draw red boundaries
  const outputData = Buffer.from(data);
  const labelMap = new Int32Array(info.width * info.height).fill(-1);

  for (let ri = 0; ri < result.regions.length; ri++) {
    const region = result.regions[ri];
    const scale = 1 / (region.scaleFactor || 1);
    const b = region.bounds;
    const maskData = region.mask.data;
    for (let my = 0; my < b.height; my++) {
      for (let mx = 0; mx < b.width; mx++) {
        if (maskData[my * b.width + mx] > 0) {
          const gx = Math.round((b.x + mx) * scale);
          const gy = Math.round((b.y + my) * scale);
          if (gx >= 0 && gx < info.width && gy >= 0 && gy < info.height) {
            labelMap[gy * info.width + gx] = ri;
          }
        }
      }
    }
  }

  let boundaryCount = 0;
  for (let y = 0; y < info.height; y++) {
    for (let x = 0; x < info.width; x++) {
      const label = labelMap[y * info.width + x];
      let isBoundary = false;
      for (let dy = -1; dy <= 1 && !isBoundary; dy++) {
        for (let dx = -1; dx <= 1 && !isBoundary; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < info.width && ny >= 0 && ny < info.height) {
            if (labelMap[ny * info.width + nx] !== label) isBoundary = true;
          }
        }
      }
      if (isBoundary) {
        const idx = (y * info.width + x) * 4;
        outputData[idx] = 255; outputData[idx+1] = 0; outputData[idx+2] = 0; outputData[idx+3] = 255;
        boundaryCount++;
      }
    }
  }

  console.log(`Boundaries: ${boundaryCount} pixels (${(boundaryCount/(info.width*info.height)*100).toFixed(1)}%)`);

  await sharp(outputData, { raw: { width: info.width, height: info.height, channels: 4 } })
    .png().toFile(outputPath);
  console.log(`Saved: ${outputPath}`);

  // Cleanup
  mat.delete();
  if (result.edgeMap) result.edgeMap.delete();
  for (const r of result.regions) {
    if (r.contour && !r.contour.isDeleted()) r.contour.delete();
    if (r.mask && !r.mask.isDeleted()) r.mask.delete();
  }
}

main().catch(err => { console.error('FATAL:', err.message || err); process.exit(1); });
