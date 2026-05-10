# AWS-Artists-With-Soul-

Adversarial Image Poisoning tool to protect artwork from unauthorized AI training. 

## Changelog

### Performance Optimization Update
- **Removed Celery Network Timeout**: Completely bypassed the Celery queue in `app.py`. The app no longer waits for a 30-second Redis timeout when processing locally.
- **Fixed Grad-CAM CPU Bottleneck**: Grad-CAM overlays are now computed using the downscaled 512px tensors instead of running heavy ResNet-50 backward passes on massive 12-Megapixel original images.
- **Optimized Image Resizing**: Switched from the extremely slow `Image.LANCZOS` convolution filter to `Image.BILINEAR` for resizing heatmaps and upscaling perturbations, saving several seconds per image.
- **Base64 JSON Reduction**: Response payload preview images are now capped at 1200px and 80% JPEG quality, dropping JSON serialization time from minutes to milliseconds (full-res images are still preserved in a separate `poisoned_download` field).

*(Logs will be updated here with every new change going forward).*