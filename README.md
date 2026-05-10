# AWS-Artists-With-Soul-

Adversarial Image Poisoning tool to protect artwork from unauthorized AI training. 

## Changelog

### Performance Optimization Update
- **Removed Celery Network Timeout**: Completely bypassed the Celery queue in `app.py`. The app no longer waits for a 30-second Redis timeout when processing locally.
- **Fixed Grad-CAM CPU Bottleneck**: Grad-CAM overlays are now computed using the downscaled 512px tensors instead of running heavy ResNet-50 backward passes on massive 12-Megapixel original images.
- **Optimized Image Resizing**: Switched from the extremely slow `Image.LANCZOS` convolution filter to `Image.BILINEAR` for resizing heatmaps and upscaling perturbations, saving several seconds per image.
- **Base64 JSON Reduction**: Response payload preview images are now capped at 1200px and 80% JPEG quality, dropping JSON serialization time from minutes to milliseconds (full-res images are still preserved in a separate `poisoned_download` field).
- **GPU Environment Setup**: Switched to a standard Windows Python 3.13 environment (`venv_gpu_313`) to support official NVIDIA CUDA 12.4 binaries.

*(Logs will be updated here with every new change going forward).*

### Setting up GPU Acceleration (CUDA)
By default, the app will run on CPU if you only installed standard `torch`. To unlock massive performance gains, install PyTorch with CUDA support.

Run this in your terminal:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
The application (`poison_engine.py`) automatically detects CUDA and switches computing from CPU to GPU without any code changes required.