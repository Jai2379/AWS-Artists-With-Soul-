import time
from PIL import Image
import io
import json
from poison_engine import _load_image_as_tensor, _get_model, fgsm_attack, compute_psnr, _tensor_to_image, get_predictions, embed_watermark, generate_gradcam_overlay
from app import _process_single_image

# 1. Create a dummy high-res image (2000x2000)
print("Generating dummy 2000x2000 image...")
img = Image.new('RGB', (2000, 2000), color = 'red')

print("Running _process_single_image...")
start_time = time.time()
result = _process_single_image(img, 0.01, "fgsm", None, "")
end_time = time.time()

print(f"\n--- TIMING METRICS ---")
print(f"Total time (including base64 encoding): {end_time - start_time:.2f} seconds")
print("\n--- DETAILED BREAKDOWN ---")
print(json.dumps(result["timings"], indent=4))
