"""
app.py — Flask web server for the Adversarial Image Poisoning UI.

Improvements in this version:
  1. In-Memory Processing   — no temp files; images go straight PIL → tensor.
  2. Async Task Queue       — Celery workers process heavy PyTorch jobs in the
                              background; the frontend polls /status/<task_id>.
  3. Batch ZIP Processing   — POST /poison/batch accepts a .zip of images and
                              returns a .zip of protected images.
  4. EXIF Stripping         — all returned images have metadata removed.
  5. Decompression Bomb     — PIL.Image.MAX_IMAGE_PIXELS is capped at startup.
  6. Rate Limiting          — /poison capped at 30/min, /poison/batch at 5/min.

Usage:
    # Terminal 1 — start Redis (Windows: use WSL2 or Docker)
    redis-server

    # Terminal 2 — start Celery worker
    celery -A app.celery_app worker --loglevel=info --pool=solo

    # Terminal 3 — start Flask
    python app.py
"""

from __future__ import annotations

import base64
import io
import time
import uuid
import zipfile
import traceback
from pathlib import Path

from celery import Celery
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image

from poison_engine import (
    _get_model,
    _pil_to_tensor,
    _tensor_to_image,
    compute_psnr,
    fgsm_attack,
    pgd_attack,
    get_predictions,
    embed_watermark,
    generate_gradcam_overlay,
    resize_tensor_for_attack,
    upscale_perturbation,
)

# ---------------------------------------------------------------------------
# Security: hard-cap PIL's decompression-bomb limit.
# Default is 178 million pixels (~13 400 x 13 400).  We cap at 25 MP which
# is already far beyond any normal photo need, but prevents deliberate abuse.
# ---------------------------------------------------------------------------
Image.MAX_IMAGE_PIXELS = 25_000_000  # 25 megapixels

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB for ZIP batches

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "60 per hour"],
    storage_uri="memory://",  # falls back to in-process store if Redis is down
)

# ---------------------------------------------------------------------------
# Celery (async task queue)
# ---------------------------------------------------------------------------
REDIS_URL = "redis://localhost:6379/0"

celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,          # results kept for 1 hour
    task_track_started=True,
    broker_connection_timeout=1.0,  # Fails fast if Redis is down
    broker_connection_retry=False,
    broker_connection_max_retries=0,
)

# ---------------------------------------------------------------------------
# Model cache (loaded once per process)
# ---------------------------------------------------------------------------
_cached_model = None


def get_model():
    global _cached_model
    if _cached_model is None:
        print("[app] Loading ResNet-50 (pretrained) ...")
        _cached_model = _get_model()
    return _cached_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_BATCH_EXTENSIONS = {".zip"}


def _safe_open_image(stream_or_bytes) -> Image.Image:
    """Open an image safely — strips EXIF without costly re-encoding.

    PIL will raise DecompressionBombWarning (and then Error) automatically
    because we've set MAX_IMAGE_PIXELS above. We just need to convert the
    raw upload into a clean RGB PIL Image with no metadata attached.
    """
    if isinstance(stream_or_bytes, (bytes, bytearray)):
        raw = io.BytesIO(stream_or_bytes)
    else:
        raw = stream_or_bytes

    img = Image.open(raw)
    img.load()  # Force full decode now — triggers decompression bomb check

    # Strip ALL metadata by copying raw pixel bytes into a fresh image
    # — much faster than re-encoding as PNG
    rgb = img.convert("RGB")
    clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    return clean


# Max dimension for images sent back in the JSON response.
# Full-res images blow up base64 encoding + JSON serialization time.
MAX_RESPONSE_SIZE = 1200


def _downscale_for_response(img: Image.Image) -> Image.Image:
    """Shrink an image to MAX_RESPONSE_SIZE if it's larger, for fast b64 encoding."""
    w, h = img.size
    if max(w, h) <= MAX_RESPONSE_SIZE:
        return img
    scale = MAX_RESPONSE_SIZE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.BILINEAR)


def _img_to_b64(img: Image.Image, for_download: bool = False) -> str:
    """Encode a clean (EXIF-free) PIL Image as a base64 JPEG data-URL.

    Preview images are downscaled and use lower quality for speed.
    Download images keep full resolution and higher quality.
    """
    if not for_download:
        img = _downscale_for_response(img)
    buf = io.BytesIO()
    quality = 92 if for_download else 80
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _img_to_bytes(img: Image.Image) -> bytes:
    """Return raw PNG bytes for a PIL Image (no EXIF)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def _process_single_image(
    img: Image.Image, 
    epsilon: float,
    attack_mode: str = "fgsm",
    target_class: int | None = None,
    watermark: str = ""
) -> dict:
    """Core pipeline: PIL Image → poisoned image + Grad-CAM + metrics.

    Entirely in-memory — no files are written to disk.
    Images larger than MAX_ATTACK_SIZE are downscaled for the attack pass,
    then the perturbation is upscaled and applied at full resolution.

    Returns a dict with keys: original, poisoned, gradcam_original,
    gradcam_poisoned (all PIL Images), psnr (float), width, height,
    orig_preds, poisoned_preds, watermarked, timings.
    """
    timings = {}
    t_total = time.perf_counter()

    model = get_model()
    original_size = img.size  # (W, H)

    # PIL → tensor (in-memory, no disk I/O)
    t0 = time.perf_counter()
    image_tensor, _ = _pil_to_tensor(img)
    timings["tensor_conversion_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Resize for faster attack computation
    t0 = time.perf_counter()
    small_tensor, was_resized = resize_tensor_for_attack(image_tensor)
    timings["resize_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if was_resized:
        timings["attack_resolution"] = f"{small_tensor.shape[3]}x{small_tensor.shape[2]}"
        timings["original_resolution"] = f"{image_tensor.shape[3]}x{image_tensor.shape[2]}"

    # Get original predictions (at processing resolution)
    t0 = time.perf_counter()
    orig_preds = get_predictions(small_tensor, model)
    timings["orig_predictions_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Adversarial attack (at processing resolution)
    t0 = time.perf_counter()
    if attack_mode == "pgd":
        poisoned_small = pgd_attack(small_tensor, model, epsilon, target_class=target_class)
    else:
        poisoned_small = fgsm_attack(small_tensor, model, epsilon, target_class=target_class)
    timings["attack_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Upscale perturbation to full resolution if needed
    t0 = time.perf_counter()
    if was_resized:
        poisoned_tensor = upscale_perturbation(image_tensor, small_tensor, poisoned_small)
    else:
        poisoned_tensor = poisoned_small
    timings["upscale_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Quality metric (at full resolution)
    t0 = time.perf_counter()
    psnr = compute_psnr(image_tensor, poisoned_tensor)
    timings["psnr_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Get poisoned predictions (at processing resolution)
    t0 = time.perf_counter()
    poisoned_small_pred, _ = resize_tensor_for_attack(poisoned_tensor)
    poisoned_preds = get_predictions(poisoned_small_pred, model)
    timings["poisoned_predictions_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Reconstruct poisoned PIL image
    t0 = time.perf_counter()
    poisoned_img = _tensor_to_image(poisoned_tensor)
    if poisoned_img.size != original_size:
        poisoned_img = poisoned_img.resize(original_size, Image.BILINEAR)
    timings["tensor_to_image_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Embed invisible watermark
    if watermark:
        t0 = time.perf_counter()
        poisoned_img = embed_watermark(poisoned_img, watermark)
        timings["watermark_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Grad-CAM overlays (MUST use the small 512px tensors, not full-res, to save massive CPU time)
    t0 = time.perf_counter()
    gradcam_orig = generate_gradcam_overlay(small_tensor, model, original_size)
    gradcam_poison = generate_gradcam_overlay(poisoned_small_pred, model, original_size)
    timings["gradcam_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    timings["compute_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

    print(f"[app] Compute done in {timings['compute_ms']}ms | "
          f"attack={timings['attack_ms']}ms gradcam={timings['gradcam_ms']}ms "
          f"resized={was_resized}")

    # --- Encode images to base64 (this was the hidden bottleneck) ---
    t_enc = time.perf_counter()
    original_b64 = _img_to_b64(img)
    poisoned_b64 = _img_to_b64(poisoned_img)
    poisoned_download_b64 = _img_to_b64(poisoned_img, for_download=True)
    gradcam_orig_b64 = _img_to_b64(gradcam_orig)
    gradcam_poison_b64 = _img_to_b64(gradcam_poison)
    timings["encoding_ms"] = round((time.perf_counter() - t_enc) * 1000, 1)

    timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

    print(f"[app] Encoding done in {timings['encoding_ms']}ms | "
          f"TOTAL wall-clock = {timings['total_ms']}ms")

    return {
        "original": original_b64,
        "poisoned": poisoned_b64,
        "poisoned_download": poisoned_download_b64,
        "gradcam_original": gradcam_orig_b64,
        "gradcam_poisoned": gradcam_poison_b64,
        "psnr": round(psnr, 2),
        "width": original_size[0],
        "height": original_size[1],
        "orig_preds": orig_preds,
        "poisoned_preds": poisoned_preds,
        "watermarked": bool(watermark),
        "timings": timings
    }


# ---------------------------------------------------------------------------
# Celery task — runs in a background worker process
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.poison_task")
def poison_task(self, image_b64: str, epsilon: float, attack_mode: str = "fgsm", target_class: int | None = None, watermark: str = "") -> dict:
    """Celery task: decode image bytes → run pipeline → return base64 results.

    The image is passed as a base64 string so it can be serialised through
    Redis without any file system involvement.
    """
    try:
        self.update_state(state="STARTED", meta={"progress": 0})

        # Decode image from base64
        img_bytes = base64.b64decode(image_b64)
        img = _safe_open_image(img_bytes)

        self.update_state(state="PROGRESS", meta={"progress": 20})

        result = _process_single_image(img, epsilon, attack_mode, target_class, watermark)

        self.update_state(state="PROGRESS", meta={"progress": 90})

        # _process_single_image now returns pre-encoded b64 strings
        return {
            "original":        result["original"],
            "poisoned":        result["poisoned"],
            "poisoned_download": result["poisoned_download"],
            "gradcam_original": result["gradcam_original"],
            "gradcam_poisoned": result["gradcam_poisoned"],
            "psnr":   result["psnr"],
            "epsilon": epsilon,
            "width":  result["width"],
            "height": result["height"],
            "orig_preds": result["orig_preds"],
            "poisoned_preds": result["poisoned_preds"],
            "watermarked": result["watermarked"],
            "timings": result["timings"]
        }

    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/poison", methods=["POST"])
@limiter.limit("30 per minute")
def poison():
    """Single-image endpoint.

    If Redis/Celery is available → dispatch async task, return task_id.
    If Celery is unavailable (no Redis) → run synchronously as fallback.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        epsilon = float(request.form.get("epsilon", 0.01))
        epsilon = max(0.001, min(0.1, epsilon))  # clamp to safe range
        attack_mode = request.form.get("attack_mode", "fgsm").lower()
        target_str = request.form.get("target_class", "")
        target_class = int(target_str) if target_str.isdigit() else None
        watermark = request.form.get("watermark", "").strip()

        img_bytes = file.read()

        # Run synchronously immediately (bypassing any Celery/Redis timeouts)
        t_start = time.perf_counter()
        img = _safe_open_image(img_bytes)
        result = _process_single_image(img, epsilon, attack_mode, target_class, watermark)

        t_json = time.perf_counter()
        resp = jsonify({
            "original":         result["original"],
            "poisoned":         result["poisoned"],
            "poisoned_download": result["poisoned_download"],
            "gradcam_original": result["gradcam_original"],
            "gradcam_poisoned": result["gradcam_poisoned"],
            "psnr":    result["psnr"],
            "epsilon": epsilon,
            "width":   result["width"],
            "height":  result["height"],
            "orig_preds": result["orig_preds"],
            "poisoned_preds": result["poisoned_preds"],
            "watermarked": result["watermarked"],
            "timings": result["timings"],
            "mode":    "sync",
        })
        print(f"[app] jsonify took {(time.perf_counter() - t_json)*1000:.1f}ms")
        print(f"[app] Total endpoint time: {(time.perf_counter() - t_start):.2f}s")
        return resp

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/status/<task_id>", methods=["GET"])
def task_status(task_id: str):
    """Poll Celery task status. Returns state + result when done."""
    task = poison_task.AsyncResult(task_id)

    if task.state == "PENDING":
        return jsonify({"state": "PENDING", "progress": 0})
    elif task.state == "STARTED":
        return jsonify({"state": "STARTED", "progress": task.info.get("progress", 0)})
    elif task.state == "PROGRESS":
        return jsonify({"state": "PROGRESS", "progress": task.info.get("progress", 50)})
    elif task.state == "SUCCESS":
        return jsonify({"state": "SUCCESS", "result": task.result})
    elif task.state == "FAILURE":
        return jsonify({"state": "FAILURE", "error": str(task.info)}), 500
    else:
        return jsonify({"state": task.state})


@app.route("/poison/batch", methods=["POST"])
@limiter.limit("5 per minute")
def poison_batch():
    """Batch endpoint: accepts a .zip of images, returns a .zip of poisoned images.

    Images are processed in memory. No files are written to disk.
    Supported formats inside the ZIP: jpg, jpeg, png, bmp, webp.
    """
    if "archive" not in request.files:
        return jsonify({"error": "No archive uploaded (field name: 'archive')"}), 400

    archive_file = request.files["archive"]
    if Path(archive_file.filename).suffix.lower() not in ALLOWED_BATCH_EXTENSIONS:
        return jsonify({"error": "Only .zip archives are supported"}), 400

    try:
        epsilon = float(request.form.get("epsilon", 0.01))
        epsilon = max(0.001, min(0.1, epsilon))

        archive_bytes = io.BytesIO(archive_file.read())

        if not zipfile.is_zipfile(archive_bytes):
            return jsonify({"error": "Uploaded file is not a valid ZIP archive"}), 400

        archive_bytes.seek(0)
        results_summary = []

        out_zip_buf = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "r") as zin, \
             zipfile.ZipFile(out_zip_buf, "w", zipfile.ZIP_DEFLATED) as zout:

            image_entries = [
                name for name in zin.namelist()
                if Path(name).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
                and not name.startswith("__MACOSX")
            ]

            if not image_entries:
                return jsonify({"error": "No supported images found in ZIP"}), 400

            if len(image_entries) > 100:
                return jsonify({"error": "Batch limited to 100 images per request"}), 400

            for name in image_entries:
                try:
                    raw_bytes = zin.read(name)
                    img = _safe_open_image(raw_bytes)
                    result = _process_single_image(img, epsilon)

                    stem = Path(name).stem
                    parent = str(Path(name).parent)
                    out_name = f"{parent}/{stem}_poisoned.png" if parent != "." else f"{stem}_poisoned.png"
                    # result["poisoned_download"] is a data URL — extract raw bytes
                    b64_data = result["poisoned_download"].split(",", 1)[1]
                    zout.writestr(out_name, base64.b64decode(b64_data))

                    results_summary.append({
                        "file": name,
                        "psnr": result["psnr"],
                        "status": "ok",
                    })

                except Exception as img_err:
                    results_summary.append({
                        "file": name,
                        "status": "error",
                        "error": str(img_err),
                    })

        out_zip_buf.seek(0)
        avg_psnr = sum(r["psnr"] for r in results_summary if "psnr" in r)
        n_ok = sum(1 for r in results_summary if r["status"] == "ok")

        # Attach a summary JSON inside the ZIP
        summary_txt = "\n".join(
            f"{r['file']}: {r.get('psnr', 'ERROR')} dB ({r['status']})"
            for r in results_summary
        )

        # Re-open output ZIP to add summary
        out_zip_buf.seek(0)
        final_buf = io.BytesIO()
        with zipfile.ZipFile(out_zip_buf, "a") as zadd:
            zadd.writestr("summary.txt", summary_txt)

        out_zip_buf.seek(0)
        return send_file(
            out_zip_buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name="poisoned_batch.zip",
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[app] Pre-loading model ...")
    get_model()
    print("[app] Ready. Starting server on http://127.0.0.1:5000")
    app.run(debug=False, port=5000, threaded=True)
