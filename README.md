# Artists With Souls (AWS)

Adversarial perturbation tool that makes artwork unreliable input for the vision models behind scraping and generative-AI training, while keeping the change below what a viewer notices. Built by a team at GDGHacks 2026 (University of Guelph). [Devpost](https://devpost.com/software/artists-with-souls)

Pipeline: an L∞-bounded FGSM or PGD perturbation against a pretrained ResNet-50 surrogate, computed at reduced resolution and applied at full resolution, with Grad-CAM and top-3 predictions before and after as evidence the model's reading changed. It ships as a Flask service with a Celery/Redis job path and as a standalone CLI.

## Attacks

Both attacks run in pixel space `x ∈ [0,1]`. ImageNet mean/std normalization happens inside the forward pass, so `ε` is measured in raw pixel units.

**FGSM** (`fgsm_attack`), one step:

```
untargeted:  x' = clip[0,1]( x + ε · sign(∇x L(f(x), ŷ)) ),   ŷ = argmax f(x)
targeted:    x' = clip[0,1]( x − ε · sign(∇x L(f(x), t)) )
```

**PGD** (`pgd_attack`), iterative with projection:

```
x₀ = x
x_{k+1} = clip[0,1]( x + clip[−ε,ε]( x_k ± α · sign(∇x L) − x ) )
```

10 iterations, step size `α = 2ε / 10`. The prediction `ŷ` is fixed from the clean image before the first step. Each iterate is projected back into the ε-ball around the original, which makes PGD the stronger of the two at equal `ε`.

`L` is cross-entropy. The surrogate is `torchvision.models.resnet50` with `IMAGENET1K_V1` weights, in eval mode, loaded once per process.

## Resolution strategy

The attack and both Grad-CAM passes run on a copy downscaled to a 512 px long side. The perturbation is then transferred to the original:

```
δ_small = x'_small − x_small
δ_full  = bilinear_upsample(δ_small, size = original)
x'_full = clip[0,1]( x_full + δ_full )
```

This cut end-to-end processing time by more than 90% compared with attacking at full resolution on CPU. Every stage is timed with `perf_counter` and returned in the response as `timings`: tensor conversion, resize, predictions, attack, upscale, PSNR, Grad-CAM and base64 encoding.

## Perturbation budget

PSNR is computed at full resolution as `10 · log10(1 / MSE)`.

| Preset | ε | Approx. PSNR |
|:--|:--|:--|
| ultra-subtle | 0.005 | 46+ dB |
| recommended (default) | 0.01 | 40 dB |
| moderate | 0.02 | 34 dB |
| aggressive | 0.04 | 28 dB |

The API clamps `ε` to `[0.001, 0.1]`.

## Grad-CAM

The custom `GradCAM` class hooks `model.layer4[-1]` (forward activations, full backward gradients). Channel weights come from global-average-pooling the gradients. The weighted activation sum goes through ReLU, bilinear upsampling and min-max normalization, then a jet-style colormap written in NumPy (no OpenCV or matplotlib) and an alpha blend at 0.45. Maps are produced for the clean and the perturbed image so the attention shift is visible side by side.

## Watermark

Optional ownership tag embedded by `embed_watermark`: the text is encoded as 8-bit ASCII plus a null terminator and written into the least significant bit of the flattened RGB bytes with a `0xFE` clear mask. It is fully vectorized in NumPy, with no per-pixel Python loop.

## HTTP API

| Method | Route | Input | Output |
|:--|:--|:--|:--|
| `GET` | `/` | | Web UI (`templates/index.html`) |
| `POST` | `/poison` | multipart: `image`, `epsilon`, `attack_mode` (`fgsm` \| `pgd`), `target_class` (ImageNet index), `watermark` | JSON: previews, full-res download, Grad-CAM pair, PSNR, top-3 predictions before/after, `timings` |
| `GET` | `/status/<task_id>` | | Celery task state, progress, result |
| `POST` | `/poison/batch` | multipart: `archive` (.zip), `epsilon` | ZIP of `<name>_poisoned.png` files plus `summary.txt` with per-file PSNR |

Previews are capped at 1200 px and encoded as JPEG q80 to keep JSON payloads small. The download copy keeps full resolution at q92. Batches allow up to 100 images and skip `__MACOSX` entries.

## Execution model

- **Celery over Redis.** `poison_task` decodes a base64 image, runs the pipeline and reports `STARTED` and `PROGRESS` states, polled through `/status/<task_id>`. JSON serialization, 1 h result expiry, 1 s broker connection timeout with retries disabled, so a missing broker fails fast.
- **In-process.** `/poison` runs the same `_process_single_image` pipeline inside the Flask worker, so a request never waits on broker availability.

## Hardening

- **Metadata.** EXIF and all other metadata are stripped by rebuilding each upload from raw RGB bytes (`Image.frombytes`), without re-encoding.
- **Decompression bombs.** `Image.MAX_IMAGE_PIXELS = 25_000_000`, and uploads are fully decoded on open so the check fires early.
- **Request size.** `MAX_CONTENT_LENGTH` is 200 MB. Image and archive extensions are allowlisted.
- **Rate limits.** Flask-Limiter, per IP: 30/min on `/poison`, 5/min on `/poison/batch`, with defaults of 200/day and 60/hour.
- **No disk I/O.** Images move PIL → tensor → PIL in memory. Nothing is written to disk during a request.

## Setup

Python 3.10+, Redis, optional NVIDIA GPU.

```bash
git clone https://github.com/Jai2379/AWS-Artists-With-Soul-.git
cd AWS-Artists-With-Soul-
pip install -r requirements.txt

# optional CUDA build of PyTorch
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python test_cuda.py
```

Run the service, one process per terminal:

```bash
redis-server
celery -A app.celery_app worker --loglevel=info --pool=solo   # solo pool for Windows
python app.py                                                 # http://127.0.0.1:5000
```

CLI, single image, full-resolution untargeted FGSM:

```bash
python poison_engine.py artwork.jpg -e 0.01 -o protected.png
```

## Repository

```
.
├── poison_engine.py        FGSM, PGD, Grad-CAM, PSNR, LSB watermark, CLI
├── app.py                  Flask routes, Celery task, rate limiting, upload sanitizing
├── templates/index.html    single-page frontend
├── test_direct.py          timing breakdown of the pipeline on a 2000×2000 image
├── test_client.py          end-to-end multipart request against /poison
├── test_celery_timeout.py  broker fail-fast check
├── test_cuda.py            CUDA availability check
└── requirements.txt
```

## Limitations and next steps

- Perturbations are optimized against a single surrogate (ResNet-50). Transfer to other architectures is not measured yet. The next step is ensemble optimization across CNN and ViT models.
- Downloads are JPEG-encoded (q92), and lossy re-encoding attenuates part of the perturbation. A lossless PNG path would keep the full signal.
