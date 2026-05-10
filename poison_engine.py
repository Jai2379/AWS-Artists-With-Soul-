"""
poison_engine.py — Adversarial Image Poisoning via FGSM

Applies nearly-invisible adversarial noise to images so they become unreliable
training samples for AI models, while preserving visual quality for humans.

Algorithm:
    1. Load image and pre-process it for a pretrained classifier (ResNet-50).
    2. Forward-pass to get the model's top predicted class.
    3. Back-propagate the loss for that class to obtain input gradients.
    4. Apply the Fast Gradient Sign Method (FGSM): perturb = ε · sign(∇_x L)
    5. Clamp the result to valid pixel range and save as PNG.

Dependencies (all local, no cloud APIs):
    pip install torch torchvision Pillow numpy

Usage:
    python poison_engine.py photo.jpg              # default ε = 0.01
    python poison_engine.py photo.jpg --epsilon 0.03
    python poison_engine.py photo.jpg -e 0.02 -o protected.png

Author:  Auto-generated adversarial-ML utility
License: MIT
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet normalisation parameters (used by every torchvision pretrained model)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Device configuration for GPU acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recommended epsilon ranges for different use-cases
EPSILON_GUIDE = {
    "ultra-subtle":  0.005,   # PSNR ≈ 46+ dB — virtually invisible
    "recommended":   0.01,    # PSNR ≈ 40 dB  — good balance (default)
    "moderate":      0.02,    # PSNR ≈ 34 dB  — noticeable on zoom
    "aggressive":    0.04,    # PSNR ≈ 28 dB  — visible artifacts
}


# ---------------------------------------------------------------------------
# Image ↔ Tensor helpers
# ---------------------------------------------------------------------------

def _load_image_as_tensor(image_path: str | Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an image file and return a [1, 3, H, W] float tensor in [0, 1]
    plus the original (width, height) for later saving."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)

    to_tensor = transforms.ToTensor()  # scales uint8 → [0, 1]
    tensor = to_tensor(img).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
    return tensor, original_size


def _pil_to_tensor(img: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert an already-open PIL Image directly to a [1, 3, H, W] float
    tensor in [0, 1] — no disk I/O involved.

    Args:
        img: A PIL Image (any mode; will be converted to RGB internally).

    Returns:
        Tuple of (tensor, (width, height)).
    """
    img = img.convert("RGB")
    original_size = img.size  # (W, H)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
    return tensor, original_size


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] float tensor in [0, 1] back to a PIL Image."""
    arr = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Quality metric
# ---------------------------------------------------------------------------

def compute_psnr(original: torch.Tensor, poisoned: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio between two [0, 1] tensors (in dB).

    PSNR > 30 dB is generally considered imperceptible for photographic images.
    PSNR > 40 dB is virtually indistinguishable.
    """
    mse = F.mse_loss(original, poisoned).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Core poisoning logic
# ---------------------------------------------------------------------------

def _get_model() -> torch.nn.Module:
    """Return a pretrained ResNet-50 in eval mode."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.to(DEVICE)
    model.eval()
    return model


def fgsm_attack(
    image_tensor: torch.Tensor,
    model: torch.nn.Module,
    epsilon: float,
    target_class: int | None = None,
) -> torch.Tensor:
    """Apply the Fast Gradient Sign Method (FGSM) to perturb the image.
    
    If target_class is provided, minimizes loss for that class (Targeted).
    Otherwise, maximizes loss for the original predicted class (Untargeted).
    """
    x = image_tensor.clone().detach().requires_grad_(True)
    x_norm = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    logits = model(x_norm)

    if target_class is not None:
        # Targeted: we want to MINIMIZE the loss for the target class
        # Gradient ascent on loss increases it, so we subtract to minimize
        loss = F.cross_entropy(logits, torch.tensor([target_class], device=x.device))
        model.zero_grad()
        loss.backward()
        grad_sign = x.grad.data.sign()
        x_poisoned = x.data - epsilon * grad_sign
    else:
        # Untargeted: we want to MAXIMIZE the loss for the true/predicted class
        pred_class = logits.argmax(dim=1)
        loss = F.cross_entropy(logits, pred_class)
        model.zero_grad()
        loss.backward()
        grad_sign = x.grad.data.sign()
        x_poisoned = x.data + epsilon * grad_sign

    return x_poisoned.clamp(0.0, 1.0)


def pgd_attack(
    image_tensor: torch.Tensor,
    model: torch.nn.Module,
    epsilon: float,
    alpha: float | None = None,
    num_iters: int = 10,
    target_class: int | None = None,
) -> torch.Tensor:
    """Projected Gradient Descent (PGD) — stronger, iterative adversarial attack."""
    if alpha is None:
        alpha = (epsilon / num_iters) * 2  # Standard heuristic

    x = image_tensor.clone().detach()
    x_adv = x.clone().detach()
    
    if target_class is None:
        with torch.no_grad():
            orig_logits = model((x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device))
            orig_pred_class = orig_logits.argmax(dim=1)

    for _ in range(num_iters):
        x_adv.requires_grad_(True)
        x_norm = (x_adv - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        logits = model(x_norm)

        if target_class is not None:
            loss = F.cross_entropy(logits, torch.tensor([target_class], device=x.device))
            model.zero_grad()
            loss.backward()
            grad_sign = x_adv.grad.data.sign()
            x_adv = x_adv.data - alpha * grad_sign
        else:
            loss = F.cross_entropy(logits, orig_pred_class)
            model.zero_grad()
            loss.backward()
            grad_sign = x_adv.grad.data.sign()
            x_adv = x_adv.data + alpha * grad_sign

        # Project back into epsilon-ball around original image
        eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + eta, min=0.0, max=1.0).detach()

    return x_adv


def get_predictions(tensor: torch.Tensor, model: torch.nn.Module, k: int = 3) -> list[dict]:
    """Get top-k class predictions and confidence scores."""
    model.eval()
    categories = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    x_norm = (tensor.clone().detach() - IMAGENET_MEAN.to(tensor.device)) / IMAGENET_STD.to(tensor.device)
    
    with torch.no_grad():
        logits = model(x_norm)
        
    probs = F.softmax(logits, dim=1)[0]
    top_prob, top_catid = torch.topk(probs, k)
    
    results = []
    for i in range(k):
        results.append({
            "class_id": top_catid[i].item(),
            "label": categories[top_catid[i].item()].split(",")[0].title(), # Clean label
            "confidence": round(top_prob[i].item() * 100, 1)
        })
    return results


def embed_watermark(img: Image.Image, text: str) -> Image.Image:
    """Embed invisible text into the least significant bits of the image (LSB)."""
    if not text:
        return img
    
    binary_data = ''.join(format(ord(char), '08b') for char in text) + '00000000' # Null terminator
    data_len = len(binary_data)
    
    img_arr = np.array(img.convert('RGB'))
    flat = img_arr.flatten()
    
    if data_len > len(flat):
        binary_data = binary_data[:len(flat)]
        data_len = len(binary_data)
        
    for i in range(data_len):
        flat[i] = (flat[i] & ~1) | int(binary_data[i])
        
    return Image.fromarray(flat.reshape(img_arr.shape), 'RGB')


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM: extract class-discriminative heatmaps from a CNN.

    Registers forward/backward hooks on a target convolutional layer to
    capture activations and their gradients, then combines them into a
    spatial attention map that highlights the regions the model focuses on.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a given input.

        Args:
            input_tensor: [1, 3, H, W] float tensor (already normalised).
            class_idx:    Target class.  If None, uses the top prediction.

        Returns:
            2-D numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Back-propagate the score for the target class
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        # Global-average-pool the gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)  # Only positive contributions

        # Up-sample to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Clean up registered hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def generate_gradcam_overlay(
    image_tensor: torch.Tensor,
    model: torch.nn.Module,
    original_size: tuple[int, int] | None = None,
    alpha: float = 0.45,
) -> Image.Image:
    """Generate a Grad-CAM heatmap overlay on top of the source image.

    Args:
        image_tensor: [1, 3, H, W] float tensor in [0, 1] (un-normalised).
        model:        Pre-trained model (ResNet-50).
        original_size: (W, H) to resize the result to. None keeps tensor dims.
        alpha:        Opacity of the heatmap overlay.

    Returns:
        PIL Image with the Grad-CAM heatmap blended on top.
    """
    # Target: last conv block in ResNet-50 (layer4[-1])
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # Normalise for the model
    x_norm = (image_tensor.clone() - IMAGENET_MEAN.to(image_tensor.device)) / IMAGENET_STD.to(image_tensor.device)

    cam = grad_cam.generate(x_norm)
    grad_cam.remove_hooks()

    # Convert heatmap to colour using a simple jet-like colormap (no OpenCV needed)
    heatmap_rgb = _apply_colormap(cam)  # [H, W, 3] uint8

    # Build the base image
    base = _tensor_to_image(image_tensor)
    if original_size and base.size != original_size:
        base = base.resize(original_size, Image.LANCZOS)

    heatmap_img = Image.fromarray(heatmap_rgb).resize(base.size, Image.LANCZOS)

    # Blend
    overlay = Image.blend(base, heatmap_img, alpha)
    return overlay


def _apply_colormap(cam: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] 2-D array to a jet-like RGB heatmap (uint8).

    Implements a simplified jet colormap without requiring matplotlib or OpenCV.
    """
    # Clamp
    cam = np.clip(cam, 0, 1)

    # Jet-like: blue (0) -> cyan (0.25) -> green (0.5) -> yellow (0.75) -> red (1)
    r = np.clip(1.5 - np.abs(cam - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(cam - 0.5) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(cam - 0.25) * 4, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def poison_image(
    image_path: str | Path,
    epsilon: float = 0.01,
    output_path: str | Path | None = None,
    *,
    verbose: bool = True,
) -> Path:
    """Apply adversarial poison to an image and save the result as PNG.

    Args:
        image_path:  Path to the source image (JPEG, PNG, BMP, etc.).
        epsilon:     Perturbation strength.  See ``EPSILON_GUIDE`` for
                     recommended values.  Default is 0.01 (≈ 40 dB PSNR).
        output_path: Where to save the poisoned image.  Defaults to
                     ``<original_stem>_poisoned.png``.
        verbose:     If True, print progress and quality metrics.

    Returns:
        Path to the saved poisoned image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_poisoned.png")
    output_path = Path(output_path)

    if verbose:
        print(f"[poison_engine] Loading image : {image_path}")
        print(f"[poison_engine] Epsilon (eps) : {epsilon}")

    # Load
    image_tensor, original_size = _load_image_as_tensor(image_path)

    # Load model
    if verbose:
        print("[poison_engine] Loading ResNet-50 (pretrained) ...")
    model = _get_model()

    # Attack
    if verbose:
        print("[poison_engine] Computing FGSM adversarial perturbation ...")
    poisoned_tensor = fgsm_attack(image_tensor, model, epsilon)

    # Quality check
    psnr = compute_psnr(image_tensor, poisoned_tensor)
    if verbose:
        print(f"[poison_engine] PSNR          : {psnr:.2f} dB", end="")
        if psnr >= 40:  # noqa: PLR2004
            print("  (virtually invisible)")
        elif psnr >= 30:
            print("  (imperceptible to most viewers)")
        elif psnr >= 25:
            print("  (minor artifacts - may be visible on close inspection)")
        else:
            print("  (WARNING: visible noise - consider lowering epsilon)")

    # Save
    result_img = _tensor_to_image(poisoned_tensor)
    # Resize back to original dimensions in case tensor ops changed them
    if result_img.size != original_size:
        result_img = result_img.resize(original_size, Image.LANCZOS)
    result_img.save(output_path, format="PNG")

    if verbose:
        print(f"[poison_engine] Saved to      : {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply adversarial FGSM poison to an image to protect it from AI training.",
        epilog=(
            "Epsilon guide:\n"
            "  0.005  -- ultra-subtle  (PSNR ~ 46 dB, virtually invisible)\n"
            "  0.01   -- recommended   (PSNR ~ 40 dB, good balance)       [default]\n"
            "  0.02   -- moderate      (PSNR ~ 34 dB, noticeable on zoom)\n"
            "  0.04   -- aggressive    (PSNR ~ 28 dB, visible artifacts)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", type=str, help="Path to the input image file.")
    parser.add_argument(
        "-e", "--epsilon",
        type=float,
        default=0.01,
        help="Perturbation strength (default: 0.01).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: <input>_poisoned.png).",
    )

    args = parser.parse_args()
    poison_image(args.image, epsilon=args.epsilon, output_path=args.output)


if __name__ == "__main__":
    main()
