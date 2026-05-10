# Artists With Souls (AWS)

Protecting creative work in the age of generative AI.

Artists With Souls (AWS) is a local-first adversarial image protection tool built for artists who want more control over how their work is used online. AWS applies carefully generated perturbations to artwork that are nearly invisible to humans, but disruptive to computer vision systems commonly used in large-scale scraping and AI training pipelines.

Instead of focusing on watermarking alone, AWS treats protection as an adversarial machine learning problem — using the same techniques that expose weaknesses in neural networks to help creators defend their work.

---

# Inspiration

As AI models continue to evolve, the line between inspiration and scraping has become increasingly blurred. Many creators feel that the essence of their work is being absorbed into massive training datasets without permission or attribution.

Artists With Souls (AWS) was created as a form of digital self-defense — a way for artists to protect their visual identity and retain control over how their work is used online.

---

# How It Works

## Adversarial Protection Pipeline

At the core of AWS is an adversarial machine learning pipeline designed to disrupt computer vision models commonly used for large-scale image scraping and dataset generation.

AWS uses a pre-trained ResNet-50 model as a proxy for modern vision architectures and computes the gradient of the model’s loss with respect to the input image.

Two attack methods are currently implemented:

### FGSM (Fast Gradient Sign Method)

A fast single-step perturbation method optimized for speed and lightweight processing.

### PGD (Projected Gradient Descent)

A stronger iterative attack that remains more resilient after compression, resizing, or minor image transformations.

The perturbation process follows the adversarial formulation:

```math
x_{adv} = \text{Clip}_{x,\epsilon}\left\{x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))\right\}
```

The generated perturbations are nearly imperceptible to the human eye but significantly interfere with how neural networks interpret the image.

---

## Grad-CAM Visualization

AWS integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize how a model’s attention changes before and after protection is applied.

In many cases, the model’s focus shifts away from the primary subject toward irrelevant regions after perturbation, demonstrating how the protection disrupts feature recognition.

---

# Performance Optimizations

Early versions of AWS processed full-resolution images directly, resulting in extremely slow runtimes on CPU hardware.

To solve this, the adversarial pipeline was redesigned around a lower-resolution attack pass:

- Perturbations are generated on a downscaled 512px tensor
- Noise is then upscaled and blended back into the original image
- All processing occurs entirely in memory to avoid unnecessary disk I/O

This reduced processing time by over 90% while maintaining strong adversarial effectiveness.

---

# Privacy Features

AWS was designed around a local-first workflow focused on creator privacy.

Features include:

- Automatic EXIF metadata stripping
- No external image uploads or cloud inference requirements
- In-memory image processing pipeline
- Optional invisible LSB watermark embedding for ownership tracking

---

# Application Stack

AWS was built as a lightweight full-stack application.

## Backend

- Flask
- PyTorch
- Pillow
- NumPy
- Torchvision
- Celery + Redis for asynchronous processing

## Frontend

- Custom HTML/CSS interface
- Glassmorphism-inspired UI
- Spline 3D integrations for interactive visuals

---

# Features

- Multi-attack adversarial protection
- Batch ZIP processing
- Real-time preview generation
- Grad-CAM attention visualization
- PSNR quality monitoring
- GPU acceleration support (CUDA)
- Web UI and CLI support

---

# Protection Strength Guide

| Setting | Epsilon | Visual Quality | Visibility |
|---|---|---|---|
| Ultra-Subtle | 0.005 | Very High | Virtually invisible |
| Recommended | 0.010 | High | Indistinguishable |
| Moderate | 0.020 | Medium | Minor grain on zoom |
| Aggressive | 0.040 | Lower | Visible artifacts |

---

# Installation

## Prerequisites

- Python 3.10+
- Redis
- NVIDIA GPU with CUDA support (recommended)

---

## Clone the Repository

```bash
git clone https://github.com/taranprits2/AWS-Artists-With-Soul-.git
cd AWS-Artists-With-Soul-
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Optional CUDA Setup

For NVIDIA GPU acceleration:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

# Running AWS

## Web Application

Start the following services in separate terminals.

### Terminal 1 — Redis

```bash
redis-server
```

### Terminal 2 — Celery Worker

```bash
celery -A app.celery_app worker --loglevel=info --pool=solo
```

### Terminal 3 — Flask Application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## CLI Mode

For quick single-image protection:

```bash
python poison_engine.py my_artwork.jpg --epsilon 0.01 --output protected.png
```

---

# What We Learned

Building AWS highlighted how fragile modern deep learning systems can be. Even highly capable vision models are vulnerable to extremely small, carefully structured perturbations.

We also learned the importance of optimization when deploying ML systems on consumer hardware. Designing around lower-resolution adversarial passes made real-time processing practical without sacrificing meaningful attack strength.

---

# Future Plans

Future versions of AWS will move beyond a single-model approach by implementing ensemble-based adversarial generation.

Instead of targeting only ResNet-50, upcoming versions aim to generate perturbations effective across multiple architectures, including:

- Vision Transformers (ViTs)
- Convolutional Neural Networks (CNNs)
- Hybrid vision-language models

The long-term goal is to provide creators with a more universal layer of protection against automated scraping and unauthorized AI training pipelines.
