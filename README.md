# 🎨 AWS: Artists With Soul

![Artists With Soul Hero](aws_hero_image_1778435298904.png)

**Protecting Creativity in the Age of Generative AI.**

AWS (Artists With Soul) is a high-performance, privacy-centric adversarial poisoning tool designed to empower artists. It injects nearly invisible "noise" into digital artwork, making the images unreliable for AI training models (like Stable Diffusion or Midjourney) while preserving visual beauty for human viewers.

---

## ✨ Key Features

- **🛡️ Multi-Attack Defense**: Implements both **FGSM** (Fast Gradient Sign Method) and **PGD** (Projected Gradient Descent) to maximize protection strength.
- **⚡ High Performance**:
  - **Downscaled Computation**: Attacks are computed on 512px tensors and upscaled, allowing for near real-time processing even on 20MP images.
  - **In-Memory Pipeline**: Zero disk I/O during processing; everything happens in RAM for maximum speed.
- **🔍 X-Ray Vision (Grad-CAM)**: Visualizes exactly where the AI model is looking, demonstrating how the "poison" successfully redirects the model's focus.
- **🕵️ Privacy First**: Automatically strips all EXIF metadata (GPS, camera info, timestamps) from every processed image.
- **🏷️ Stealth Watermarking**: Embeds invisible LSB (Least Significant Bit) watermarks for ownership tracking without altering the aesthetic.
- **📦 Batch Processing**: Upload a ZIP archive and get back a ZIP of protected images in seconds.
- **🖥️ Dual Mode**: Full-featured Web UI (Flask) and a lightweight CLI for power users.

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.10 or higher.
- **Redis**: Required for the asynchronous task queue (Celery).
- **GPU (Recommended)**: NVIDIA GPU with CUDA support for 10x faster processing.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/taranprits2/AWS-Artists-With-Soul-.git
   cd AWS-Artists-With-Soul-
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Setup CUDA (NVIDIA GPU)**:
   If you have an NVIDIA GPU, install the optimized PyTorch binaries:
   ```bash
   pip uninstall torch torchvision -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

---

## 🛠️ Usage

### 1. Web Interface (Recommended)

To run the full web application, you need to start three components:

- **Terminal 1 (Redis)**:
  ```bash
  redis-server
  ```
- **Terminal 2 (Celery Worker)**:
  ```bash
  celery -A app.celery_app worker --loglevel=info --pool=solo
  ```
- **Terminal 3 (Flask App)**:
  ```bash
  python app.py
  ```
Visit `http://127.0.0.1:5000` in your browser.

### 2. Command Line Interface (CLI)

For quick, single-image poisoning:
```bash
python poison_engine.py my_artwork.jpg --epsilon 0.01 --output protected.png
```

---

## 🧠 Technology Stack

- **Framework**: [Flask](https://flask.palletsprojects.com/) (Web Server)
- **Deep Learning**: [PyTorch](https://pytorch.org/) (ResNet-50 Backend)
- **Task Queue**: [Celery](https://docs.celeryq.dev/) + [Redis](https://redis.io/)
- **Image Processing**: [Pillow](https://python-pillow.org/) + [NumPy](https://numpy.org/)
- **UI Architecture**: In-memory Base64 serialization for zero-latency previews.

---

## 📊 Poison Strength Guide

| Setting | Epsilon | PSNR (Quality) | Visibility |
| :--- | :--- | :--- | :--- |
| **Ultra-Subtle** | 0.005 | ~46 dB | Virtually Invisible |
| **Recommended** | 0.010 | ~40 dB | Indistinguishable |
| **Moderate** | 0.020 | ~34 dB | Minor grain on zoom |
| **Aggressive** | 0.040 | ~28 dB | Visible artifacts |

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information (if applicable).

---

Developed with ❤️ for the artist community. 🎨🛡️