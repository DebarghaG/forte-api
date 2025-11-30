# Installation

## Requirements

- Python 3.9+
- CUDA 11.8+ (optional, for GPU)

## Dependencies

Core (installed automatically):
- torch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- pillow >= 9.0.0
- tqdm >= 4.65.0

## PyPI

```bash
pip install forte-detector
```

Optional extras:
```bash
pip install forte-detector[dev]   # pytest, black, flake8, mypy
pip install forte-detector[docs]  # mkdocs, mkdocs-material
pip install forte-detector[viz]   # matplotlib
pip install forte-detector[all]   # all optional dependencies
```

## From Source

```bash
git clone https://github.com/debarghag/forte-detector.git
cd forte-detector
pip install -e ".[dev]"
```

## Verify

```python
from forte import ForteOODDetector
print(ForteOODDetector.__module__)
```

## GPU Setup

### CUDA

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```python
import torch
print(torch.cuda.is_available())
```

### MPS (Apple Silicon)

```python
import torch
print(torch.backends.mps.is_available())
```

## First Run

First call to `fit()` downloads pretrained models (~2GB total):
- `openai/clip-vit-base-patch32`
- `facebook/vit-msn-base`
- `facebook/dinov2-base`

Models cached to `~/.cache/huggingface/`.
