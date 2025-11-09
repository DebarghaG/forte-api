# Installation Guide

Get started with Forte in just a few minutes!

## Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher
- CUDA 11.0+ (optional, for GPU acceleration)

## Install from PyPI

The easiest way to install Forte is via pip:

```bash
pip install forte-detector
```

This will install Forte along with all required dependencies.

### Optional Dependencies

For visualization support (matplotlib):

```bash
pip install forte-detector[viz]
```

For development (includes testing and linting tools):

```bash
pip install forte-detector[dev]
```

For documentation building:

```bash
pip install forte-detector[docs]
```

Install everything:

```bash
pip install forte-detector[all]
```

## Install from Source

For the latest development version:

```bash
# Clone the repository
git clone https://github.com/debarghag/forte-detector.git
cd forte-detector

# Install in editable mode
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

## Verify Installation

Test your installation:

```python
import forte
print(forte.__version__)  # Should print: 0.1.0

# Quick test
from forte import ForteOODDetector
detector = ForteOODDetector(device='cpu')
print("Forte installed successfully!")
```

## GPU Setup

### CUDA (NVIDIA GPUs)

Forte will automatically use CUDA if available. Verify CUDA installation:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

If CUDA is not available, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon (MPS)

On macOS with Apple Silicon, Forte supports MPS acceleration:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## Troubleshooting

### Issue: "No module named 'forte'"

**Solution**: Make sure you installed the package correctly:

```bash
pip install forte-detector
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or use CPU:

```python
detector = ForteOODDetector(batch_size=8, device='cpu')
```

### Issue: Model download failures

**Solution**: Check your internet connection. Models are downloaded from Hugging Face Hub on first use.

### Issue: Import errors for transformers

**Solution**: Update transformers:

```bash
pip install --upgrade transformers
```

## Docker Support

A Dockerfile will be provided in future releases. For now, use the standard Python installation.

## Next Steps

- [Quick Start Tutorial](quickstart.md) - Build your first OOD detector
- [User Guide](user-guide.md) - Learn about all features
- [Examples](examples.md) - See real-world applications
