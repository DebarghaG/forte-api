# Forte: Finding Outliers with Representation Typicality Estimation

[![PyPI version](https://badge.fury.io/py/forte-detector.svg)](https://badge.fury.io/py/forte-detector)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-red.svg)](https://openreview.net/forum?id=7XNgVPxCiA)

**Forte** is a state-of-the-art PyTorch library for out-of-distribution (OOD) detection using topology-aware representation learning from multiple pretrained vision models.

!!! paper "ICLR 2025 Paper"
    This work was published at the Thirteenth International Conference on Learning Representations (ICLR 2025).

    **[Read the paper on OpenReview →](https://openreview.net/forum?id=7XNgVPxCiA)**

## Overview

Out-of-distribution detection is crucial for deploying machine learning models safely in real-world applications. Forte provides an easy-to-use solution that:

- ✨ **Works with any computer vision model** - Just provide image paths, no model training required
- 🚀 **GPU-accelerated** - Fast inference with CUDA and Apple Silicon (MPS) support
- 📊 **Multiple detection methods** - Choose from GMM, KDE, or One-Class SVM
- 🎯 **State-of-the-art performance** - Leverages CLIP, ViT-MSN, and DINOv2 features
- 🔧 **Easy integration** - Simple Python API, works with existing pipelines

## How It Works

Forte uses a three-stage pipeline:

1. **Multi-Model Feature Extraction**: Extract semantic features using pretrained models (CLIP, ViT-MSN, DINOv2)
2. **PRDC Computation**: Compute topology-aware features (Precision, Recall, Density, Coverage)
3. **Anomaly Detection**: Train a detector (GMM/KDE/OCSVM) on PRDC features

## Key Features

### 🎨 Flexible Feature Extraction

Forte automatically extracts features using three complementary pretrained models:

- **CLIP** (OpenAI): Text-image aligned representations
- **ViT-MSN** (Facebook): Self-supervised vision transformer
- **DINOv2** (Facebook): Self-distilled vision features

### 📈 Topology-Aware Scoring

Uses PRDC metrics to capture the distributional properties of image representations:

- **Precision**: Fidelity of generated/test samples
- **Recall**: Coverage of reference distribution
- **Density**: Local density estimation
- **Coverage**: Mode coverage

### ⚡ GPU Acceleration

Custom PyTorch implementations of detection algorithms optimized for GPU:

- TorchGMM: Gaussian Mixture Models
- TorchKDE: Kernel Density Estimation
- TorchOCSVM: One-Class Support Vector Machines

### 💾 Intelligent Caching

Automatically caches extracted features to disk, making repeated experiments fast and efficient.

## Quick Example

```python
from forte import ForteOODDetector

# Initialize detector
detector = ForteOODDetector(
    method='gmm',      # Detection method: 'gmm', 'kde', or 'ocsvm'
    nearest_k=5,       # Number of neighbors for PRDC
    device='cuda:0'    # Use GPU acceleration
)

# Fit on in-distribution images
detector.fit(id_image_paths)

# Detect outliers
predictions = detector.predict(test_image_paths)  # Returns 1 (ID) or -1 (OOD)
scores = detector.predict_proba(test_image_paths) # Returns [0, 1] scores

# Evaluate performance
metrics = detector.evaluate(id_test_paths, ood_test_paths)
print(f"AUROC: {metrics['AUROC']:.4f}")
print(f"FPR@95TPR: {metrics['FPR@95TPR']:.4f}")
```

## Performance

Forte achieves state-of-the-art results on standard OOD detection benchmarks:

| Dataset (ID vs OOD) | AUROC ↑ | FPR@95TPR ↓ | AUPRC ↑ |
|---------------------|---------|-------------|---------|
| CIFAR-10 vs CIFAR-100 | 0.92+ | <0.15 | 0.90+ |
| ImageNet vs Textures | 0.95+ | <0.10 | 0.94+ |

## Use Cases

Forte is designed for ease-of-use across various scenarios:

- 🏥 **Medical Imaging**: Detect anomalous scans without retraining models
- 🚗 **Autonomous Vehicles**: Identify novel road scenarios
- 🏭 **Quality Control**: Spot manufacturing defects
- 🔍 **Content Moderation**: Flag unusual or inappropriate content
- 🧪 **Scientific Research**: Identify outliers in experimental data

## Why Forte?

| Feature | Forte | Traditional Methods |
|---------|-------|-------------------|
| **No Training Required** | ✅ Use pretrained models | ❌ Requires model training |
| **Multi-Model Ensemble** | ✅ 3 complementary models | ❌ Single model |
| **Topology-Aware** | ✅ PRDC features | ❌ Simple distances |
| **GPU Accelerated** | ✅ Custom PyTorch implementations | ⚠️ Often CPU-only |
| **Automatic Caching** | ✅ Smart feature caching | ❌ Manual management |

## Next Steps

- [Installation Guide](installation.md) - Get started in 5 minutes
- [Quick Start Tutorial](quickstart.md) - Your first OOD detector
- [User Guide](user-guide.md) - Deep dive into features
- [API Reference](api-reference.md) - Complete API documentation
- [Examples](examples.md) - Real-world use cases
- [Citation](citation.md) - How to cite this work

## Citation

If you use Forte in your research, please cite our ICLR 2025 paper:

```bibtex
@inproceedings{ganguly2025forte,
  title={Forte: Finding Outliers with Representation Typicality Estimation},
  author={Debargha Ganguly and Warren Richard Morningstar and Andrew Seohwan Yu and Vipin Chaudhary},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=7XNgVPxCiA}
}
```

## License

Forte is released under the MIT License. See [LICENSE](https://github.com/debarghag/forte-detector/blob/main/LICENSE) for details.

## Acknowledgements

This work was supported by the NSF ICICLE grant. We thank the open-source community for their foundational work on CLIP, ViT, and DINOv2.
