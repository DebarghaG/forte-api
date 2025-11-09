# Quick Start Guide

Get up and running with Forte in 5 minutes!

## Your First OOD Detector

This tutorial shows you how to build an out-of-distribution detector using Forte.

### Step 1: Install Forte

```bash
pip install forte-detector
```

### Step 2: Prepare Your Data

Forte works with image file paths. Organize your images:

```python
# In-distribution images (e.g., normal samples)
id_train_paths = [
    "/path/to/normal/image1.jpg",
    "/path/to/normal/image2.jpg",
    # ... more images
]

# Test images (mix of ID and OOD)
id_test_paths = [...]  # Normal test images
ood_test_paths = [...]  # Anomalous test images
```

### Step 3: Create and Train Detector

```python
from forte import ForteOODDetector

# Initialize the detector
detector = ForteOODDetector(
    method='gmm',       # Detection method: 'gmm', 'kde', or 'ocsvm'
    nearest_k=5,        # Neighbors for PRDC computation
    batch_size=32,      # Batch size for processing
    device='cuda:0'     # Use 'cuda:0', 'mps', or 'cpu'
)

# Train on in-distribution data
detector.fit(id_train_paths, val_split=0.2)
```

!!! tip "Training Time"
    First run downloads pretrained models (~2GB) and may take 10-15 minutes depending on your dataset size. Subsequent runs use cached features and are much faster!

### Step 4: Make Predictions

```python
# Get binary predictions (1 = in-distribution, -1 = out-of-distribution)
predictions = detector.predict(id_test_paths + ood_test_paths)

# Get probability scores (higher = more likely in-distribution)
scores = detector.predict_proba(id_test_paths + ood_test_paths)

print(f"Predictions: {predictions}")
print(f"Scores: {scores}")
```

### Step 5: Evaluate Performance

```python
# Compute standard OOD detection metrics
metrics = detector.evaluate(id_test_paths, ood_test_paths)

print(f"AUROC: {metrics['AUROC']:.4f}")
print(f"FPR at 95% TPR: {metrics['FPR@95TPR']:.4f}")
print(f"AUPRC: {metrics['AUPRC']:.4f}")
print(f"Best F1 Score: {metrics['F1']:.4f}")
```

## Complete Example: CIFAR-10 vs CIFAR-100

Here's a complete working example using CIFAR datasets:

```python
import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from forte import ForteOODDetector

# Download CIFAR datasets
transform = transforms.ToTensor()
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Helper function to save images
def save_dataset_as_png(dataset, save_dir, num_images=1000):
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i in range(min(num_images, len(dataset))):
        image, label = dataset[i]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        path = os.path.join(save_dir, f"{i}.png")
        image.save(path)
        paths.append(path)
    return paths

# Save images
id_train_paths = save_dataset_as_png(cifar10_train, "data/cifar10/train", num_images=5000)
id_test_paths = save_dataset_as_png(cifar10_test, "data/cifar10/test", num_images=1000)
ood_test_paths = save_dataset_as_png(cifar100_test, "data/cifar100/test", num_images=1000)

# Create and train detector
detector = ForteOODDetector(method='gmm', device='cuda:0' if torch.cuda.is_available() else 'cpu')
detector.fit(id_train_paths)

# Evaluate
metrics = detector.evaluate(id_test_paths, ood_test_paths)
print(f"Results: {metrics}")
```

Expected output:
```
AUROC: 0.9250
FPR at 95% TPR: 0.1234
AUPRC: 0.9012
Best F1 Score: 0.8567
```

## Visualization Example

Visualize the score distribution:

```python
import matplotlib.pyplot as plt
import numpy as np

# Get scores for both distributions
id_scores = detector.predict_proba(id_test_paths)
ood_scores = detector.predict_proba(ood_test_paths)

# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(id_scores, bins=50, alpha=0.7, label='In-Distribution', density=True)
plt.hist(ood_scores, bins=50, alpha=0.7, label='Out-of-Distribution', density=True)
plt.xlabel('OOD Score')
plt.ylabel('Density')
plt.title('Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('score_distribution.png')
```

## Understanding the Output

### Predictions
- `1`: Image is likely in-distribution (normal)
- `-1`: Image is likely out-of-distribution (anomalous)

### Scores
- Higher values (close to 1.0): More confident the image is in-distribution
- Lower values (close to 0.0): More confident the image is out-of-distribution

### Metrics
- **AUROC**: Area under ROC curve (higher is better, max 1.0)
- **FPR@95TPR**: False positive rate at 95% true positive rate (lower is better)
- **AUPRC**: Area under precision-recall curve (higher is better)
- **F1**: Best F1 score across all thresholds (higher is better)

## Next Steps

- [User Guide](user-guide.md) - Learn about advanced features
- [Examples](examples.md) - See more real-world applications
- [API Reference](api-reference.md) - Detailed API documentation
- [Methods](methods.md) - Understand the algorithms

## Tips for Best Results

!!! tip "Dataset Size"
    Use at least 500-1000 training images for best results. More is better!

!!! tip "Detection Method"
    - **GMM**: Best for most cases, automatically selects components
    - **KDE**: Good for small datasets (<1000 samples)
    - **OCSVM**: Fast, works well with clear boundaries

!!! tip "Hyperparameters"
    - `nearest_k`: Use 5-10 for most datasets. Larger values (10-20) for noisy data.
    - `batch_size`: Increase for faster processing on GPU (32-128).

!!! warning "Memory Usage"
    Each model processes images in batches. If you encounter out-of-memory errors, reduce `batch_size` or use CPU mode.
