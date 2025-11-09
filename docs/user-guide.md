# User Guide

Complete guide to using Forte for out-of-distribution detection.

## Overview

Forte provides a simple yet powerful API for detecting out-of-distribution images using pretrained vision models and topology-aware features.

## Core Concepts

### Feature Extraction

Forte uses three pretrained models to extract complementary features:

1. **CLIP** (`openai/clip-vit-base-patch32`): 512-dimensional features, text-image aligned
2. **ViT-MSN** (`facebook/vit-msn-base`): 768-dimensional features, self-supervised
3. **DINOv2** (`facebook/dinov2-base`): 768-dimensional features, self-distilled

### PRDC Features

For each model's features, Forte computes 4 topology-aware metrics:

- **Precision**: Measures if test samples fall within the manifold of reference data
- **Recall**: Measures coverage of the reference distribution
- **Density**: Local density estimation using k-NN
- **Coverage**: Mode coverage of the distribution

This results in 12 total features (3 models × 4 PRDC metrics) used for detection.

### Detection Methods

Forte supports three anomaly detection methods:

#### Gaussian Mixture Models (GMM)
- Automatically selects the number of components (1-64) using BIC
- Best for complex, multi-modal distributions
- Recommended for most use cases

#### Kernel Density Estimation (KDE)
- Non-parametric density estimation
- Good for small datasets (<1000 samples)
- Uses Scott's rule for bandwidth selection

#### One-Class SVM (OCSVM)
- Learns a decision boundary around in-distribution data
- Fast inference
- Good when ID and OOD are clearly separated

## API Usage

### Initialization

```python
from forte import ForteOODDetector

detector = ForteOODDetector(
    batch_size=32,                    # Batch size for processing
    device='cuda:0',                  # Device: 'cuda:0', 'mps', or 'cpu'
    embedding_dir='./embeddings',     # Cache directory for features
    nearest_k=5,                       # k for k-NN in PRDC
    method='gmm'                      # Detection method
)
```

**Parameters:**

- `batch_size` (int, default=32): Number of images to process at once. Increase for faster GPU processing.
- `device` (str, optional): Computation device. Auto-detected if not specified.
- `embedding_dir` (str, default='./embeddings'): Directory to cache extracted features.
- `nearest_k` (int, default=5): Number of nearest neighbors for PRDC computation.
- `method` (str, default='gmm'): Detection method - 'gmm', 'kde', or 'ocsvm'.

### Training

```python
detector.fit(
    id_image_paths,     # List of paths to in-distribution images
    val_split=0.2,      # Validation split fraction
    random_state=42     # Random seed for reproducibility
)
```

**Parameters:**

- `id_image_paths` (list): Paths to in-distribution training images
- `val_split` (float, default=0.2): Fraction of data for validation
- `random_state` (int, default=42): Random seed

**Returns:** `self` (the fitted detector)

### Prediction

#### Binary Prediction

```python
predictions = detector.predict(image_paths)
# Returns: numpy array of 1 (ID) or -1 (OOD)
```

#### Probability Scores

```python
scores = detector.predict_proba(image_paths)
# Returns: numpy array of values in [0, 1]
# Higher values = more likely in-distribution
```

### Evaluation

```python
metrics = detector.evaluate(id_test_paths, ood_test_paths)
# Returns dict with: AUROC, FPR@95TPR, AUPRC, F1
```

## Advanced Features

### Feature Caching

Forte automatically caches extracted features to speed up repeated experiments:

```python
# First run: extracts and caches features
detector1 = ForteOODDetector(embedding_dir='./my_cache')
detector1.fit(train_paths)

# Second run: loads cached features (much faster!)
detector2 = ForteOODDetector(embedding_dir='./my_cache')
detector2.fit(train_paths)  # Reuses cached features
```

To force recomputation:
```bash
rm -rf ./my_cache
```

### Device Selection

#### Automatic Device Selection

```python
# Automatically selects best available device
detector = ForteOODDetector()  # cuda:0 > mps > cpu
```

#### Manual Device Selection

```python
# Force CPU (useful for debugging)
detector = ForteOODDetector(device='cpu')

# Specific CUDA device
detector = ForteOODDetector(device='cuda:1')

# Apple Silicon
detector = ForteOODDetector(device='mps')
```

### Method Comparison

Compare different detection methods:

```python
results = {}
for method in ['gmm', 'kde', 'ocsvm']:
    detector = ForteOODDetector(method=method, embedding_dir=f'./cache_{method}')
    detector.fit(train_paths)
    results[method] = detector.evaluate(id_test_paths, ood_test_paths)

# Print comparison
for method, metrics in results.items():
    print(f"{method.upper()}: AUROC={metrics['AUROC']:.4f}, FPR@95TPR={metrics['FPR@95TPR']:.4f}")
```

### Hyperparameter Tuning

#### nearest_k

```python
# Try different k values
for k in [3, 5, 10, 20]:
    detector = ForteOODDetector(nearest_k=k)
    detector.fit(train_paths)
    metrics = detector.evaluate(id_test_paths, ood_test_paths)
    print(f"k={k}: AUROC={metrics['AUROC']:.4f}")
```

#### Validation Split

```python
# Use more data for training (less for validation)
detector.fit(train_paths, val_split=0.1)  # 90% train, 10% val
```

## Best Practices

### Data Preparation

✅ **Do:**
- Use high-quality images (>224×224 pixels)
- Ensure consistent image format (JPEG, PNG)
- Have at least 500-1000 training images
- Balance your test set (equal ID and OOD samples)

❌ **Don't:**
- Mix very different image types in ID data
- Use corrupted or very low-resolution images
- Have class imbalance in training data

### Performance Optimization

**For Speed:**
```python
detector = ForteOODDetector(
    batch_size=128,      # Large batches on GPU
    device='cuda:0',     # Use GPU
    method='ocsvm'       # Fastest method
)
```

**For Accuracy:**
```python
detector = ForteOODDetector(
    batch_size=16,       # Smaller batches, more stable
    nearest_k=10,        # More neighbors for PRDC
    method='gmm'         # Most accurate method
)
```

**For Memory:**
```python
detector = ForteOODDetector(
    batch_size=8,        # Small batches
    device='cpu',        # Use CPU if GPU OOM
    method='kde'         # Memory-efficient
)
```

### Common Patterns

#### Cross-Validation

```python
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
aurocs = []

for train_idx, val_idx in kf.split(all_id_paths):
    train_paths = [all_id_paths[i] for i in train_idx]
    val_paths = [all_id_paths[i] for i in val_idx]

    detector = ForteOODDetector()
    detector.fit(train_paths, val_split=0)  # No internal validation
    metrics = detector.evaluate(val_paths, ood_paths)
    aurocs.append(metrics['AUROC'])

print(f"Mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
```

#### Threshold Selection

```python
# Get scores for validation set
val_scores = detector.predict_proba(id_val_paths)

# Set threshold for 95% TPR
threshold = np.percentile(val_scores, 5)

# Apply threshold
test_scores = detector.predict_proba(test_paths)
predictions = (test_scores > threshold).astype(int) * 2 - 1  # Convert to -1/1
```

## Troubleshooting

### Out of Memory Errors

```python
# Reduce batch size
detector = ForteOODDetector(batch_size=4)

# Or use CPU
detector = ForteOODDetector(device='cpu')
```

### Slow Performance

```python
# Check if features are being cached
import os
cache_dir = './embeddings'
if os.path.exists(cache_dir):
    print(f"Cached files: {len(os.listdir(cache_dir))}")

# Increase batch size for GPU
detector = ForteOODDetector(batch_size=64, device='cuda:0')
```

### Poor Detection Performance

- Ensure sufficient training data (>500 images)
- Check that ID and OOD are actually different distributions
- Try different methods (GMM usually best)
- Increase `nearest_k` for noisy data

## Next Steps

- [Examples](examples.md) - Real-world use cases
- [Methods](methods.md) - Technical details
- [API Reference](api-reference.md) - Complete API docs
