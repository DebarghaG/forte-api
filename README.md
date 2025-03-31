# Forte API Documentation

## Overview

The Forte library provides robust out-of-distribution (OOD) detection capabilities through the `ForteOODDetector` class. The core algorithm is built on the principle of **F**inding **O**utliers using **R**epresentation **T**ypicality **E**stimation, which:

1. Uses self-supervised vision models to extract semantic features
2. Incorporates manifold estimation to account for local topology
3. Requires no class labels or exposure to OOD data during training

This makes Forte particularly useful for real-world applications where anomalous data may be unexpected or unknown at training time.


## Key Features

- **Multiple feature extractors**: Leverages CLIP, ViT-MSN, and DINOv2 models for robust semantic representation
- **Topology-aware scoring**: Uses Precision, Recall, Density, and Coverage (PRDC) metrics to capture manifold structure
- **Multiple detection methods**: Supports Gaussian Mixture Models (GMM), Kernel Density Estimation (KDE), and One-Class SVM (OCSVM)
- **Automatic hyperparameter selection**: Optimizes model hyperparameters using validation data
- **Caching for efficiency**: Saves extracted features to avoid redundant computation

## API Reference

### `ForteOODDetector`

The main class for OOD detection.

```python
detector = ForteOODDetector(
    batch_size=32,
    device=None,
    embedding_dir="./embeddings",
    nearest_k=5,
    method='gmm'
)
```

#### Parameters

- **batch_size** (int, default=32): Batch size for processing images during feature extraction
- **device** (str, default=None): Device to use for computation (e.g., 'cuda:0', 'cpu'). If None, uses CUDA if available
- **embedding_dir** (str, default='./embeddings'): Directory to store extracted features for caching
- **nearest_k** (int, default=5): Number of nearest neighbors for PRDC computation
- **method** (str, default='gmm'): Method to use for OOD detection. Options:
  - 'gmm': Gaussian Mixture Model (best for clustered data)
  - 'kde': Kernel Density Estimation (best for smooth distributions)
  - 'ocsvm': One-Class SVM (best for complex boundaries)

### Methods

#### `fit(id_image_paths, val_split=0.2, random_state=42)`

Fits the OOD detector on in-distribution data.

**Parameters:**
- **id_image_paths** (list): List of paths to in-distribution images
- **val_split** (float, default=0.2): Fraction of data to use for validation
- **random_state** (int, default=42): Random seed for reproducibility

**Returns:**
- The fitted detector object

**Process:**
1. Splits data into training and validation sets
2. Extracts features using pretrained models
3. Computes PRDC features
4. Trains the OOD detector (GMM, KDE, or OCSVM)

```python
detector.fit(id_image_paths, val_split=0.2, random_state=42)
```

#### `predict(image_paths)`

Predicts if samples are OOD.

**Parameters:**
- **image_paths** (list): List of paths to images

**Returns:**
- Binary array (1 for in-distribution, -1 for OOD)

```python
predictions = detector.predict(test_image_paths)
```

#### `predict_proba(image_paths)`

Returns normalized probability scores for OOD detection.

**Parameters:**
- **image_paths** (list): List of paths to images

**Returns:**
- Array of normalized scores (higher values indicate in-distribution)

```python
scores = detector.predict_proba(test_image_paths)
```

#### `evaluate(id_image_paths, ood_image_paths)`

Evaluates the OOD detector on in-distribution and out-of-distribution data.

**Parameters:**
- **id_image_paths** (list): List of paths to in-distribution images
- **ood_image_paths** (list): List of paths to out-of-distribution images

**Returns:**
- Dictionary of evaluation metrics:
  - **AUROC**: Area Under the Receiver Operating Characteristic curve
  - **FPR@95TPR**: False Positive Rate at 95% True Positive Rate
  - **AUPRC**: Area Under the Precision-Recall Curve
  - **F1**: Maximum F1 score

```python
metrics = detector.evaluate(id_image_paths, ood_image_paths)
print(f"AUROC: {metrics['AUROC']:.4f}")
```

## Usage Examples

### Basic Usage

```python
from forte_api import ForteOODDetector
import glob

# Collect in-distribution images
id_images = glob.glob("data/normal_class/*.jpg")

# Split for training and testing
train_images = id_images[:800]
test_id_images = id_images[800:]

# Collect OOD images
ood_images = glob.glob("data/anomalies/*.jpg")

# Create and train detector
detector = ForteOODDetector(
    batch_size=32,
    device="cuda:0",
    method="gmm"
)

# Train the detector
detector.fit(train_images)

# Evaluate performance
metrics = detector.evaluate(test_id_images, ood_images)
print(f"AUROC: {metrics['AUROC']:.4f}")
print(f"FPR@95TPR: {metrics['FPR@95TPR']:.4f}")

# Get predictions
predictions = detector.predict(ood_images)
```

### Complete Example with CIFAR-10/CIFAR-100

For a complete example using CIFAR-10 as in-distribution and CIFAR-100 as out-of-distribution data, see the [examples/cifar_demo.py](examples/cifar_demo.py) script in the repository.

### Experimenting with Different Methods

```python
# Try different detection methods
methods = ['gmm', 'kde', 'ocsvm']
results = {}

for method in methods:
    detector = ForteOODDetector(method=method)
    detector.fit(train_images)
    results[method] = detector.evaluate(test_id_images, ood_images)

# Compare results
for method, metrics in results.items():
    print(f"{method.upper()} - AUROC: {metrics['AUROC']:.4f}, FPR@95TPR: {metrics['FPR@95TPR']:.4f}")
```

## Model Details

### Feature Extraction Models

Forte uses three pretrained models for feature extraction:

1. **CLIP** (Contrastive Language-Image Pretraining): Captures semantic information aligned with natural language concepts
2. **ViT-MSN** (Vision Transformer with Masked Self-supervised Network): Captures fine-grained visual patterns
3. **DINOv2** (Self-supervised Vision Transformer): Captures hierarchical visual representations

You may modify the code to use your own encoder if you wish. This may be a CNN or a ViT. Anything you want.

