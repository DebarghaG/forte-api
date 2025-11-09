# API Reference

Complete API documentation for Forte.

## Main Classes

### ForteOODDetector

::: forte.ForteOODDetector
    options:
      show_source: true
      members:
        - __init__
        - fit
        - predict
        - predict_proba
        - evaluate

---

## Model Classes

Custom PyTorch implementations for GPU-accelerated anomaly detection.

### TorchGMM

::: forte.TorchGMM
    options:
      show_source: true
      members:
        - __init__
        - fit
        - score_samples
        - bic

### TorchKDE

::: forte.TorchKDE
    options:
      show_source: true
      members:
        - __init__
        - fit
        - evaluate
        - logpdf
        - scotts_factor
        - silverman_factor

### TorchOCSVM

::: forte.TorchOCSVM
    options:
      show_source: true
      members:
        - __init__
        - fit
        - decision_function
        - predict

---

## Module Information

### Package Version

```python
import forte
print(forte.__version__)  # '0.1.0'
```

### Available Imports

```python
from forte import (
    ForteOODDetector,  # Main detector class
    TorchGMM,          # Gaussian Mixture Model
    TorchKDE,          # Kernel Density Estimation
    TorchOCSVM,        # One-Class SVM
    __version__,       # Package version
)
```

---

## Type Signatures

For type hints and IDE support:

```python
from typing import List, Dict, Tuple
import numpy as np
import torch

class ForteOODDetector:
    def __init__(
        self,
        batch_size: int = 32,
        device: Optional[str] = None,
        embedding_dir: str = "./embeddings",
        nearest_k: int = 5,
        method: str = 'gmm'
    ) -> None: ...

    def fit(
        self,
        id_image_paths: List[str],
        val_split: float = 0.2,
        random_state: int = 42
    ) -> 'ForteOODDetector': ...

    def predict(
        self,
        image_paths: List[str]
    ) -> np.ndarray: ...

    def predict_proba(
        self,
        image_paths: List[str]
    ) -> np.ndarray: ...

    def evaluate(
        self,
        id_image_paths: List[str],
        ood_image_paths: List[str]
    ) -> Dict[str, float]: ...
```

---

## Constants and Defaults

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `batch_size` | 32 | Batch size for image processing |
| `device` | Auto-detect | Computation device |
| `embedding_dir` | "./embeddings" | Feature cache directory |
| `nearest_k` | 5 | k for k-NN in PRDC |
| `method` | 'gmm' | Detection algorithm |
| `val_split` | 0.2 | Validation split fraction |
| `random_state` | 42 | Random seed |

## Return Types

### detector.predict()

Returns `numpy.ndarray` of shape `(n_samples,)` with values:
- `1`: In-distribution
- `-1`: Out-of-distribution

### detector.predict_proba()

Returns `numpy.ndarray` of shape `(n_samples,)` with values in `[0, 1]`:
- Values close to `1.0`: High confidence in-distribution
- Values close to `0.0`: High confidence out-of-distribution

### detector.evaluate()

Returns `dict` with keys:
```python
{
    'AUROC': float,        # Area under ROC curve [0, 1]
    'FPR@95TPR': float,    # FPR at 95% TPR [0, 1]
    'AUPRC': float,        # Area under PR curve [0, 1]
    'F1': float            # Best F1 score [0, 1]
}
```

---

## Examples

### Basic Usage

```python
from forte import ForteOODDetector

# Initialize
detector = ForteOODDetector(method='gmm', device='cuda:0')

# Fit
detector.fit(train_image_paths)

# Predict
predictions = detector.predict(test_image_paths)
scores = detector.predict_proba(test_image_paths)

# Evaluate
metrics = detector.evaluate(id_test_paths, ood_test_paths)
```

### Advanced Usage

```python
from forte import ForteOODDetector, TorchGMM
import torch

# Custom detector with specific parameters
detector = ForteOODDetector(
    batch_size=64,
    device='cuda:0',
    embedding_dir='./my_features',
    nearest_k=10,
    method='gmm'
)

# Fit with custom validation split
detector.fit(
    id_image_paths=train_paths,
    val_split=0.15,  # Use 15% for validation
    random_state=123
)

# Get detailed predictions
predictions = detector.predict(test_paths)
scores = detector.predict_proba(test_paths)

# Evaluate with custom test sets
metrics = detector.evaluate(
    id_image_paths=id_validation_paths,
    ood_image_paths=ood_validation_paths
)

print(f"AUROC: {metrics['AUROC']:.4f}")
```

### Using Individual Models

```python
from forte.models import TorchGMM, TorchKDE, TorchOCSVM
import torch

# Prepare features (example with random data)
features = torch.randn(1000, 12, device='cuda:0')

# GMM
gmm = TorchGMM(n_components=4, device='cuda:0')
gmm.fit(features)
scores_gmm = gmm.score_samples(features)
bic = gmm.bic(features)

# KDE
kde = TorchKDE(features.T, bw_method='scott', device='cuda:0')
scores_kde = kde.logpdf(features)

# OCSVM
ocsvm = TorchOCSVM(nu=0.1, n_iters=500, device='cuda:0')
ocsvm.fit(features)
scores_ocsvm = ocsvm.decision_function(features)
```

---

## Error Handling

### RuntimeError

Raised when detector is used before fitting:

```python
detector = ForteOODDetector()
try:
    predictions = detector.predict(test_paths)
except RuntimeError as e:
    print(e)  # "Detector must be fitted before prediction"
```

### ValueError

Raised for invalid parameters:

```python
# Invalid covariance type for TorchGMM
from forte.models import TorchGMM
try:
    gmm = TorchGMM(covariance_type='diagonal')
except NotImplementedError as e:
    print(e)  # "Only 'full' covariance is implemented"
```

---

## Notes

!!! note "GPU Memory"
    The detector loads three large pretrained models (CLIP, ViT-MSN, DINOv2). Expect ~2-3GB GPU memory usage.

!!! warning "First Run"
    The first call to `fit()` downloads pretrained models from Hugging Face (~2GB total). This happens once and is cached locally.

!!! tip "Reproducibility"
    For reproducible results, set `random_state` in `fit()` and ensure PyTorch determinism:
    ```python
    import torch
    import numpy as np

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    ```
