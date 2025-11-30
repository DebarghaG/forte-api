# API Reference

## ForteOODDetector

Main class for out-of-distribution detection.

### Constructor

```python
ForteOODDetector(
    batch_size: int = 32,
    device: str = None,
    embedding_dir: str = "./embeddings",
    nearest_k: int = 5,
    method: str = "gmm"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 32 | Images per forward pass |
| `device` | str | None | `'cuda:N'`, `'mps'`, or `'cpu'`. Auto-detects if None. |
| `embedding_dir` | str | `'./embeddings'` | Directory for cached features |
| `nearest_k` | int | 5 | k for k-NN in PRDC computation |
| `method` | str | `'gmm'` | Detection backend: `'gmm'`, `'kde'`, `'ocsvm'` |

### Methods

#### fit

```python
fit(id_image_paths: List[str], val_split: float = 0.2, random_state: int = 42) -> ForteOODDetector
```

Train detector on in-distribution images.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id_image_paths` | List[str] | required | Paths to ID training images |
| `val_split` | float | 0.2 | Fraction for hyperparameter tuning |
| `random_state` | int | 42 | Random seed |

Returns: `self`

#### predict

```python
predict(image_paths: List[str]) -> np.ndarray
```

Binary OOD classification.

Returns: `np.ndarray` of shape `(n,)` with dtype `int64`. Values: `1` (in-distribution), `-1` (out-of-distribution).

#### predict_proba

```python
predict_proba(image_paths: List[str]) -> np.ndarray
```

Normalized OOD scores.

Returns: `np.ndarray` of shape `(n,)` with dtype `float64`. Range `[0, 1]`. Higher values indicate in-distribution.

#### evaluate

```python
evaluate(id_image_paths: List[str], ood_image_paths: List[str]) -> Dict[str, float]
```

Compute evaluation metrics on labeled test data.

Returns: `dict` with keys:
- `AUROC`: Area under ROC curve
- `FPR@95TPR`: False positive rate at 95% true positive rate
- `AUPRC`: Area under precision-recall curve
- `F1`: Maximum F1 score across thresholds

---

## TorchGMM

GPU-accelerated Gaussian Mixture Model.

### Constructor

```python
TorchGMM(
    n_components: int = 1,
    covariance_type: str = "full",
    max_iter: int = 100,
    tol: float = 1e-3,
    reg_covar: float = 1e-6,
    device: str = "cuda"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | 1 | Number of mixture components |
| `covariance_type` | str | `"full"` | Only `"full"` supported |
| `max_iter` | int | 100 | Maximum EM iterations |
| `tol` | float | 1e-3 | Convergence threshold |
| `reg_covar` | float | 1e-6 | Covariance regularization |
| `device` | str | `"cuda"` | Computation device |

### Methods

#### fit

```python
fit(X: torch.Tensor) -> TorchGMM
```

Fit GMM via EM algorithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | torch.Tensor | Shape `(n_samples, n_features)` |

Returns: `self`

#### score_samples

```python
score_samples(X: torch.Tensor) -> torch.Tensor
```

Compute log-likelihood per sample.

Returns: `torch.Tensor` of shape `(n_samples,)`

#### bic

```python
bic(X: torch.Tensor) -> float
```

Bayesian Information Criterion.

Returns: `float`. Lower is better.

---

## TorchKDE

GPU-accelerated Kernel Density Estimation.

### Constructor

```python
TorchKDE(
    dataset: torch.Tensor,
    bw_method: Optional[Union[str, float, Callable]] = None,
    weights: Optional[torch.Tensor] = None,
    device: str = "cuda"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | torch.Tensor | required | Shape `(d, n)` where d=dimension, n=samples |
| `bw_method` | str/float/Callable | None | `'scott'`, `'silverman'`, or scalar. None defaults to Scott. |
| `weights` | torch.Tensor | None | Sample weights of shape `(n,)` |
| `device` | str | `"cuda"` | Computation device |

### Methods

#### evaluate

```python
evaluate(points: torch.Tensor) -> torch.Tensor
```

Evaluate density at given points.

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | torch.Tensor | Shape `(d, m)` or `(m, d)` |

Returns: `torch.Tensor` of shape `(m,)`

#### logpdf

```python
logpdf(points: torch.Tensor) -> torch.Tensor
```

Log probability density.

Returns: `torch.Tensor` of shape `(m,)`

#### scotts_factor

```python
scotts_factor() -> float
```

Returns: Scott's bandwidth factor: $n_{\text{eff}}^{-1/(d+4)}$

#### silverman_factor

```python
silverman_factor() -> float
```

Returns: Silverman's bandwidth factor: $(n_{\text{eff}}(d+2)/4)^{-1/(d+4)}$

---

## TorchOCSVM

GPU-accelerated One-Class SVM.

### Constructor

```python
TorchOCSVM(
    nu: float = 0.1,
    n_iters: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nu` | float | 0.1 | Upper bound on outlier fraction (0, 1) |
| `n_iters` | int | 1000 | Optimization iterations |
| `lr` | float | 1e-3 | Adam learning rate |
| `device` | str | `"cuda"` | Computation device |

### Methods

#### fit

```python
fit(X: torch.Tensor) -> TorchOCSVM
```

Fit via gradient descent on primal objective.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | torch.Tensor | Shape `(n_samples, n_features)` |

Returns: `self`

#### decision_function

```python
decision_function(X: torch.Tensor) -> torch.Tensor
```

Signed distance to decision boundary.

Returns: `torch.Tensor` of shape `(n_samples,)`. Positive = inlier.

#### predict

```python
predict(X: torch.Tensor) -> torch.Tensor
```

Binary classification.

Returns: `torch.Tensor` of shape `(n_samples,)`. Values: `1` (inlier), `-1` (outlier).

---

## Module Exports

```python
from forte import (
    ForteOODDetector,
    TorchGMM,
    TorchKDE,
    TorchOCSVM,
    __version__,
)
```
