# Configuration

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `'gmm'` | `'gmm'`, `'kde'`, or `'ocsvm'` |
| `nearest_k` | int | 5 | k for k-NN manifold estimation |
| `batch_size` | int | 32 | Images per GPU forward pass |
| `device` | str | auto | `'cuda:N'`, `'mps'`, `'cpu'` |
| `embedding_dir` | str | `'./embeddings'` | Feature cache directory |

## Method Selection

| Method | Best for | Hyperparameter tuning |
|--------|----------|----------------------|
| GMM | Multi-modal distributions | Components via BIC (1-64) |
| KDE | Small datasets, smooth boundaries | Bandwidth via Scott's rule |
| OCSVM | Large datasets, fast inference | nu via validation (0.01-0.5) |

## Device Selection

Auto-detection priority: CUDA > MPS > CPU

```python
# Force specific device
detector = ForteOODDetector(device='cuda:1')
detector = ForteOODDetector(device='cpu')
```

## Caching

Features cached to `{embedding_dir}/{name}_{model}_features.pt`

Cache is reused if file exists and sample count matches. Delete to force recomputation:

```bash
rm -rf ./embeddings
```

## Memory

GPU memory usage:
- Models: ~2-3 GB (CLIP + ViT-MSN + DINOv2)
- Features: ~4 bytes × n_samples × 2048 (all model dims)
- PRDC distances: O(n²) temporary

Reduce `batch_size` if OOM.

## Hyperparameter Tuning

### nearest_k

Controls manifold resolution. Larger k = smoother estimates, less sensitive to noise.

| Dataset size | Recommended k |
|--------------|---------------|
| <1000 | 3-5 |
| 1000-10000 | 5-10 |
| >10000 | 10-20 |

### val_split

Fraction of training data used for hyperparameter selection.

```python
detector.fit(paths, val_split=0.1)  # 90% train, 10% validation
```

## Reproducibility

```python
import torch
import numpy as np

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

detector.fit(paths, random_state=42)
```
