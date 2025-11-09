# Technical Methods

Deep dive into the algorithms and techniques used in Forte.

## Overview

Forte combines three key components for effective out-of-distribution detection:

1. **Multi-Model Feature Extraction** - Leveraging pretrained vision models
2. **PRDC Topology Estimation** - Computing distributional metrics
3. **Density-Based Detection** - Identifying anomalies in feature space

## Feature Extraction

### Pretrained Models

Forte uses three complementary pretrained vision models:

#### CLIP (Contrastive Language-Image Pre-training)
- **Model**: `openai/clip-vit-base-patch32`
- **Architecture**: Vision Transformer (ViT-B/32)
- **Features**: 512-dimensional embeddings
- **Training**: Contrastive learning on 400M image-text pairs
- **Strengths**: Captures semantic and text-aligned concepts

$$\text{CLIP}(x) = f_{\text{visual}}(x) \in \mathbb{R}^{512}$$

#### ViT-MSN (Vision Transformer with Masked Siamese Networks)
- **Model**: `facebook/vit-msn-base`
- **Architecture**: Vision Transformer Base
- **Features**: 768-dimensional embeddings (CLS token)
- **Training**: Self-supervised masked image modeling
- **Strengths**: Strong spatial and structural understanding

$$\text{ViT-MSN}(x) = h_{\text{CLS}}(x) \in \mathbb{R}^{768}$$

#### DINOv2 (Self-Distillation with No Labels v2)
- **Model**: `facebook/dinov2-base`
- **Architecture**: Vision Transformer Base
- **Features**: 768-dimensional embeddings
- **Training**: Self-supervised distillation
- **Strengths**: Robust to distribution shifts, excellent for dense predictions

$$\text{DINOv2}(x) = g_{\text{CLS}}(x) \in \mathbb{R}^{768}$$

### Feature Concatenation

For each image $x$, we extract features from all three models:

$$\phi(x) = [\text{CLIP}(x), \text{ViT-MSN}(x), \text{DINOv2}(x)]$$

## PRDC Metrics

PRDC (Precision, Recall, Density, Coverage) provides a topology-aware characterization of distributions.

### Mathematical Formulation

Given:
- Reference features: $\mathbf{X}_{\text{ref}} = \{x_1, \ldots, x_n\}$
- Query features: $\mathbf{X}_{\text{query}} = \{y_1, \ldots, y_m\}$
- k-NN radius for $x$: $r_k(x)$ (distance to k-th nearest neighbor)

### Precision

Measures if query samples fall within the manifold of reference data:

$$\text{Precision} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}\left[\exists x \in \mathbf{X}_{\text{ref}} : \|y_i - x\| < r_k(x)\right]$$

### Recall

Measures coverage of the reference distribution:

$$\text{Recall} = \frac{1}{n \cdot m} \sum_{i=1}^{m} \left|\{x \in \mathbf{X}_{\text{ref}} : \|y_i - x\| < r_k(y_i)\}\right|$$

### Density

Local density estimation using k-NN:

$$\text{Density} = \frac{1}{km} \sum_{i=1}^{m} \left|\{x \in \mathbf{X}_{\text{ref}} : \|y_i - x\| < r_k(x)\}\right|$$

### Coverage

Mode coverage of the distribution:

$$\text{Coverage} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}\left[\min_{x \in \mathbf{X}_{\text{ref}}} \|y_i - x\| < r_k(y_i)\right]$$

### PRDC Feature Vector

For each model's features, we compute all 4 PRDC metrics, resulting in a 12-dimensional feature vector:

$$\text{PRDC}(x) = [P_1, R_1, D_1, C_1, P_2, R_2, D_2, C_2, P_3, R_3, D_3, C_3] \in \mathbb{R}^{12}$$

where subscripts 1, 2, 3 correspond to CLIP, ViT-MSN, and DINOv2 respectively.

## Detection Methods

### Gaussian Mixture Models (GMM)

Models the distribution of PRDC features as a mixture of Gaussians:

$$p(\mathbf{z}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{z} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where:
- $K$ is the number of components (selected via BIC)
- $\pi_k$ are mixture weights
- $\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$ are mean and covariance of component $k$

**Training**: Expectation-Maximization (EM) algorithm

**Scoring**: Log-likelihood under the mixture:

$$s_{\text{GMM}}(\mathbf{z}) = \log \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{z} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

**Model Selection**: Bayesian Information Criterion (BIC):

$$\text{BIC} = -2\log\mathcal{L} + p\log(n)$$

where $p$ is the number of parameters and $n$ is the number of samples.

### Kernel Density Estimation (KDE)

Non-parametric density estimation using Gaussian kernels:

$$p(\mathbf{z}) = \frac{1}{n} \sum_{i=1}^{n} K_h(\mathbf{z} - \mathbf{z}_i)$$

where $K_h$ is a Gaussian kernel with bandwidth $h$:

$$K_h(\mathbf{u}) = \frac{1}{(2\pi h^2)^{d/2}} \exp\left(-\frac{\|\mathbf{u}\|^2}{2h^2}\right)$$

**Bandwidth Selection**: Scott's rule:

$$h = n^{-1/(d+4)} \cdot \sigma$$

where $\sigma$ is the standard deviation of the data.

**Scoring**: Log probability density:

$$s_{\text{KDE}}(\mathbf{z}) = \log p(\mathbf{z})$$

### One-Class SVM (OCSVM)

Learns a decision boundary enclosing in-distribution data:

$$\min_{\mathbf{w}, \rho, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 - \rho + \frac{1}{\nu n}\sum_{i=1}^{n} \xi_i$$

subject to:
$$\mathbf{w}^T\phi(\mathbf{z}_i) \geq \rho - \xi_i, \quad \xi_i \geq 0$$

where:
- $\mathbf{w}$ is the normal vector
- $\rho$ is the offset
- $\boldsymbol{\xi}$ are slack variables
- $\nu \in (0, 1)$ bounds the fraction of outliers

**Scoring**: Decision function:

$$s_{\text{OCSVM}}(\mathbf{z}) = \mathbf{w}^T\mathbf{z} - \rho$$

## GPU Acceleration

Forte implements custom PyTorch versions of all detection algorithms for GPU acceleration.

### TorchGMM

- Full covariance matrices stored as tensors
- Batched E-step using `torch.logsumexp`
- Efficient M-step with matrix operations
- ~10-50x faster than scikit-learn on GPU

### TorchKDE

- Cholesky decomposition for covariance
- Batched kernel evaluation
- Memory-efficient for large datasets
- ~20-100x faster than scipy on GPU

### TorchOCSVM

- Gradient-based optimization (Adam)
- Soft margin with clamped slack variables
- Iterative refinement of decision boundary
- ~5-20x faster than scikit-learn on GPU

## Training Pipeline

### 1. Feature Extraction

```
For each image x in training set:
    Extract CLIP features f1(x)
    Extract ViT-MSN features f2(x)
    Extract DINOv2 features f3(x)
    Cache to disk
```

### 2. PRDC Computation

```
For each model m:
    Split features into two halves: F_ref, F_query
    For each query feature q in F_query:
        Compute k-NN radii
        Compute PRDC(q) = [P, R, D, C]
    Concatenate PRDC features
```

### 3. Detector Training

```
Input: PRDC features Z = [z1, ..., zn]

If method = GMM:
    For k in [1, 2, 4, 8, 16, 32, 64]:
        Fit GMM with k components
        Compute BIC(k)
    Select k* = argmin BIC

If method = KDE:
    Compute bandwidth h using Scott's rule
    Fit KDE with bandwidth h

If method = OCSVM:
    For nu in [0.01, 0.05, 0.1, 0.2, 0.5]:
        Fit OCSVM with nu
        Evaluate on validation set
    Select nu* with best accuracy
```

### 4. Inference

```
For each test image x:
    Extract features [f1(x), f2(x), f3(x)]
    Compute PRDC(x) using cached training features
    score = detector.score(PRDC(x))
    prediction = 1 if score > threshold else -1
```

## Complexity Analysis

Let $n$ be the number of training images, $m$ the number of test images, and $d$ the feature dimension.

### Time Complexity

| Operation | Complexity |
|-----------|-----------|
| Feature Extraction | $O(n \cdot T)$ where $T$ is model forward pass time |
| PRDC Computation | $O(n^2 \cdot d)$ for pairwise distances |
| GMM Training | $O(K \cdot I \cdot n \cdot d^2)$ where $I$ is EM iterations |
| KDE Training | $O(n \cdot d)$ |
| OCSVM Training | $O(T_{\text{opt}} \cdot n \cdot d)$ where $T_{\text{opt}}$ is optimization steps |
| Inference (per image) | $O(d + n)$ for PRDC + scoring |

### Space Complexity

| Component | Complexity |
|-----------|-----------|
| Cached Features | $O(n \cdot d)$ |
| PRDC Features | $O(n \cdot 12)$ |
| GMM Parameters | $O(K \cdot d^2)$ |
| KDE Data | $O(n \cdot d)$ |
| OCSVM Parameters | $O(d)$ |

## Implementation Details

### Numerical Stability

- Add small regularization ($10^{-6}$) to covariance matrices
- Use log-space computations for GMM
- Clamp very small/large values in KDE
- Normalize features before OCSVM

### Caching Strategy

Features are cached with naming convention:
```
{embedding_dir}/{dataset_name}_{model_name}_features.pt
```

Cached features are automatically loaded if:
1. Cache file exists
2. Number of cached features matches number of images

### Reproducibility

Set random seeds for reproducibility:
```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
```

## Performance Characteristics

### Method Comparison

| Method | Speed | Accuracy | Memory | Best For |
|--------|-------|----------|--------|----------|
| GMM | Medium | High | Medium | Most datasets, multi-modal distributions |
| KDE | Slow | High | High | Small datasets, complex boundaries |
| OCSVM | Fast | Medium | Low | Large datasets, simple boundaries |

### Scalability

- **Small datasets** (<1K images): All methods work well
- **Medium datasets** (1K-10K): GMM recommended
- **Large datasets** (>10K): OCSVM for speed, GMM for accuracy

## References

1. **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
2. **ViT-MSN**: Assran et al., "Masked Siamese Networks for Label-Efficient Learning", ECCV 2022
3. **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv 2023
4. **PRDC**: Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models", NeurIPS 2019

## Next Steps

- [Examples](examples.md) - See practical applications
- [User Guide](user-guide.md) - Learn to use the API
- [API Reference](api-reference.md) - Detailed documentation
