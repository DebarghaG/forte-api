# Algorithm

## Problem

Given reference data $\mathbf{X}_{\text{ref}} = \{x_i^r\}_{i=1}^m \sim P$ and test data $\mathbf{X}_{\text{test}} = \{x_j^g\}_{j=1}^n \sim \alpha P + (1-\alpha) Q$ where $Q$ is an unknown OOD distribution and $\alpha \in [0,1]$ is unknown, determine which $x_j^g \notin \text{supp}(P)$.

## Notation

| Symbol | Definition |
|--------|------------|
| $\text{NND}_k(x)$ | Distance from $x$ to its $k$-th nearest neighbor |
| $B(x, r)$ | Closed ball $\{y : \|x - y\| \leq r\}$ |
| $S(\mathbf{X})$ | $\bigcup_{i} B(x_i, \text{NND}_k(x_i))$ |
| $\mathbf{1}[\cdot]$ | Indicator function |

## Per-Point PRDC Metrics

For each test point $x_j^g$, compute four statistics relative to $\mathbf{X}_{\text{ref}}$:

**Precision** (binary):
$$\text{precision}_j = \mathbf{1}\left[x_j^g \in S(\mathbf{X}_{\text{ref}})\right]$$

**Recall** (continuous):
$$\text{recall}_j = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\left[x_i^r \in B(x_j^g, \text{NND}_k(x_j^g))\right]$$

**Density** (continuous):
$$\text{density}_j = \frac{1}{km} \sum_{i=1}^{m} \mathbf{1}\left[x_j^g \in B(x_i^r, \text{NND}_k(x_i^r))\right]$$

**Coverage** (binary):
$$\text{coverage}_j = \mathbf{1}\left[\min_i \|x_j^g - x_i^r\| < \text{NND}_k(x_j^g)\right]$$

These metrics capture local manifold geometry. OOD samples fall outside high-density regions, yielding low metric values. See [paper](https://openreview.net/pdf?id=7XNgVPxCiA) Section 3 for theoretical analysis.

## Feature Extraction

| Model | Dim | HuggingFace ID |
|-------|-----|----------------|
| CLIP ViT-B/32 | 512 | `openai/clip-vit-base-patch32` |
| ViT-MSN | 768 | `facebook/vit-msn-base` |
| DINOv2 | 768 | `facebook/dinov2-base` |

For each image, extract CLS token embeddings from all three models. PRDC computed independently per model, then concatenated: 4 metrics × 3 models = 12-dimensional feature vector.

## Training Procedure

```
Input: ID image paths, method ∈ {gmm, kde, ocsvm}, k
Output: Fitted detector

1. Extract features F_ref for all images
2. Split F_ref into F_train (50%) and F_val (50%)
3. For each model m ∈ {clip, vitmsn, dinov2}:
     Compute NND_k radii on F_train[m]
     Compute PRDC(F_train[m], F_val[m]) → 4-dim vector per sample
4. Concatenate PRDC vectors → Z ∈ R^{n×12}
5. Fit density estimator on Z:
     GMM: Select components via BIC from {1,2,4,8,16,32,64}
     KDE: Bandwidth via Scott's rule
     OCSVM: Select ν from {0.01,0.05,0.1,0.2,0.5} by validation accuracy
```

## Inference

```
Input: Test image paths
Output: Scores (higher = more likely ID)

1. Extract features F_test
2. For each model m:
     Compute PRDC(F_train[m], F_test[m])
3. Concatenate → Z_test ∈ R^{n×12}
4. Score:
     GMM: log p(z)
     KDE: log p(z)
     OCSVM: decision function value
```

## Density Estimators

### GMM

Mixture of $K$ Gaussians:
$$p(z) = \sum_{k=1}^{K} \pi_k \mathcal{N}(z \mid \mu_k, \Sigma_k)$$

Component count selected by minimizing BIC:
$$\text{BIC} = -2 \log \mathcal{L} + p \log n$$

### KDE

Non-parametric density with Gaussian kernel:
$$p(z) = \frac{1}{n} \sum_{i=1}^{n} K_h(z - z_i)$$

Bandwidth $h$ via Scott's rule: $h = n^{-1/(d+4)} \sigma$

### OCSVM

Finds hyperplane separating origin from data:
$$\min_{w,\rho,\xi} \frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n} \sum_i \xi_i$$
subject to $w^\top z_i \geq \rho - \xi_i$, $\xi_i \geq 0$

Score: $w^\top z - \rho$

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Feature extraction | $O(n \cdot T_{\text{forward}})$ | $O(n \cdot d)$ |
| Pairwise distances | $O(n^2 \cdot d)$ | $O(n^2)$ |
| GMM training | $O(K \cdot I \cdot n \cdot d^2)$ | $O(K \cdot d^2)$ |
| KDE evaluation | $O(n_{\text{train}} \cdot n_{\text{test}})$ | $O(n_{\text{train}} \cdot d)$ |
| OCSVM training | $O(T_{\text{opt}} \cdot n \cdot d)$ | $O(d)$ |

Where $K$ = GMM components, $I$ = EM iterations, $d$ = feature dimension.

## Method Selection

| Method | Use when |
|--------|----------|
| GMM | Default choice. Multi-modal ID distributions. |
| KDE | Small datasets (<1000). Smooth decision boundaries. |
| OCSVM | Large datasets. Fast inference required. |
