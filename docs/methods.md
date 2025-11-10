# Technical Methods

Deep dive into the algorithms and techniques used in Forte.

## Overview

Forte combines three key components for effective out-of-distribution detection:

1. **Multi-Model Feature Extraction** - Leveraging pretrained vision models
2. **PRDC Topology Estimation** - Computing distributional metrics
3. **Density-Based Detection** - Identifying anomalies in feature space

## Problem Formulation

Forte addresses the out-of-distribution (OOD) detection problem where we aim to identify inputs that are atypical compared to a reference distribution.

### Setup

We start with a dataset $X = \{x_i^r\}_{i=1}^m$ sampled independently and identically from an unknown true distribution $p$, where each $x_i \in \mathbb{R}^{d}$. During deployment, unseen data $\{x_j^g\}_{j=1}^n$ may come from a mixture of the true distribution $p$ and an unknown confounding distribution $\tilde{p}$ (e.g., OOD benchmarks, synthetic data from generative models):

$$\grave{X} \sim \alpha p(\grave{X}) + (1 - \alpha)\tilde{p}(\grave{X})$$

where $\alpha$ is an unknown mixing parameter. Since both $\alpha$ and $\tilde{p}$ are unknown, we cannot directly sample from $\tilde{p}$ or make assumptions about these parameters.

### Objective

The goal is to develop a decision rule that determines when input data $\grave{X}$ is atypical without requiring:
- Class labels
- Exposure to OOD data during training
- Assumptions about the architecture of generative models

### Approach

Forte builds on Density of States Estimation (DoSE) but extends beyond likelihood-based generative models. Instead of relying on generative model likelihoods, which can be suboptimal for OOD detection, we:

1. Create summary statistics that capture local geometric properties of data manifolds in feature space
2. Use self-supervised representations that focus on semantic content while discarding confounding features
3. Model the distribution of these statistics using non-parametric density estimation
4. Score test samples based on their typicality relative to the reference distribution

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

## Per-Point PRDC Metrics

PRDC (Precision, Recall, Density, Coverage) provides a topology-aware characterization of distributions through **per-point summary statistics**. Unlike aggregate metrics that summarize entire distributions, these per-point metrics capture local geometric properties for each individual sample in the feature space, enabling fine-grained anomaly detection.

### Notation

Given:
- Reference features: $\mathbf{X}_{\text{ref}} = \{x_i^r\}_{i=1}^{m}$ from the in-distribution
- Test features: $\mathbf{X}_{\text{test}} = \{x_j^g\}_{j=1}^{n}$ from unseen data
- Indicator function: $\mathds{1}(\cdot)$ returns 1 if condition is true, 0 otherwise
- k-NN distance: $\mathrm{NND}_k(x_i^r)$ is the distance between $x_i^r$ and its k-th nearest neighbor
- Neighborhood: $S(\{x_i^r\}_{i=1}^m) = \bigcup_{i=1}^m B(x_i^r, \mathrm{NND}_k(x_i^r))$, where $B(x, r)$ is a Euclidean ball centered at $x$ with radius $r$

### Precision Per Point (`precision_pp`)

**Binary statistic** indicating whether each test point falls within the nearest neighbor distance of any reference point:

$$\mathrm{precision_{pp}^{(j)}} = \mathds{1}\left(x_j^g \in S(\{x_i^r\}_{i=1}^m)\right)$$

**Interpretation**: A high value indicates the test sample is closely aligned and similar to the reference data distribution. Test points with low precision are likely OOD.

### Recall Per Point (`recall_pp`)

**Continuous statistic** counting the number of reference points within each test point's nearest neighbor distance:

$$\mathrm{recall_{pp}^{(j)}} = \frac{1}{m} \sum_{i=1}^m \mathds{1}\left(x_i^r \in B(x_j^g, \mathrm{NND}_k(x_j^g))\right)$$

**Interpretation**: High recall implies the test distribution collectively covers a significant portion of the reference data, indicating diversity and representation across different regions of the reference manifold.

### Density Per Point (`density_pp`)

**Continuous statistic** measuring expected likelihood by counting reference points that contain the test point within their neighborhoods:

$$\mathrm{density_{pp}^{(j)}} = \frac{1}{km} \sum_{i=1}^m \mathds{1}\left(x_j^g \in B(x_i^r, \mathrm{NND}_k(x_i^r))\right)$$

**Interpretation**: High density suggests the test point is located in a high-probability region of the reference distribution. This provides a more informative measure than binary precision by quantifying how typical the location is.

### Coverage Per Point (`coverage_pp`)

**Binary statistic** checking if the distance to the nearest reference point is less than the test point's own nearest neighbor distance:

$$\mathrm{coverage_{pp}^{(j)}} = \mathds{1}\left(\min_{i} d(x_j^g, x_i^r) < \mathrm{NND}_k(x_j^g)\right)$$

**Interpretation**: High coverage indicates test samples are well-distributed across the support of the reference distribution. This improves upon the original recall metric by building manifolds around reference points, making it more robust to outliers.

### Theoretical Justification

Under certain theoretical assumptions, these per-point metrics effectively distinguish between in-distribution (ID) and out-of-distribution (OOD) data. Specifically, when reference data $\{x_j^r\}_{j=1}^m$ and test data $\{x_i^g\}_{i=1}^n$ are drawn from Gaussian distributions with the same covariance but different means (with significant mean difference), the expected values differ markedly:

**For ID data:**
- Expected precision_pp and coverage_pp: $\approx 1 - e^{-k}$
- Expected recall_pp: $\approx k/m$
- Expected density_pp: $\approx 1$

**For OOD data:**
- All expected values: $\approx 0$

This substantial disparity occurs because OOD samples fall outside the typical regions of the reference distribution due to the large mean difference. This provides a strong theoretical foundation for using these metrics as effective summary statistics for OOD detection.

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

## Decision Rules and Thresholding

The per-point summary statistics enable us to develop non-parametric density estimators as anomaly detection models. The decision rule is based on modeling the typical set of the reference distribution and identifying samples that fall outside this set.

### Training Strategy

To understand what the summary statistics look like when test data matches the reference distribution (i.e., $P \overset{d}{=} Q$), we split the reference data into three parts:

1. **Reference distribution** (1/3): Used to compute per-point metrics for other samples
2. **Test distribution** (1/3): Drawn from the reference distribution, used to compute statistics and train density models
3. **Held-out test set** (1/3): Reserved for evaluation

The density estimation models (GMM, KDE, OCSVM) are trained on the summary statistics from the test distribution, learning a decision boundary that encloses the typical set of the reference data distribution.

### Atypicality Scoring

During inference, we evaluate a test sample's atypicality by:

1. Computing its per-point metrics relative to the reference distribution
2. Scoring these metrics using the trained density model
3. Comparing the score against a threshold

Samples with scores below the threshold (falling outside the typical set) are classified as OOD.

### Threshold Selection

The decision threshold is selected to balance the trade-off between:
- **True Positive Rate (TPR)**: Correctly identifying OOD samples
- **False Positive Rate (FPR)**: Incorrectly flagging ID samples as OOD

Common strategies include:
- Fixed threshold based on validation set performance
- Adaptive threshold targeting a specific FPR (e.g., FPR@95TPR)
- Percentile-based threshold on training scores

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

## Evaluation Metrics

To assess the performance of OOD detection models, we use metrics that measure the ability to discriminate between in-distribution and out-of-distribution samples across different decision thresholds.

### AUROC (Area Under the ROC Curve)

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings:

$$\text{TPR} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

$$\text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}$$

The AUROC summarizes the ROC curve into a single scalar value between 0 and 1:
- **AUROC = 1.0**: Perfect discrimination (all OOD samples scored lower than all ID samples)
- **AUROC = 0.5**: Random discrimination (no better than chance)
- **AUROC < 0.5**: Worse than random (inverted predictions)

AUROC measures the probability that a randomly chosen OOD sample receives a lower score than a randomly chosen ID sample, making it threshold-independent and robust to class imbalance.

### FPR@95 (False Positive Rate at 95% True Positive Rate)

FPR@95TPR measures the proportion of in-distribution samples incorrectly classified as OOD when the model correctly identifies 95% of true OOD samples:

$$\text{FPR@95} = \text{FPR at threshold where TPR} = 0.95$$

This metric is particularly important for OOD detection because:
- It reflects real-world deployment scenarios where we want to catch most anomalies
- Lower values indicate fewer false alarms on normal data
- It provides a practical operating point rather than an aggregate measure

**Target values:**
- **FPR@95 = 0%**: Ideal performance (no false alarms while detecting 95% of OOD)
- **FPR@95 < 10%**: Excellent performance
- **FPR@95 > 50%**: Poor performance (too many false alarms)

### Why These Metrics for OOD Detection

Traditional classification metrics (accuracy, precision, recall) can be misleading for OOD detection because:
1. Class imbalance varies significantly between deployment scenarios
2. The cost of false positives vs. false negatives is application-dependent
3. We need threshold-independent measures (AUROC) and practical operating points (FPR@95)

Together, AUROC and FPR@95 provide complementary views:
- **AUROC**: Overall discriminative ability
- **FPR@95**: Practical performance at a specific operating point

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

## Experimental Overview

Forte has been extensively evaluated across multiple domains and scenarios to validate its effectiveness for OOD detection.

### Benchmark Datasets

**Natural Images:**
- CIFAR-10/100: Standard benchmark for OOD detection
- ImageNet-1k: Large-scale dataset with 1000 object classes
- iNaturalist, Texture, OpenImage-O: Far-OOD evaluation sets
- NINCO, SSB-Hard: Challenging near-OOD datasets
- ImageNet-C, ImageNet-R, ImageNet-V2: Covariate shift and robustness testing

**Medical Imaging:**
- FastMRI: Multi-coil knee MRI scans with varying acquisition protocols
- OAI (Osteoarthritis Initiative): Knee MRI with different sequences (TSE, T1, MPR)
- Application: Detecting batch effects and protocol differences

**Synthetic Data:**
- Generated using Stable Diffusion 2.0 with multiple approaches:
  - **Img2Img**: Varying strength parameters (0.3, 0.5, 0.7, 0.9, 1.0) controlling input image influence
  - **Caption-based**: Generated from BLIP-generated captions of real images
  - **Class-based**: Generated directly from class names (e.g., "a photo of a monarch butterfly")

### Baseline Comparisons

**Unsupervised Methods:**
- DoSE (Density of States Estimation): State-of-the-art unsupervised baseline using Glow models
- WAIC (Watanabe-Akaike Information Criterion)
- TT (Single-sample Typicality Test)
- LLR (Likelihood Ratio method)
- Single-sided threshold

**Supervised Methods:**
- NNGuide: Nearest-neighbor guidance for OOD detection
- ViM (Virtual Logit Matching)
- OpenOOD v1.5: ViT-B with cross-entropy + RMDS/MLS postprocessors
- DINOv2+MLS: Linear probe on DINOv2 features

**Distribution Metrics:**
- Fréchet Distance (FD) and $FD_\infty$ with DINOv2 encoder
- CMMD (CLIP Maximum Mean Discrepancy)
- Statistical tests: Kolmogorov-Smirnov, Mann-Whitney U, Z-test
- Divergence measures: KL, JS, Wasserstein, Bhattacharyya distances

## Results Summary

Forte consistently achieves state-of-the-art performance across diverse OOD detection scenarios, outperforming both supervised and unsupervised baselines.

### Key Findings

**1. Superior Performance on Standard Benchmarks**

Forte+GMM demonstrates exceptional performance on established OOD detection benchmarks:
- **iNaturalist (Far-OOD)**: AUROC 99.67%, FPR@95 0.64% (vs. best supervised baseline 99.57% / 1.83%)
- **NINCO (Near-OOD)**: AUROC 98.34%, FPR@95 5.18% (vs. best supervised baseline 88.38% / 41.02%)
- **SSB-Hard (Challenging Near-OOD)**: AUROC 94.95%, FPR@95 22.30% (vs. best supervised baseline 77.28% / 72.90%)

Forte significantly outperforms on challenging datasets where supervised methods struggle, particularly on near-OOD scenarios with semantic similarity to in-distribution data.

**2. Dominance Over Unsupervised Baselines**

On CIFAR-10 in-distribution detection:
- **CIFAR-100 (OOD)**: Forte+GMM achieves 97.63% AUROC vs. DoSE's 56.90%
- **Celeb-A (OOD)**: Perfect 100% AUROC, 0% FPR@95 (DoSE: 97.60% / 12.82%)
- **SVHN (OOD)**: 99.49% AUROC, 0% FPR@95 (DoSE: 97.30% / 13.16%)

Forte demonstrates substantial improvements over likelihood-based methods, validating the approach of using semantic representations and per-point metrics.

**3. Multi-Model Ensemble Benefits**

Ablation studies on ImageNet hierarchy classification show combining representations improves performance:
- **Far-OOD Detection**: CLIP+MSN+DINOv2 achieves 100% AUROC vs. 99.13-99.79% for individual models
- **Near-OOD Detection**: CLIP+DINOv2 reaches 91.35% AUROC, 26.89% FPR@95 (best two-model combination)
- **Individual Models**: Each model provides complementary information about different aspects of the data manifold

The multi-model approach captures diverse semantic properties, enhancing robustness across different OOD types.

**4. Effective Synthetic Image Detection**

Forte successfully detects synthetic images generated by Stable Diffusion across varying generation settings:
- **High-strength img2img (S=0.9, 1.0)**: AUROC >97%, FPR@95 <15%
- **Caption-based generation**: AUROC 96.77%, FPR@95 18.90%
- **Class-based generation**: AUROC 98.26%, FPR@95 10.22%

Performance improves as generated images diverge from the reference distribution (higher diffusion strength). Distribution-level metrics (FD, CMMD) and statistical tests show inconsistent patterns, highlighting the advantage of per-point detection.

**5. Medical Imaging Applications**

Near-perfect performance detecting batch effects and protocol differences in MRI datasets:
- **FastMRI vs. OAI datasets**: Forte+SVM achieves 100% AUROC, 0% FPR@95
- **Forte+GMM**: 99.91-99.95% AUROC across different protocol pairs

This demonstrates zero-shot applicability to high-stakes domains where distribution shift detection is critical for model deployment and data harmonization.

### Limitations and Considerations

- **Low-strength synthetic images** (img2img S<0.5): Detection becomes challenging when generated images are very similar to reference data
- **Computational cost**: Multi-model feature extraction and PRDC computation scale quadratically with dataset size
- **Mode collapse scenarios**: Performance may degrade when generative models produce limited diversity (e.g., volleyball class example)

## References

### Core Methods

1. **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
2. **ViT-MSN**: Assran et al., "Masked Siamese Networks for Label-Efficient Learning", ECCV 2022
3. **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv 2023
4. **PRDC**: Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models", NeurIPS 2019
5. **DoSE**: Morningstar et al., "Density of States Estimation for Out-of-Distribution Detection", AISTATS 2021

### Baseline Methods

6. **WAIC**: Choi et al., "WAIC, but Why? Generative Ensembles for Robust Anomaly Detection", arXiv 2018
7. **Typicality Test**: Nalisnick et al., "Do Deep Generative Models Know What They Don't Know?", ICLR 2019
8. **Likelihood Ratio**: Ren et al., "Likelihood Ratio for Out-of-Distribution Detection", NeurIPS 2019
9. **NNGuide**: Park et al., "Nearest Neighbor Guidance for Out-of-Distribution Detection", ICCV 2023
10. **ViM**: Wang et al., "Virtual Logit Matching for Out-of-Distribution Detection", arXiv 2022
11. **OpenOOD**: Zhang et al., "OpenOOD v1.5: Benchmarking Out-of-Distribution Detection", NeurIPS 2024

### Generative Models and Evaluation

12. **Stable Diffusion**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
13. **BLIP**: Li et al., "BLIP: Bootstrapping Language-Image Pre-training", ICML 2022
14. **Fréchet Distance**: Stein et al., "Exposing Flaws of Generative Model Evaluation Metrics", arXiv 2024
15. **CMMD**: Jayasumana et al., "Rethinking FID: Towards a Better Evaluation Metric for Image Generation", CVPR 2024

## Next Steps

- [Examples](examples.md) - See practical applications
- [User Guide](user-guide.md) - Learn to use the API
- [API Reference](api-reference.md) - Detailed documentation
