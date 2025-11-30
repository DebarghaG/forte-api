# Forte

[![PyPI](https://badge.fury.io/py/forte-detector.svg)](https://pypi.org/project/forte-detector/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-red.svg)](https://openreview.net/pdf?id=7XNgVPxCiA)

Out-of-distribution detection via per-point manifold estimation on self-supervised representations.

**Paper**: [PDF](https://openreview.net/pdf?id=7XNgVPxCiA) | [arXiv](https://arxiv.org/abs/2410.01322)

**Documentation**: [debarghag.github.io/forte-detector](https://debarghag.github.io/forte-detector)

## Installation

```bash
pip install forte-detector
```

## Usage

```python
from forte import ForteOODDetector

detector = ForteOODDetector(method='gmm', device='cuda:0')
detector.fit(train_paths)
predictions = detector.predict(test_paths)
metrics = detector.evaluate(id_test_paths, ood_test_paths)
```

## Method

Forte detects OOD samples by:
1. Extracting features from CLIP, ViT-MSN, and DINOv2
2. Computing per-point PRDC metrics using k-NN manifold geometry
3. Fitting a density estimator (GMM, KDE, or OCSVM) on PRDC features
4. Scoring test samples by typicality under the learned density

No class labels or OOD exposure required during training.

## Citation

```bibtex
@inproceedings{ganguly2025forte,
  title={Forte: Finding Outliers with Representation Typicality Estimation},
  author={Ganguly, Debargha and Morningstar, Warren Richard and Yu, Andrew Seohwan and Chaudhary, Vipin},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/pdf?id=7XNgVPxCiA}
}
```

## License

MIT. Supported by NSF ICICLE (OAC 2112606).
