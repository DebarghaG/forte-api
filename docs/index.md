# Forte

Out-of-distribution detection via per-point manifold estimation on self-supervised representations.

**Paper**: [ICLR 2025](https://openreview.net/pdf?id=7XNgVPxCiA)

## Method

Forte detects OOD samples by:

1. Extracting features from CLIP, ViT-MSN, and DINOv2
2. Computing per-point PRDC metrics using k-NN manifold geometry
3. Fitting a density estimator (GMM, KDE, or OCSVM) on the PRDC feature space
4. Scoring test samples by their typicality under the learned density

No class labels or OOD exposure required during training.

## Installation

```bash
pip install forte-detector
```

## Example

```python
from forte import ForteOODDetector

detector = ForteOODDetector(method='gmm', device='cuda:0')
detector.fit(train_paths)
predictions = detector.predict(test_paths)
metrics = detector.evaluate(id_test_paths, ood_test_paths)
```

## Documentation

- [Quickstart](quickstart.md)
- [Algorithm](methods.md)
- [API Reference](api-reference.md)
- [Configuration](user-guide.md)
- [Examples](examples.md)

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

MIT. See [LICENSE](https://github.com/debarghag/forte-detector/blob/main/LICENSE).

Supported by NSF ICICLE (OAC 2112606).
