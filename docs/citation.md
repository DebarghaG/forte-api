# Citation & Acknowledgements

## Citing Forte

If you use Forte in your research, please cite our ICLR 2025 paper:

### BibTeX

```bibtex
@inproceedings{ganguly2025forte,
  title={Forte: Finding Outliers with Representation Typicality Estimation},
  author={Debargha Ganguly and Warren Richard Morningstar and Andrew Seohwan Yu and Vipin Chaudhary},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=7XNgVPxCiA}
}
```

### Text Citation

Debargha Ganguly, Warren Richard Morningstar, Andrew Seohwan Yu, and Vipin Chaudhary. "Forte: Finding Outliers with Representation Typicality Estimation." In *The Thirteenth International Conference on Learning Representations* (ICLR 2025). [https://openreview.net/forum?id=7XNgVPxCiA](https://openreview.net/forum?id=7XNgVPxCiA)

## Paper Links

- **OpenReview**: [https://openreview.net/forum?id=7XNgVPxCiA](https://openreview.net/forum?id=7XNgVPxCiA)
- **Conference**: ICLR 2025
- **PDF**: Available on OpenReview

## Software Citation

For the software package itself:

```bibtex
@software{forte_detector_2025,
  author = {Debargha Ganguly and Warren Richard Morningstar and Andrew Seohwan Yu and Vipin Chaudhary},
  title = {Forte Detector: PyTorch library for out-of-distribution detection},
  year = {2025},
  publisher = {PyPI},
  version = {0.1.0},
  url = {https://github.com/debargha/forte-detector}
}
```

## Acknowledgements

### Funding

This work was supported by the **NSF ICICLE (Intelligent CyberInfrastructure with Computational Learning in the Environment)** grant. We gratefully acknowledge this support.

### Open Source Libraries

Forte builds upon several excellent open-source projects:

#### Core Dependencies

- **PyTorch** - Deep learning framework
  Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS 2019

- **Hugging Face Transformers** - Pretrained models
  Wolf et al., "Transformers: State-of-the-Art Natural Language Processing", EMNLP 2020

- **scikit-learn** - Machine learning utilities
  Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 2011

- **NumPy** - Numerical computing
  Harris et al., "Array programming with NumPy", Nature 2020

- **SciPy** - Scientific computing
  Virtanen et al., "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python", Nature Methods 2020

#### Pretrained Models

- **CLIP** (OpenAI)
  Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
  Model: `openai/clip-vit-base-patch32`

- **ViT-MSN** (Meta AI)
  Assran et al., "Masked Siamese Networks for Label-Efficient Learning", ECCV 2022
  Model: `facebook/vit-msn-base`

- **DINOv2** (Meta AI)
  Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv 2023
  Model: `facebook/dinov2-base`

#### PRDC Metrics

- **Improved Precision and Recall Metric**
  Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models", NeurIPS 2019

### Development Tools

- **MkDocs Material** - Documentation
- **pytest** - Testing framework
- **GitHub Actions** - CI/CD

## Authors

### Debargha Ganguly
- **Affiliation**: [Your Institution]
- **Email**: debargha.ganguly@gmail.com
- **Role**: Lead developer, primary author

### Warren Richard Morningstar
- **Affiliation**: [Your Institution]
- **Role**: Co-author

### Andrew Seohwan Yu
- **Affiliation**: [Your Institution]
- **Role**: Co-author

### Vipin Chaudhary
- **Affiliation**: [Your Institution]
- **Role**: Principal investigator

## Contributing

We welcome contributions from the community! Please see our [contributing guidelines](https://github.com/debargha/forte-detector/blob/main/CONTRIBUTING.md) for more information.

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Reporting Issues

Please report bugs and feature requests on our [GitHub Issues](https://github.com/debargha/forte-detector/issues) page.

## License

Forte is released under the **MIT License**:

```
MIT License

Copyright (c) 2025 Debargha Ganguly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Related Work

If you're interested in out-of-distribution detection, you may also find these works relevant:

1. **ODIN** - Liang et al., "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks", ICLR 2018

2. **Mahalanobis Distance** - Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018

3. **Energy-based OOD** - Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020

4. **OpenOOD** - Zhang et al., "OpenOOD: Benchmarking Generalized Out-of-Distribution Detection", NeurIPS 2022

5. **ViM** - Wang et al., "ViM: Out-Of-Distribution with Virtual-logit Matching", CVPR 2022

## Contact

For questions, comments, or collaborations:

- **Email**: debargha.ganguly@gmail.com
- **GitHub**: [https://github.com/debargha/forte-detector](https://github.com/debargha/forte-detector)
- **Issues**: [https://github.com/debargha/forte-detector/issues](https://github.com/debargha/forte-detector/issues)

## Community

- **Discussions**: [GitHub Discussions](https://github.com/debargha/forte-detector/discussions)
- **Twitter**: [Coming soon]
- **Discord**: [Coming soon]

---

Thank you for using Forte! We hope it helps advance your research and applications.
