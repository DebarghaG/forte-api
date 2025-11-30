"""Forte: Finding Outliers with Representation Typicality Estimation.

A PyTorch-based library for out-of-distribution (OOD) detection using
topology-aware representation learning from multiple pretrained vision models.

Based on the ICLR 2025 paper:
Ganguly, D., Morningstar, W. R., Yu, A. S., & Chaudhary, V. (2025).
Forte: Finding Outliers with Representation Typicality Estimation.
In The Thirteenth International Conference on Learning Representations.
"""

__version__ = "0.1.0"
__author__ = "Debargha Ganguly"
__email__ = "debargha.ganguly@gmail.com"
__license__ = "MIT"

from .detector import ForteOODDetector
from .models import TorchGMM, TorchKDE, TorchOCSVM

__all__ = [
    "ForteOODDetector",
    "TorchGMM",
    "TorchKDE",
    "TorchOCSVM",
    "__version__",
]
