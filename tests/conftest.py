"""
Pytest configuration and shared fixtures for forte-detector tests.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(scope="session")
def device():
    """Determine the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture(scope="session")
def tmp_dir():
    """Create a temporary directory for test artifacts."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup after all tests
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_image_paths(tmp_dir):
    """Create mock image files for testing."""
    image_dir = os.path.join(tmp_dir, "mock_images")
    os.makedirs(image_dir, exist_ok=True)

    paths = []
    for i in range(10):
        # Create a small random RGB image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        path = os.path.join(image_dir, f"image_{i}.png")
        img.save(path)
        paths.append(path)

    return paths


@pytest.fixture
def small_mock_images(tmp_dir):
    """Create a small set of mock images for quick tests."""
    image_dir = os.path.join(tmp_dir, "small_mock_images")
    os.makedirs(image_dir, exist_ok=True)

    paths = []
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        path = os.path.join(image_dir, f"small_image_{i}.png")
        img.save(path)
        paths.append(path)

    return paths


@pytest.fixture
def mock_features(device):
    """Create mock feature tensors for testing."""
    # Simulate features from 3 models
    n_samples = 20
    feature_dims = [512, 768, 768]  # CLIP, ViTMSN, DINOv2

    features = {}
    for i, dim in enumerate(feature_dims):
        model_name = ["clip", "vitmsn", "dinov2"][i]
        features[model_name] = torch.randn(n_samples, dim, device=device)

    return features


@pytest.fixture
def mock_prdc_features(device):
    """Create mock PRDC features for testing detectors."""
    # PRDC features have 4 dimensions per model (precision, recall, density, coverage)
    # With 3 models, total dimension is 12
    n_samples = 50
    n_features = 12  # 4 PRDC metrics * 3 models

    return torch.randn(n_samples, n_features, device=device)


@pytest.fixture
def embedding_dir(tmp_dir):
    """Create a temporary embedding directory."""
    emb_dir = os.path.join(tmp_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    return emb_dir


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility in all tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def sample_dataset():
    """Create a small synthetic dataset for testing."""
    # In-distribution: samples from N(0, 1)
    id_samples = torch.randn(100, 10)
    # Out-of-distribution: samples from N(5, 2)
    ood_samples = torch.randn(100, 10) * 2 + 5

    return {"id": id_samples, "ood": ood_samples}
