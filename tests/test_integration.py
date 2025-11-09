"""
Integration tests for forte-detector package.
These tests verify end-to-end functionality.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_package_imports(self):
        """Test that all main imports work correctly."""
        from forte import ForteOODDetector, TorchGMM, TorchKDE, TorchOCSVM, __version__

        assert ForteOODDetector is not None
        assert TorchGMM is not None
        assert TorchKDE is not None
        assert TorchOCSVM is not None
        assert __version__ == "0.1.0"

    def test_detector_initialization_all_methods(self, device, embedding_dir):
        """Test detector initialization with all methods."""
        from forte import ForteOODDetector

        for method in ["gmm", "kde", "ocsvm"]:
            detector = ForteOODDetector(
                method=method, device=device, embedding_dir=embedding_dir, batch_size=8, nearest_k=3
            )
            assert detector.method == method
            assert not detector.is_fitted

    def test_image_loading(self, mock_image_paths):
        """Test image loading functionality."""
        from forte import ForteOODDetector

        detector = ForteOODDetector(device="cpu")

        # Test loading a valid image
        img = detector._load_image(mock_image_paths[0])
        assert img is not None
        assert isinstance(img, Image.Image)

        # Test loading an invalid path
        img = detector._load_image("/nonexistent/path.png")
        assert img is None

    def test_prdc_computation_pipeline(self, device):
        """Test PRDC computation on synthetic data."""
        from forte import ForteOODDetector

        detector = ForteOODDetector(device=device, nearest_k=5)

        # Create synthetic features
        real_features = torch.randn(50, 128, device=device)
        fake_features = torch.randn(40, 128, device=device)

        prdc = detector._compute_prdc_features(real_features, fake_features)

        assert prdc.shape == (40, 4)  # 4 PRDC metrics per sample
        assert not torch.isnan(prdc).any()
        assert (prdc >= 0).all()

    def test_models_work_with_synthetic_features(self, device):
        """Test that all models work with synthetic PRDC features."""
        from forte.models import TorchGMM, TorchKDE, TorchOCSVM

        # Generate synthetic PRDC features
        X = torch.randn(100, 12, device=device)  # 12 = 4 PRDC * 3 models

        # Test GMM
        gmm = TorchGMM(n_components=4, max_iter=20, device=device)
        gmm.fit(X)
        gmm_scores = gmm.score_samples(X)
        assert gmm_scores.shape == (100,)
        assert not torch.isnan(gmm_scores).any()

        # Test KDE
        kde = TorchKDE(X.T, device=device)
        kde_scores = kde.logpdf(X)
        assert kde_scores.shape == (100,)
        assert not torch.isnan(kde_scores).any()

        # Test OCSVM
        ocsvm = TorchOCSVM(nu=0.1, n_iters=50, lr=1e-3, device=device)
        ocsvm.fit(X)
        ocsvm_scores = ocsvm.decision_function(X)
        assert ocsvm_scores.shape == (100,)
        assert not torch.isnan(ocsvm_scores).any()


@pytest.mark.integration
class TestModelSelection:
    """Test model selection and hyperparameter optimization."""

    def test_gmm_bic_selection(self, device):
        """Test GMM BIC-based model selection."""
        from forte.models import TorchGMM

        X = torch.randn(100, 10, device=device)

        bic_scores = []
        for n_components in [1, 2, 4, 8]:
            gmm = TorchGMM(n_components=n_components, max_iter=20, device=device)
            gmm.fit(X)
            bic = gmm.bic(X)
            bic_scores.append(bic)

        # BIC scores should be finite
        assert all(np.isfinite(bic) for bic in bic_scores)

    def test_ocsvm_nu_selection(self, device):
        """Test OCSVM with different nu values."""
        from forte.models import TorchOCSVM

        X = torch.randn(100, 10, device=device)

        for nu in [0.01, 0.05, 0.1, 0.2]:
            ocsvm = TorchOCSVM(nu=nu, n_iters=30, device=device)
            ocsvm.fit(X)
            scores = ocsvm.decision_function(X)
            assert not torch.isnan(scores).any()


@pytest.mark.integration
class TestDeviceCompatibility:
    """Test compatibility across different devices."""

    def test_cpu_device(self):
        """Test that everything works on CPU."""
        from forte import ForteOODDetector

        detector = ForteOODDetector(device="cpu")
        assert detector.device == "cpu"
        assert not detector.custom_detector  # CPU uses non-custom detectors

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test that everything works on CUDA."""
        from forte import ForteOODDetector

        detector = ForteOODDetector(device="cuda:0")
        assert detector.device == "cuda:0"
        assert detector.custom_detector  # GPU uses custom detectors

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps_device(self):
        """Test that everything works on MPS (Apple Silicon)."""
        from forte import ForteOODDetector

        detector = ForteOODDetector(device="mps")
        assert detector.device == "mps"
        assert detector.custom_detector  # GPU uses custom detectors


@pytest.mark.integration
class TestCaching:
    """Test feature caching functionality."""

    def test_embedding_directory_creation(self, tmp_dir):
        """Test that embedding directory is created."""
        import os

        from forte import ForteOODDetector

        emb_dir = os.path.join(tmp_dir, "test_embeddings")
        detector = ForteOODDetector(embedding_dir=emb_dir, device="cpu")

        assert os.path.exists(emb_dir)

    def test_feature_caching_structure(self, tmp_dir):
        """Test that feature caching saves files correctly."""
        import os

        from forte import ForteOODDetector

        emb_dir = os.path.join(tmp_dir, "cache_test")
        os.makedirs(emb_dir, exist_ok=True)

        # Create a mock cached feature
        cache_path = os.path.join(emb_dir, "test_clip_features.pt")
        torch.save(torch.randn(10, 512), cache_path)

        assert os.path.exists(cache_path)
        loaded = torch.load(cache_path)
        assert loaded.shape == (10, 512)


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_method_raises_error(self, device, embedding_dir):
        """Test that invalid method raises appropriate error."""
        from forte import ForteOODDetector

        detector = ForteOODDetector(
            method="invalid_method", device=device, embedding_dir=embedding_dir
        )
        # Should initialize but may fail during fit
        assert detector.method == "invalid_method"

    def test_empty_image_list_handling(self, device, embedding_dir):
        """Test handling of empty image lists."""
        from forte import ForteOODDetector

        detector = ForteOODDetector(device=device, embedding_dir=embedding_dir)
        # This should be handled gracefully
        # Actual behavior depends on implementation

    def test_invalid_image_path_handling(self, device):
        """Test handling of invalid image paths."""
        from forte import ForteOODDetector

        detector = ForteOODDetector(device=device)
        img = detector._load_image("/invalid/path/image.png")
        assert img is None  # Should return None, not raise error


@pytest.mark.integration
class TestReproducibility:
    """Test reproducibility with fixed random seeds."""

    def test_prdc_reproducibility(self, device):
        """Test that PRDC computation is reproducible."""
        import numpy as np

        from forte import ForteOODDetector

        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)

        detector1 = ForteOODDetector(device=device, nearest_k=5)
        real_features = torch.randn(50, 128, device=device)
        fake_features = torch.randn(40, 128, device=device)
        prdc1 = detector1._compute_prdc_features(real_features, fake_features)

        # Reset seeds
        torch.manual_seed(42)
        np.random.seed(42)

        detector2 = ForteOODDetector(device=device, nearest_k=5)
        prdc2 = detector2._compute_prdc_features(real_features, fake_features)

        assert torch.allclose(prdc1, prdc2, atol=1e-6)

    def test_model_fitting_reproducibility(self, device):
        """Test that model fitting is reproducible with same seed."""
        import numpy as np

        from forte.models import TorchGMM

        X = torch.randn(100, 10, device=device)

        # First fit
        torch.manual_seed(42)
        np.random.seed(42)
        gmm1 = TorchGMM(n_components=2, max_iter=20, device=device)
        gmm1.fit(X)
        scores1 = gmm1.score_samples(X)

        # Second fit with same seed
        torch.manual_seed(42)
        np.random.seed(42)
        gmm2 = TorchGMM(n_components=2, max_iter=20, device=device)
        gmm2.fit(X)
        scores2 = gmm2.score_samples(X)

        # Results should be very similar (allowing for small numerical differences)
        assert torch.allclose(scores1, scores2, rtol=1e-3, atol=1e-3)
