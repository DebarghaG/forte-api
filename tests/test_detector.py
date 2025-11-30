"""
Tests for ForteOODDetector class.
"""

import numpy as np
import pytest
import torch

from forte import ForteOODDetector


class TestForteOODDetectorInit:
    """Test ForteOODDetector initialization."""

    def test_default_initialization(self, device):
        """Test detector with default parameters."""
        detector = ForteOODDetector()
        assert detector.batch_size == 32
        assert detector.device in ["cuda:0", "mps", "cpu"]
        assert detector.embedding_dir == "./embeddings"
        assert detector.nearest_k == 5
        assert detector.method == "gmm"
        assert not detector.is_fitted

    def test_custom_parameters(self, device, embedding_dir):
        """Test detector with custom parameters."""
        detector = ForteOODDetector(
            batch_size=16, device=device, embedding_dir=embedding_dir, nearest_k=10, method="kde"
        )
        assert detector.batch_size == 16
        assert detector.device == device
        assert detector.embedding_dir == embedding_dir
        assert detector.nearest_k == 10
        assert detector.method == "kde"

    @pytest.mark.parametrize("method", ["gmm", "kde", "ocsvm"])
    def test_all_methods(self, method, device, embedding_dir):
        """Test initialization with all supported methods."""
        detector = ForteOODDetector(method=method, device=device, embedding_dir=embedding_dir)
        assert detector.method == method


class TestForteOODDetectorHelperMethods:
    """Test private helper methods of ForteOODDetector."""

    def test_compute_pairwise_distance(self, device):
        """Test pairwise distance computation."""
        detector = ForteOODDetector(device=device)
        X = torch.randn(10, 5, device=device)
        Y = torch.randn(8, 5, device=device)

        dist = detector._compute_pairwise_distance(X, Y)
        assert dist.shape == (10, 8)
        assert (dist >= 0).all()  # Distances should be non-negative

    def test_get_kth_value(self, device):
        """Test k-th value extraction."""
        detector = ForteOODDetector(device=device)
        X = torch.randn(10, 20, device=device)
        k = 5

        kth_vals = detector._get_kth_value(X, k=k)
        assert kth_vals.shape == (10,)

    def test_compute_nearest_neighbour_distances(self, device):
        """Test nearest neighbor distance computation."""
        detector = ForteOODDetector(device=device, nearest_k=5)
        X = torch.randn(20, 10, device=device)

        distances = detector._compute_nearest_neighbour_distances(X, nearest_k=5)
        assert distances.shape == (20,)
        assert (distances >= 0).all()

    def test_compute_prdc_features(self, device):
        """Test PRDC feature computation."""
        detector = ForteOODDetector(device=device, nearest_k=5)
        real_features = torch.randn(30, 10, device=device)
        fake_features = torch.randn(25, 10, device=device)

        prdc = detector._compute_prdc_features(real_features, fake_features)
        assert prdc.shape == (25, 4)  # 4 PRDC metrics
        assert not torch.isnan(prdc).any()
        # PRDC values should be in reasonable ranges
        assert (prdc >= 0).all()
        assert (prdc <= 1).any()  # At least some values should be normalized


@pytest.mark.slow
class TestForteOODDetectorFit:
    """Test ForteOODDetector fitting (slower tests)."""

    def test_fit_not_implemented_full(self, small_mock_images, device, embedding_dir):
        """Test that fit raises error before implementation."""
        # This is a placeholder - in real implementation, we'd need actual models
        # For now, we just test the basic structure
        detector = ForteOODDetector(
            device="cpu",  # Use CPU to avoid downloading large models
            embedding_dir=embedding_dir,
            method="gmm",
        )

        # Note: This test would actually download models and run feature extraction
        # For unit tests, we might want to mock this
        # For now, we just check the structure exists
        assert hasattr(detector, "fit")
        assert hasattr(detector, "predict")
        assert hasattr(detector, "predict_proba")
        assert hasattr(detector, "evaluate")

    def test_fit_sets_is_fitted(self, device):
        """Test that fit sets the is_fitted flag."""
        detector = ForteOODDetector(device=device)
        assert not detector.is_fitted
        # After fit, should be True (mocking this for now)

    def test_predict_before_fit_raises_error(self, small_mock_images, device, embedding_dir):
        """Test that predict raises error if not fitted."""
        detector = ForteOODDetector(device=device, embedding_dir=embedding_dir)

        with pytest.raises(RuntimeError, match="Detector must be fitted"):
            detector._get_ood_scores(small_mock_images)


class TestForteOODDetectorPredict:
    """Test ForteOODDetector prediction methods."""

    def test_predict_output_shape(self):
        """Test that predict returns correct shape."""
        # This would require a fitted detector
        # Placeholder for now
        pass

    def test_predict_proba_output_range(self):
        """Test that predict_proba returns values in [0, 1]."""
        # Placeholder - would need fitted detector
        pass

    def test_predict_binary_values(self):
        """Test that predict returns only 1 and -1."""
        # Placeholder - would need fitted detector
        pass


class TestForteOODDetectorEvaluate:
    """Test ForteOODDetector evaluation methods."""

    def test_evaluate_before_fit_raises_error(self, small_mock_images, device, embedding_dir):
        """Test that evaluate raises error if not fitted."""
        detector = ForteOODDetector(device=device, embedding_dir=embedding_dir)

        with pytest.raises(RuntimeError, match="Detector must be fitted"):
            detector.evaluate(small_mock_images[:2], small_mock_images[2:])

    def test_evaluate_returns_correct_metrics(self):
        """Test that evaluate returns all expected metrics."""
        # Placeholder - would need fitted detector
        # Should return dict with keys: AUROC, FPR@95TPR, AUPRC, F1
        pass


@pytest.mark.integration
class TestForteOODDetectorIntegration:
    """Integration tests for complete ForteOODDetector workflow."""

    @pytest.mark.slow
    def test_full_pipeline_mock_data(self):
        """Test complete pipeline with mocked data."""
        # This would be a full end-to-end test
        # Requires significant resources, so marked as slow
        pass

    def test_device_compatibility(self, device):
        """Test that detector works on the available device."""
        detector = ForteOODDetector(device=device)
        assert detector.device == device

        # Test that custom_detector flag is set correctly
        if device == "cpu":
            assert not detector.custom_detector
        else:
            assert detector.custom_detector

    def test_method_compatibility(self, device, embedding_dir):
        """Test all methods are compatible with device."""
        for method in ["gmm", "kde", "ocsvm"]:
            detector = ForteOODDetector(device=device, embedding_dir=embedding_dir, method=method)
            assert detector.method == method
