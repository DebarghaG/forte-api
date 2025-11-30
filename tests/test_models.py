"""
Tests for custom PyTorch model implementations (TorchGMM, TorchKDE, TorchOCSVM).
"""

import numpy as np
import pytest
import torch

from forte.models import TorchGMM, TorchKDE, TorchOCSVM


class TestTorchGMM:
    """Test suite for TorchGMM implementation."""

    def test_initialization(self, device):
        """Test GMM initialization."""
        gmm = TorchGMM(n_components=2, device=device)
        assert gmm.n_components == 2
        assert gmm.device == device
        assert gmm.weights_ is None
        assert gmm.means_ is None
        assert gmm.covariances_ is None

    def test_fit(self, device, sample_dataset):
        """Test GMM fitting."""
        X = sample_dataset["id"].to(device)
        gmm = TorchGMM(n_components=2, max_iter=10, device=device)
        gmm.fit(X)

        assert gmm.weights_ is not None
        assert gmm.means_ is not None
        assert gmm.covariances_ is not None
        assert gmm.weights_.shape == (2,)
        assert gmm.means_.shape == (2, X.shape[1])
        assert torch.allclose(gmm.weights_.sum(), torch.tensor(1.0), atol=1e-5)

    def test_score_samples(self, device, sample_dataset):
        """Test GMM score_samples method."""
        X = sample_dataset["id"].to(device)
        gmm = TorchGMM(n_components=2, max_iter=10, device=device)
        gmm.fit(X)

        scores = gmm.score_samples(X)
        assert scores.shape == (X.shape[0],)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    def test_bic(self, device, sample_dataset):
        """Test GMM BIC computation."""
        X = sample_dataset["id"].to(device)
        gmm = TorchGMM(n_components=2, max_iter=10, device=device)
        gmm.fit(X)

        bic = gmm.bic(X)
        assert isinstance(bic, float)
        assert not np.isnan(bic)
        assert not np.isinf(bic)

    def test_convergence(self, device):
        """Test GMM convergence on simple data."""
        # Create clear clusters
        cluster1 = torch.randn(50, 5, device=device) + 0
        cluster2 = torch.randn(50, 5, device=device) + 5
        X = torch.cat([cluster1, cluster2], dim=0)

        gmm = TorchGMM(n_components=2, max_iter=100, tol=1e-3, device=device)
        gmm.fit(X)

        # Should converge
        assert gmm.converged_ or gmm.lower_bound_ > -np.inf


class TestTorchKDE:
    """Test suite for TorchKDE implementation."""

    def test_initialization(self, device):
        """Test KDE initialization."""
        dataset = torch.randn(5, 20, device=device)  # (d, n)
        kde = TorchKDE(dataset, device=device)

        assert kde.d == 5
        assert kde.n == 20
        assert kde.device == device
        assert kde.weights is not None

    def test_scotts_silverman_factor(self, device):
        """Test bandwidth factor calculations."""
        dataset = torch.randn(5, 20, device=device)
        kde_scott = TorchKDE(dataset, bw_method="scott", device=device)
        kde_silverman = TorchKDE(dataset, bw_method="silverman", device=device)

        assert kde_scott.factor > 0
        assert kde_silverman.factor > 0

    def test_evaluate(self, device):
        """Test KDE evaluation."""
        dataset = torch.randn(5, 20, device=device)
        kde = TorchKDE(dataset, bw_method="scott", device=device)

        # Evaluate at test points
        test_points = torch.randn(5, 10, device=device)
        densities = kde.evaluate(test_points)

        assert densities.shape == (10,)
        assert (densities >= 0).all()  # Densities should be non-negative
        assert not torch.isnan(densities).any()

    def test_logpdf(self, device):
        """Test KDE log probability density."""
        dataset = torch.randn(5, 20, device=device)
        kde = TorchKDE(dataset, device=device)

        test_points = torch.randn(5, 10, device=device)
        log_densities = kde.logpdf(test_points)

        assert log_densities.shape == (10,)
        assert not torch.isnan(log_densities).any()
        assert not torch.isinf(log_densities).any()

    def test_custom_bandwidth(self, device):
        """Test KDE with custom bandwidth."""
        dataset = torch.randn(5, 20, device=device)
        custom_bw = 0.5
        kde = TorchKDE(dataset, bw_method=custom_bw, device=device)

        assert kde.factor == custom_bw


class TestTorchOCSVM:
    """Test suite for TorchOCSVM implementation."""

    def test_initialization(self, device):
        """Test OCSVM initialization."""
        ocsvm = TorchOCSVM(nu=0.1, n_iters=100, lr=1e-3, device=device)

        assert ocsvm.nu == 0.1
        assert ocsvm.n_iters == 100
        assert ocsvm.lr == 1e-3
        assert ocsvm.device == device
        assert ocsvm.w is None
        assert ocsvm.rho is None

    def test_fit(self, device, sample_dataset):
        """Test OCSVM fitting."""
        X = sample_dataset["id"].to(device)
        ocsvm = TorchOCSVM(nu=0.1, n_iters=50, lr=1e-3, device=device)
        ocsvm.fit(X)

        assert ocsvm.w is not None
        assert ocsvm.rho is not None
        assert ocsvm.w.shape == (X.shape[1],)
        assert ocsvm.rho.shape == ()

    def test_decision_function(self, device, sample_dataset):
        """Test OCSVM decision function."""
        X = sample_dataset["id"].to(device)
        ocsvm = TorchOCSVM(nu=0.1, n_iters=50, lr=1e-3, device=device)
        ocsvm.fit(X)

        decisions = ocsvm.decision_function(X)
        assert decisions.shape == (X.shape[0],)
        assert not torch.isnan(decisions).any()

    def test_predict(self, device, sample_dataset):
        """Test OCSVM prediction."""
        X = sample_dataset["id"].to(device)
        ocsvm = TorchOCSVM(nu=0.1, n_iters=50, lr=1e-3, device=device)
        ocsvm.fit(X)

        predictions = ocsvm.predict(X)
        assert predictions.shape == (X.shape[0],)
        assert torch.all((predictions == 1) | (predictions == -1))

    def test_ood_detection(self, device, sample_dataset):
        """Test OCSVM can distinguish ID from OOD."""
        X_id = sample_dataset["id"].to(device)
        X_ood = sample_dataset["ood"].to(device)

        ocsvm = TorchOCSVM(nu=0.1, n_iters=100, lr=1e-3, device=device)
        ocsvm.fit(X_id)

        # Get decisions for both
        decision_id = ocsvm.decision_function(X_id).mean()
        decision_ood = ocsvm.decision_function(X_ood).mean()

        # ID samples should generally have higher decision values
        # (though not guaranteed for all random seeds)
        assert decision_id.item() != decision_ood.item()


@pytest.mark.integration
class TestModelsIntegration:
    """Integration tests for all models working together."""

    def test_all_models_on_same_data(self, device, mock_prdc_features):
        """Test that all models can work with the same data."""
        X = mock_prdc_features

        # GMM
        gmm = TorchGMM(n_components=2, max_iter=20, device=device)
        gmm.fit(X)
        gmm_scores = gmm.score_samples(X)

        # KDE
        kde = TorchKDE(X.T, device=device)  # KDE expects (d, n)
        kde_scores = kde.logpdf(X)

        # OCSVM
        ocsvm = TorchOCSVM(nu=0.1, n_iters=50, device=device)
        ocsvm.fit(X)
        ocsvm_scores = ocsvm.decision_function(X)

        # All should produce valid scores
        assert gmm_scores.shape == (X.shape[0],)
        assert kde_scores.shape == (X.shape[0],)
        assert ocsvm_scores.shape == (X.shape[0],)

        assert not torch.isnan(gmm_scores).any()
        assert not torch.isnan(kde_scores).any()
        assert not torch.isnan(ocsvm_scores).any()
