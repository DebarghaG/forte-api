"""
Custom PyTorch implementations of OOD detection models.

This module provides GPU-accelerated implementations of:
- Gaussian Mixture Models (GMM)
- Kernel Density Estimation (KDE)
- One-Class Support Vector Machines (OCSVM)
"""

import math
import numpy as np
import torch


class TorchGMM:
    """PyTorch implementation of Gaussian Mixture Model with GPU acceleration."""

    def __init__(self, n_components=1, covariance_type='full', max_iter=100, tol=1e-3, reg_covar=1e-6, device='cuda'):
        """
        A PyTorch implementation of a Gaussian Mixture Model that closely follows
        scikit-learn's GaussianMixture (for the 'full' covariance case).

        Parameters:
            n_components (int): Number of mixture components.
            covariance_type (str): Only 'full' is implemented in this example.
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence threshold.
            reg_covar (float): Non-negative regularization added to the diagonal of covariance matrices.
            device (str): 'cuda' or 'cpu'.
        """
        if covariance_type != 'full':
            raise NotImplementedError("Only 'full' covariance is implemented.")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.device = device

        # Parameters to be learned
        self.weights_ = None   # shape: (n_components,)
        self.means_ = None     # shape: (n_components, n_features)
        # shape: (n_components, n_features, n_features)
        self.covariances_ = None
        self.converged_ = False
        self.lower_bound_ = -np.inf

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        K = self.n_components
        # Initialize weights uniformly
        self.weights_ = torch.full((K,), 1.0 / K, device=self.device)
        # Initialize means by randomly selecting K samples
        indices = torch.randperm(n_samples, device=self.device)[:K]
        self.means_ = X[indices].clone()
        # Initialize covariances as diagonal matrices based on sample variance
        variance = torch.var(X, dim=0) + self.reg_covar
        self.covariances_ = torch.stack(
            [torch.diag(variance) for _ in range(K)], dim=0)

    def _estimate_log_gaussian_prob(self, X):
        # X: (n_samples, n_features)
        n_samples, n_features = X.shape
        # Create a batched MultivariateNormal distribution for each component
        mvn = torch.distributions.MultivariateNormal(
            self.means_,
            covariance_matrix=self.covariances_ + self.reg_covar *
            torch.eye(n_features, device=self.device)
        )
        # X has shape (n_samples, n_features); unsqueeze to (n_samples, 1, n_features) to broadcast over components
        # Expected shape: (n_samples, n_components)
        log_prob = mvn.log_prob(X.unsqueeze(1))
        return log_prob

    def _e_step(self, X):
        # Compute log probabilities for each sample and each component
        log_prob = self._estimate_log_gaussian_prob(
            X)  # shape: (n_samples, n_components)
        # Add log weights
        weighted_log_prob = log_prob + torch.log(self.weights_ + 1e-10)
        # Compute log-sum-exp for each sample
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        # Compute responsibilities: r_ik = exp(weighted_log_prob - log_prob_norm)
        log_resp = weighted_log_prob - log_prob_norm
        resp = torch.exp(log_resp)
        return resp, log_prob_norm.sum().item()

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        Nk = resp.sum(dim=0)  # shape: (n_components,)
        self.weights_ = Nk / n_samples
        # Update means
        self.means_ = (resp.t() @ X) / (Nk.unsqueeze(1) + 1e-10)
        # Update covariances
        K = self.n_components
        covariances = []
        for k in range(K):
            diff = X - self.means_[k]
            weighted_diff = diff * resp[:, k].unsqueeze(1)
            cov_k = (weighted_diff.t() @ diff) / (Nk[k] + 1e-10)
            # Add regularization for numerical stability
            cov_k = cov_k + self.reg_covar * \
                torch.eye(n_features, device=self.device)
            covariances.append(cov_k)
        self.covariances_ = torch.stack(covariances, dim=0)

    def fit(self, X):
        """
        Fit the GMM model on data X.

        Parameters:
            X (torch.Tensor): Input data of shape (n_samples, n_features) on self.device.

        Returns:
            self
        """
        X = X.to(self.device)
        self._initialize_parameters(X)
        lower_bound = -np.inf

        for i in range(self.max_iter):
            resp, curr_lower_bound = self._e_step(X)
            self._m_step(X, resp)
            change = abs(curr_lower_bound - lower_bound)
            lower_bound = curr_lower_bound
            if change < self.tol:
                self.converged_ = True
                break
        self.lower_bound_ = lower_bound
        return self

    def score_samples(self, X):
        """
        Compute the log-likelihood of each sample under the model.

        Parameters:
            X (torch.Tensor): Data of shape (n_samples, n_features) on self.device.

        Returns:
            torch.Tensor: Log probability for each sample.
        """
        X = X.to(self.device)
        log_prob = self._estimate_log_gaussian_prob(X)
        weighted_log_prob = log_prob + torch.log(self.weights_ + 1e-10)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        return log_prob_norm

    def bic(self, X):
        """
        Bayesian Information Criterion for the current model.

        Parameters:
            X (torch.Tensor): Data of shape (n_samples, n_features) on self.device.

        Returns:
            float: BIC value.
        """
        n_samples, n_features = X.shape
        p = (self.n_components - 1) + self.n_components * n_features + \
            self.n_components * n_features * (n_features + 1) / 2
        log_likelihood = self.score_samples(X).sum().item()
        return -2 * log_likelihood + p * np.log(n_samples)


class TorchKDE:
    """PyTorch implementation of Kernel Density Estimation with GPU acceleration."""

    def __init__(self, dataset, bw_method=None, weights=None, device='cuda'):
        """
        Initialize Kernel Density Estimator.

        Parameters:
            dataset (torch.Tensor): Data points of shape (d, n) where d is dimensionality.
            bw_method (str or float): Bandwidth method ('scott', 'silverman', or float value).
            weights (torch.Tensor, optional): Sample weights.
            device (str): Device for computation ('cuda', 'mps', or 'cpu').
        """
        # Use float32 for MPS devices, otherwise float64.
        dtype = torch.float32 if "mps" in device.lower() else torch.float64
        self.device = device
        self.dataset = dataset  # shape: (d, n)
        self.d, self.n = self.dataset.shape

        # Process weights (assumed to be a torch.Tensor on device if provided).
        if weights is not None:
            self.weights = (weights / weights.sum()).to(dtype=torch.float32)
            self.neff = (self.weights.sum() ** 2) / (self.weights ** 2).sum()
            # Weighted covariance: cov = sum_i w_i (x_i - mean)(x_i - mean)^T / (1 - sum(w_i^2))
            weighted_mean = (
                self.dataset * self.weights.unsqueeze(0)).sum(dim=1, keepdim=True)
            diff = self.dataset - weighted_mean
            cov = (diff * self.weights.unsqueeze(0)) @ diff.T / \
                (1 - (self.weights**2).sum())
        else:
            self.weights = torch.full(
                (self.n,), 1.0 / self.n, dtype=torch.float32, device=self.device)
            self.neff = self.n
            weighted_mean = self.dataset.mean(dim=1, keepdim=True)
            diff = self.dataset - weighted_mean
            cov = diff @ diff.T / (self.n - 1)
        self._data_covariance = cov  # computed entirely on GPU

        # Set bandwidth and compute scaled covariance.
        self.set_bandwidth(bw_method)

    def scotts_factor(self):
        """Scott's rule for bandwidth selection."""
        return self.neff ** (-1.0 / (self.d + 4))

    def silverman_factor(self):
        """Silverman's rule for bandwidth selection."""
        return (self.neff * (self.d + 2) / 4.0) ** (-1.0 / (self.d + 4))

    def set_bandwidth(self, bw_method=None):
        """Set the bandwidth for the kernel."""
        if bw_method is None or bw_method == 'scott':
            self.factor = self.scotts_factor()
        elif bw_method == 'silverman':
            self.factor = self.silverman_factor()
        elif isinstance(bw_method, (int, float)):
            self.factor = float(bw_method)
        elif callable(bw_method):
            self.factor = float(bw_method(self))
        else:
            raise ValueError("Invalid bw_method.")
        self._compute_covariance()

    def _compute_covariance(self):
        # Scale the data covariance by the bandwidth factor squared.
        self.covariance = self._data_covariance * (self.factor ** 2)
        # Increase regularization to ensure positive definiteness.
        reg = 1e-6
        self.cho_cov = torch.linalg.cholesky(
            self.covariance + reg *
            torch.eye(self.d, device=self.device, dtype=self.dataset.dtype)
        )
        self.log_det = 2. * torch.log(torch.diag(self.cho_cov)).sum()

    def evaluate(self, points):
        """
        Evaluate the KDE at given points.

        Parameters:
            points (torch.Tensor): Points to evaluate, shape (d, m) or (m, d).

        Returns:
            torch.Tensor: Density estimates.
        """
        # Assume points is already a torch.Tensor on the proper device.
        if points.dim() == 1:
            points = points.unsqueeze(0)
        # If points are provided in (n, d) format (n > d), transpose them to (d, m)
        if points.shape[0] > points.shape[1]:
            points = points.T
        if points.shape[0] != self.d:
            raise ValueError(
                f"Expected input with one dimension = {self.d}, but got shape {points.shape}")
        # Compute differences: shape (d, n, m)
        diff = self.dataset.unsqueeze(2) - points.unsqueeze(1)
        # Flatten differences for cholesky_solve: (d, n*m)
        diff_flat = diff.reshape(self.d, -1)
        sol_flat = torch.cholesky_solve(diff_flat, self.cho_cov)
        sol = sol_flat.view(diff.shape)
        energy = 0.5 * (diff * sol).sum(dim=0)  # shape: (n, m)
        result = torch.exp(-energy).T @ self.weights  # shape: (m,)
        norm_const = torch.exp(-self.log_det) / ((2 * math.pi) ** (self.d / 2))
        return result * norm_const

    def logpdf(self, points):
        """Compute log probability density at given points."""
        return torch.log(self.evaluate(points) + 1e-10)

    __call__ = evaluate


class TorchOCSVM:
    """PyTorch implementation of One-Class SVM with GPU acceleration."""

    def __init__(self, nu=0.1, n_iters=1000, lr=1e-3, device='cuda'):
        """
        Initialize One-Class SVM.

        Parameters:
            nu (float): Upper bound on fraction of outliers (between 0 and 1).
            n_iters (int): Number of optimization iterations.
            lr (float): Learning rate for Adam optimizer.
            device (str): Device for computation.
        """
        self.nu = nu
        self.n_iters = n_iters
        self.lr = lr
        self.device = device
        self.w = None
        self.rho = None

    def fit(self, X):
        """
        Fit the One-Class SVM model.

        Parameters:
            X (torch.Tensor): Training data of shape (n_samples, n_features).

        Returns:
            self
        """
        # Ensure X is on the correct device.
        X = X.to(self.device)
        n, d = X.shape
        # Initialize w and rho as nn.Parameter to ensure they are leaf tensors.
        self.w = torch.nn.Parameter(torch.randn(d, device=self.device) * 0.01)
        self.rho = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        # TODO: Adam is a good default choice, we can try SGD or adding a learning rate scheduler to adapt the learning rate during training.
        optimizer = torch.optim.Adam([self.w, self.rho], lr=self.lr)
        for i in range(self.n_iters):
            optimizer.zero_grad()
            scores = X @ self.w  # shape: (n,)
            # Compute slack = max(0, rho - w^T x) for each sample.
            # apply a smooth approximation?
            slack = torch.clamp(self.rho - scores, min=0)
            loss = 0.5 * torch.norm(self.w) ** 2 - \
                self.rho + (1 / (self.nu * n)) * slack.sum()
            loss.backward()
            optimizer.step()
            if (i + 1) % 200 == 0:
                print(
                    f"OCSVM iter {i+1}/{self.n_iters}, loss: {loss.item():.4f}")
        return self

    def decision_function(self, X):
        """
        Compute the decision function for samples.

        Parameters:
            X (torch.Tensor): Samples of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Decision values.
        """
        X = X.to(self.device)
        return (X @ self.w - self.rho)

    def predict(self, X):
        """
        Predict class labels.

        Parameters:
            X (torch.Tensor): Samples of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Predictions (1 for inlier, -1 for outlier).
        """
        decision = self.decision_function(X)
        return torch.where(decision >= 0, 1, -1)
