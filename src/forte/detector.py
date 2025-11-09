"""
Forte OOD Detector: Finding Outliers with Representation Typicality Estimation.

This module implements the main ForteOODDetector class based on the ICLR 2025 paper.
"""

import os
import time

import numpy as np
import torch
from PIL import Image
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    ViTMSNModel,
)

from .models import TorchGMM, TorchKDE, TorchOCSVM


class ForteOODDetector:
    """
    Forte OOD Detector: Finding Outliers Using Representation Typicality Estimation.

    This class implements the Forte method for OOD detection. It extracts features using
    pretrained models and computes PRDC features using PyTorch tensors on GPU.

    Detector training can use either a custom GPU-based implementation
    or fall back to CPU-based detectors from scikit-learn/SciPy.

    Example:
        >>> detector = ForteOODDetector(method='gmm', nearest_k=5)
        >>> detector.fit(id_image_paths)
        >>> predictions = detector.predict(test_image_paths)
        >>> metrics = detector.evaluate(id_test_paths, ood_test_paths)
    """

    def __init__(
        self, batch_size=32, device=None, embedding_dir="./embeddings", nearest_k=5, method="gmm"
    ):
        """
        Initialize the ForteOODDetector.

        Args:
            batch_size (int): Batch size for processing images.
            device (str): Device to use for computation (e.g., 'cuda:0' or 'cpu').
            embedding_dir (str): Directory to store embeddings.
            nearest_k (int): Number of nearest neighbors for PRDC computation.
            method (str): Detector method ('gmm', 'kde', or 'ocsvm').
        """
        self.batch_size = batch_size
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.embedding_dir = embedding_dir
        self.nearest_k = nearest_k
        self.method = method
        self.custom_detector = self.device != "cpu"
        self.models = None
        self.is_fitted = False

        # These will be set during fit
        self.id_train_features = None  # GPU tensors for feature extraction
        self.id_train_prdc = None  # Combined PRDC features (GPU tensor)
        self.detector = None

        os.makedirs(self.embedding_dir, exist_ok=True)

    def _load_image(self, path):
        """Load an image from path."""
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def _init_models(self):
        """Initialize the models used for feature extraction."""
        print(f"Initializing models on {self.device}...")
        device = self.device
        models = [
            (
                "clip",
                CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device),
                CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            ),
            (
                "vitmsn",
                ViTMSNModel.from_pretrained("facebook/vit-msn-base").to(device),
                AutoFeatureExtractor.from_pretrained("facebook/vit-msn-base"),
            ),
            (
                "dinov2",
                AutoModel.from_pretrained("facebook/dinov2-base").to(device),
                AutoImageProcessor.from_pretrained("facebook/dinov2-base"),
            ),
        ]
        return models

    def _extract_features_batch(self, image_paths, batch_idx=0):
        """
        Extract features for a batch of images using multiple models.

        Args:
            image_paths (list): List of image paths.
            batch_idx (int): Batch index for progress tracking.

        Returns:
            dict: Dictionary of features for each model (torch tensors on GPU).
        """
        # Load images using the helper method and filter out failures
        images = [self._load_image(path) for path in image_paths]
        images = [img for img in images if img is not None]

        if not images:
            return {
                model_name: torch.empty(0, device=self.device) for model_name, _, _ in self.models
            }

        all_features = {}
        # Process each model using its corresponding processor
        for model_name, model, processor in self.models:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(self.device)
            try:
                with torch.no_grad():
                    if model_name == "clip":
                        features = model.get_image_features(**inputs)
                    elif model_name in ["vitmsn", "dinov2"]:
                        features = model(**inputs).last_hidden_state[:, 0, :]
                    else:
                        raise ValueError(f"Unsupported model: {model_name}")
                all_features[model_name] = features
            except Exception as e:
                print(f"Error extracting features with {model_name}: {e}")
                all_features[model_name] = torch.empty(0, device=self.device)
        return all_features

    def _extract_features(self, image_paths, name="tmp"):
        """
        Extract features from all images using the models.

        Args:
            image_paths (list): List of image paths.
            name (str): Identifier for caching.

        Returns:
            dict: Dictionary of features for each model (torch tensors on GPU).
        """
        if self.models is None:
            self.models = self._init_models()

        all_features = {model_name: [] for model_name, _, _ in self.models}
        models_to_process = []

        for model_name, _, _ in self.models:
            embedding_file = os.path.join(self.embedding_dir, f"{name}_{model_name}_features.pt")
            if os.path.exists(embedding_file):
                print(f"Loading pre-computed features from {embedding_file}")
                loaded = torch.load(embedding_file, map_location=self.device)
                all_features[model_name] = loaded
                if loaded.size(0) != len(image_paths):
                    print(
                        f"Warning: Cached features count ({loaded.size(0)}) doesn't "
                        f"match image count ({len(image_paths)}). "
                        f"Recomputing for {model_name}."
                    )
                    all_features[model_name] = []
                    models_to_process.append(model_name)
                else:
                    print(f"Feature shape for {model_name}: {loaded.shape}")
            else:
                models_to_process.append(model_name)

        if not models_to_process:
            return all_features

        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Extracting features"):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_features = self._extract_features_batch(batch_paths, i // self.batch_size)
            for model_name, features in batch_features.items():
                if features.numel() > 0 and model_name in models_to_process:
                    all_features[model_name].append(features)

        for model_name in models_to_process:
            if all_features[model_name]:
                all_features[model_name] = torch.cat(all_features[model_name], dim=0)
                embedding_file = os.path.join(
                    self.embedding_dir, f"{name}_{model_name}_features.pt"
                )
                torch.save(all_features[model_name], embedding_file)
                print(
                    f"Saved {model_name} features with shape "
                    f"{all_features[model_name].shape} to {embedding_file}"
                )
            else:
                all_features[model_name] = torch.empty(0, device=self.device)

        return all_features

    def _compute_pairwise_distance(self, data_x, data_y=None):
        """
        Compute pairwise distances between two sets of points using torch operations.

        Args:
            data_x (torch.Tensor): Data points.
            data_y (torch.Tensor, optional): Data points.

        Returns:
            torch.Tensor: Pairwise distances.
        """
        if data_y is None:
            data_y = data_x
        return torch.cdist(data_x, data_y, p=2)

    def _get_kth_value(self, unsorted, k, axis=-1):
        """
        Get the kth smallest values along an axis using torch.topk.

        Args:
            unsorted (torch.Tensor): Input tensor.
            k (int): k value.
            axis (int): Axis.

        Returns:
            torch.Tensor: kth smallest values along the specified axis.
        """
        values, _ = torch.topk(unsorted, k, largest=False)
        return values.max(dim=axis).values

    def _compute_nearest_neighbour_distances(self, input_features, nearest_k):
        """
        Compute distances to kth nearest neighbours using torch operations.

        Args:
            input_features (torch.Tensor): Input features.
            nearest_k (int): Number of nearest neighbors.

        Returns:
            torch.Tensor: Distances to kth nearest neighbours.
        """
        distances = self._compute_pairwise_distance(input_features)
        radii = self._get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def _compute_prdc_features(self, real_features, fake_features):
        """
        Compute PRDC features using GPU-based tensor operations.

        Args:
            real_features (torch.Tensor): Reference features.
            fake_features (torch.Tensor): Query features.

        Returns:
            torch.Tensor: PRDC features (recall, density, precision, coverage).
        """
        num_real = real_features.size(0)
        real_distances = self._compute_nearest_neighbour_distances(real_features, self.nearest_k)
        fake_distances = self._compute_nearest_neighbour_distances(fake_features, self.nearest_k)
        distance_matrix = self._compute_pairwise_distance(real_features, fake_features)

        precision = (distance_matrix < real_distances.unsqueeze(1)).any(dim=0).float()
        recall = (distance_matrix < fake_distances).sum(dim=0).float() / num_real
        density = (1.0 / float(self.nearest_k)) * (
            distance_matrix < real_distances.unsqueeze(1)
        ).sum(dim=0).float()
        coverage = (distance_matrix.min(dim=0).values < fake_distances).float()

        return torch.stack((recall, density, precision, coverage), dim=1)

    def fit(self, id_image_paths, val_split=0.2, random_state=42):
        """
        Fit the OOD detector on in-distribution images.

        Args:
            id_image_paths (list): Paths to in-distribution images.
            val_split (float): Fraction for validation.
            random_state (int): Random seed.

        Returns:
            self: The fitted detector.
        """
        start_time = time.time()
        print(f"Fitting ForteOODDetector on {len(id_image_paths)} images...")

        # Split paths into training and validation
        id_train_paths, id_val_paths = train_test_split(
            id_image_paths, test_size=val_split, random_state=random_state
        )

        print(f"Extracting features from {len(id_train_paths)} training images...")
        self.id_train_features = self._extract_features(id_train_paths, name="id_train")

        print(f"Extracting features from {len(id_val_paths)} validation images...")
        id_val_features = self._extract_features(id_val_paths, name="id_val")

        # Compute PRDC features for each model using GPU tensor operations
        print("Computing PRDC features...")
        X_id_train_prdc = []
        X_id_val_prdc = []
        for model_name in self.id_train_features:
            print(f"Computing PRDC for {model_name}...")
            features = self.id_train_features[model_name]
            # Use torch-based splitting on GPU
            train_idx = torch.randperm(features.size(0), device=self.device)
            split = int(features.size(0) * 0.5)
            id_train_part1 = features[train_idx[:split]]
            id_train_part2 = features[train_idx[split:]]

            print(f"  Training PRDC: {id_train_part1.shape} vs {id_train_part2.shape}")
            train_prdc = self._compute_prdc_features(id_train_part1, id_train_part2)
            X_id_train_prdc.append(train_prdc)

            val_feats = id_val_features[model_name]
            print(f"  Validation PRDC: {id_train_part1.shape} vs {val_feats.shape}")
            val_prdc = self._compute_prdc_features(id_train_part1, val_feats)
            X_id_val_prdc.append(val_prdc)

        self.id_train_prdc = torch.cat(X_id_train_prdc, dim=1)  # still on GPU
        id_val_prdc = torch.cat(X_id_val_prdc, dim=1)
        print(
            f"Combined PRDC features - Training: {self.id_train_prdc.shape}, "
            f"Validation: {id_val_prdc.shape}"
        )

        print(f"Training detector ({self.method}) with custom_detector={self.custom_detector}...")
        if self.method == "gmm":
            best_bic = np.inf
            best_n_components = 1
            best_model = None
            for n_components in [1, 2, 4, 8, 16, 32, 64]:
                if self.custom_detector:
                    gmm = TorchGMM(
                        n_components=n_components, max_iter=100, tol=1e-3, device=self.device
                    )
                    gmm.fit(self.id_train_prdc)
                    bic_val = gmm.bic(self.id_train_prdc)
                else:
                    id_train_prdc_cpu = self.id_train_prdc.cpu().numpy()
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type="full",
                        random_state=random_state,
                        max_iter=100,
                    )
                    gmm.fit(id_train_prdc_cpu)
                    bic_val = gmm.bic(id_train_prdc_cpu)
                if bic_val < best_bic:
                    best_bic = bic_val
                    best_n_components = n_components
                    best_gmm = gmm
            print(f"Selected {best_n_components} components for GMM with BIC={best_bic:.2f}")
            self.detector = best_gmm

        elif self.method == "kde":
            self.detector = (
                TorchKDE(self.id_train_prdc.T, bw_method="scott", device=self.device)
                if self.custom_detector
                else gaussian_kde(self.id_train_prdc.cpu().numpy().T, bw_method="scott")
            )

        elif self.method == "ocsvm":
            if self.custom_detector:
                best_accuracy = 0
                best_nu = 0.01
                best_model = None
                for nu in [0.01, 0.05, 0.1, 0.2, 0.5]:
                    model = TorchOCSVM(nu=nu, n_iters=1000, lr=1e-3, device=self.device)
                    model.fit(self.id_train_prdc)
                    decision = model.decision_function(self.id_train_prdc)
                    accuracy = (
                        torch.where(decision.detach() >= 0, 1, -1).float().mean().item() + 1
                    ) / 2.0
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_nu = nu
                        best_model = model
                print(f"Selected nu={best_nu} for TorchOCSVM with accuracy {best_accuracy:.4f}")
                self.detector = best_model
            else:
                best_accuracy = 0
                best_nu = 0.01
                for nu in [0.01, 0.05, 0.1, 0.2, 0.5]:
                    try:
                        id_train_prdc_cpu = self.id_train_prdc.cpu().numpy()
                        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
                        ocsvm.fit(id_train_prdc_cpu)
                        val_pred = ocsvm.predict(id_train_prdc_cpu)
                        accuracy = np.mean(val_pred == 1)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_nu = nu
                    except Exception as e:
                        print(f"Error with nu={nu}: {e}")
                        continue
                print(f"Selected nu={best_nu} for OCSVM with accuracy {best_accuracy:.4f}")
                id_train_prdc_cpu = self.id_train_prdc.cpu().numpy()
                self.detector = OneClassSVM(kernel="rbf", gamma="scale", nu=best_nu)
                self.detector.fit(id_train_prdc_cpu)

        self.is_fitted = True
        fit_time = time.time() - start_time
        print(f"ForteOODDetector fitted in {fit_time:.2f} seconds.")
        return self

    def _get_ood_scores(self, image_paths, cache_name="test"):
        """
        Get OOD scores for a set of images.

        Args:
            image_paths (list): Paths to images.
            cache_name (str): Identifier for caching.

        Returns:
            np.ndarray: Array of scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        test_features = self._extract_features(image_paths, name=cache_name)
        X_test_prdc = []
        for model_name in test_features:
            ref_features = self.id_train_features[model_name]
            train_idx = torch.randperm(ref_features.size(0), device=self.device)
            split = int(ref_features.size(0) * 0.5)
            id_train_part1 = ref_features[train_idx[:split]]
            test_tensor = test_features[model_name]
            print(
                f"Computing test PRDC for {model_name}: "
                f"{id_train_part1.shape} vs {test_tensor.shape}"
            )
            test_prdc = self._compute_prdc_features(id_train_part1, test_tensor)
            X_test_prdc.append(test_prdc)

        X_test_prdc = torch.cat(X_test_prdc, dim=1)
        print(f"Combined test PRDC shape: {X_test_prdc.shape}")

        # For custom (GPU-based) detectors, use torch outputs; then convert to numpy if needed.
        if self.custom_detector:
            if self.method == "gmm":
                scores = self.detector.score_samples(X_test_prdc)
                scores = scores.cpu().numpy()
            elif self.method == "kde":
                scores = self.detector.logpdf(X_test_prdc)
                scores = scores.cpu().numpy()
            elif self.method == "ocsvm":
                scores = self.detector.decision_function(X_test_prdc)
                scores = scores.detach().cpu().numpy()
        else:
            X_test_prdc_cpu = X_test_prdc.cpu().numpy()
            if self.method == "gmm":
                scores = self.detector.score_samples(X_test_prdc_cpu)
            elif self.method == "kde":
                scores = self.detector.logpdf(X_test_prdc_cpu.T)
            elif self.method == "ocsvm":
                scores = self.detector.decision_function(X_test_prdc_cpu)
        return scores

    def predict(self, image_paths):
        """
        Predict OOD status.

        Args:
            image_paths (list): Paths to images.

        Returns:
            np.ndarray: Binary predictions (1 for in-distribution, -1 for OOD).
        """
        scores = self._get_ood_scores(image_paths)
        if self.method == "ocsvm":
            threshold = 0
        else:
            if self.custom_detector:
                ref_features = self.id_train_prdc
                # Use a simple split for threshold estimation
                train_idx = torch.randperm(ref_features.size(0), device=self.device)
                split = int(ref_features.size(0) * 0.5)
                id_train_part1 = ref_features[train_idx[:split]]
                if self.method == "gmm":
                    id_scores = self.detector.score_samples(id_train_part1).cpu().numpy()
                elif self.method == "kde":
                    id_scores = self.detector.score_samples(id_train_part1).cpu().numpy()
            else:
                id_train_part1_np, _ = train_test_split(
                    self.id_train_prdc.cpu().numpy(), test_size=0.5, random_state=42
                )
                if self.method == "gmm":
                    id_scores = self.detector.score_samples(id_train_part1_np)
                elif self.method == "kde":
                    id_scores = self.detector.logpdf(id_train_part1_np.T)
            threshold = np.percentile(id_scores, 5)
        return np.where(scores > threshold, 1, -1)

    def predict_proba(self, image_paths):
        """
        Return normalized probability scores for OOD detection.

        Args:
            image_paths (list): Paths to images.

        Returns:
            np.ndarray: Normalized scores.
        """
        scores = self._get_ood_scores(image_paths)
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores) * 0.5
        return normalized_scores

    def evaluate(self, id_image_paths, ood_image_paths):
        """
        Evaluate the detector.

        Args:
            id_image_paths (list): In-distribution image paths.
            ood_image_paths (list): OOD image paths.

        Returns:
            dict: Evaluation metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before evaluation")

        print(f"Evaluating on {len(id_image_paths)} ID and {len(ood_image_paths)} OOD images...")

        # Fuse ID and OOD samples for processing together
        all_image_paths = id_image_paths + ood_image_paths
        all_scores = self._get_ood_scores(all_image_paths, cache_name="eval_fused")

        # Split the scores back to ID and OOD
        id_scores = all_scores[: len(id_image_paths)]
        ood_scores = all_scores[len(id_image_paths) :]

        print("\nScore Statistics:")
        print(
            f"ID  - Mean: {np.mean(id_scores):.4f}, Std: {np.std(id_scores):.4f}, "
            f"Min: {np.min(id_scores):.4f}, Max: {np.max(id_scores):.4f}"
        )
        print(
            f"OOD - Mean: {np.mean(ood_scores):.4f}, Std: {np.std(ood_scores):.4f}, "
            f"Min: {np.min(ood_scores):.4f}, Max: {np.max(ood_scores):.4f}"
        )

        labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        scores_all = np.concatenate([id_scores, ood_scores])
        auroc = roc_auc_score(labels, scores_all)
        fpr, tpr, _ = roc_curve(labels, scores_all)
        idx = np.argmin(np.abs(tpr - 0.95))
        fpr95 = fpr[idx] if idx < len(fpr) else 1.0
        precision_vals, recall_vals, _ = precision_recall_curve(labels, scores_all)
        auprc = average_precision_score(labels, scores_all)
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
        f1_score = np.max(f1_scores)
        return {"AUROC": auroc, "FPR@95TPR": fpr95, "AUPRC": auprc, "F1": f1_score}
