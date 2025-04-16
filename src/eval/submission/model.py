"""Model for submission."""

from typing import Any, Mapping

import torch
from anomalib.data import ImageBatch
from torch import nn

from .gmm import GMM
from .pca import PCA


class Model(nn.Module):
    """Grid-based GMM model for anomaly detection.

    Does not produce a meaningful result but demonstrates how to use URL for weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.grid_size = 8  # Split image into 8x8 grid
        self.pca = PCA(n_components=32)  # Reduce features to 32 dimensions
        # Register GMMs as ModuleList so they are included in state_dict
        self.gmms = nn.ModuleList([GMM(n_components=2) for _ in range(self.grid_size * self.grid_size)])
        self.is_setup = False

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        pca_state_dict = {
            "components": state_dict["pca.components"],
            "mean": state_dict["pca.mean"],
            "explained_variance": state_dict["pca.explained_variance"],
        }
        self.pca.load_state_dict(pca_state_dict, strict, assign)
        for i, gmm in enumerate(self.gmms):
            gmm_state_dict = {
                "weights": state_dict[f"gmms.{i}.weights"],
                "means": state_dict[f"gmms.{i}.means"],
                "covariances": state_dict[f"gmms.{i}.covariances"],
                "precision": state_dict[f"gmms.{i}.precision"],
            }
            gmm.load_state_dict(gmm_state_dict, strict, assign)

    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:
        """Setup the model by fitting GMMs to the few-shot images.

        Args:
            setup_data (dict[str, torch.Tensor]): The setup data containing few-shot samples.
        """
        return None

    def weights_url(self, category: str) -> str | None:
        """URL to the model weights.

        You can optionally use the category to download specific weights for each category.
        """
        urls = {
            "breakfast_box": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/breakfast_box.pth",
            "juice_bottle": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/juice_bottle.pth",
            "pushpins": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/pushpins.pth",
            "screw_bag": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/screw_bag.pth",
            "splicing_connectors": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/splicing_connectors.pth",
        }
        return urls[category]

    def forward(self, image: torch.Tensor) -> ImageBatch:
        """Forward pass of the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            ImageBatch: The output image batch with anomaly scores.
        """
        # if not self.is_setup:
        #     raise RuntimeError("Model must be setup before inference")

        batch_size, channels, height, width = image.shape
        grid_h = height // self.grid_size
        grid_w = width // self.grid_size

        # Initialize anomaly scores
        anomaly_scores = torch.zeros(batch_size, device=image.device)

        for b in range(batch_size):
            max_score = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Extract grid cell
                    grid_cell = image[b, :, i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w]
                    # Flatten features
                    features = grid_cell.reshape(-1).cpu().numpy()
                    # Transform with PCA
                    reduced_features = self.pca.transform(features.reshape(1, -1))
                    # Get negative log likelihood from GMM
                    score = -self.gmms[i * self.grid_size + j].gmm.score_samples(reduced_features)[0]
                    max_score = max(max_score, score)

            # Normalize score
            anomaly_scores[b] = torch.tensor(max_score, device=image.device)

        # Normalize scores across batch
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-6)

        return ImageBatch(
            image=image,
            pred_score=anomaly_scores,
        )
