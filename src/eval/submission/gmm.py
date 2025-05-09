"""GMM model."""

from typing import Any, Mapping

import torch
from sklearn.mixture import GaussianMixture as SKLearnGMM
from torch import nn


class GMM(nn.Module):
    """GMM model."""

    def __init__(self, n_components: int):
        super().__init__()
        self.gmm = SKLearnGMM(n_components=n_components, covariance_type="diag")
        self.weights = self.register_buffer("weights", torch.zeros(n_components))
        self.means = self.register_buffer("means", torch.zeros(n_components, n_components))
        self.covariances = self.register_buffer("covariances", torch.zeros(n_components, n_components))
        self.precision = self.register_buffer("precision", torch.zeros(n_components, n_components))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gmm.fit_transform(x)

    def fit(self, x: torch.Tensor) -> None:
        self.gmm.fit(x)
        self.weights = torch.from_numpy(self.gmm.weights_)
        self.means = torch.from_numpy(self.gmm.means_)
        self.covariances = torch.from_numpy(self.gmm.covariances_)
        self.precision = torch.from_numpy(self.gmm.precisions_cholesky_)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.weights = state_dict["weights"]
        self.means = state_dict["means"]
        self.covariances = state_dict["covariances"]
        self.precision = state_dict["precision"]
        self.gmm.weights_ = self.weights.cpu().numpy()
        self.gmm.means_ = self.means.cpu().numpy()
        self.gmm.covariances_ = self.covariances.cpu().numpy()
        self.gmm.precisions_cholesky_ = self.precision.cpu().numpy()
