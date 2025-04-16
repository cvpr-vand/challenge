"""PCA model."""

from typing import Any, Mapping

import torch
from sklearn.decomposition import PCA as SKLearnPCA
from torch import nn


class PCA(nn.Module):
    """PCA model."""

    def __init__(self, n_components: int):
        super().__init__()
        self.pca = SKLearnPCA(n_components=n_components)
        self.components = self.register_buffer("components", torch.zeros(n_components, n_components))
        self.mean = self.register_buffer("mean", torch.zeros(n_components))
        self.explained_variance = self.register_buffer("explained_variance", torch.zeros(n_components))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pca.fit_transform(x)

    def fit(self, x: torch.Tensor) -> None:
        self.pca.fit(x)
        self.components = torch.from_numpy(self.pca.components_)
        self.mean = torch.from_numpy(self.pca.mean_)
        self.explained_variance = torch.from_numpy(self.pca.explained_variance_)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.pca.transform(x)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.components = state_dict["components"]
        self.mean = state_dict["mean"]
        self.explained_variance = state_dict["explained_variance"]
        self.pca.components_ = self.components.cpu().numpy()
        self.pca.mean_ = self.mean.cpu().numpy()
        self.pca.explained_variance_ = self.explained_variance.cpu().numpy()
