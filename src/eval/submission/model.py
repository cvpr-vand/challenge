"""Model for submission."""

import torch
from anomalib.data import ImageBatch
from torch import nn


class Model(nn.Module):
    """TODO: Implement your model here"""

    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:
        """Setup the model.

        Optional: Use this to pass few-shot images and dataset category to the model.

        Args:
            setup_data (dict[str, torch.Tensor]): The setup data.
        """
        pass

    def weights_url(self, category: str) -> str | None:
        """URL to the model weights.

        You can optionally use the category to download specific weights for each category.
        """
        # TODO: Implement this if you want to download the weights from a URL
        return None

    def forward(self, image: torch.Tensor) -> ImageBatch:
        """Forward pass of the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            ImageBatch: The output image batch.
        """
        # TODO: Implement the forward pass of the model.
        return ImageBatch(
            image=image,
            pred_score=torch.zeros(image.shape[0]),
            anomaly_map=torch.zeros(1, *image.shape[2:]),
        )
