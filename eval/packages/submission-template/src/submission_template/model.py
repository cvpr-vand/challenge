"""Model for submission."""

import torch
from anomalib.data import ImageBatch
from anomalib.models.image.winclip.torch_model import WinClipModel
from torch import nn


class Model(nn.Module):
    """TODO: Implement your model here"""
    def __init__(self):
        super().__init__()
        self.winclip = WinClipModel()
        self.transform = self.winclip.transform

    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:
        """Setup the model.

        Optional: Use this to pass few-shot images and dataset category to the model.

        Args:
            setup_data (dict[str, torch.Tensor]): The setup data.
        """
        # get class name
        class_name = setup_data.get("dataset_category")
        # get few shot images
        images = setup_data.get("few_shot_samples")
        if images is not None:
            device = images.device
            images = torch.stack([self.winclip.transform(image) for image in images])
            images = images.to(device)
        # setup model
        self.winclip.setup(class_name, images)

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
        device = image.device
        image = self.winclip.transform(image.squeeze())
        predictions = self.winclip(image.unsqueeze(0).to(device))
        return ImageBatch(
            image=image,
            pred_score=predictions.pred_score,
            anomaly_map=predictions.anomaly_map,
        )