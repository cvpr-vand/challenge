"""Model for submission."""

import torch
from anomalib.data import ImageBatch
from anomalib.models.image.winclip.torch_model import WinClipModel
from torch import nn
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize


class Model(nn.Module):
    """TODO: Implement your model here"""

    def __init__(self):
        super().__init__()
        self.winclip = WinClipModel()
        self.transform = Compose(
            [
                Resize((240, 240), antialias=True, interpolation=InterpolationMode.BICUBIC),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ]
        )

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

    @property
    def batch_size(self) -> int:
        """Batch size of the model."""
        # TODO: Reduce the batch size in case your model is too large to fit in memory.
        return 32

    def forward(self, image: torch.Tensor) -> ImageBatch:
        """Forward pass of the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            ImageBatch: The output image batch.
        """
        # TODO: Implement the forward pass of the model.
        device = image.device
        image = self.transform(image)
        predictions = self.winclip(image.to(device))
        return ImageBatch(
            image=image,
            pred_score=predictions.pred_score,
            anomaly_map=predictions.anomaly_map,
        )
