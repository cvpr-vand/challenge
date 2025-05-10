"""Evaluation script"""

import argparse
import json
from pathlib import Path
from tempfile import gettempdir
from typing import cast
from urllib.request import urlretrieve

import pandas as pd
import torch
from anomalib.data import ImageBatch, MVTecLOCODataset
from anomalib.data.utils import Split
from anomalib.data.utils.download import DownloadProgressBar
from anomalib.metrics.f1_score import _F1Max
from sklearn.metrics import auc
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize
from tqdm import tqdm

from eval.submission.model import Model

CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[42, 0, 1234],
        help="List of seed values for reproducibility. Default is [42, 0, 1234].",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        help="Single value for few-shot learning. Overwrites --k_shots if provided.",
    )
    parser.add_argument(
        "--k_shots",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="List of integers for few-shot learning samples.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default="./datasets/mvtec_loco",
        help="Path to the MVTEC LOCO dataset.",
    )

    args = parser.parse_args()

    if args.k_shot is not None:
        args.k_shots = [args.k_shot]

    return args


def get_dataloaders(dataset_path: Path | str, category: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Get the MVTec LOCO dataloader.

    Args:
        dataset_path (Path | str): Path to the dataset.
        category (str): Category of the MVTec dataset.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        tuple[DataLoader, DataLoader]: Tuple of train and test dataloaders.
    """
    # Create the dataset
    # NOTE: We fix the image size to (256, 256) for consistent evaluation across all models.
    train_dataset = MVTecLOCODataset(
        root=dataset_path,
        category=category,
        split=Split.TRAIN,
        augmentations=Resize((256, 256)),
    )
    test_dataset = MVTecLOCODataset(
        root=dataset_path,
        category=category,
        split=Split.TEST,
        augmentations=Resize((256, 256)),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=train_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.collate_fn,
    )

    return train_dataloader, test_dataloader


def download(url: str) -> Path:
    """Download a file from a URL.

    Args:
        url (str): URL of the file to download.

    Returns:
        Path: Path to the downloaded file.
    """
    root = Path(gettempdir())
    downloaded_file_path = root / url.split("/")[-1]
    if not downloaded_file_path.exists():  # Check if file already exists
        if url.startswith("http://") or url.startswith("https://"):
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as progress_bar:
                urlretrieve(  # noqa: S310  # nosec B310
                    url=f"{url}",
                    filename=downloaded_file_path,
                    reporthook=progress_bar.update_to,
                )
        else:
            message = f"URL {url} is not valid. Please check the URL."
            raise ValueError(message)
    return downloaded_file_path


def get_model(
    category: str,
) -> nn.Module:
    """Instantiate and potentially load weights for the model.

    Args:
        category (str): Category of the dataset. Used for potentially loading category-specific weights.

    Returns:
        nn.Module: Loaded model moved to the specified device.
    """
    model = Model()
    model.eval()

    weights_url = model.weights_url(category)
    if weights_url is not None:
        weights_path = download(weights_url)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    return model.to(DEVICE)


def compute_kshot_metrics(
    train_dataset: MVTecLOCODataset,
    test_dataloader: DataLoader,
    model: nn.Module,
    k_shot: int,
    seed: int,
) -> dict[str, float]:
    """Compute metrics for a specific k-shot setting.

    Args:
        train_dataset (MVTecLOCODataset): Training dataset (used for sampling few-shot images).
        test_dataloader (DataLoader): Test dataloader for the category.
        model (nn.Module): The model instance (already on the correct device).
        k_shot (int): The number of few-shot samples (k).
        seed (int): Seed value for reproducibility of few-shot sampling.

    Returns:
        dict[str, float]: Computed metrics for this k-shot setting.
    """
    image_metric = _F1Max().to(DEVICE)

    # Sample k_shot images from the training set deterministically
    torch.manual_seed(seed)
    k_shot_idxs = torch.randperm(len(train_dataset))[:k_shot].tolist()

    # Pass few-shot images and dataset category to model's setup method
    few_shot_images = torch.stack([cast(ImageBatch, train_dataset[idx]).image for idx in k_shot_idxs]).to(DEVICE)
    setup_data = {
        "few_shot_samples": few_shot_images,
        "dataset_category": train_dataset.category,
    }
    model.setup(setup_data)

    # Inference loop
    model.eval()  # Ensure model is in eval mode
    with torch.no_grad():  # Disable gradient calculations for inference
        for data in tqdm(test_dataloader, desc=f"k={k_shot} Inference", leave=False):
            output = model(data.image.to(DEVICE))
            image_metric.update(output.pred_score, data.gt_label.to(DEVICE))

    # Compute final metrics
    k_shot_metrics = {"image_score": image_metric.compute().item()}

    return k_shot_metrics


def compute_average_metrics(
    metrics: list[dict[str, int | float]] | pd.DataFrame,
) -> dict[str, float]:
    """Compute the average metrics across all seeds and categories.

    Args:
        metrics (list[dict[str, int | float]] | pd.DataFrame): Collected metrics.

    Returns:
        dict[str, float]: Average metrics across all seeds and categories.
    """
    # Convert the metrics list to a pandas DataFrame
    if not isinstance(metrics, pd.DataFrame):
        df = pd.DataFrame(metrics)
        df.to_csv("metrics.csv")

    # Compute the average metrics for each seed and k_shot across categories
    average_seed_performance = df.groupby(["k_shot", "category"])[["image_score"]].mean().reset_index()

    # Calculate the mean image and pixel performance for each k-shot
    k_shot_performance = average_seed_performance.groupby("k_shot")[["image_score"]].mean().reset_index()

    # Extract the k-shot numbers and their corresponding average image scores
    k_shot_numbers = k_shot_performance["k_shot"]
    average_image_scores = k_shot_performance["image_score"]

    # Calculate the area under the F1-max curve (AUFC)
    aufc = auc(k_shot_numbers, average_image_scores)

    # Get the normalized aufc score
    normalized_k_shot_numbers = (k_shot_numbers - k_shot_numbers.min()) / (k_shot_numbers.max() - k_shot_numbers.min())
    normalized_aufc = auc(normalized_k_shot_numbers, average_image_scores)

    # Directly calculate the average image score across all k-shot performances
    avg_image_score = k_shot_performance["image_score"].mean()

    # Output the final average metrics and AUFC
    final_avg_metrics = {
        "aufc": aufc,
        "normalized_aufc": normalized_aufc,
        "avg_image_score": avg_image_score,
    }

    return final_avg_metrics


def evaluate_submission(
    seeds: list[int],
    k_shots: list[int],
    dataset_path: Path | str,
) -> dict[str, float]:
    """Run the full evaluation across seeds, categories, and k-shots.

    Args:
        seeds (list[int]): List of seed values.
        k_shots (list[int]): List of k-shot values.
        dataset_path (Path | str): Path to the dataset.

    Returns:
        dict[str, float]: Final averaged metrics.
    """
    metrics = []
    print(f"Using device: {DEVICE}")

    for category in tqdm(CATEGORIES, desc="Processing Categories"):
        # --- Per-Category Setup ---
        # Load model once per category
        model = get_model(category)
        # Load data once per category
        train_dataloader, test_dataloader = get_dataloaders(dataset_path, category, batch_size=model.batch_size)
        train_dataset = cast(MVTecLOCODataset, train_dataloader.dataset)  # Get underlying dataset

        for seed in tqdm(seeds, desc=f"Category {category} Seeds", leave=False):
            for k_shot in k_shots:  # No tqdm here, handled in compute_kshot_metrics
                # Compute metrics for this specific seed/category/k-shot combination
                k_shot_metrics = compute_kshot_metrics(
                    train_dataset=train_dataset,
                    test_dataloader=test_dataloader,
                    model=model,
                    k_shot=k_shot,
                    seed=seed,
                )

                # Append results
                metrics.append(
                    {
                        "seed": seed,
                        "k_shot": k_shot,
                        "category": category,
                        "image_score": k_shot_metrics["image_score"],
                    }
                )

    final_average_metrics = compute_average_metrics(metrics)
    print("Final Average Metrics Across All Seeds:", final_average_metrics)

    return final_average_metrics


def eval():
    args = parse_args()
    result = evaluate_submission(
        seeds=args.seeds,
        k_shots=args.k_shots,
        dataset_path=args.dataset_path,
    )
    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    eval()
