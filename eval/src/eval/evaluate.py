"""Evaluation script"""

import argparse
import json
from pathlib import Path
from tempfile import gettempdir
from urllib.request import urlretrieve

import pandas as pd
import torch
from anomalib.data import ImageBatch, MVTecLOCO
from anomalib.data.utils.download import DownloadProgressBar
from anomalib.metrics import F1Max
from sklearn.metrics import auc
from submission_template.model import Model
from torch import nn
from tqdm import tqdm

CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]


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
        help="List of seed values for reproducibility. Ignored if --seed is provided. Default is [42, 0, 1234].",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        help="Single value for few-shot learning. This overwrites --k_shots if both are provided.",
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


def get_datamodule(dataset_path: Path | str, category: str) -> MVTecLOCO:
    """Get the MVTec LOCO datamodule.

    Args:
        dataset_path (Path | str): Path to the dataset.
        category (str): Category of the MVTec dataset.

    Returns:
        MVTec: MVTec datamodule.
    """
    # Create the dataset
    # NOTE: We fix the image size to (256, 256) for consistent evaluation across all models.
    datamodule = MVTecLOCO(
        root=dataset_path,
        category=category,
        eval_batch_size=1,
    )
    datamodule.setup()

    return datamodule


def download(url: str) -> Path:
    root = Path(gettempdir())
    downloaded_file_path = root / url.split("/")[-1]
    if url.startswith("http://") or url.startswith("https://"):
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as progress_bar:
            urlretrieve(  # noqa: S310  # nosec B310
                url=f"{url}",
                filename=downloaded_file_path,
                reporthook=progress_bar.update_to,
            )
    else:
        message = f"URL {url} is not valid. Please check the URL."
        raise ValueError(message)
    return downloaded_file_path


def get_model(category: str) -> nn.Module:
    """Get model.

    Args:
        category (str): Category of the dataset. This can be used to download specific weights for each category.

    Returns:
        nn.Module: Loaded model.
    """

    # instantiate model
    model = Model()
    model.eval()

    # load weights
    if model.weights_url(category) is not None:
        weights_path = download(model.weights_url)
        model.load_state_dict(torch.load(weights_path))

    return model


def compute_category_metrics(
    datamodule: MVTecLOCO,
    model: nn.Module,
    k_shot: int,
    seed: int,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute category-wise metrics.

    Args:
        datamodule (MVTecLOCO): MVTec LOCO datamodule for the category.
        model (nn.Module): Model for the category.
        k_shot (int): k-shot
        seed (int): Seed value for reproducibility.
        device (torch.device): Device to run the model on.

    Returns:
        dict[str, float]: Category-wise metrics.
    """
    # Get the device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model.to(device)

    # Create the metrics
    image_metric, pixel_metric = (
        F1Max(fields=["pred_score", "gt_label"], prefix="image_"),
        F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
    )

    # Sample k_shot images from the training set.
    torch.manual_seed(seed)
    k_shot_idxs = torch.randperm(len(datamodule.train_data))[:k_shot].tolist()

    # pass few-shot images and dataset category to model
    setup_data = {
        "few_shot_samples": torch.stack(
            [datamodule.train_data[idx].image for idx in k_shot_idxs]
        ).to(device),
        "dataset_category": datamodule.category,
    }
    model.setup(setup_data)

    # Loop over the test set and compute the metrics
    for data in datamodule.test_dataloader():
        output = model(data.image.to(device))

        metric_batch = ImageBatch(
            image=data.image,
            pred_score=output.pred_score,
            anomaly_map=output.anomaly_map if "anomaly_map" in output else None,
            gt_label=data.gt_label,
            gt_mask=data.gt_mask,
        )
        # Update the image metric
        image_metric.update(metric_batch)

        # TODO: Do not compute pixel metrics as it is already computation intensive.
        # Update the pixel metric
        if "anomaly_map" in output:
            pixel_metric.update(metric_batch)

    # Compute the metrics
    # TODO: Double check if pixel_score is required.
    category_metrics = {"image_score": image_metric.compute().item()}
    if pixel_metric.update_called:
        category_metrics["pixel_score"] = pixel_metric.compute().item()

    return category_metrics


def compute_average_metrics(
    metrics: list[dict[str, int | float]] | pd.DataFrame,
) -> dict[str, float]:
    """Compute the average metrics across all seeds and categories.

    Args:
        metrics (list[dict[str, int  |  float]] | pd.DataFrame): List of metrics
            for each seed and category.

    Returns:
        dict[str, float]: Average metrics across all seeds and categories.
    """
    # Convert the metrics list to a pandas DataFrame
    if not isinstance(metrics, pd.DataFrame):
        df = pd.DataFrame(metrics)
        df.to_csv("metrics.csv")

    # Compute the average metrics for each seed and k_shot across categories
    average_seed_performance = (
        df.groupby(["k_shot", "category"])[["image_score"]].mean().reset_index()
    )

    # Calculate the mean image and pixel performance for each k-shot
    k_shot_performance = (
        average_seed_performance.groupby("k_shot")[["image_score"]].mean().reset_index()
    )

    # Extract the k-shot numbers and their corresponding average image scores
    k_shot_numbers = k_shot_performance["k_shot"]
    average_image_scores = k_shot_performance["image_score"]

    # Calculate the area under the F1-max curve (AUFC)
    aufc = auc(k_shot_numbers, average_image_scores)

    # Get the normalized aufc score
    normalized_k_shot_numbers = (k_shot_numbers - k_shot_numbers.min()) / (
        k_shot_numbers.max() - k_shot_numbers.min()
    )
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
    """Evaluate the submission.

    Args:
        seeds (list[int]): List of seed values to generate random perturbations.
        k_shots (list[int]): List of integers for few-shot learning samples.
        dataset_path (Path | str): Path to the dataset.
    """
    # Initialize a list to store the metrics for each seed and category.
    metrics = []
    for seed in tqdm(seeds, desc="Processing Seeds"):
        for category in tqdm(CATEGORIES, desc="Processing Categories", leave=False):
            # Create the datamodule.
            datamodule = get_datamodule(dataset_path, category)
            for k_shot in tqdm(k_shots, desc="Processing k-shots", leave=False):
                # Initialize the model for the seed, category, and k-shot.
                model = get_model(category)
                # Compute the category-wise metrics.
                category_metrics = compute_category_metrics(
                    datamodule=datamodule,
                    model=model,
                    k_shot=k_shot,
                    seed=seed,
                )

                # Append the metrics to the metrics list.
                metrics.append(
                    {
                        "seed": seed,
                        "k_shot": k_shot,
                        "category": category,
                        "image_score": category_metrics["image_score"],
                        "pixel_score": category_metrics.get(
                            "pixel_score", float("nan")
                        ),
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
        json.dump(result, f)


if __name__ == "__main__":
    eval()
