# VAND 2025 Evaluation Framework

This repository contains the official evaluation script and framework for the **Visual Anomaly and Novelty Detection (VAND) 2025 Challenge**. It focuses on evaluating few-shot anomaly detection performance on the MVTec LOCO dataset.

> [!IMPORTANT] Participants should only modify the code within the `packages/submission-template` directory to implement their solution. Do **not** modify the core evaluation scripts in `src/eval`. Refer to `RULES.md` (if provided) and the documentation within `packages/submission-template` for detailed submission guidelines.

## Overview

The evaluation framework is designed to:

1.  Load the MVTec LOCO dataset for specific categories.
2.  Instantiate the participant's anomaly detection model (defined in `packages/submission-template`).
3.  Perform a few-shot evaluation protocol:
    - For each category, seed, and specified `k_shot` value:
      - Randomly sample `k_shot` images from the training set for model setup/adaptation.
      - Evaluate the model on the test set.
      - Calculate the image-level F1Max score.
4.  Aggregate results across different seeds and k-shot values.
5.  Compute final metrics: Area Under the F1-max Curve (AUFC), normalized AUFC, and the average image F1Max score.

## Project Structure

```
eval/
├── .venv/                     # Virtual environment (managed by uv)
├── packages/
│   └── submission-template/   # <-- PARTICIPANT CODE GOES HERE
│       ├── src/
│       │   └── submission_template/
│       │       └── model.py   # Define your Model class here
│       ├── pyproject.toml
│       └── README.md
├── src/
│   └── eval/
│       └── evaluate.py        # Core evaluation script (DO NOT MODIFY)
├── datasets/                  # (Optional) Default location for datasets
│   └── mvtec_loco/
├── results.json               # Final aggregated evaluation results
├── metrics.csv                # Detailed results per category/seed/k-shot
├── pyproject.toml             # Main project configuration (dependencies, scripts)
├── uv.lock                    # Dependency lock file
└── README.md                  # This file
```

## Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for package management. Key dependencies include:

- `anomalib`: For MVTec LOCO dataset handling and metrics.
- `scikit-learn`: For calculating the Area Under Curve (AUC).
- `torch`: The deep learning framework.
- `submission-template`: The local package containing the participant's model.

## How to Use

1.  **Install Dependencies:**

    - Ensure you have `uv` installed (`pip install uv`).
    - Navigate to the `eval/` directory.
    - Sync the environment:
      `uv sync`
      This installs all necessary dependencies defined in `pyproject.toml` and `uv.lock` into a virtual environment (`.venv/`).

2.  **Implement Your Model:**

    - Go to `packages/submission-template/src/submission_template/model.py`.
    - Implement your anomaly detection logic within the `Model` class, adhering to the required interface (methods like `__init__`, `setup`, `forward`, `weights_url`). See the README within `packages/submission-template` for specifics.

3.  **Prepare Dataset:**

    - Download the MVTec LOCO dataset.
    - By default, the script expects it at `./datasets/mvtec_loco`. You can change this using the `--dataset_path` argument.

4.  **Run Evaluation:**
    - Activate the virtual environment: `source .venv/bin/activate` (or equivalent for your shell).
    - Run the evaluation script using the `eval` command (defined in `pyproject.toml`).
    - Use `--help` to see available options:
      ```bash
      eval --help
      ```
    - Example command:
      ```bash
      eval --k_shots 1 2 4 8 --seeds 42 0 1234 --dataset_path /path/to/your/mvtec_loco
      ```
      - `--k_shots`: Specifies the list of k-values for few-shot learning.
      - `--seeds`: Specifies the random seeds for reproducibility. Multiple seeds are run to ensure robustness.
      - `--dataset_path`: Overrides the default dataset location.

## Evaluation Details

- **Dataset:** MVTec LOCO Anomaly Detection Dataset. The evaluation runs on the following categories:
  - `breakfast_box`
  - `juice_bottle`
  - `pushpins`
  - `screw_bag`
  - `splicing_connectors`
- **Metrics:**
  - **Image-Level F1Max:** The primary metric, computed for each category/seed/k-shot.
  - **Area Under the F1-max Curve (AUFC):** Calculated by plotting the average Image F1Max score (averaged across categories and seeds) against the `k_shot` values.
  - **Normalized AUFC:** AUFC calculated with k-shot values normalized to the [0, 1] range.
  - **Average Image Score:** The mean Image F1Max score across all k-shot values (after averaging over categories and seeds).
  - _(Note: Pixel-level metrics are currently disabled in the script but might be included in the future.)_
- **Few-Shot Protocol:** For each run, `k_shot` images are randomly selected _without replacement_ from the category's training set using the specified `seed`. These samples are passed to the `model.setup()` method.
- **Input Image Size:** All images are resized to 256x256 before being passed to the model.

## Output Files

- `metrics.csv`: A CSV file logging the individual results for every run:
  - `seed`: The random seed used.
  - `k_shot`: The number of few-shot samples used.
  - `category`: The MVTec LOCO category.
  - `image_score`: The computed Image-Level F1Max score.
  - `pixel_score`: (Currently likely empty/NaN as pixel metrics are disabled).
- `results.json`: A JSON file containing the final, aggregated performance metrics:
  - `aufc`: The calculated Area Under the F1-max Curve.
  - `normalized_aufc`: The normalized AUFC score.
  - `avg_image_score`: The average image F1Max score across all k-shots.
