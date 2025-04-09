# VAND 2025 Evaluation Framework

This repository contains the official evaluation script and framework for the **Visual Anomaly and Novelty Detection (VAND) 2025 Challenge**. It focuses on evaluating few-shot anomaly detection performance on the MVTec LOCO dataset.

> [!IMPORTANT]
> Participants should only modify the code within the `src/eval/submission` directory to implement their solution. Do **not** modify the core evaluation scripts in `src/eval`. Refer to `RULES.md` and the documentation within `src/eval/submission` for detailed submission guidelines.

This repository contains:

- The evaluation framework and scripts located in the `src/eval/` directory.
- A submission template for participants in `src/eval/submission/`.
- The official challenge rules in `RULES.md`.

## Getting Started

1.  **Familiarize yourself with the rules:** Read `RULES.md` carefully.
2.  **Explore the evaluation framework:** Understand how submissions will be evaluated by reviewing the documentation in `src/eval/submission/README.md`.
3.  **Implement your solution:** Modify the code within `src/eval/submission/` according to the guidelines.

## Overview

The evaluation framework is designed to:

1.  Load the MVTec LOCO dataset for specific categories.
2.  Instantiate the participant's anomaly detection model (defined in `src/eval/submission/model.py`).
3.  Perform a few-shot evaluation protocol:
    - For each category, seed, and specified `k_shot` value:
      - Randomly sample `k_shot` images from the training set for model setup/adaptation.
      - Evaluate the model on the test set.
      - Calculate the image-level F1Max score.
4.  Aggregate results across different seeds and k-shot values.
5.  Compute final metrics: Area Under the F1-max Curve (AUFC), normalized AUFC, and the average image F1Max score.

## Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for package management. Key dependencies include:

- `anomalib`: For MVTec LOCO dataset handling and metrics.
- `scikit-learn`: For calculating the Area Under Curve (AUC).
- `torch`: The deep learning framework.

## Project Structure

```
.
├── src/
│   └── eval/
│       ├── evaluate.py        # Core evaluation script (DO NOT MODIFY)
│       └── submission/ # <-- PARTICIPANT CODE GOES HERE
│           ├── model.py       # Define your Model class here
│           ├── __init__.py
│           └── README.md      # Submission-specific instructions
├── datasets/                  # (Optional) Default location for datasets
│   └── mvtec_loco/
├── results.json               # Final aggregated evaluation results (auto generated)
├── metrics.csv                # Detailed results per category/seed/k-shot (auto generated)
├── pyproject.toml             # Main project configuration (dependencies, scripts) # <-- Add optional dependencies here
├── uv.lock                    # Dependency lock file
├── README.md                  # Top-level README for the challenge
└── RULES.md                   # Official challenge rules

```

## How to Use

1.  **Install Dependencies:**

    - Ensure you have `uv` installed.
    - Sync the environment:
      `uv sync`
      This installs all necessary dependencies defined in `pyproject.toml` and `uv.lock` into a virtual environment (`.venv/`).

2.  **Implement Your Model:**

    - Go to `src/eval/submission/model.py`.
    - Implement your anomaly detection logic within the `Model` class, adhering to the required interface (methods like `__init__`, `setup`, `forward`, `weights_url`). See the README within `src/eval/submission` for specifics.

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

## Output Files

- `metrics.csv`: A CSV file logging the individual results for every run.
- `results.json`: A JSON file containing the final, aggregated performance metrics.

## Questions?

Refer to the contact information provided in `RULES.md` or the official challenge communication channels.
