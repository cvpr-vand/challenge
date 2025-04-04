# VAND 2025 Challenge Repository

Welcome to the official repository for the **Visual Anomaly and Novelty Detection (VAND) 2025 Challenge**.

This repository contains:

- The evaluation framework and scripts located in the `eval/` directory.
- A submission template for participants in `eval/packages/submission-template/`.
- The official challenge rules in `RULES.md`.

## Getting Started

1.  **Familiarize yourself with the rules:** Read `RULES.md` carefully.
2.  **Explore the evaluation framework:** Understand how submissions will be evaluated by reviewing the documentation in `eval/README.md`.
3.  **Implement your solution:** Modify the code within `eval/packages/submission-template/` according to the guidelines.

## Local Testing

You can test your submission locally using the provided evaluation script.

1.  **Install `uv` (if you haven't already):**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env # Or restart your shell
    ```

2.  **Set up the environment and run evaluation:**

    ```bash
    cd eval
    # Install dependencies (this creates a .venv folder)
    uv sync
    # Activate the virtual environment
    source .venv/bin/activate
    # Run the evaluation script (replace with the actual path to your dataset)
    uv run eval --dataset_path /path/to/your/mvtec_loco
    ```

3.  **Check the results:** The script will generate `metrics.csv` (detailed scores) and `results.json` (final aggregated metrics) in the `eval/` directory.

## Submission

Please refer to `RULES.md` for detailed submission instructions.

## Questions?

Refer to the contact information provided in `RULES.md` or the official challenge communication channels.
