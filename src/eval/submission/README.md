# VAND 2025 Challenge Submission Template

This directory contains the template for participants to implement their solution.

## To Create a Submission

1. Fork the repository
2. Modify the `model.py` file to implement your solution. You can introduce any other files/directories in this directory.
3. Make a pull request to the repository.
4. If successful, the evaluation pipeline will run and you will be able to see the results on the leaderboard, and a comment will be added to the pull request with the results.

## Evaluation

The evaluation pipeline is defined in `src/eval/evaluate.py`. It is used to evaluate the performance of the submitted model. DO NOT modify the evaluation pipeline.

The repository uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies, and running the evaluation script.

1.  **Install `uv` (if you haven't already):**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env # Or restart your shell
    ```

2.  **Set up the environment and run evaluation:**

To test your submission locally, run the following command from the repository root:

```bash
uv run eval --dataset_path /path/to/your/mvtec_loco
```

This will automatically create a virtual environment and install the dependencies.

If the above command fails, you can try the following steps manually:

```bash
# Install dependencies (this creates a .venv folder)
uv sync
# Activate the virtual environment
source .venv/bin/activate
# Run the evaluation script (replace with the actual path to your dataset)
uv run eval --dataset_path /path/to/your/mvtec_loco
```

3.  **Check the results:** The script will generate `metrics.csv` (detailed scores) and `results.json` (final aggregated metrics).
