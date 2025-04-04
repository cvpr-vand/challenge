# Repository for evaluating VAND 2025 submissions

> [!NOTE]
> Please make all changes to `src/eval/submission`.
> Refer to RULES.md for more details.

## To test your submission locally

First, install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then you can just run the evaluation script

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run eval --dataset_path=/path-to-dataset
```

This should generate metrics.csv and metrics.json