# Repository for evaluating VAND 2025 submissions

> [!NOTE]
> Please make all changes to `eval/packages/submission-template`.
> Refer to RULES.md for more details.

## To test your submission locally

```bash
cd eval
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run eval --dataset_path=/path-to-dataset
```

This should generate metrics.csv and metrics.json