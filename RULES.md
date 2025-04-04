# VAND 2025 Challenge Rules

**Version:** 1.0 (Last Updated: YYYY-MM-DD)

Please read these rules carefully before participating.

## 1. Introduction

- **Challenge Goal:** Briefly describe the main objective (e.g., develop few-shot anomaly detection models for industrial inspection).
- **Organizers:** List the organizing committee or institutions.
- **Website:** Link to the official challenge website.
- **Timeline:** Key dates (registration deadline, submission deadline, results announcement).

## 2. Eligibility

- Who can participate? (e.g., individuals, teams, academic/industry restrictions).
- Team size limits (if any).
- Registration requirements.

## 3. Task Definition

- **Problem:** Few-shot anomaly detection and/or segmentation.
- **Dataset:** MVTec LOCO (specify categories used: `breakfast_box`, `juice_bottle`, `pushpins`, `screw_bag`, `splicing_connectors`). Link to the dataset page.
- **Input:** Describe the input provided to the model (e.g., k-shot support images, test images).
- **Output:** Describe the expected output from the model (e.g., image-level anomaly score, pixel-level anomaly map).
- **Few-Shot Protocol:** Detail how the `k` support samples are selected and used (refer to `eval/README.md` for the implementation details).

## 4. Evaluation

- **Metrics:** Define the primary and secondary evaluation metrics (Image F1Max, AUFC, Normalized AUFC, Average Image Score). Refer to `eval/README.md` for precise definitions and calculation methods.
- **Evaluation Server/Platform:** Describe how submissions will be evaluated (e.g., CodaLab, EvalAI, custom platform).
- **Leaderboard:** Explain how the leaderboard will be maintained (e.g., public/private test sets, update frequency).

## 5. Submission Guidelines

- **Format:**
  - Code must be packaged within the provided `eval/packages/submission-template/` directory.
  - Participants must implement the `Model` class in `model.py`.
  - Specify any file size limits or required files (e.g., trained weights, technical report).
- **Code Requirements:**
  - Programming language(s) allowed (e.g., Python 3.10+).
  - Required libraries (must be installable via `uv` or standard `pip`). List any disallowed libraries.
  - Model weight handling (how to load weights, `weights_url` function).
  - Resource constraints (e.g., maximum inference time per image, GPU memory limits during evaluation).
- **Submission Process:** Detail the steps to submit (e.g., zip the `submission-template` directory, upload to the platform).
- **Number of Submissions:** Limits on the number of submissions per participant/team.

## 6. Allowed Resources & Pre-training

- **External Data:** Clearly state whether external datasets are allowed for pre-training. If allowed, specify any approved datasets.
- **Pre-trained Models:** Specify if pre-trained backbones (e.g., ImageNet weights) are allowed.
- **Training Data Usage:** Clarify how the MVTec LOCO training set can be used (e.g., only the provided k-shot samples during `model.setup`, or allowed for unsupervised pre-training _before_ the challenge evaluation phase).

## 7. Code Release & Reproducibility

- Requirement for top teams to release their code.
- Code verification process.
- Emphasis on reproducible results (use of fixed seeds as provided by the evaluation script).

## 8. General Rules & Conduct

- Prohibition of cheating or exploiting the evaluation system.
- Rules regarding multiple accounts or submissions.
- Consequences of rule violations.

## 9. Contact & Support

- Provide contact information (email address, forum link) for questions regarding the rules or evaluation.

## 10. Amendments

- Statement that organizers reserve the right to amend the rules, with notification to participants.
