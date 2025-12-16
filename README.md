Project: Hull Tactical — Market Prediction (2025)

This repository contains the code, experiments and helper tools used to compete in the "Hull Tactical - Market Prediction" Kaggle challenge. The project aims to predict daily excess returns and produce an allocation value (0–2) while controlling for volatility. It includes data preparation, feature engineering, model training/tuning, evaluation, and tools to create a Kaggle-ready notebook submission.

**Key Highlights**
- Embedded feature pipeline for efficient inference (precomputed warmup features to meet Kaggle runtime limits).
- Multiple experiment scripts and notebooks for training, backtests, and model analysis.
- Utilities to package warmup features and prepare Kaggle-friendly artifacts.

**Repository Structure**
- **data/**: raw and processed data (input for feature building).
- **artifacts/**: generated artifacts used for inference (e.g., `warmup_features.parquet`, model outputs, and submission files).
- **dataset** / **dataset_kaggle** / **dataset_models**: helper dataset metadata and packaged artifacts for Kaggle use.
- **notebooks/**: EDA and production-ready inference notebook (`fortress_inference_prod.ipynb`).
- **scripts/**: command-line helpers for feature building, packaging, evaluation, and creating submissions (e.g., `create_warmup_features.py`, `package_warmup_dataset.py`, `create_submission_local.py`).
- **experiments/**: results, backtests, and diagnostics from model runs.
- **src/**: core package and model code powering training and inference.

**Quick Start (overview)**
- Prepare dependencies and an appropriate Python environment.
- Generate the required warmup features used for fast inference.
- Run the repository's inference utilities to produce a submission file.
- See `scripts/` and `README_SUBMISSION.md` for script names and the submission checklist.

**Kaggle submission notes**
- See `README_SUBMISSION.md` for a checklist to make the notebook run reliably on Kaggle (include `artifacts/warmup_features.parquet`, use the fast `FORTRESS.predict_batch(...)` path, set reproducible env variables, etc.).

- View submissions for this competition (replace the slug if different):

- Competition submissions Link: https://www.kaggle.com/code/djt5ingh/fortress-inference-kernel