Submission notes — ensure fast (featured) inference path

- Purpose: ensure the notebook submission uses precomputed features (fast path) to meet Kaggle runtime limits and to avoid per-row feature recomputation.

Checklist before committing notebook for Kaggle:

- Include `artifacts/warmup_features.parquet` alongside your notebook (Kaggle "Input" dataset or in the notebook folder). This file should contain recent historical featured rows (>= 252 rows recommended).
- Confirm the notebook calls the fast path: `FORTRESS.predict_batch(df_with_features)` where `df_with_features` contains the same feature columns used in training.
- Set environment variables for reproducible runs in the notebook header if needed:
  - `WARMUP_ROWS=252`
  - `ENFORCE_CHRONO=1`
  - `COLLECT_DIAGNOSTICS=1`

How to generate `warmup_features.parquet` locally (if missing):

- Use the included helper (recommended):

```powershell
# from repo root
.\.venv\Scripts\python.exe .\scripts\create_warmup_features.py --nrows 252 --out .\artifacts
```

- Or run the simple preparer included: `scripts/check_and_prepare_warmup.py` (it will attempt to create the warmup file from `data/processed/features_post_gfc.parquet` or the sample CSV).

Why this matters:
- Precomputed features avoid expensive rolling/window recomputation per incoming row (our profiling showed ~20x speedup).
- Kaggle notebooks must finish within runtime limits and run with internet disabled; shipping `warmup_features.parquet` is the robust, offline-friendly approach.

If you'd like, I can add the warmup file into a dataset you can attach to the Kaggle notebook (requires you to upload it).