# Final Model Report — Optuna Narrow (Winning) Params

Date: 2025-10-10

## Winning hyperparameters

The winning hyperparameter set (declared official) is the output of the second-stage narrow Optuna run and is saved at:

`experiments/backtest/optuna_reg_narrow_best_params.json`

Content (summary):

- lambda_l1: 0.009889110290106218
- lambda_l2: 53.14044364006851
- num_leaves: 44
- min_child_samples: 136
- feature_fraction: 0.5034248963552715

These parameters are hereby declared the official winning configuration from the model tuning phase.

## Performance summary (penalized Sharpe across folds)

We re-evaluated the winning params across the five walk-forward folds and computed the per-fold penalized Sharpe ratio. The results are saved in:

`experiments/backtest/narrow_eval/narrow_eval_summary.csv`

Per-fold penalized Sharpe (fold 1 → fold 5):

- Fold 1: 0.991287
- Fold 2: 0.772112
- Fold 3: 0.578434
- Fold 4: -0.053552
- Fold 5: 0.008036

Aggregate statistics:

- Mean penalized Sharpe: 0.459263
- Std  penalized Sharpe: 0.464142

For comparison, the earlier stability-aware best (optuna_reg_stability) produced:

- Mean penalized Sharpe: 0.352166
- Std  penalized Sharpe: 0.595313

## Why this is our chosen (best) model

1. Improved average performance: The narrow-Optuna configuration increased the mean penalized Sharpe from ~0.352 (stability-best) to ~0.459, a substantial lift in expected risk-adjusted performance across folds.

2. Lower dispersion: Standard deviation across folds dropped from ~0.595 to ~0.464, indicating a more consistent model (less spread in performance).

3. Major improvement in Fold 4: The largest practical improvement relative to prior models is observed in Fold 4. Prior to the narrow search, Fold 4's penalized Sharpe for the reg-stability best was -0.4939 (severe underperformance). With the narrow-Optuna params, Fold 4 moved up to -0.0536 — a meaningful reduction in the tail loss and a material improvement in robustness. This single-fold improvement materially reduces downside skew in cross-fold performance and improves the practical deployability of the model.

4. Regularization & model shape: The narrow-Optuna solution favors relatively strong L2 regularization combined with moderate tree complexity (num_leaves=44, min_child_samples=136) and a moderate feature_fraction (~0.50). This combination appears to strike a favorable bias-variance/stability balance for the problem and dataset.

## Files produced / where to look next

- Winning params: `experiments/backtest/optuna_reg_narrow_best_params.json`
- Narrow evaluation summary: `experiments/backtest/narrow_eval/narrow_eval_summary.csv`
- Per-fold details: `experiments/backtest/narrow_eval/fold_{1..5}_details.csv`
- Comparison plot/csv: `experiments/backtest/comparison/per_fold_penalized_sharpe_comparison.*`
- Narrow decile diagnostics: `experiments/feature_diagnostics/narrow_deciles/fold_{1..5}_deciles.csv`

## Recommended next steps

1. Finalize allocation parameters (scale, turnover_penalty) and run a full-sample training to produce predictions for the holdout/test set.
2. Run a small ensemble of the top nearby parameterizations to smooth remaining fold-level noise.
3. Package the winning params into `experiments/backtest/optuna_best_params.json` and include them in the final training/deployment script.

---

This report concludes the model tuning phase; the `optuna_reg_narrow_best_params.json` should be used as the production hyperparameter set for final submission and downstream evaluation.
