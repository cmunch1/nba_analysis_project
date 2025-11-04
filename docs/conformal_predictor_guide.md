# Conformal Probability Guide

This guide explains how to enable and tune split conformal prediction on top of the existing probability calibration pipeline. It assumes you are already using the `ProbabilityCalibrator` during model evaluation (`ModelTester.calibrate_probabilities`).

## 1. Prerequisites
- A calibrated probability stream (`results.calibrated_predictions`) produced by `ProbabilityCalibrator`.
- Sufficient hold-out data: conformal quantiles are reliable only if the calibration split contains at least `min_calibration_samples` points (default `50`).
- `configs/core/postprocessing_config.yaml` available to toggle conformal options.

## 2. Enable Conformal Prediction

1. Open `configs/core/postprocessing_config.yaml`.
2. Set `conformal.enable: true`.
3. Adjust other options as needed:
   ```yaml
   conformal:
     enable: true
     method: split
     alphas:
       prediction_set: 0.1        # 90% target coverage for prediction sets
       probability_interval: 0.2  # 80% central probability interval
     score_function: probability_shortfall
     class_labels: [away, home]
     allow_empty_set: false
     min_calibration_samples: 50
     save_metrics: true
     save_diagnostics: true
     save_to_registry: true
   ```
4. Re-run model evaluation. `ModelTrainingResults` now contains:
   - `conformal_prediction_sets`: per-game label sets meeting the coverage target.
   - `conformal_probability_intervals`: `[lower, upper]` probability bands.
   - `conformal_metrics`: empirical coverage + diagnostics.
   - `conformal_artifact` & `conformal_metadata`: serialized predictor with quantiles.

## 3. Interpreting Outputs

- **Prediction sets**  
  Each entry contains the labels whose conformal score is below the learned quantile.  
  - `['home']`: model is confident the home team wins while meeting coverage.
  - `['home', 'away']`: model cannot exclude either outcome at the chosen `alpha`.
  - Empty sets occur only if `allow_empty_set: true`.

- **Probability intervals**  
  Symmetric bands around the calibrated probability (`p̂ ± q_interval`). The default alpha of `0.2` yields 80% expected coverage of the true binary outcome.

- **Diagnostics** (`conformal_metrics`)
  - `empirical_coverage_prediction_set` vs `target_coverage_prediction_set` highlight how close the calibration split coverage is to the target.
  - `empirical_interval_width` shows average width of the probability interval (`2 * q_interval`).
  - Logged via `structured_log` and persisted when `save_metrics` is `true`.

## 4. Fine-Tuning `alpha` Values

### 4.1 Understand the Trade-off
- Lower `alpha` (e.g., `0.05`) → higher coverage target (95%) → larger prediction sets / wider intervals.
- Higher `alpha` (e.g., `0.2`) → lower coverage (80%) → narrower sets but more uncovered outcomes.

Adjust `alphas.prediction_set` and `alphas.probability_interval` independently:
- Want fewer ambiguous games? Increase `alpha.prediction_set` slightly (e.g., `0.15` for ~85% coverage), then monitor the drop in empirical coverage.
- Need more conservative probability intervals for high-stakes decisions? Decrease `alpha.probability_interval` to widen the bands.

### 4.2 Calibration Loop
1. Pick an initial alpha based on business requirements.
2. Run evaluation and inspect `conformal_metrics`.
3. If `empirical_coverage_prediction_set` < `target_coverage_…`, reduce `alpha` (stricter coverage).
4. If empirical coverage is significantly above the target and prediction sets are too wide, increase `alpha`.
5. Repeat until empirical coverages align with targets on multiple validation splits.

### 4.3 Cross-Validation / Time-Based Splits
- For time-series leagues, ensure the calibration split mirrors deployment order. Non-random splits help maintain validity.
- When using cross-validation, aggregate conformal metrics across folds to choose global alphas.

## 5. Score Function Selection

- `probability_shortfall` *(default)*: scores each example by how far the calibrated probability falls short of certainty for the true label (`1 - p_true`). Use when you trust the calibrated probabilities.
- `absolute_error`: symmetric error between label and probability (`|y - p|`). Useful if you want intervals aligned with MAE behaviour.

Switch via `conformal.score_function`. Refit and compare coverage + interval width.

## 6. Diagnostics & Logging

- Set `save_diagnostics: true` to persist metrics for dashboards or notebooks. Pair this with custom visualizations (e.g., coverage over time or by matchup tier).
- In the model registry, conformal artifacts are saved when `save_to_registry: true`. Downstream inference can load the same quantiles to produce deployment-time intervals.

## 7. Practical Tips

- **Data Sufficiency**: If conformal fitting is skipped (log message indicates insufficient samples), gather more validation data or lower `min_calibration_samples`.
- **Class Labels**: Ensure `class_labels` matches your target ordering. The second entry is treated as the positive class.
- **Empty Sets**: Keep `allow_empty_set: false` unless you explicitly want to signal “no decision.” Otherwise an empty set defaults to both labels.
- **Scaling / On-line updates**: Conformal quantiles assume exchangeability. If the data distribution drifts, periodically refresh the calibration split and recompute quantiles.
- **Testing**: Run `pytest tests/postprocessing/test_conformal_predictor.py` after changing core logic to ensure quantile calculations remain consistent.

## 8. Next Steps

1. Automate alpha search: integrate a simple grid over candidate alphas and pick the pair that balances coverage vs. set size on your validation data.
2. Visualize conformal outputs: plot empirical coverage vs. alpha choices, or track the proportion of ambiguous prediction sets per team/week.
3. Extend to multi-class tasks: `ConformalPredictor` currently assumes binary labels; generalisation would require class-conditional scoring.

With these guidelines, you can tailor conformal prediction to your NBA win probability pipeline, ensuring your probability outputs are accompanied by transparent uncertainty signals.***
