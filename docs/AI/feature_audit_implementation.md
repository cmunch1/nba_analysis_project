# Feature Audit Implementation

**Date:** 2025-10-12
**Status:** Core Implementation Complete
**Related:** [Core Framework Usage](core_framework_usage.md), [Model Registry and Inference](model_registry_and_inference.md)

---

## Overview

The feature audit system automatically generates comprehensive reports about feature quality, importance, and pruning candidates during model training. It integrates seamlessly with your existing ML framework and follows all established patterns.

## Key Features

### 1. **Automated Audit Generation**
- Triggered automatically after model evaluation
- Reuses existing computation (SHAP, feature importance, CV folds)
- No manual intervention required

### 2. **Comprehensive Metrics**
- **Statistical**: coverage, missing_rate, variance, cardinality
- **Importance**: SHAP, permutation importance, model-specific gain
- **Redundancy**: pairwise correlation, optional VIF
- **Stability**: cross-validation fold consistency
- **Composite**: drop_candidate_score for prioritization

### 3. **Framework Integration**
- Uses dependency injection via `di_container.py`
- Follows `app_logger` structured logging patterns
- Uses `error_handler` for typed exceptions
- Config-driven via [model_testing_config.yaml](../../configs/core/model_testing_config.yaml)

---

## Architecture

### Component Structure

```
src/ml_framework/model_testing/
├── feature_auditing/
│   ├── __init__.py
│   ├── base_feature_auditor.py      # Abstract interface
│   ├── feature_auditor.py           # Concrete implementation
│   └── audit_metrics.py             # Helper functions
├── main.py                          # Modified to call auditor
└── di_container.py                  # Registered feature_auditor
```

### Data Flow

```
process_model_evaluation() [main.py:241]
├─> Train model (existing)
├─> Calculate metrics (existing)
├─> Generate charts (existing)
└─> Generate feature audit (NEW)
    ├─> Extract from ModelTrainingResults
    │   ├─> SHAP values (reused)
    │   ├─> Feature importance (reused)
    │   └─> Feature data (reused)
    ├─> Compute additional metrics
    │   ├─> Permutation importance (new)
    │   ├─> Correlation matrix (new)
    │   ├─> Target correlation (new)
    │   └─> Stability scores (placeholder)
    ├─> Compute drop candidate scores
    └─> Save versioned audit CSV/parquet
```

---

## Configuration

### Config Schema ([model_testing_config.yaml:89-127](../../configs/core/model_testing_config.yaml))

```yaml
# Feature audit settings
generate_feature_audit: true # Enable feature audit generation
save_feature_audit: true # Save audit results to CSV/parquet
feature_audit_format: csv # Output format: 'csv' or 'parquet'

# Feature audit metrics configuration
feature_audit_metrics:
  compute_basic_stats: true # coverage, missing_rate, unique_values, variance
  compute_shap_importance: true # Requires calculate_shap_values: true
  compute_permutation_importance: true # Computed on validation set
  permutation_n_repeats: 5 # Number of permutation repeats
  compute_correlation_matrix: true # Pairwise feature correlations
  compute_vif: false # Variance Inflation Factor (expensive)
  correlation_method: pearson # 'pearson', 'spearman', 'kendall'
  compute_stability_score: true # Requires perform_oof_cross_validation: true
  stability_top_k: 20 # Number of top features to track
  compute_target_correlation: true # Correlation with target

# Feature audit thresholds (for drop_candidate_score)
feature_audit_thresholds:
  near_zero_importance_percentile: 10 # Bottom 10% of features by SHAP
  near_zero_importance_threshold: 0.0 # Absolute permutation importance threshold
  high_missing_threshold: 0.4 # Flag features with >40% missing values
  high_collinearity_threshold: 0.95 # Flag features with >0.95 correlation
  low_stability_threshold: 0.2 # Flag features in <20% of top-k across folds
  drop_candidate_threshold: 2 # Minimum score to flag as drop candidate

# Feature audit output
feature_audit_output_dir: "${PROJECT_ROOT}/feature_audits"
feature_audit_versioning: true # Include run_id and timestamp in filename
```

### Enable/Disable Features

To disable feature auditing entirely:
```yaml
generate_feature_audit: false
```

To compute but not save:
```yaml
generate_feature_audit: true
save_feature_audit: false
```

To disable specific metrics:
```yaml
feature_audit_metrics:
  compute_permutation_importance: false  # Skip expensive permutation test
  compute_vif: false  # VIF is expensive for many features
```

---

## Output Format

### Audit CSV Columns

```csv
feature_name                    # Feature identifier
data_type                       # numeric/categorical/binary
coverage                        # Non-null ratio (0-1)
missing_rate                    # Null ratio (0-1)
unique_values                   # Number of unique values
variance                        # Variance (numeric only)
cardinality                     # Unique value count (categorical)
categorical_flag                # Boolean: is in config.categorical_features
shap_mean_abs                   # Mean absolute SHAP value
shap_std_abs                    # Std of absolute SHAP values
shap_rank                       # Rank by SHAP importance (1=most important)
permutation_importance_mean     # Mean permutation importance
permutation_importance_std      # Std of permutation importance
model_gain_importance           # Model-specific importance (e.g., XGBoost gain)
pairwise_max_corr               # Max correlation with any other feature
target_correlation              # Correlation with target variable
stability_score                 # Fraction of CV folds where feature is top-k (gain importance)
stability_score_shap            # Fraction of CV folds where feature is top-k (SHAP importance)
near_zero_importance            # Flag: 1 if low SHAP & low permutation
high_missing_flag               # Flag: 1 if missing_rate > threshold
high_collinearity_flag          # Flag: 1 if pairwise_max_corr > threshold
unstable_flag                   # Flag: 1 if stability_score < threshold
leakage_flag                    # Flag: manual annotation (defaults to 0)
drop_candidate_score            # Sum of flags (higher = more likely to drop)
notes                           # Free text for manual annotations
model_name                      # Model type (e.g., XGBoost)
run_id                          # MLflow run ID (if available)
experiment_id                   # MLflow experiment ID (if available)
audit_timestamp                 # ISO timestamp of audit creation
```

### Example Output

```csv
feature_name,data_type,coverage,missing_rate,shap_mean_abs,shap_rank,permutation_importance_mean,pairwise_max_corr,drop_candidate_score,model_name
h_team_elo,numeric,1.0,0.0,0.052,1,0.043,0.82,0,XGBoost
v_team_elo,numeric,1.0,0.0,0.048,2,0.041,0.82,0,XGBoost
h_net_rating_l5,numeric,0.98,0.02,0.031,3,0.028,0.35,0,XGBoost
obsolete_feature_1,numeric,0.45,0.55,0.001,89,0.0,0.12,2,XGBoost
```

### File Naming Convention

**With versioning enabled:**
```
feature_audit_XGBoost_validation_abc12345_20251012_103045.csv
                └─model   └─eval_type └─run_id └─timestamp
```

**Without versioning:**
```
feature_audit_XGBoost_validation_20251012_103045.csv
```

---

## Usage Examples

### Example 1: Basic Feature Pruning Workflow

```python
# Run model training (audit generated automatically)
uv run -m src.ml_framework.model_testing.main

# Load audit results
import pandas as pd
audit = pd.read_csv('feature_audits/feature_audit_XGBoost_validation_20251012_103045.csv')

# Identify drop candidates
drop_candidates = audit[audit['drop_candidate_score'] >= 2]
print(f"Found {len(drop_candidates)} drop candidates")

# Review reasons
print(drop_candidates[['feature_name', 'drop_candidate_score',
                      'near_zero_importance', 'high_missing_flag',
                      'high_collinearity_flag']])

# Features to drop
features_to_drop = drop_candidates['feature_name'].tolist()
```

### Example 2: Analyzing Feature Importance

```python
# Sort by SHAP importance
top_features = audit.nsmallest(20, 'shap_rank')

print("Top 20 Features by SHAP:")
print(top_features[['feature_name', 'shap_mean_abs', 'shap_rank',
                    'permutation_importance_mean', 'stability_score']])

# Check for disagreement between importance metrics
disagreement = audit[
    (audit['shap_rank'] <= 20) &
    (audit['permutation_importance_mean'] < 0.01)
]
print(f"Features with high SHAP but low permutation: {len(disagreement)}")
```

### Example 3: Redundancy Analysis

```python
# Find highly correlated feature pairs
high_corr = audit[audit['pairwise_max_corr'] > 0.9]

print("Highly Correlated Features:")
print(high_corr[['feature_name', 'pairwise_max_corr', 'shap_rank']])

# Strategy: Keep the more important feature from correlated pairs
# (Manual review recommended)
```

### Example 4: Missing Data Analysis

```python
# Features with high missing rates
high_missing = audit[audit['missing_rate'] > 0.3]

print("Features with >30% Missing:")
print(high_missing[['feature_name', 'missing_rate', 'shap_rank']])

# Keep high-importance features even if missing
keep_despite_missing = high_missing[high_missing['shap_rank'] <= 10]
print(f"High-importance features to keep despite missing data: {len(keep_despite_missing)}")
```

### Example 5: Stability Analysis (when implemented)

```python
# Unstable features across CV folds
unstable = audit[audit['stability_score'] < 0.5]

print("Unstable Features (< 50% fold consistency):")
print(unstable[['feature_name', 'stability_score', 'shap_mean_abs']])

# Flag for further investigation
```

---

## Implementation Details

### Reused Computation

The auditor **reuses** these from `ModelTrainingResults`:
- ✅ SHAP values → `results.shap_values`
- ✅ Feature importance → `results.feature_importance_scores`
- ✅ Feature data → `results.feature_data`
- ✅ Target data → `results.target_data`
- ✅ Trained model → `results.model`

### New Computation

The auditor **computes**:
- Permutation importance (on validation set)
- Pairwise correlation matrix
- Target correlation
- Stability scores (placeholder - needs CV fold tracking)

### Performance Considerations

**Fast operations:**
- Basic stats (coverage, missing, variance): ~1ms
- Correlation matrix: ~10-100ms depending on feature count
- SHAP statistics: instant (already computed)

**Slow operations:**
- Permutation importance: ~1-10s depending on model complexity
- VIF: ~1-60s for >50 features (disabled by default)

**Recommendation:**
- Keep `compute_permutation_importance: true` (useful metric)
- Keep `compute_vif: false` unless specifically needed

---

## Integration with MLflow

### Audit as Artifact

The audit CSV can be logged to MLflow as an artifact:

```python
# In experiment_logger.py (future enhancement)
if hasattr(results, 'feature_audit') and results.feature_audit is not None:
    audit_path = feature_auditor.save_audit(...)
    mlflow.log_artifact(audit_path, artifact_path="feature_audits")
```

### Run Metadata

The audit includes:
- `run_id`: MLflow run identifier
- `experiment_id`: MLflow experiment identifier
- `audit_timestamp`: Creation timestamp

This enables tracking feature evolution across experiments.

---

## Known Limitations & Future Work

### Current Limitations

1. **Stability Score**: ✅ **FULLY IMPLEMENTED** (2025-10-12)
   - Tracks both gain importance AND SHAP importance per CV fold
   - Returns stability score (0.0-1.0) indicating fraction of folds where feature is in top-k
   - `stability_score`: Based on model gain importance (e.g., XGBoost gain)
   - `stability_score_shap`: Based on mean absolute SHAP values per fold
   - Unstable flag triggered if EITHER metric shows low stability

2. **Leakage Detection**: Manual only
   - `leakage_flag` defaults to 0
   - Users must manually annotate suspected leakage
   - **TODO**: Implement automated leakage detection heuristics

3. **VIF Computation**: Disabled by default
   - Expensive for high-dimensional data
   - Only supports numeric features
   - **TODO**: Add approximate VIF for large feature sets

4. **Categorical Correlation**: Basic support
   - Currently uses 0.0 for categorical-categorical correlation
   - **TODO**: Implement Cramér's V or Theil's U

### Planned Enhancements

1. **Fold-Level Tracking**: ✅ **COMPLETED** (2025-10-12)
   - Implemented in [model_tester.py:277-299](../../src/ml_framework/model_testing/model_tester.py#L277-L299)
   - Captures gain importance per fold → `results.fold_importances`
   - Captures mean absolute SHAP per fold → `results.fold_shap_importances`
   - Both used for stability scoring in feature audit

2. **Automated Leakage Detection**
   - High target correlation + suspicious feature names
   - Features with perfect splits in trees
   - Time-based leakage detection for time series

3. **Interactive Dashboard**
   - Plotly/Streamlit dashboard for audit exploration
   - Filter by drop_candidate_score
   - Visualize importance distributions

4. **Audit Comparison**
   - Compare audits across experiments
   - Track feature importance drift over time
   - Automated alerts for unstable features

5. **MLflow Integration**
   - Log audit as MLflow artifact automatically
   - Query audits via MLflow API
   - Audit-based model comparison

---

## Testing

### Unit Tests (TODO)

```python
# tests/ml_framework/model_testing/feature_auditing/test_feature_auditor.py

def test_basic_stats_computation():
    """Test basic statistics are computed correctly."""
    pass

def test_importance_metrics_extraction():
    """Test SHAP and permutation importance extraction."""
    pass

def test_redundancy_metrics():
    """Test correlation and VIF computation."""
    pass

def test_drop_candidate_scoring():
    """Test composite drop score calculation."""
    pass

def test_audit_save_and_load():
    """Test audit serialization to CSV/parquet."""
    pass
```

### Integration Test

```python
# Test full audit generation during model training
def test_audit_generation_during_training():
    """Run model training and verify audit is generated."""
    # Run main.py
    # Load audit file
    # Verify columns and row count
    pass
```

---

## FAQ

### Q: How do I disable feature auditing temporarily?

**A:** Set `generate_feature_audit: false` in [model_testing_config.yaml:90](../../configs/core/model_testing_config.yaml)

### Q: Why is stability_score NaN for validation set testing?

**A:** Stability scores require cross-validation (multiple folds). If you only run validation set testing (`perform_oof_cross_validation: false`), stability scores will be NaN. Enable OOF cross-validation to get stability metrics.

### Q: Can I use this for feature selection?

**A:** Yes! Sort by `drop_candidate_score` and manually review features with score >= 2. Always validate on a holdout set.

### Q: How do I annotate leakage manually?

**A:** Edit the saved CSV and set `leakage_flag=1` for suspected features, then recompute `drop_candidate_score`.

### Q: What if permutation importance fails?

**A:** The implementation has error handling and returns 0.0 values. Check logs for error details.

### Q: Can I audit features from multiple models?

**A:** Yes! Each model generates a separate audit file. Compare them to find consistent drop candidates.

---

## References

- [Core Framework Usage](core_framework_usage.md) - DI patterns, logging, error handling
- [Model Registry and Inference](model_registry_and_inference.md) - Model persistence
- [BaseFeatureAuditor](../../src/ml_framework/model_testing/feature_auditing/base_feature_auditor.py) - Abstract interface
- [FeatureAuditor](../../src/ml_framework/model_testing/feature_auditing/feature_auditor.py) - Concrete implementation
- [audit_metrics.py](../../src/ml_framework/model_testing/feature_auditing/audit_metrics.py) - Helper functions

---

## Changelog

**2025-10-12** - Initial implementation
- Created base_feature_auditor.py abstract interface
- Implemented FeatureAuditor concrete class
- Created audit_metrics.py helper module
- Added configuration schema to model_testing_config.yaml
- Integrated into main.py process_model_evaluation()
- Registered in di_container.py
- Added feature_audit field to ModelTrainingResults

**2025-10-12** - Fold-level importance tracking
- Added `fold_importances` list to ModelTrainingResults ([training.py:53](../../src/ml_framework/framework/data_classes/training.py#L53))
- Added `fold_shap_importances` list to ModelTrainingResults ([training.py:54](../../src/ml_framework/framework/data_classes/training.py#L54))
- Implemented per-fold capture in model_tester.py ([model_tester.py:277-299](../../src/ml_framework/model_testing/model_tester.py#L277-L299))
- Updated compute_stability_scores() to use real fold data ([feature_auditor.py:336-404](../../src/ml_framework/model_testing/feature_auditing/feature_auditor.py#L336-L404))
- Added `stability_score_shap` column for SHAP-based stability
- Updated unstable_flag logic to consider both gain and SHAP stability
