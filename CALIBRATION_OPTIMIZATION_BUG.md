# Calibration Optimization Indexing Bug

## Status
**OPEN** - Temporarily worked around by disabling optimization

## Summary
The calibration optimization feature (which automatically selects the best calibration method via grid search with cross-validation) fails with an indexing error when enabled. This prevents the model testing pipeline from completing within reasonable timeframes.

## Error Details

### Error Message
```
ERROR: Error during calibration optimization | Context: {
    'error_type': 'PostprocessingError',
    'original_error': "'[1, 2, 4, 5, 6, 7, 9, 10, 11, 12, ...] not in index'",
    'n_samples': 18330
}
```

### Location
- **Module**: `ml_framework.postprocessing.calibration_optimizer`
- **Function**: Likely in the cross-validation fold splitting or data indexing within the optimization loop
- **Config**: [configs/core/postprocessing_config.yaml:13](configs/core/postprocessing_config.yaml#L13)

### When It Occurs
- When `calibration.optimize: true` is set in postprocessing config
- During the automatic calibration method selection process
- Happens after model training, during the postprocessing phase
- Specifically during cross-validation evaluation of calibration methods

## Configuration

### Current Workaround
Set `calibration.optimize: false` in [configs/core/postprocessing_config.yaml](configs/core/postprocessing_config.yaml):

```yaml
calibration:
  enable: true
  optimize: false  # WORKAROUND: Disabled due to indexing bug
```

This causes the pipeline to use manual calibration with the specified method (sigmoid or isotonic) instead of automatic optimization.

### Optimization Settings (When Enabled)
```yaml
calibration:
  optimize: true
  optimization:
    methods: [sigmoid, isotonic]
    evaluation_bins: [5, 7, 9, 12, 15, 20]
    selection_metric: brier_score
    cv_folds: 5
    use_cv: true
```

## Impact

### What Works
- ✅ Manual calibration (when `optimize: false`)
- ✅ Single method calibration (sigmoid or isotonic)
- ✅ Model training and prediction
- ✅ Conformal prediction
- ✅ Model registry saving

### What's Broken
- ❌ Automatic calibration method selection
- ❌ Grid search over multiple calibration methods
- ❌ Cross-validation evaluation of calibration quality

### Performance Impact
- The optimization loop appears to get stuck or run very slowly
- Model testing times out (180-300 seconds) before completing
- Prevents full pipeline execution when enabled

## Root Cause Analysis

### Likely Issues

1. **Index Mismatch in CV Splits**
   - The error message shows a list of indices that are "not in index"
   - Suggests that cross-validation fold splitting is creating indices that don't exist in the data
   - Possible DataFrame vs array indexing confusion

2. **DataFrame Indexing Problem**
   - The calibration data might be a DataFrame with non-integer index
   - CV fold splitting returns integer positions, but code tries to use them on labeled index
   - Need to reset index or use `.iloc` instead of `.loc`

3. **Subset Selection Error**
   - During CV, a subset of data is selected for each fold
   - The subset selection might be using wrong indexing method
   - List of indices from CV split don't match the DataFrame's actual index

### Files to Investigate

1. **CalibrationOptimizer Class**
   - Location: `src/ml_framework/postprocessing/calibration_optimizer.py` (likely)
   - Check the CV fold splitting logic
   - Look for index handling in `optimize()` or `evaluate()` methods

2. **Data Flow**
   - Check how predictions are passed from ModelTester to CalibrationOptimizer
   - Verify if DataFrame index is preserved or reset
   - Look for `.loc[]` vs `.iloc[]` usage

3. **Cross-Validation**
   - Check if using sklearn's `cross_validate` or custom CV
   - Verify fold indices are correctly applied to data

## Debug Steps

### 1. Add Logging to CalibrationOptimizer
```python
# Before CV split
self.app_logger.structured_log(
    logging.DEBUG,
    "Pre-CV data info",
    data_shape=y_pred.shape,
    data_type=type(y_pred),
    index_type=type(y_pred.index) if hasattr(y_pred, 'index') else None,
    index_sample=y_pred.index[:10].tolist() if hasattr(y_pred, 'index') else None
)

# After CV split
for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
    self.app_logger.structured_log(
        logging.DEBUG,
        "CV fold indices",
        fold=fold_idx,
        train_idx_sample=train_idx[:10].tolist(),
        val_idx_sample=val_idx[:10].tolist(),
        train_idx_max=train_idx.max(),
        val_idx_max=val_idx.max(),
        data_length=len(y_pred)
    )
```

### 2. Check Index Reset
Look for code that selects fold data:
```python
# Likely incorrect:
y_train_fold = y_pred[train_idx]  # If y_pred is DataFrame with non-integer index

# Should be:
y_train_fold = y_pred.iloc[train_idx]  # or ensure y_pred.reset_index(drop=True)
```

### 3. Reproduce in Isolation
Create a minimal test case:
```python
# Test script: test_calibration_optimizer.py
from ml_framework.postprocessing.calibration_optimizer import CalibrationOptimizer
import numpy as np
import pandas as pd

# Create test data with non-integer index (simulating the issue)
y_true = np.random.randint(0, 2, size=1000)
y_pred = np.random.rand(1000)

# Try with array (should work)
optimizer.optimize(y_pred=y_pred, y_true=y_true)

# Try with DataFrame (might fail)
df_pred = pd.DataFrame({'pred': y_pred}, index=range(100, 1100))
optimizer.optimize(y_pred=df_pred['pred'].values, y_true=y_true)
```

## Recommended Fix

### Short-term (Already Applied)
- Disable optimization: `calibration.optimize: false`
- Use manual method selection (sigmoid works well for most cases)
- Re-enable after fix is implemented

### Long-term Fix

1. **Add Index Reset in CalibrationOptimizer**
   ```python
   def optimize(self, y_pred, y_true, **kwargs):
       # Ensure we're working with arrays or reset DataFrame index
       if isinstance(y_pred, pd.Series):
           y_pred = y_pred.reset_index(drop=True)
       if isinstance(y_true, pd.Series):
           y_true = y_true.reset_index(drop=True)

       # Convert to numpy arrays for CV
       y_pred = np.asarray(y_pred)
       y_true = np.asarray(y_true)

       # Now CV fold indices will work correctly
       # ... rest of optimization logic
   ```

2. **Use iloc for DataFrame Indexing**
   ```python
   for train_idx, val_idx in cv.split(y_pred):
       # Use iloc if working with DataFrames
       if isinstance(y_pred, (pd.Series, pd.DataFrame)):
           y_train = y_pred.iloc[train_idx]
           y_val = y_pred.iloc[val_idx]
       else:
           y_train = y_pred[train_idx]
           y_val = y_pred[val_idx]
   ```

3. **Add Input Validation**
   ```python
   def optimize(self, y_pred, y_true, **kwargs):
       # Validate inputs
       assert len(y_pred) == len(y_true), "Prediction and label lengths must match"

       # Convert to consistent format
       y_pred = self._to_array(y_pred)
       y_true = self._to_array(y_true)

       # Validate indices if DataFrame
       if isinstance(y_pred, pd.Series):
           assert y_pred.index.equals(y_true.index), "Index mismatch between pred and true"
   ```

## Testing After Fix

### Test Cases

1. **With Integer Index**
   ```python
   y_pred = pd.Series([0.1, 0.2, 0.3, ...], index=range(1000))
   optimizer.optimize(y_pred, y_true)
   ```

2. **With Non-Integer Index**
   ```python
   y_pred = pd.Series([0.1, 0.2, 0.3, ...], index=range(100, 1100))
   optimizer.optimize(y_pred, y_true)
   ```

3. **With Non-Sequential Index**
   ```python
   idx = [1, 5, 10, 15, 20, ...]  # Non-sequential
   y_pred = pd.Series([0.1, 0.2, 0.3, ...], index=idx)
   optimizer.optimize(y_pred, y_true)
   ```

4. **With NumPy Arrays**
   ```python
   y_pred = np.array([0.1, 0.2, 0.3, ...])
   optimizer.optimize(y_pred, y_true)
   ```

### Validation
After fix, enable optimization and run full pipeline:
```bash
# Re-enable in config
sed -i 's/optimize: false/optimize: true/' configs/core/postprocessing_config.yaml

# Run model testing
uv run -m src.ml_framework.model_testing.main

# Verify:
# 1. No indexing errors
# 2. Optimization completes successfully
# 3. Best method is selected and logged
# 4. Pipeline completes within timeout
```

## Related Files

- [configs/core/postprocessing_config.yaml](configs/core/postprocessing_config.yaml) - Configuration
- `src/ml_framework/postprocessing/calibration_optimizer.py` - Likely location of bug
- `src/ml_framework/postprocessing/probability_calibrator.py` - Calibration implementation
- `src/ml_framework/model_testing/model_tester.py` - Calls calibration optimization
- [logs/feature_engineering.log](logs/feature_engineering.log) - May contain error traces

## Timeline

- **Discovered**: 2025-10-23
- **Workaround Applied**: 2025-10-23 (disabled optimization)
- **Priority**: Medium (workaround available, functionality not critical for MVP)
- **Target Fix**: Before production deployment

## Additional Notes

- Calibration optimization is a "nice to have" feature for automatic hyperparameter tuning
- Manual method selection (sigmoid) works well and is commonly used in practice
- The optimization was working in some scenarios but consistently fails with the current dataset
- May be related to specific data characteristics (18,330 samples with certain index structure)
- Consider adding unit tests for CalibrationOptimizer with various index types

## References

- Issue observed in model testing logs: `/tmp/model_test_with_registry.log`
- Similar issues in pandas/sklearn interoperability: https://github.com/scikit-learn/scikit-learn/issues/16079
- Best practices for CV with pandas: https://scikit-learn.org/stable/common_pitfalls.html#inconsistent-preprocessing
