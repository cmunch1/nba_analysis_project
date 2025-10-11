# Preprocessing Architecture Implementation

**Date:** 2025-10-11
**Status:** Core Implementation Complete
**Next Steps:** Update remaining trainers, create inference module

---

## Overview

This document describes the implementation of a clean, extensible preprocessing architecture that separates domain-specific feature engineering from model-specific preprocessing transformations.

## Design Principles

### 1. **Separation of Concerns**

```
┌─────────────────────────────────────────────────────────────┐
│              nba_app (Domain Layer)                          │
│  Responsible for: Basketball-specific features               │
│  - Rolling averages (FG%, points, rebounds, etc.)           │
│  - Win/loss streaks                                          │
│  - Home/visitor patterns                                     │
│  - ELO ratings                                               │
│  - Opponent-adjusted statistics                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
              (feature_schema.json + training.csv)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           ml_framework (Model Layer)                         │
│  Responsible for: Algorithm-specific transforms             │
│  - Scaling/normalization (for linear models, neural nets)   │
│  - Categorical encoding (one-hot, ordinal)                   │
│  - Imputation (when model doesn't handle missing)           │
│  - Feature selection (variance, mutual info)                │
└─────────────────────────────────────────────────────────────┘
```

### 2. **Runtime Transforms (No Saved Datasets)**

- Preprocessing transforms are applied **in-memory** during training/inference
- **Fit on training data only**, transform on validation/test data
- **Never save scaled/transformed datasets** to disk
- Keep one canonical feature-engineered dataset from `nba_app`

### 3. **Reproducible Inference**

- Fitted preprocessor is **persisted alongside trained model**
- At inference time: load model + preprocessor → transform new data → predict
- **Never refit preprocessor** at inference time

---

## Implementation Details

### File Structure

```
src/
├── nba_app/
│   └── feature_engineering/
│       ├── feature_engineer.py          # Domain feature engineering
│       ├── feature_schema.py            # Feature schema dataclass
│       └── main.py                      # Entry point
├── ml_framework/
│   ├── preprocessing/
│   │   ├── base_preprocessor.py         # Abstract interface
│   │   └── preprocessor.py              # Concrete implementation
│   ├── model_testing/
│   │   └── trainers/
│   │       ├── base_trainer.py          # Updated with preprocessor param
│   │       └── xgboost_trainer.py       # Example trainer implementation
│   └── framework/
│       └── data_classes/
│           ├── training.py              # Updated with preprocessing_artifact
│           └── preprocessing.py         # PreprocessingResults
└── configs/
    └── core/
        └── preprocessing_config.yaml    # Model-specific preprocessing profiles
```

---

## Key Components

### 1. Feature Schema ([feature_schema.py](../../src/nba_app/feature_engineering/feature_schema.py))

**Purpose:** Define and track feature types produced by feature engineering.

```python
@dataclass
class FeatureSchema:
    """Schema describing features and their types."""
    numeric_features: List[str]
    categorical_features: List[str]
    binary_features: List[str]
    ordinal_features: List[str]
    datetime_features: List[str]
    target_column: str
    game_id_column: str
    feature_descriptions: Dict[str, str]
    feature_groups: Dict[str, List[str]]
```

**Key Methods:**
- `from_dataframe()` - Auto-infer schema from DataFrame
- `to_json()` - Export to JSON for preprocessor consumption
- `validate_dataframe()` - Ensure data conforms to schema
- `summary()` - Human-readable schema summary

**Usage:**
```python
# In feature_engineer.py
schema = FeatureSchema.from_dataframe(
    df,
    target_column='home_team_won',
    game_id_column='game_id',
    exclude_columns=['game_id', 'team_id', 'date']
)
schema.to_json(Path('data/processed/feature_schema.json'))
```

---

### 2. Enhanced Preprocessing Config ([preprocessing_config.yaml](../../configs/core/preprocessing_config.yaml))

**Purpose:** Define model-family specific preprocessing profiles.

**Structure:**
```yaml
preprocessing:
  default:
    numerical:
      imputation: null
      scaling: null
      handling_outliers: null
    categorical:
      encoding: null
      handling_missing: null

  model_specific:
    # Tree models - no scaling needed
    XGBoost:
      numerical:
        imputation: null  # XGBoost handles missing natively
        scaling: null     # Tree models don't need scaling
      categorical:
        encoding: ordinal

    # Linear models - need scaling
    LogisticRegression:
      numerical:
        imputation: mean
        scaling: standard  # Critical for linear models
      categorical:
        encoding: one_hot

    # Neural networks - need scaling + outlier handling
    PyTorch:
      numerical:
        imputation: mean
        scaling: standard
        handling_outliers: clip
      categorical:
        encoding: one_hot
```

---

### 3. Preprocessor with Persistence ([preprocessor.py](../../src/ml_framework/preprocessing/preprocessor.py))

**Purpose:** Apply model-specific transforms and persist fitted state.

**New Methods:**

#### `get_preprocessor_artifact()` → Dict
Packages fitted preprocessor for saving with model:
```python
artifact = preprocessor.get_preprocessor_artifact()
# Returns:
# {
#   'preprocessor': fitted ColumnTransformer,
#   'model_name': 'XGBoost',
#   'feature_names_in': [...],
#   'feature_names_out': [...],
#   'preprocessing_results': PreprocessingResults(...)
# }
```

#### `load_preprocessor_artifact(artifact: Dict)`
Restores preprocessor from saved state for inference:
```python
preprocessor.load_preprocessor_artifact(artifact)
X_new_transformed = preprocessor.transform(X_new)
```

#### `is_fitted()` → bool
Check if preprocessor is ready for transform.

---

### 4. Updated Trainers ([xgboost_trainer.py](../../src/ml_framework/model_testing/trainers/xgboost_trainer.py))

**Purpose:** Apply preprocessing during training and store artifact with model.

**Updated Signature:**
```python
def train(self, X_train, y_train, X_val, y_val, fold, model_params, results,
          preprocessor: Optional[BasePreprocessor] = None) -> ModelTrainingResults:
```

**Preprocessing Flow:**
```python
if preprocessor:
    # Fit on training data and transform
    X_train_transformed, prep_results = preprocessor.fit_transform(
        X_train, y_train, model_name='XGBoost'
    )

    # Transform validation data (don't refit!)
    X_val_transformed = preprocessor.transform(X_val)

    # Store artifact in results
    results.preprocessing_artifact = preprocessor.get_preprocessor_artifact()
else:
    X_train_transformed = X_train
    X_val_transformed = X_val

# Train model on transformed data
model = train_model(X_train_transformed, y_train)
```

---

### 5. Model Training Results ([training.py](../../src/ml_framework/framework/data_classes/training.py))

**Purpose:** Store preprocessing artifact alongside model.

**New Field:**
```python
@dataclass
class ModelTrainingResults:
    # ... existing fields ...
    preprocessing_artifact: Optional[Dict[str, Any]] = None  # Fitted preprocessor
```

---

## Complete Pipeline Flow

### Training Flow

```
1. Data Processing (nba_app.data_processing)
   └─> Clean raw scraped data → data/processed/*.csv

2. Feature Engineering (nba_app.feature_engineering)
   ├─> Engineer domain features (rolling avgs, ELO, streaks)
   ├─> Export feature_schema.json
   └─> Save to data/training/training.csv

3. Model Training (ml_framework.model_testing)
   └─> For each model:
       ├─> Load training data
       ├─> Create Preprocessor instance
       ├─> For each CV fold:
       │   ├─> preprocessor.fit_transform(X_train) → X_train_scaled
       │   ├─> preprocessor.transform(X_val) → X_val_scaled
       │   ├─> train_model(X_train_scaled, y_train)
       │   └─> validate(X_val_scaled, y_val)
       ├─> Final fit on full training set
       └─> Save model + preprocessor artifact
```

### Inference Flow (Future)

```
1. Load Artifacts
   ├─> Load trained model from disk
   └─> Load fitted preprocessor from artifact

2. Prepare Features
   └─> Load new game features (from feature engineering)

3. Transform & Predict
   ├─> X_new_transformed = preprocessor.transform(X_new)
   └─> predictions = model.predict(X_new_transformed)
```

---

## Configuration by Model Family

### Tree-Based Models

**Models:** XGBoost, LightGBM, CatBoost, RandomForest, HistGradientBoosting

**Why no scaling?**
- Tree models split on thresholds, not distances
- Scale-invariant by design
- Scaling provides no benefit and adds complexity

**Why handle missing natively?**
- Modern tree implementations (XGBoost, LightGBM, HGB, CatBoost) treat missing as information
- Can learn optimal direction for missing values during training

**Configuration:**
```yaml
XGBoost:
  numerical:
    imputation: null      # Handles missing natively
    scaling: null         # Doesn't benefit from scaling
  categorical:
    encoding: ordinal     # Works well with ordinal encoding
```

### Linear Models

**Models:** LogisticRegression, LinearSVM, ElasticNet

**Why scaling required?**
- Gradient descent converges faster with standardized features
- Regularization (L1/L2) penalizes features equally when scaled
- Prevents features with large ranges from dominating

**Why one-hot encoding?**
- Linear models need explicit feature representation
- One-hot preserves no ordinal relationship assumption

**Configuration:**
```yaml
LogisticRegression:
  numerical:
    imputation: mean      # Can't handle missing
    scaling: standard     # Critical for convergence
  categorical:
    encoding: one_hot     # Explicit feature representation
  feature_selection:
    method: variance_threshold  # Remove low-variance features
```

### Neural Networks

**Models:** PyTorch, TensorFlow, Keras

**Why scaling + outlier handling?**
- Gradient descent requires normalized inputs
- Extreme values can cause gradient explosion/vanishing
- Clipping outliers improves training stability

**Configuration:**
```yaml
PyTorch:
  numerical:
    imputation: mean           # Neural nets need complete data
    scaling: standard          # Required for gradient descent
    handling_outliers: clip    # Prevent gradient issues
  categorical:
    encoding: one_hot          # Or use embeddings in model architecture
```

---

## Implementation Status

### ✅ Completed

1. **Enhanced Configuration**
   - [x] Extended [preprocessing_config.yaml](../../configs/core/preprocessing_config.yaml) with model profiles
   - [x] Added documentation and design principles
   - [x] Backward compatibility with legacy config keys

2. **Feature Schema Management**
   - [x] Created [feature_schema.py](../../src/nba_app/feature_engineering/feature_schema.py)
   - [x] Auto-inference from DataFrames
   - [x] Validation and summary methods
   - [x] JSON export/import

3. **Feature Engineering Updates**
   - [x] Added schema export to [feature_engineer.py](../../src/nba_app/feature_engineering/feature_engineer.py)
   - [x] Feature grouping (rolling avgs, streaks, ELO, temporal)
   - [x] Optional export parameter

4. **Preprocessor Enhancements**
   - [x] Added `get_preprocessor_artifact()` to [preprocessor.py](../../src/ml_framework/preprocessing/preprocessor.py)
   - [x] Added `load_preprocessor_artifact()`
   - [x] Added `is_fitted()` helper
   - [x] State tracking for persistence

5. **Trainer Updates**
   - [x] Updated [base_trainer.py](../../src/ml_framework/model_testing/trainers/base_trainer.py) interface
   - [x] Updated [xgboost_trainer.py](../../src/ml_framework/model_testing/trainers/xgboost_trainer.py) as reference
   - [x] Preprocessor integration with fit/transform discipline
   - [x] Artifact storage in results

6. **Data Classes**
   - [x] Added `preprocessing_artifact` to [training.py](../../src/ml_framework/framework/data_classes/training.py)

### 🚧 Remaining Work

1. **Update Remaining Trainers** (follow XGBoost pattern)
   - [ ] [lightgbm_trainer.py](../../src/ml_framework/model_testing/trainers/lightgbm_trainer.py)
   - [ ] [catboost_trainer.py](../../src/ml_framework/model_testing/trainers/catboost_trainer.py)
   - [ ] [sklearn_trainer.py](../../src/ml_framework/model_testing/trainers/sklearn_trainer.py)
   - [ ] [pytorch_trainer.py](../../src/ml_framework/model_testing/trainers/pytorch_trainer.py)

2. **Model Persistence**
   - [ ] Update model saving to persist preprocessor artifact
   - [ ] Create helper to save model + preprocessor together
   - [ ] Implement versioning for artifacts

3. **Inference Module**
   - [ ] Create `ml_framework/inference/predictor.py`
   - [ ] Implement model + preprocessor loading
   - [ ] Implement transform + predict pipeline

4. **Integration**
   - [ ] Update [feature_engineering/main.py](../../src/nba_app/feature_engineering/main.py) to export schema
   - [ ] Update model testing to instantiate and pass preprocessor
   - [ ] Add configuration flag to enable/disable preprocessing

5. **Testing**
   - [ ] Test preprocessing with each model family
   - [ ] Verify artifact persistence/loading
   - [ ] Test inference pipeline end-to-end

6. **Documentation**
   - [ ] Add usage examples to docstrings
   - [ ] Create inference tutorial
   - [ ] Update project README

---

## Usage Examples

### Training with Preprocessing

```python
from ml_framework.preprocessing.preprocessor import Preprocessor
from ml_framework.model_testing.trainers import XGBoostTrainer

# Initialize preprocessor
preprocessor = Preprocessor(config, app_logger, app_file_handler, error_handler)

# Initialize trainer
trainer = XGBoostTrainer(config, app_logger, error_handler)

# Train with preprocessing (happens inside trainer.train())
results = trainer.train(
    X_train, y_train,
    X_val, y_val,
    fold=0,
    model_params=params,
    results=ModelTrainingResults(X_train.shape),
    preprocessor=preprocessor  # Preprocessing applied automatically
)

# Save model + preprocessor artifact together
artifacts = {
    'model': results.model,
    'preprocessor_artifact': results.preprocessing_artifact,
    'model_name': 'XGBoost',
    'timestamp': datetime.now()
}
save_artifacts(artifacts, 'model_artifacts.pkl')
```

### Inference with Preprocessor (Future)

```python
from ml_framework.inference.predictor import ModelPredictor

# Load model + preprocessor
predictor = ModelPredictor(config, app_logger, error_handler)
predictor.load_artifacts('model_artifacts.pkl')

# Make predictions (preprocessing applied automatically)
predictions = predictor.predict(X_new)
```

### Feature Engineering with Schema Export

```python
from nba_app.feature_engineering import FeatureEngineer

feature_engineer = FeatureEngineer(config, app_logger)

# Engineer features and export schema
df_engineered = feature_engineer.engineer_features(
    df_processed,
    export_schema=True  # Saves to data/processed/feature_schema.json
)
```

---

## Benefits

### ✅ Clean Architecture
- Domain logic (NBA features) separated from model logic (scaling/encoding)
- Each layer has clear responsibility
- Easy to modify without breaking other parts

### ✅ Flexibility
- Easy to experiment with different preprocessing strategies
- Config-driven: change preprocessing without code changes
- Model-aware: each model gets appropriate transforms

### ✅ Reproducibility
- Fitted preprocessor saved with model
- Inference uses exact same transforms as training
- No risk of train/test preprocessing mismatch

### ✅ Type Safety
- Feature schema prevents type confusion
- Validates data conforms to expected structure
- Catches issues early in pipeline

### ✅ Performance
- No unnecessary transforms (tree models skip scaling)
- Runtime transforms avoid disk I/O
- Single source of truth for features

---

## Common Patterns

### Pattern 1: No Preprocessing (Tree Models)

```python
# Training - no preprocessor needed
results = xgboost_trainer.train(
    X_train, y_train, X_val, y_val,
    fold, params, results,
    preprocessor=None  # Tree models don't need preprocessing
)
```

### Pattern 2: Scaling for Linear Models

```python
# Preprocessor automatically applies standard scaling
results = logistic_trainer.train(
    X_train, y_train, X_val, y_val,
    fold, params, results,
    preprocessor=preprocessor  # Applies StandardScaler based on config
)
```

### Pattern 3: Custom Preprocessing

```yaml
# Add to preprocessing_config.yaml
CustomModel:
  numerical:
    imputation: median
    scaling: robust        # Robust to outliers
    handling_outliers: winsorize
  feature_selection:
    method: mutual_info    # Information-based selection
    params:
      k: 50
```

---

## Troubleshooting

### Issue: Preprocessor not fitted error

**Cause:** Calling `transform()` before `fit_transform()`

**Solution:** Ensure `fit_transform()` is called on training data first:
```python
# Fit on training data
X_train_t, _ = preprocessor.fit_transform(X_train, y_train, model_name='XGBoost')

# Then transform validation data
X_val_t = preprocessor.transform(X_val)
```

### Issue: Feature names mismatch

**Cause:** Inference data has different columns than training data

**Solution:** Use feature schema to validate:
```python
schema = FeatureSchema.from_json('data/processed/feature_schema.json')
validation_results = schema.validate_dataframe(X_new)

if validation_results['missing_columns']:
    print(f"Missing: {validation_results['missing_columns']}")
```

### Issue: Tree model performance degraded

**Cause:** Accidentally applied scaling to tree model

**Solution:** Check config and set scaling to `null`:
```yaml
XGBoost:
  numerical:
    scaling: null  # Tree models don't benefit from scaling
```

---

## References

- [Core Framework Usage Guide](core_framework_usage.md)
- [Feature Engineering Documentation](../../src/nba_app/feature_engineering/README.md)
- [Model Testing Documentation](../../src/ml_framework/model_testing/README.md)
- [Configuration Reference](config_reference.txt)

---

## Changelog

**2025-10-11** - Initial implementation
- Created feature schema management
- Enhanced preprocessing config with model profiles
- Added preprocessor persistence
- Updated XGBoostTrainer as reference implementation
- Updated BaseTrainer interface
