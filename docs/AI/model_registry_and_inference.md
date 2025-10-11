# Model Registry and Inference Architecture

**Date:** 2025-10-11
**Status:** Core Implementation Complete
**Related:** [Preprocessing Architecture](preprocessing_architecture.md)

---

## Overview

This document describes the abstracted model registry and inference architecture that enables swappable registry backends (MLflow, custom, cloud services) while providing a consistent interface for model persistence and retrieval.

## Design Principles

### 1. **Abstraction via Dependency Injection**

```
┌──────────────────────────────────────────────────────────────┐
│              BaseModelRegistry (Abstract)                     │
│  Interface for: save, load, list, delete, versioning         │
└──────────────────────────────────────────────────────────────┘
                            ↑
                            │ implements
            ┌───────────────┴───────────────┐
            │                               │
┌───────────────────────┐    ┌──────────────────────────┐
│ MLflowModelRegistry   │    │ CustomModelRegistry      │
│ (Concrete)            │    │ (Future)                 │
└───────────────────────┘    └──────────────────────────┘
```

- Model registry is injected into components (trainers, predictors)
- Easy to swap implementations without changing client code
- Consistent API across all registry types

### 2. **Model + Preprocessor Persistence**

Models are saved with their complete inference pipeline:
- **Model object** - Trained weights and architecture
- **Preprocessor artifact** - Fitted transformations (scaling, encoding)
- **Metadata** - Hyperparameters, metrics, feature names
- **Signature** - Input/output schema for validation

### 3. **Clean Inference Path**

```
Input Data → Preprocessor.transform() → Model.predict() → Predictions
```

The preprocessor is loaded with the model and automatically applied during inference.

---

## Architecture Components

### 1. Base Model Registry ([base_model_registry.py](../../src/ml_framework/model_registry/base_model_registry.py))

**Abstract interface** defining registry operations:

```python
class BaseModelRegistry(ABC):
    @abstractmethod
    def save_model(self, model, model_name, preprocessor_artifact=None, ...):
        """Save model with artifacts to registry."""

    @abstractmethod
    def load_model(self, model_identifier):
        """Load model and artifacts from registry."""

    @abstractmethod
    def list_models(self, model_name=None, tags=None):
        """List available models."""

    @abstractmethod
    def delete_model(self, model_identifier):
        """Delete a model from registry."""

    @abstractmethod
    def transition_model_stage(self, model_identifier, stage):
        """Move model to different stage (staging, production, etc.)."""

    @abstractmethod
    def get_model_by_stage(self, model_name, stage):
        """Load model in specific stage (e.g., production)."""
```

### 2. MLflow Model Registry ([mlflow_model_registry.py](../../src/ml_framework/model_registry/mlflow_model_registry.py))

**Concrete implementation** using MLflow Model Registry:

**Key Features:**
- Automatic model flavor detection (XGBoost, LightGBM, CatBoost, PyTorch, Sklearn)
- Preprocessor saved as artifact alongside model
- Version management and staging (Development → Staging → Production)
- Integration with MLflow tracking for full lineage

**Usage:**
```python
from ml_framework.model_registry import MLflowModelRegistry

registry = MLflowModelRegistry(config, app_logger, error_handler)

# Save model with preprocessor
model_uri = registry.save_model(
    model=trained_model,
    model_name="nba_win_predictor",
    preprocessor_artifact=results.preprocessing_artifact,
    metadata={'fold': 0, 'accuracy': 0.85},
    tags={'model_type': 'XGBoost', 'version': 'v1.0'}
)

# Load model
model_data = registry.load_model("models:/nba_win_predictor/Production")
```

### 3. Model Predictor ([model_predictor.py](../../src/ml_framework/inference/model_predictor.py))

**Unified inference interface** that works with any registry:

**Key Features:**
- Registry-agnostic (works with MLflow, custom, etc.)
- Automatic preprocessing application
- Batch prediction support
- Input validation
- Probability and class predictions

**Usage:**
```python
from ml_framework.inference import ModelPredictor
from ml_framework.model_registry import MLflowModelRegistry

# Initialize with registry
registry = MLflowModelRegistry(config, app_logger, error_handler)
predictor = ModelPredictor(config, app_logger, error_handler, registry)

# Load model
predictor.load_model("models:/nba_win_predictor/Production")

# Make predictions (preprocessing applied automatically)
probabilities = predictor.predict(X_new, return_probabilities=True)
classes = predictor.predict(X_new, return_probabilities=False)

# Batch predictions for large datasets
predictions = predictor.predict_batch(X_large, batch_size=1000)

# Validate input features
validation = predictor.validate_input(X_new)
if not validation['valid']:
    print(f"Missing features: {validation['missing_features']}")
```

---

## Complete Pipeline Flow

### Training & Registration

```
1. Feature Engineering (nba_app)
   └─> Domain features → training.csv

2. Preprocessing (ml_framework)
   └─> Model-specific transforms

3. Training (ml_framework.trainers)
   ├─> Fit preprocessor on training data
   ├─> Transform train/validation data
   ├─> Train model
   └─> Store preprocessing_artifact in results

4. Model Registry (ml_framework.model_registry)
   ├─> Save model object
   ├─> Save preprocessor artifact
   ├─> Save metadata (params, metrics)
   ├─> Register model name/version
   └─> Return model URI

5. MLflow Tracking
   └─> Full experiment lineage preserved
```

### Inference

```
1. Load Model (via ModelPredictor)
   ├─> Load model from registry
   ├─> Load preprocessor artifact
   └─> Restore preprocessor state

2. Validate Input
   └─> Check features match expected schema

3. Transform Input
   └─> Apply fitted preprocessor transforms

4. Predict
   └─> Model makes predictions on transformed data

5. Return Results
   └─> Probabilities or class predictions
```

---

## MLflow Integration Details

### Model Saving

```python
# In trainer after training completes
registry.save_model(
    model=results.model,
    model_name="nba_win_predictor_xgboost",
    preprocessor_artifact=results.preprocessing_artifact,
    signature=infer_signature(X_train, y_train),
    input_example=X_train.head(1),
    metadata={
        'fold': fold,
        'accuracy': metrics.accuracy,
        'auc': metrics.auc,
        'model_params': results.model_params
    },
    tags={
        'model_family': 'tree',
        'framework': 'XGBoost',
        'season': '2024'
    }
)
```

**What Gets Saved:**
1. **Model artifact** - Under `runs:/<run_id>/model`
2. **Preprocessor** - Under `runs:/<run_id>/artifacts/preprocessor_artifact.pkl`
3. **Metadata** - As MLflow params
4. **Tags** - As MLflow tags
5. **Signature** - Model input/output schema
6. **Input example** - Sample for validation

### Model Loading

```python
# Load by stage (recommended for production)
model_data = registry.load_model("models:/nba_win_predictor/Production")

# Load by version
model_data = registry.load_model("models:/nba_win_predictor/3")

# Load by run ID
model_data = registry.load_model("runs:/abc123/model")
```

**What Gets Loaded:**
```python
{
    'model': <loaded model object>,
    'preprocessor_artifact': {
        'preprocessor': <fitted ColumnTransformer>,
        'model_name': 'XGBoost',
        'feature_names_in': [...],
        'feature_names_out': [...]
    },
    'metadata': {
        'params': {...},
        'metrics': {...},
        'tags': {...}
    },
    'run_id': 'abc123'
}
```

### Model Versioning & Staging

```python
# Transition to staging
registry.transition_model_stage(
    model_identifier="nba_win_predictor/3",
    stage="Staging"
)

# Test in staging
staging_model = registry.get_model_by_stage(
    model_name="nba_win_predictor",
    stage="Staging"
)

# Promote to production
registry.transition_model_stage(
    model_identifier="nba_win_predictor/3",
    stage="Production"
)

# Load production model
prod_model = registry.get_model_by_stage(
    model_name="nba_win_predictor",
    stage="Production"
)
```

---

## Adding Custom Registry Implementations

To add a new registry backend (e.g., AWS SageMaker, Azure ML, custom file-based):

### Step 1: Create Concrete Implementation

```python
# src/ml_framework/model_registry/custom_model_registry.py

from .base_model_registry import BaseModelRegistry

class CustomModelRegistry(BaseModelRegistry):
    def __init__(self, config, app_logger, error_handler):
        super().__init__(config, app_logger, error_handler)
        # Initialize custom backend

    def save_model(self, model, model_name, preprocessor_artifact=None, ...):
        # Implement custom save logic
        pass

    def load_model(self, model_identifier):
        # Implement custom load logic
        pass

    # Implement other abstract methods...
```

### Step 2: Register in __init__.py

```python
# src/ml_framework/model_registry/__init__.py

from .base_model_registry import BaseModelRegistry
from .mlflow_model_registry import MLflowModelRegistry
from .custom_model_registry import CustomModelRegistry

__all__ = ['BaseModelRegistry', 'MLflowModelRegistry', 'CustomModelRegistry']
```

### Step 3: Use via Dependency Injection

```python
# No code changes needed in trainers or predictors!

# Just inject different registry implementation
registry = CustomModelRegistry(config, app_logger, error_handler)
predictor = ModelPredictor(config, app_logger, error_handler, registry)
```

---

## Usage Examples

### Example 1: Training with Model Registry

```python
from ml_framework.model_testing.trainers import XGBoostTrainer
from ml_framework.preprocessing import Preprocessor
from ml_framework.model_registry import MLflowModelRegistry

# Initialize components
preprocessor = Preprocessor(config, app_logger, app_file_handler, error_handler)
trainer = XGBoostTrainer(config, app_logger, error_handler)
registry = MLflowModelRegistry(config, app_logger, error_handler)

# Train model with preprocessing
results = trainer.train(
    X_train, y_train, X_val, y_val,
    fold=0,
    model_params=params,
    results=ModelTrainingResults(X_train.shape),
    preprocessor=preprocessor
)

# Save to registry
model_uri = registry.save_model(
    model=results.model,
    model_name="nba_win_predictor",
    preprocessor_artifact=results.preprocessing_artifact,
    metadata={'accuracy': 0.85, 'auc': 0.90}
)

print(f"Model saved: {model_uri}")
```

### Example 2: Inference

```python
from ml_framework.inference import ModelPredictor
from ml_framework.model_registry import MLflowModelRegistry

# Initialize
registry = MLflowModelRegistry(config, app_logger, error_handler)
predictor = ModelPredictor(config, app_logger, error_handler, registry)

# Load production model
predictor.load_model("models:/nba_win_predictor/Production")

# Get model info
info = predictor.get_model_info()
print(f"Loaded: {info['model_type']}")
print(f"Has preprocessor: {info['has_preprocessor']}")

# Validate input
validation = predictor.validate_input(X_new)
if validation['valid']:
    # Make predictions
    win_probabilities = predictor.predict(X_new, return_probabilities=True)
    print(f"Home team win probability: {win_probabilities[0]:.2%}")
else:
    print(f"Invalid input: {validation}")
```

### Example 3: Batch Inference

```python
# Load model
predictor.load_model("models:/nba_win_predictor/Production")

# Process large dataset in batches
predictions = predictor.predict_batch(
    X_large_dataset,
    batch_size=1000,
    return_probabilities=True
)

# Save predictions
results_df = pd.DataFrame({
    'game_id': X_large_dataset['game_id'],
    'home_win_probability': predictions
})
results_df.to_csv('predictions.csv', index=False)
```

### Example 4: Model Staging Workflow

```python
# Train new model version
model_uri = registry.save_model(
    model=new_model,
    model_name="nba_win_predictor",
    preprocessor_artifact=preprocessor_artifact,
    tags={'version': 'v2.0', 'improvement': 'new features'}
)

# Automatically registered as version 4
# Transition to staging for testing
registry.transition_model_stage("nba_win_predictor/4", "Staging")

# Test in staging environment
staging_predictor = ModelPredictor(config, app_logger, error_handler, registry)
staging_predictor.load_model("models:/nba_win_predictor/Staging")

# Run validation tests
test_accuracy = evaluate_model(staging_predictor, X_test, y_test)

if test_accuracy > threshold:
    # Promote to production
    registry.transition_model_stage("nba_win_predictor/4", "Production")
    print("New model deployed to production!")
else:
    # Archive and keep old production model
    registry.transition_model_stage("nba_win_predictor/4", "Archived")
    print("New model did not meet threshold, keeping current production model")
```

---

## Benefits

### ✅ Registry Abstraction
- Swap implementations (MLflow → Custom → Cloud) without code changes
- Test with different registries easily
- Consistent API across all backends

### ✅ Complete Inference Pipeline
- Model + preprocessor saved together
- Reproducible predictions (same transforms as training)
- No manual preprocessing at inference time

### ✅ MLflow Integration
- Full experiment tracking lineage
- Model versioning and staging
- Built-in model signatures and validation
- UI for model exploration

### ✅ Production Ready
- Stage-based deployment (Development → Staging → Production)
- Batch inference support
- Input validation
- Error handling and logging

---

## File Structure

```
src/ml_framework/
├── model_registry/
│   ├── __init__.py
│   ├── base_model_registry.py          # Abstract interface
│   └── mlflow_model_registry.py        # MLflow implementation
└── inference/
    ├── __init__.py
    └── model_predictor.py               # Unified predictor
```

---

## Next Steps (Remaining Work)

1. **Integration with model_tester**
   - Instantiate registry in model_tester
   - Save models after training via registry
   - Add config flag to enable/disable registry

2. **Feature engineering schema export**
   - Update feature_engineering/main.py
   - Export schema after feature engineering

3. **Testing**
   - Test save/load cycle
   - Test inference with different model types
   - Test staging workflow

4. **Documentation**
   - Add API reference
   - Create deployment guide
   - Document staging workflow

---

## References

- [Preprocessing Architecture](preprocessing_architecture.md)
- [Core Framework Usage](core_framework_usage.md)
- [MLflow Model Registry Docs](https://mlflow.org/docs/latest/model-registry.html)

---

## Changelog

**2025-10-11** - Initial implementation
- Created BaseModelRegistry abstract interface
- Implemented MLflowModelRegistry
- Created ModelPredictor with registry abstraction
- Full preprocessor artifact integration
