# NBA Prediction Project v2 🏀

> **Note:** This project is currently a work in progress.

This is a comprehensive rework of my original NBA prediction machine learning project, focusing on improved reliability, engineering practices, and prediction accuracy.

## 🎯 Project Goals

### Enhanced Reliability & Open Source Focus
- Migration from SaaS solutions to open-source alternatives
  - Using GitHub as primary data store (previously Hopsworks)
  - Implementing MLFlow for experiment tracking (replacing Neptune)
  - MLFlow for model registry management (replacing Hopsworks)
- Redundant deployment architecture
- Container-based deployment strategy

### Software Engineering Improvements
- Object-Oriented Programming implementation
- Enhanced modularity and component isolation
- Comprehensive logging system
- Robust error handling
- Extensive testing infrastructure
- Dependency injection throughout
- Abstract interfaces for swappable implementations

### Architecture Highlights
- **Model-Aware Preprocessing**: Automatic preprocessing based on model family (tree models skip scaling, linear models get standardization)
- **Abstracted Model Registry**: Swappable registry backends (MLflow, custom, cloud services)
- **Clean Separation of Concerns**: Domain features (NBA stats) separate from model preprocessing (scaling/encoding)
- **Reproducible Inference**: Fitted preprocessors saved with models for identical transforms at inference
- **Production-Ready**: Versioning, staging workflows, batch inference, input validation

### Model Enhancements
- Expanded data collection through advanced scraping
- Integration of ELO scoring system
- Advanced feature engineering (rolling averages, streaks, opponent-adjusted stats)
- Model-specific preprocessing pipelines
- Increased experimental iterations
- Target accuracy improvement to ~65%

### Documentation & Transparency
- Detailed technical documentation in [docs/AI/](docs/AI/)
- Process discussion and methodology explanations
- Decision-making rationale
- Architecture guides and usage examples

### Deployment Strategy
- Multi-provider deployment architecture
- Cloud platform integration
- Redundant deployment systems
- Stage-based model deployment (Development → Staging → Production)

## 📚 Documentation

### Core Architecture
- [Preprocessing Architecture](docs/AI/preprocessing_architecture.md) - Model-aware preprocessing system
- [Model Registry & Inference](docs/AI/model_registry_and_inference.md) - Abstracted registry and inference pipeline
- [Core Framework Usage](docs/AI/core_framework_usage.md) - Dependency injection and design patterns
- [Interfaces](docs/AI/interfaces.md) - Abstract interfaces and implementations

### Technical Reference
- [Config Reference](docs/AI/config_reference.txt) - Configuration system overview
- [Directory Structure](docs/AI/directory_tree.txt) - Project layout

## 🏗️ Key Components

### Feature Engineering (`nba_app`)
- Domain-specific NBA feature creation (rolling averages, streaks, ELO ratings)
- Feature schema export for preprocessing
- Game-centric data merging

### Preprocessing (`ml_framework.preprocessing`)
- Model-specific transforms (scaling, encoding, imputation)
- Runtime fit/transform discipline (no saved scaled datasets)
- Preprocessor persistence with trained models

### Model Registry (`ml_framework.model_registry`)
- Abstract registry interface for swappable backends
- MLflow implementation with versioning and staging
- Automatic model flavor detection

### Inference (`ml_framework.inference`)
- Unified predictor interface
- Automatic preprocessing application
- Batch prediction support

### Model Training (`ml_framework.model_testing`)
- Support for 6 model families (XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression, PyTorch)
- Automated preprocessing pipeline
- Hyperparameter management
- Cross-validation and metrics tracking

## 🔄 Previous Version
The original version of this project can be found [here](https://github.com/cmunch1/nba-prediction).

