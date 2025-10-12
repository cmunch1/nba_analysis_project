# Project Documentation

This directory contains comprehensive documentation for the NBA Prediction Project v2.

## 📚 Documentation Index

### Core Architecture Documentation

Located in [AI/](AI/) subdirectory:

- **[preprocessing_architecture.md](AI/preprocessing_architecture.md)** - Complete guide to the model-aware preprocessing system
  - Separation of domain features from model preprocessing
  - Config-driven preprocessing profiles
  - Runtime fit/transform discipline
  - Preprocessor persistence

- **[model_registry_and_inference.md](AI/model_registry_and_inference.md)** - Model registry and inference architecture
  - Abstracted registry interface
  - MLflow implementation
  - ModelPredictor for unified inference
  - Staging workflows

- **[core_framework_usage.md](AI/core_framework_usage.md)** - Core framework patterns
  - Dependency injection
  - Configuration management
  - Logging and error handling
  - Design patterns

- **[interfaces.md](AI/interfaces.md)** - Abstract interfaces and implementations
  - Base classes
  - Implementation guidelines

### Technical Reference

Located in [AI/](AI/) subdirectory:

- **[config_reference.txt](AI/config_reference.txt)** - Configuration system overview
- **[config_tree.txt](AI/config_tree.txt)** - Configuration hierarchy
- **[directory_tree.txt](AI/directory_tree.txt)** - Project structure

### Data Documentation

Located in [data/](data/) subdirectory:

- **[nba-boxscore-data-dictionary.md](data/nba-boxscore-data-dictionary.md)** - NBA data schema

### Commentary & Process

- **[commentary/Webscraping.md](commentary/Webscraping.md)** - Web scraping implementation notes
- **[future_refactors.md](future_refactors.md)** - Planned improvements
- **[validation_scraper_implementation.md](validation_scraper_implementation.md)** - Scraper validation approach

---

## 🔄 Development Pipeline

### 1. Data Collection (`nba_app.webscraping`)
- Scrape NBA boxscores and upcoming games from NBA.com
- Validate scraped data
- Store in GitHub as primary data store

### 2. Data Processing (`nba_app.data_processing`)
- Clean and normalize scraped data
- Handle missing values
- Create consistent schema

### 3. Feature Engineering (`nba_app.feature_engineering`)
- Domain-specific NBA features:
  - Rolling averages (FG%, points, rebounds, etc.)
  - Win/loss streaks
  - Home/visitor patterns
  - ELO ratings
  - Opponent-adjusted statistics
- Export feature schema for preprocessing
- Game-centric data format

### 4. Preprocessing (`ml_framework.preprocessing`)
- Model-specific transforms:
  - Tree models: No scaling (XGBoost, LightGBM, CatBoost)
  - Linear models: Standardization (LogisticRegression)
  - Neural networks: Standardization + outlier handling (PyTorch)
- Runtime fit/transform (no saved scaled datasets)
- Preprocessor persisted with trained models

### 5. Model Training (`ml_framework.model_testing`)
- Support for 6 model families:
  - XGBoost
  - LightGBM
  - CatBoost
  - RandomForest
  - LogisticRegression
  - PyTorch neural networks
- Automated preprocessing pipeline
- Hyperparameter management
- Cross-validation
- Metrics tracking via MLflow

### 6. Model Registry (`ml_framework.model_registry`)
- Save trained models with preprocessing artifacts
- Version management
- Stage-based deployment:
  - Development → Staging → Production
- MLflow integration with swappable backend

### 7. Inference (`ml_framework.inference`)
- Load models from registry
- Automatic preprocessing application
- Batch prediction support
- Input validation

### 8. Deployment (Future)
- API for model access
- Dashboard for predictions
- Multi-provider architecture

### 9. Monitoring (Future)
- Model performance tracking
- API access monitoring
- Drift detection

---

## 🏗️ Project Structure

```
nba_analysis_project/
├── configs/              # YAML configuration files
│   └── core/            # Core configurations
│       ├── preprocessing_config.yaml
│       └── models/      # Model-specific configs
├── data/                # Data storage
│   ├── raw/            # Scraped data
│   ├── processed/      # Cleaned data + feature_schema.json
│   └── training/       # Training datasets
├── docs/               # Documentation (you are here)
│   └── AI/            # Technical architecture docs
├── logs/              # Application logs
├── mlruns/            # MLflow tracking
├── notebooks/         # Exploratory analysis
├── src/               # Source code
│   ├── ml_framework/  # ML infrastructure
│   │   ├── core/                # Core utilities (config, logging, errors)
│   │   ├── preprocessing/       # Model-specific preprocessing
│   │   ├── model_testing/       # Training and evaluation
│   │   ├── model_registry/      # Model persistence (abstracted)
│   │   ├── inference/           # Prediction pipeline
│   │   └── visualization/       # Charts and plots
│   └── nba_app/       # NBA-specific logic
│       ├── webscraping/         # Data collection
│       ├── data_processing/     # Data cleaning
│       └── feature_engineering/ # Domain features
└── tests/             # Unit and integration tests
```

---

## 🎯 Design Principles

### Separation of Concerns
- **Domain layer** (`nba_app`): Basketball-specific logic
- **ML layer** (`ml_framework`): Reusable ML infrastructure
- **Configuration layer**: YAML-driven behavior

### Dependency Injection
- Abstract interfaces for all major components
- Concrete implementations injected at runtime
- Easy to swap implementations (e.g., different model registries)

### Clean Architecture
- Domain features separate from model preprocessing
- Preprocessing separate from model training
- Training separate from inference

### Reproducibility
- Fitted preprocessors saved with models
- Configuration tracked with experiments
- Full lineage in MLflow

### Production Ready
- Comprehensive error handling
- Structured logging throughout
- Input validation
- Staging workflows
- Batch inference support

---

## 📖 Getting Started

1. **Read core architecture docs first:**
   - [preprocessing_architecture.md](AI/preprocessing_architecture.md)
   - [model_registry_and_inference.md](AI/model_registry_and_inference.md)

2. **Understand the framework:**
   - [core_framework_usage.md](AI/core_framework_usage.md)

3. **Review configurations:**
   - [config_reference.txt](AI/config_reference.txt)

4. **Explore the code:**
   - Start with `src/nba_app/` for NBA-specific logic
   - Then `src/ml_framework/` for ML infrastructure
