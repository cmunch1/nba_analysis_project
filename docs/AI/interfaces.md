# NBA Analysis Platform - Interfaces & Architecture

## Overview
This document provides a comprehensive overview of all abstract base classes and interfaces in the NBA Analysis Platform. The project follows a three-tier architecture separating reusable ML infrastructure from domain-specific implementations.

## Platform Core - Reusable ML Infrastructure

### Core Infrastructure
Located in `src/ml_framework/core/` - Generic application infrastructure components.

#### Configuration Management
- **BaseConfigManager** - `src/ml_framework/core/config_management/base_config_manager.py`
  - Manages configuration loading and merging from multiple sources
  - Provides nested configuration support with deep merging capabilities
  - Converts configurations to SimpleNamespace objects for easy access

#### Logging
- **BaseAppLogger** - `src/ml_framework/core/app_logging/base_app_logger.py`
  - Structured logging interface with performance monitoring
  - Provides decorators for function performance tracking
  - Context manager support for adding contextual information to logs

#### File Handling
- **BaseAppFileHandler** - `src/ml_framework/core/app_file_handling/base_app_file_handler.py`
  - Abstract interface for file I/O operations
  - Standardizes file handling across the application

#### Error Management
- **BaseErrorHandler** - `src/ml_framework/core/error_handling/base_error_handler.py`
  - Centralized error handling and reporting
  - Standardized error management across all components

### Framework Layer
Located in `src/ml_framework/framework/` - ML framework components and base classes.

#### Data Access
- **BaseDataAccess** - `src/ml_framework/framework/data_access/base_data_access.py`
  - Abstract interface for data persistence and retrieval operations
  - Database and file system abstraction layer

#### Data Validation
- **BaseDataValidator** - `src/ml_framework/framework/base_data_validator.py`
  - Data validation and integrity checking interface
  - Ensures data quality throughout the pipeline

### Preprocessing
- **BasePreprocessor** - `src/ml_framework/preprocessing/base_preprocessor.py`
  - Data preprocessing pipeline interface
  - Methods: `fit_transform()`, `transform()`
  - Integrates with PreprocessingResults for tracking transformations

### Uncertainty Quantification
- **UncertaintyCalibrator** - `src/ml_framework/uncertainty/uncertainty_calibrator.py`
  - Model uncertainty quantification and calibration (concrete implementation)
  - Supports various calibration methods for prediction confidence

### Model Registry
Located in `src/ml_framework/model_registry/` - Model persistence and versioning infrastructure.

- **BaseModelRegistry** - `src/ml_framework/model_registry/base_model_registry.py`
  - Abstract interface for model registry implementations
  - Methods: `save_model()`, `load_model()`, `list_models()`, `delete_model()`, `get_model_metadata()`
  - Methods: `register_model_version()`, `transition_model_stage()`, `get_model_by_stage()`
  - Supports MLflow, custom file-based, and cloud service backends
  - Enables model versioning and stage-based deployment (Development → Staging → Production)

- **MLflowModelRegistry** - `src/ml_framework/model_registry/mlflow_model_registry.py`
  - Concrete implementation using MLflow Model Registry
  - Automatic model flavor detection (XGBoost, LightGBM, CatBoost, PyTorch, Sklearn)
  - Integrated with preprocessing artifacts for complete model persistence

### Inference
Located in `src/ml_framework/inference/` - Production model serving and prediction.

- **ModelPredictor** - `src/ml_framework/inference/model_predictor.py`
  - Unified inference interface that works with any model registry
  - Methods: `load_model()`, `predict()`, `predict_batch()`, `get_model_info()`, `validate_input()`, `is_loaded()`
  - Automatically applies preprocessing transformations from saved artifacts
  - Supports batch predictions for large datasets
  - Input feature validation and error handling

### Model Testing & Training
Located in `src/ml_framework/model_testing/` - Generic ML testing and training infrastructure.

#### Model Testing
- **BaseModelTester** - `src/ml_framework/model_testing/base_model_testing.py`
  - Core model evaluation and testing interface
  - Methods: `prepare_data()`, `perform_oof_cross_validation()`
  - Integrates with ModelTrainingResults for comprehensive result tracking

#### Training
- **BaseTrainer** - `src/ml_framework/model_testing/trainers/base_trainer.py`
  - Model training interface for different ML algorithms
  - Methods: `train()`, feature importance calculation, metric conversion
  - Handles learning curves and performance metrics

#### Hyperparameter Optimization
- **BaseHyperparamsOptimizer** - `src/ml_framework/model_testing/hyperparams_optimizers/base_hyperparams_optimizer.py`
  - Hyperparameter optimization interface (supports Optuna, etc.)
  - Methods: `optimize()`, `get_best_params()`

- **BaseHyperparamsManager** - `src/ml_framework/model_testing/hyperparams_managers/base_hyperparams_manager.py`
  - Hyperparameter management and persistence interface

#### Experiment Logging
- **BaseExperimentLogger** - `src/ml_framework/model_testing/experiment_loggers/base_experiment_logger.py`
  - Experiment tracking and result logging interface
  - Supports MLflow, Weights & Biases, and other experiment tracking systems

### Visualization
Located in `src/ml_framework/visualization/` - Generic visualization components.

#### Charts
- **BaseChart** - `src/ml_framework/visualization/charts/base_chart.py`
  - Base interface for creating visualization charts
  - Methods: `create_figure()`
  - Integrates with matplotlib for consistent chart generation

#### Chart Orchestration
- **BaseChartOrchestrator** - `src/ml_framework/visualization/orchestration/base_chart_orchestrator.py`
  - Coordinates multiple chart creation and layout management

#### Exploratory Analysis
- **BaseExplorer** - `src/ml_framework/visualization/exploratory/base_explorer.py`
  - Exploratory data analysis and visualization (concrete implementation)
  - Provides statistical summaries and visual exploration tools

## NBA Application - Domain-Specific Implementation

### Data Processing
Located in `src/nba_app/data_processing/` - NBA-specific data processing operations.

- **BaseNBADataProcessor** - `src/nba_app/data_processing/base_data_processing_classes.py`
  - Core interface for NBA data processing operations
  - Methods: `process_data()`, `merge_team_data()`

### Feature Engineering
Located in `src/nba_app/feature_engineering/` - NBA-specific feature creation and selection.

- **BaseFeatureEngineer** - `src/nba_app/feature_engineering/base_feature_engineering.py`
  - Feature creation and transformation interface
  - Constructor: `__init__(config: BaseConfigManager, app_logger: BaseAppLogger)`
  - Methods: `engineer_features()`, `merge_team_data()`, `encode_game_date()`
  - Creates rolling averages, win/lose streaks, ELO ratings, and temporal features
  - Implements data validation to skip incomplete game records

- **BaseFeatureSelector** - `src/nba_app/feature_engineering/base_feature_engineering.py`
  - Feature selection and data splitting interface
  - Constructor: `__init__(config: BaseConfigManager, app_logger: BaseAppLogger)`
  - Methods: `select_features()`, `split_data()`
  - Removes unnecessary features and handles missing config attributes gracefully
  - Performs stratified train/validation splits maintaining temporal consistency

### Web Scraping
Located in `src/nba_app/webscraping/` - NBA-specific web scraping components.

#### Core Scraping
- **BaseNbaScraper** - `src/nba_app/webscraping/base_scraper_classes.py`
  - High-level NBA data scraping interface
  - Methods: `scrape_and_save_all_boxscores()`, `scrape_and_save_matchups_for_day()`

#### Specialized Scrapers
- **BaseBoxscoreScraper** - `src/nba_app/webscraping/base_scraper_classes.py`
  - Boxscore data extraction interface
  - Methods: `scrape_stat_type()`, `scrape_sub_seasons()`

- **BaseScheduleScraper** - `src/nba_app/webscraping/base_scraper_classes.py`
  - Game schedule scraping interface

#### Web Infrastructure
- **BaseWebDriver** - `src/nba_app/webscraping/base_scraper_classes.py`
  - Web driver abstraction for Selenium operations

- **BasePageScraper** - `src/nba_app/webscraping/base_scraper_classes.py`
  - Low-level page scraping operations interface
  - Methods: `go_to_url()`, `get_elements_by_class()`, `scrape_page_table()`

### Data Validation
- **DataValidator** - `src/nba_app/data_validator.py`
  - NBA-specific data validation implementation
  - Extends BaseDataValidator with basketball-specific validation rules

## Key Data Classes

Located in `src/ml_framework/framework/data_classes/`:

### Metrics
- **LearningCurvePoint** - Individual learning curve data point
- **LearningCurveData** - Complete learning curve information
- **ClassificationMetrics** - Model performance metrics container

### Preprocessing
- **PreprocessingStep** - Individual preprocessing operation tracking
- **PreprocessingResults** - Complete preprocessing pipeline results

### Training
- **ModelTrainingResults** - Comprehensive model training results and metrics

## Architecture Principles

### Three-Tier Platform Architecture
1. **Platform Core** (`src/ml_framework/`) - Reusable ML infrastructure for any domain
   - **Core**: Application infrastructure (config, logging, error handling)
   - **Framework**: ML framework components (data access, base classes)
   - **Model Testing**: Generic training, optimization, and evaluation
   - **Preprocessing**: Generic data preprocessing
   - **Visualization**: Generic charts and exploratory analysis
   - **Uncertainty**: Model uncertainty quantification
   - **Model Registry**: Model persistence and versioning (MLflow, custom, cloud)
   - **Inference**: Production model serving and prediction

2. **Domain Applications** (`src/nba_app/`) - Domain-specific implementations
   - **Data Processing**: NBA-specific data transformation
   - **Feature Engineering**: Basketball-specific feature creation
   - **Web Scraping**: NBA data collection from web sources
   - **Data Validation**: Basketball-specific validation rules

3. **Configuration Layer** (`configs/`) - Environment and domain configurations
   - **Core**: Generic ML configurations (models, preprocessing, etc.)
   - **NBA**: Basketball-specific configurations (scraping endpoints, features)

### Dependency Management
- **One-way dependencies**: nba_app → ml_framework (never reverse)
- **Scalable design**: Easy to add new sports (MLB, NFL) as separate app packages
- **Clean separation**: Platform core remains domain-agnostic

### Dependency Injection
- All components accept their dependencies through constructor injection
- Enables testing, modularity, and loose coupling
- Consistent interface across all abstract base classes

### Performance Monitoring
- All interfaces include `log_performance` property for function timing
- Consistent logging and error handling across components

### Result Tracking
- Comprehensive result objects (ModelTrainingResults, PreprocessingResults)
- Enable reproducibility and experiment tracking
- Support for hyperparameter history and model comparison

## Usage Guidelines

### For AI Code Assistants
- **Platform Core**: Use for reusable ML infrastructure components
- **NBA App**: Use for basketball-specific implementations
- Focus on abstract base classes to understand component contracts
- Follow the one-way dependency rule (nba_app → ml_framework)
- Leverage the platform architecture for scalable design

### For Developers
- **Adding New Sports**: Create new app packages (e.g., `mlb_app`) that import ml_framework
- **Extending Platform**: Add new generic capabilities to ml_framework
- Implement abstract base classes for new components
- Use existing interfaces as contracts for component communication
- Follow the established patterns for logging, error handling, and configuration

### For MLOps Teams
- **Training Pipelines**: Use ml_framework for infrastructure, nba_app for domain logic
- **Inference Services**: Deploy minimal containers with ml_framework + specific app
- **CI/CD**: Separate testing for ml_framework (unit tests) and nba_app (integration tests)
- **Scalability**: Platform design supports multiple domains in single monorepo