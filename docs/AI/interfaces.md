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

### Postprocessing
- **BasePostprocessor** - `src/ml_framework/postprocessing/base_postprocessor.py`
  - Abstract base class for postprocessing model outputs
  - Methods: `fit()`, `transform()`, `fit_transform()`
  - Applies transformations to model outputs after training (calibration, threshold optimization)
  - Follows scikit-learn fit/transform pattern

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

#### Feature Analysis
- **BaseFeatureAuditor** - `src/ml_framework/model_testing/feature_auditing/base_feature_auditor.py`
  - Feature auditing and analysis interface
  - Analyzes feature importance (SHAP, permutation, model-specific)
  - Evaluates statistical properties (coverage, variance, cardinality)
  - Identifies redundancy (correlation, VIF) and stability across CV folds
  - Methods: `create_audit()`, property: `log_performance`

- **BaseFeaturePruner** - `src/ml_framework/model_testing/feature_pruning/base_feature_pruner.py`
  - Feature pruning and selection interface
  - Identifies and removes low-quality features based on audit results
  - Methods: `identify_drop_candidates()`, property: `log_performance`

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

### Metrics (`src/ml_framework/framework/data_classes/metrics.py`)

#### LearningCurvePoint
Data point for learning curves.
- `train_size: int` - Training set size
- `train_score: float` - Training score
- `val_score: float` - Validation score
- `fold: int` - Cross-validation fold number
- `iteration: Optional[int]` - Iteration number (for iterative models)
- `metric_name: Optional[str]` - Name of the metric

#### LearningCurveData
Container for learning curve data averaged across folds.
- `train_scores: List[float]` - Training scores per iteration
- `val_scores: List[float]` - Validation scores per iteration
- `iterations: List[int]` - Iteration numbers
- `metric_name: Optional[str]` - Name of the metric
- Methods: `add_iteration()`, `get_plot_data()`, `__bool__()`

#### ClassificationMetrics
Container for classification evaluation metrics.
- `accuracy: float` - Accuracy score (default: 0.0)
- `precision: float` - Precision score (default: 0.0)
- `recall: float` - Recall score (default: 0.0)
- `f1: float` - F1 score (default: 0.0)
- `auc: float` - ROC AUC score (default: 0.0)
- `optimal_threshold: float` - Optimal probability threshold (default: 0.5)
- `valid_samples: int` - Number of valid samples (default: 0)
- `total_samples: int` - Total number of samples (default: 0)
- `nan_percentage: float` - Percentage of NaN values (default: 0.0)

### Preprocessing (`src/ml_framework/framework/data_classes/preprocessing.py`)

#### PreprocessingStep
Records information about a single preprocessing step.
- `name: str` - Name of the preprocessing step
- `type: str` - Type of preprocessing operation
- `columns: List[str]` - Columns affected by this step
- `parameters: Dict[str, Any]` - Parameters used in this step
- `statistics: Dict[str, Any]` - Statistics computed during preprocessing (default: {})
- `output_features: List[str]` - Features produced by this step (default: [])
- Methods: `to_dict()`, `_convert_to_serializable()`

#### PreprocessingResults
Tracks all preprocessing information and transformations.
- `steps: List[PreprocessingStep]` - List of preprocessing steps (default: [])
- `original_features: List[str]` - Original feature names (default: [])
- `final_features: List[str]` - Final feature names after preprocessing (default: [])
- `dropped_features: Set[str]` - Features that were dropped (default: set())
- `engineered_features: Dict[str, List[str]]` - Engineered features by type (default: {})
- `feature_transformations: Dict[str, List[str]]` - Transformation history per feature (default: {})
- `final_shape: Optional[Tuple[int, int]]` - Final shape of preprocessed data
- Methods: `add_step()`, `get_feature_lineage()`, `summarize()`, `to_dict()`, `to_json()`

### Training (`src/ml_framework/framework/data_classes/training.py`)

#### ModelTrainingResults
Comprehensive container for model training results and evaluation metrics.

**Model-Specific Fields:**
- `predictions: Optional[NDArray[np.float_]]` - Model predictions
- `shap_values: Optional[NDArray[np.float_]]` - SHAP values for feature importance
- `shap_interaction_values: Optional[NDArray[np.float_]]` - SHAP interaction values
- `feature_names: List[str]` - List of feature names
- `feature_importance_scores: Optional[NDArray[np.float_]]` - Feature importance scores
- `model: Optional[Any]` - Trained model object
- `model_name: str` - Name of the model
- `model_params: Dict` - Model hyperparameters
- `num_boost_round: int` - Number of boosting rounds
- `early_stopping: int` - Early stopping rounds
- `enable_categorical: bool` - Whether categorical features are enabled
- `categorical_features: List[str]` - List of categorical feature names
- `metrics: Optional[ClassificationMetrics]` - Evaluation metrics
- `eval_metric` - Evaluation metric function

**Data Fields:**
- `feature_data: Optional[pd.DataFrame]` - Feature data
- `target_data: Optional[pd.Series]` - Target labels
- `binary_predictions: Optional[NDArray[np.int_]]` - Binary predictions (0/1)
- `probability_predictions: Optional[NDArray[np.float_]]` - Probability predictions

**Evaluation Context:**
- `is_validation: bool` - Whether this is validation data (default: False)
- `evaluation_type: str` - Type of evaluation (e.g., "oof", "validation")

**Preprocessing Results:**
- `preprocessing_results: Optional[PreprocessingResults]` - Preprocessing metadata
- `preprocessing_artifact: Optional[Dict[str, Any]]` - Fitted preprocessor for persistence

**Postprocessing Results:**
- `calibrated_predictions: Optional[NDArray[np.float_]]` - Calibrated probabilities
- `calibration_artifact: Optional[Any]` - Fitted calibrator for persistence
- `calibration_metrics: Optional[Dict[str, float]]` - Calibration quality metrics
- `calibration_curve_data: Optional[Dict[str, Any]]` - Data for calibration visualization
- `conformal_prediction_sets: Optional[List[List[str]]]` - Conformal prediction sets per sample
- `conformal_probability_intervals: Optional[NDArray[np.float_]]` - Probability intervals per sample
- `conformal_metrics: Optional[Dict[str, Any]]` - Conformal coverage diagnostics
- `conformal_artifact: Optional[Any]` - Fitted conformal predictor for persistence
- `conformal_metadata: Optional[Dict[str, Any]]` - Additional conformal configuration/quantiles

**Learning Curves:**
- `learning_curve_data: LearningCurveData` - Learning curve information
- `n_folds: int` - Number of cross-validation folds

**Feature Audit:**
- `feature_audit: Optional[pd.DataFrame]` - Feature audit results
- `fold_importances: List[Dict[str, float]]` - Model gain importance per fold
- `fold_shap_importances: List[Dict[str, float]]` - Mean absolute SHAP per fold

**Methods:**
- `add_learning_curve_point()` - Add a learning curve data point
- `update_feature_data()` - Update feature and target data
- `update_predictions()` - Update probability and binary predictions
- `prepare_for_logging()` - Prepare results for experiment logging
- `prepare_for_charting()` - Prepare data for chart generation

### Hyperparameter Management (`src/ml_framework/model_testing/hyperparams_managers/hyperparams_manager.py`)

#### HyperparameterSet
Represents a single set of hyperparameters with metadata.
- `name: str` - Name of the hyperparameter set
- `params: Dict[str, Any]` - Hyperparameter dictionary
- `num_boost_round: int` - Number of boosting rounds
- `performance_metrics: Dict[str, float]` - Performance metrics for this set
- `creation_date: str` - ISO format creation timestamp
- `experiment_id: str` - MLflow experiment ID
- `run_id: str` - MLflow run ID
- `early_stopping: int` - Early stopping rounds
- `enable_categorical: bool` - Whether categorical features are enabled
- `categorical_features: List[str]` - List of categorical feature names
- `model_version: Optional[str]` - Model version (default: None)
- `description: Optional[str]` - Description of the parameter set (default: None)
- Class method: `from_dict()`

### Feature Schema (`src/nba_app/feature_engineering/feature_schema.py`)

#### FeatureSchema
Schema describing features and their types for NBA feature engineering.
- `numeric_features: List[str]` - Continuous numeric features
- `categorical_features: List[str]` - Categorical features
- `binary_features: List[str]` - Binary features (0/1)
- `ordinal_features: List[str]` - Ordinal categorical features
- `datetime_features: List[str]` - DateTime features
- `target_column: str` - Name of the target/label column
- `game_id_column: str` - Name of the game ID column
- `team_id_column: Optional[str]` - Team ID column (default: None)
- `date_column: Optional[str]` - Date column for temporal splits (default: None)
- `feature_descriptions: Dict[str, str]` - Metadata describing each feature (default: {})
- `feature_groups: Dict[str, List[str]]` - Grouping of related features (default: {})
- Methods: `to_dict()`, `to_json()`, `from_json()`, `from_dataframe()`, `get_all_features()`, `get_modeling_features()`, `validate_dataframe()`, `summary()`

### Logging (`src/ml_framework/core/app_logging/app_logger.py`)

#### LogContext
Class to store logging context data.
- `app_version: str` - Application version
- `environment: str` - Environment name (dev, staging, prod)
- `additional_context: Dict[str, Any]` - Additional context key-value pairs (default: None)

## Architecture Principles

### Three-Tier Platform Architecture
1. **Platform Core** (`src/ml_framework/`) - Reusable ML infrastructure for any domain
   - **Core**: Application infrastructure (config, logging, error handling, file handling)
   - **Framework**: ML framework components (data access, data validation, data classes)
   - **Model Testing**: Generic training, optimization, evaluation, feature auditing, and feature pruning
   - **Preprocessing**: Generic data preprocessing transformations
   - **Postprocessing**: Model output transformations (calibration, threshold optimization)
   - **Visualization**: Generic charts and exploratory analysis
   - **Uncertainty**: Model uncertainty quantification and calibration
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
