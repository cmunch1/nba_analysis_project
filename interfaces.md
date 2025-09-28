# NBA Analysis Project - Interfaces & Architecture

## Overview
This document provides a comprehensive overview of all abstract base classes and interfaces in the NBA analysis project. These interfaces define the contracts for various components and establish the architecture's foundation.

## Core Infrastructure (Common Layer)

### Configuration Management
- **BaseConfigManager** - `src/common/core/config_management/base_config_manager.py`
  - Manages configuration loading and merging from multiple sources
  - Provides nested configuration support with deep merging capabilities
  - Converts configurations to SimpleNamespace objects for easy access

### Logging
- **BaseAppLogger** - `src/common/core/app_logging/base_app_logger.py`
  - Structured logging interface with performance monitoring
  - Provides decorators for function performance tracking
  - Context manager support for adding contextual information to logs

### File Handling
- **BaseAppFileHandler** - `src/common/core/app_file_handling/base_app_file_handler.py`
  - Abstract interface for file I/O operations
  - Standardizes file handling across the application

### Error Management
- **BaseErrorHandler** - `src/common/core/error_handling/base_error_handler.py`
  - Centralized error handling and reporting
  - Standardized error management across all components

### Data Access
- **BaseDataAccess** - `src/common/framework/data_access/base_data_access.py`
  - Abstract interface for data persistence and retrieval operations
  - Database and file system abstraction layer

### Data Validation
- **BaseDataValidator** - `src/common/framework/base_data_validator.py`
  - Data validation and integrity checking interface
  - Ensures data quality throughout the pipeline

## Data Processing Layer

### Data Processing
- **AbstractNBADataProcessor** - `src/data_processing/abstract_data_processing_classes.py`
  - Core interface for NBA data processing operations
  - Methods: `process_data()`, `merge_team_data()`

### Preprocessing
- **BasePreprocessor** - `src/preprocessing/base_preprocessor.py`
  - Data preprocessing pipeline interface
  - Methods: `fit_transform()`, `transform()`
  - Integrates with PreprocessingResults for tracking transformations

### Feature Engineering
- **AbstractFeatureEngineer** - `src/feature_engineering/abstract_feature_engineering.py`
  - Feature creation and transformation interface
  - Methods: `engineer_features()`, `merge_team_data()`

- **AbstractFeatureSelector** - `src/feature_engineering/abstract_feature_engineering.py`
  - Feature selection and data splitting interface
  - Methods: `select_features()`, `split_data()`

## Model Testing & Training Layer

### Model Testing
- **BaseModelTester** - `src/model_testing/base_model_testing.py`
  - Core model evaluation and testing interface
  - Methods: `prepare_data()`, `perform_oof_cross_validation()`
  - Integrates with ModelTrainingResults for comprehensive result tracking

### Training
- **BaseTrainer** - `src/model_testing/trainers/base_trainer.py`
  - Model training interface for different ML algorithms
  - Methods: `train()`, feature importance calculation, metric conversion
  - Handles learning curves and performance metrics

### Hyperparameter Optimization
- **BaseHyperparamsOptimizer** - `src/model_testing/hyperparams_optimizers/base_hyperparams_optimizer.py`
  - Hyperparameter optimization interface (supports Optuna, etc.)
  - Methods: `optimize()`, `get_best_params()`

- **BaseHyperparamsManager** - `src/model_testing/hyperparams_managers/base_hyperparams_manager.py`
  - Hyperparameter management and persistence interface

### Experiment Logging
- **BaseExperimentLogger** - `src/model_testing/experiment_loggers/base_experiment_logger.py`
  - Experiment tracking and result logging interface
  - Supports MLflow, Weights & Biases, and other experiment tracking systems

## Visualization Layer

### Charts
- **BaseChart** - `src/visualization/charts/base_chart.py`
  - Base interface for creating visualization charts
  - Methods: `create_figure()`
  - Integrates with matplotlib for consistent chart generation

### Chart Orchestration
- **BaseChartOrchestrator** - `src/visualization/orchestration/base_chart_orchestrator.py`
  - Coordinates multiple chart creation and layout management

### Exploratory Analysis
- **BaseExplorer** - `src/visualization/exploratory/base_explorer.py`
  - Interface for exploratory data analysis and visualization

## Web Scraping Layer

### Core Scraping
- **AbstractNbaScraper** - `src/webscraping/abstract_scraper_classes.py`
  - High-level NBA data scraping interface
  - Methods: `scrape_and_save_all_boxscores()`, `scrape_and_save_matchups_for_day()`

### Specialized Scrapers
- **AbstractBoxscoreScraper** - `src/webscraping/abstract_scraper_classes.py`
  - Boxscore data extraction interface
  - Methods: `scrape_stat_type()`, `scrape_sub_seasons()`

- **AbstractScheduleScraper** - `src/webscraping/abstract_scraper_classes.py`
  - Game schedule scraping interface

### Web Infrastructure
- **AbstractWebDriver** - `src/webscraping/abstract_scraper_classes.py`
  - Web driver abstraction for Selenium operations

- **AbstractPageScraper** - `src/webscraping/abstract_scraper_classes.py`
  - Low-level page scraping operations interface
  - Methods: `go_to_url()`, `get_elements_by_class()`, `scrape_page_table()`

## Key Data Classes

Located in `src/common/framework/data_classes/`:

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

### Three-Tier Structure
1. **Common Layer** (`src/common/`) - Core infrastructure and utilities
2. **Domain Layer** (data processing, model testing, visualization, etc.) - Business logic
3. **Application Layer** - Main entry points and orchestration

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
- Focus on abstract base classes to understand component contracts
- Use data classes to understand information flow between components
- Leverage the three-tier architecture for code organization
- Follow dependency injection patterns when adding new components

### For Developers
- Implement abstract base classes for new components
- Use existing interfaces as contracts for component communication
- Follow the established patterns for logging, error handling, and configuration
- Maintain consistency with the dependency injection approach