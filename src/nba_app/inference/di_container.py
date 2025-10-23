"""Dependency Injection Container for NBA Inference Module"""

from dependency_injector import containers, providers

from ml_framework.core.config_management.config_manager import ConfigManager
from ml_framework.core.app_logging.app_logger import AppLogger
from ml_framework.core.app_file_handling.app_file_handler import AppFileHandler
from ml_framework.core.error_handling.error_handler import ErrorHandlerFactory
from ml_framework.framework.data_access.csv_data_access import CsvDataAccess
from ml_framework.model_registry.mlflow_model_registry import MLflowModelRegistry
from ml_framework.inference.model_predictor import ModelPredictor

from nba_app.feature_engineering.feature_engineer import FeatureEngineer
from .schedule_loader import ScheduleLoader
from .game_predictor import GamePredictor


class DIContainer(containers.DeclarativeContainer):
    """
    Dependency injection container for the NBA inference module.

    Wires together:
    - Configuration management
    - Logging infrastructure
    - Data access layer
    - Model registry (MLflow)
    - Feature engineering
    - Inference pipeline
    """

    # Configuration
    config = providers.Singleton(
        ConfigManager,
        config_dirs=[
            'configs/core',
            'configs/nba'
        ]
    )

    # Core infrastructure
    app_logger = providers.Singleton(AppLogger)

    app_file_handler = providers.Singleton(
        AppFileHandler,
        config=config
    )

    error_handler_factory = providers.Factory(
        ErrorHandlerFactory,
        app_logger=app_logger
    )

    # Data access
    data_access = providers.Singleton(
        CsvDataAccess,
        config=config,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler_factory
    )

    # Model registry
    model_registry = providers.Singleton(
        MLflowModelRegistry,
        config=config,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler_factory
    )

    # Feature engineering (reused from training)
    feature_engineer = providers.Factory(
        FeatureEngineer,
        config=config,
        app_logger=app_logger
    )

    # Model inference
    model_predictor = providers.Factory(
        ModelPredictor,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler_factory,
        model_registry=model_registry
    )

    # Schedule loading
    schedule_loader = providers.Factory(
        ScheduleLoader,
        config=config,
        app_logger=app_logger,
        data_access=data_access,
        error_handler=error_handler_factory
    )

    # Game prediction orchestrator
    game_predictor = providers.Factory(
        GamePredictor,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler_factory,
        data_access=data_access,
        model_registry=model_registry,
        feature_engineer=feature_engineer,
        model_predictor=model_predictor
    )
