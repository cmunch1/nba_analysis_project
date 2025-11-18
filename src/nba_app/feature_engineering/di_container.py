from dependency_injector import containers, providers

from ml_framework.core.common_di_container import CommonDIContainer
from ..data_validator import DataValidator
from .feature_engineer import FeatureEngineer
from .feature_selector import FeatureSelector


class DIContainer(CommonDIContainer):
    """
    Feature engineering-specific DI container that inherits from CommonDIContainer.
    Adds feature engineering-specific dependencies while reusing common ones.
    """

    # Override data_validator with the specific implementation for this app
    data_validator = providers.Factory(
        DataValidator,
        config=CommonDIContainer.config,
        data_access=CommonDIContainer.data_access,
        app_logger=CommonDIContainer.app_logger,
        app_file_handler=CommonDIContainer.app_file_handler,
        error_handler=CommonDIContainer.error_handler_factory
    )

    # Feature engineering-specific dependencies
    feature_engineer = providers.Factory(
        FeatureEngineer,
        config=CommonDIContainer.config,
        app_logger=CommonDIContainer.app_logger
    )

    feature_selector = providers.Factory(
        FeatureSelector,
        config=CommonDIContainer.config,
        app_logger=CommonDIContainer.app_logger
    )