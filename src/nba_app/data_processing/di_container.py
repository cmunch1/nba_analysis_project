from dependency_injector import containers, providers

from ml_framework.core.common_di_container import CommonDIContainer
from ..data_validator import DataValidator
from .process_scraped_NBA_data import ProcessScrapedNBAData


class DIContainer(CommonDIContainer):
    """
    Data processing-specific DI container that inherits from CommonDIContainer.
    Adds data processing-specific dependencies while reusing common ones.
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

    # Data processing-specific dependencies
    process_scraped_NBA_data = providers.Factory(
        ProcessScrapedNBAData,
        config=CommonDIContainer.config,
        app_logger=CommonDIContainer.app_logger
    )
    
