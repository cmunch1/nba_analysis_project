from dependency_injector import containers, providers
from typing import Type, Any
from pathlib import Path

# concrete classes
from src.common.config_management.config_manager import ConfigManager
from src.common.data_access.csv_data_access import CSVDataAccess
from src.common.app_logging.app_logger import AppLogger
from src.common.error_handling.error_handler_factory import ErrorHandlerFactory
from src.common.app_file_handling.app_file_handler import LocalAppFileHandler
from src.common.data_validation.data_validator import DataValidator

class CommonDIContainer(containers.DeclarativeContainer):
    """Container for common application dependencies."""
    

    
    # Core file handling
    app_file_handler: providers.Provider[LocalAppFileHandler] = providers.Singleton(
        LocalAppFileHandler
    )
    
    # Configuration management
    config: providers.Provider[ConfigManager] = providers.Singleton(
        ConfigManager,
        app_file_handler=app_file_handler
    )
    
    app_logger: providers.Provider[AppLogger] = providers.Singleton(
        AppLogger,
        config=config
    )
    
    error_handler_factory: providers.Provider[ErrorHandlerFactory] = providers.Singleton(
        ErrorHandlerFactory,
        app_logger=app_logger  
    )
    
    data_access: providers.Provider[CSVDataAccess] = providers.Singleton(
        CSVDataAccess,
        config=config,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler_factory
    )

    
    # Data validation
    data_validator: providers.Provider[DataValidator] = providers.Singleton(
        DataValidator,
        config=config,
        data_access=data_access,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler_factory
    )

    @classmethod
    def configure_data_access(cls, data_access_class: Type[Any]) -> None:
        """
        Configure the container to use a different data access implementation.
        
        Args:
            data_access_class: The data access class to use
        """
        cls.data_access_factory.override(
            providers.Factory(
                DataAccessFactory,
                data_access_class=data_access_class
            )
        )