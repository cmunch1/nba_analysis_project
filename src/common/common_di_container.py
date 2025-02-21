from dependency_injector import containers, providers
from typing import Type, Any
from pathlib import Path

# concrete classes
from src.common.config_management.config_manager import ConfigManager
from src.common.data_access.csv_data_access import CSVDataAccess
from src.common.app_logging.app_logger_factory import AppLoggerFactory
from src.common.data_access.data_access_factory import DataAccessFactory
from src.common.error_handling.error_handler_factory import ErrorHandlerFactory
from src.common.app_file_handling.app_file_handler import LocalAppFileHandler
from src.common.data_validation.data_validator import DataValidator

class CommonDIContainer(containers.DeclarativeContainer):
    """Container for common application dependencies."""
    
    # Use __file__ to make path resolution more robust
    CONFIG_DIR: Path = Path(__file__).parent.parent / 'configs'
    
    # Core file handling
    app_file_handler: providers.Provider[LocalAppFileHandler] = providers.Singleton(
        LocalAppFileHandler
    )
    
    # Configuration management
    config: providers.Provider[ConfigManager] = providers.Singleton(
        ConfigManager,
        config_dir=CONFIG_DIR,
        app_file_handler=app_file_handler
    )
    
    # Logging setup
    app_logger_factory: providers.Provider[AppLoggerFactory] = providers.Factory(
        AppLoggerFactory
    )
    
    logger = providers.Factory(
        app_logger_factory.provided.create_app_logger,
        config=config
    )
    
    # Error handling setup
    error_handler_factory: providers.Provider[ErrorHandlerFactory] = providers.Singleton(
        ErrorHandlerFactory,
        logger=logger
    )
    
    # Data access setup - flexible for future data access types
    data_access_factory: providers.Provider[DataAccessFactory] = providers.Factory(
        DataAccessFactory,
        data_access_class=CSVDataAccess,  # This can be changed or made configurable
        app_file_handler=app_file_handler
    )
    
    data_access = providers.Singleton(
        data_access_factory.provided.create_data_access,
        config=config,
        logger=logger,
        app_file_handler=app_file_handler
    )
    
    # Data validation
    data_validator: providers.Provider[DataValidator] = providers.Singleton(
        DataValidator,
        config=config,
        data_access=data_access,
        app_logger=logger,
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
                data_access_class=data_access_class,
                app_file_handler=cls.app_file_handler
            )
        )