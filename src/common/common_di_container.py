from dependency_injector import containers, providers
from typing import Type
from pathlib import Path

# concrete classes
from src.common.config_management.config_manager import ConfigManager
from src.common.data_access.csv_data_access import CSVDataAccess
from src.common.app_logging.app_logger_factory import AppLoggerFactory
from src.common.data_access.data_access_factory import DataAccessFactory
from src.common.error_handling.error_handler_factory import ErrorHandlerFactory
from src.common.app_file_handling.app_file_handler import LocalAppFileHandler

CONFIG_DIR = Path('..') / 'configs'

class CommonDIContainer(containers.DeclarativeContainer):
    
    app_file_handler = providers.Singleton(LocalAppFileHandler)
    
    config = providers.Singleton(ConfigManager, config_dir=CONFIG_DIR, app_file_handler=app_file_handler)
    
    # Logger setup
    app_logger_factory = providers.Factory(AppLoggerFactory)
    logger = providers.Factory(
        lambda factory, config: factory.create_app_logger(config),
        factory=app_logger_factory,
        config=config
    )
    
    # Error handler setup
    error_handler_factory = providers.Factory(
        ErrorHandlerFactory,
        logger=logger
    )
    
    # Data access setup
    data_access_factory = providers.Factory(
        DataAccessFactory,
        data_access_class=CSVDataAccess,
        app_file_handler=app_file_handler
    )
    
    data_access = providers.Singleton(
        lambda factory, config, logger, app_file_handler: factory.create_data_access(config, logger, app_file_handler),
        factory=data_access_factory,
        config=config,
        logger=logger,
        app_file_handler=app_file_handler
    )