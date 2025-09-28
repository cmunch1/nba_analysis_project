from dependency_injector import containers, providers

from platform_core.core.config_management.config_manager import ConfigManager  
from platform_core.framework.data_access.csv_data_access import CSVDataAccess
from ..data_validation.data_validator import DataValidator
from .process_scraped_NBA_data import ProcessScrapedNBAData

class DIContainer(containers.DeclarativeContainer):
    
    config = providers.Singleton(ConfigManager)
    data_access = providers.Factory(CSVDataAccess, config=config)
    data_validator = providers.Factory(DataValidator, config=config)
    process_scraped_NBA_data = providers.Factory(ProcessScrapedNBAData, config=config)
    
