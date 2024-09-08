from dependency_injector import containers, providers

from ..config.config import Config  
from ..data_access.data_access import DataAccess
from ..data_validation.data_validator import DataValidator
from .process_scraped_NBA_data import ProcessScrapedNBAData

class DIContainer(containers.DeclarativeContainer):
    
    config = providers.Singleton(Config)
    data_access = providers.Factory(DataAccess, config=config)
    data_validator = providers.Factory(DataValidator, config=config)
    process_scraped_NBA_data = providers.Factory(ProcessScrapedNBAData, config=config)
    
