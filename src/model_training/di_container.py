from dependency_injector import containers, providers

from ..config.config import Config
from ..data_access.data_access import DataAccess
from ..data_validation.data_validator import DataValidator
from .model_trainer import ModelTrainer

class DIContainer(containers.DeclarativeContainer):
    config = providers.Singleton(Config)
    data_access = providers.Factory(DataAccess, config=config)
    data_validator = providers.Factory(DataValidator, config=config)
    model_trainer = providers.Factory(ModelTrainer, config=config)