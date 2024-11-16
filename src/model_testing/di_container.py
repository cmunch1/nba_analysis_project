from dependency_injector import containers, providers

from ..config.config import Config
from ..data_access.data_access import DataAccess
from ..data_validation.data_validator import DataValidator
from .model_tester import ModelTester
from .experiment_logger import MLFlowLogger 
class DIContainer(containers.DeclarativeContainer):
    config = providers.Singleton(Config)
    data_access = providers.Factory(DataAccess, config=config)
    data_validator = providers.Factory(DataValidator, config=config)
    model_tester = providers.Factory(ModelTester, config=config)
    experiment_logger = providers.Factory(MLFlowLogger, config=config)