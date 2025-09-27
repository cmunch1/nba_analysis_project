from dependency_injector import containers, providers

from ..config.config import Config
from ..data_access.data_access import DataAccess
from ..data_validation.data_validator import DataValidator
from .feature_engineer import FeatureEngineer
from .feature_selector import FeatureSelector

class DIContainer(containers.DeclarativeContainer):
    config = providers.Singleton(Config)
    data_access = providers.Factory(DataAccess, config=config)
    data_validator = providers.Factory(DataValidator, config=config)
    feature_engineer = providers.Factory(FeatureEngineer, config=config)
    feature_selector = providers.Factory(FeatureSelector, config=config)