from dependency_injector import containers, providers

from ml_framework.core.config_management.config_manager import ConfigManager
from ml_framework.framework.data_access.csv_data_access import CSVDataAccess
from ..data_validation.data_validator import DataValidator
from .feature_engineer import FeatureEngineer
from .feature_selector import FeatureSelector

class DIContainer(containers.DeclarativeContainer):
    config = providers.Singleton(ConfigManager)
    data_access = providers.Factory(CSVDataAccess, config=config)
    data_validator = providers.Factory(DataValidator, config=config)
    feature_engineer = providers.Factory(FeatureEngineer, config=config)
    feature_selector = providers.Factory(FeatureSelector, config=config)