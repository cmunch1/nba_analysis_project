from dependency_injector import containers, providers

from ..config.config import Config
from ..data_access.data_access import DataAccess
from ..data_validation.data_validator import DataValidator
from .model_tester import ModelTester
from .hyperparameter_manager import HyperparameterManager
from .experiment_logger_factory import ExperimentLoggerFactory, LoggerType
from .optimizer_factory import OptimizerFactory, OptimizerType

class DIContainer(containers.DeclarativeContainer):
    config = providers.Singleton(Config)
    data_access = providers.Factory(DataAccess, config=config)
    data_validator = providers.Factory(DataValidator, config=config)
    hyperparameter_manager = providers.Factory(HyperparameterManager, config=config)
    model_tester = providers.Factory(
        ModelTester, 
        config=config,
        hyperparameter_manager=hyperparameter_manager
    )
    experiment_logger = providers.Factory(
        ExperimentLoggerFactory.create_logger, 
        logger_type=LoggerType.MLFLOW, 
        config=config
    )
    optimizer_factory = providers.Factory(
        OptimizerFactory.create_optimizer, 
        optimizer_type=OptimizerType.OPTUNA, 
        config=config,
        hyperparameter_manager=hyperparameter_manager
    )