from dependency_injector import containers, providers
from ..common.common_di_container import CommonDIContainer
from .model_tester import ModelTester
from .hyperparams_managers.hyperparams_manager import HyperparameterManager
from .experiment_loggers.experiment_logger_factory import ExperimentLoggerFactory, LoggerType
from .hyperparams_optimizers.optuna_optimizer import OptunaOptimizer
from .trainers.trainer_factory import TrainerFactory

class ModelTestingDIContainer(containers.DeclarativeContainer):
    # Import common container
    common = providers.Container(CommonDIContainer)
    
    # Use common container's components
    config = common.config
    app_logger = common.logger
    app_file_handler = common.app_file_handler
    error_handler = common.error_handler_factory
    data_access = common.data_access
    data_validator = common.data_validator

    # Model testing specific components
    hyperparameter_manager = providers.Factory(
        HyperparameterManager,
        config=config,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler
    )

    trainers = providers.Factory(
        TrainerFactory.create_trainers,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler
    )

    model_tester = providers.Factory(
        ModelTester,
        config=config,
        hyperparameter_manager=hyperparameter_manager,
        trainers=trainers,
        app_logger=app_logger,
        error_handler=error_handler
    )

    experiment_logger = providers.Factory(
        ExperimentLoggerFactory.create_logger,
        logger_type=LoggerType.MLFLOW,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler
    )

    optimizer = providers.Factory(
        OptimizerFactory.create_optimizer,
        optimizer_type=OptimizerType.OPTUNA,
        config=config,
        hyperparameter_manager=hyperparameter_manager,
        app_logger=app_logger,
        error_handler=error_handler
    )