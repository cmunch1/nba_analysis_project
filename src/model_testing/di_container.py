"""
Dependency Injection container for model testing with enhanced nested configuration support
and proper logging/error handling injection.
"""

from dependency_injector import containers, providers
from src.common.common_di_container import CommonDIContainer
from ..preprocessing.di_container import PreprocessingDIContainer
from .model_tester import ModelTester
from .hyperparams_managers.hyperparams_manager import HyperparameterManager
from .experiment_loggers.experiment_logger_factory import ExperimentLoggerFactory, LoggerType
from .hyperparams_optimizers.hyperparams_optimizer_factory import OptimizerFactory, OptimizerType
from .trainers.trainer_factory import TrainerFactory
from ..visualization.orchestration.chart_orchestrator import ChartOrchestrator


class ModelTestingDIContainer(containers.DeclarativeContainer):
    # Import common container
    common = providers.Container(CommonDIContainer)
    
    # Import preprocessing container
    preprocessing = providers.Container(PreprocessingDIContainer)
    
    # Use common container's components
    config = common.config
    app_logger = common.logger
    app_file_handler = common.app_file_handler
    error_handler = common.error_handler_factory
    data_access = common.data_access
    data_validator = common.data_validator

    # Chart orchestrator with proper injection
    chart_orchestrator = providers.Singleton(
        ChartOrchestrator,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler
    )

    # Hyperparameter management with proper injection
    hyperparameter_manager = providers.Factory(
        HyperparameterManager,
        config=config,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler
    )

    # Preprocessor from the preprocessing container
    preprocessor = preprocessing.preprocessor

    # Trainers factory with proper dependency injection
    trainers = providers.Factory(
        TrainerFactory.create_trainers,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler
    )

    # Model tester with injected dependencies
    model_tester = providers.Factory(
        ModelTester,
        config=config,
        hyperparameter_manager=hyperparameter_manager,
        trainers=trainers,
        app_logger=app_logger,
        error_handler=error_handler,
        preprocessor=preprocessor
    )

    # Experiment logger with proper injection
    experiment_logger = providers.Factory(
        ExperimentLoggerFactory.create_logger,
        logger_type=LoggerType.MLFLOW,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler,
        chart_orchestrator=chart_orchestrator
    )

    # Hyperparameter optimizer with proper injection
    optimizer = providers.Factory(
        OptimizerFactory.create_optimizer,
        optimizer_type=OptimizerType.OPTUNA,
        config=config,
        hyperparameter_manager=hyperparameter_manager,
        app_logger=app_logger,
        error_handler=error_handler
    )

    @classmethod
    def configure_optimizer(cls, optimizer_type: OptimizerType) -> None:
        """Configure the container to use a different optimizer implementation."""
        cls.optimizer.override(
            providers.Factory(
                OptimizerFactory.create_optimizer,
                optimizer_type=optimizer_type,
                config=cls.config,
                hyperparameter_manager=cls.hyperparameter_manager,
                app_logger=cls.app_logger,
                error_handler=cls.error_handler
            )
        )

    @classmethod
    def configure_experiment_logger(cls, logger_type: LoggerType) -> None:
        """Configure the container to use a different experiment logger."""
        cls.experiment_logger.override(
            providers.Factory(
                ExperimentLoggerFactory.create_logger,
                logger_type=logger_type,
                config=cls.config,
                app_logger=cls.app_logger,
                error_handler=cls.error_handler,
                chart_orchestrator=cls.chart_orchestrator
            )
        )