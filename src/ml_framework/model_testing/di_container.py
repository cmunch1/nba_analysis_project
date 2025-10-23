"""
Dependency Injection container for model testing with enhanced nested configuration support
and proper logging/error handling injection.
"""

from dependency_injector import containers, providers
from ml_framework.core.common_di_container import CommonDIContainer
from ml_framework.visualization.orchestration.chart_orchestrator import ChartOrchestrator
from ml_framework.preprocessing.preprocessor import Preprocessor
from ml_framework.postprocessing.probability_calibrator import ProbabilityCalibrator

from .model_tester import ModelTester
from .hyperparams_managers.hyperparams_manager import HyperparamsManager
from .experiment_loggers.experiment_logger_factory import ExperimentLoggerFactory, LoggerType
from .hyperparams_optimizers.hyperparams_optimizer_factory import OptimizerFactory, OptimizerType
from .trainers.trainer_factory import TrainerFactory
from .feature_auditing.feature_auditor import FeatureAuditor
from .feature_pruning.feature_pruner import FeaturePruner
from .feature_pruning.pruning_comparison import PruningComparison



class ModelTestingDIContainer(containers.DeclarativeContainer):
    # Import common container
    common = providers.Container(CommonDIContainer)
    
    # Use common container's components
    config = common.config
    app_logger = common.app_logger
    app_file_handler = common.app_file_handler
    error_handler = common.error_handler_factory
    data_access = common.data_access
    data_validator = common.data_validator
    model_registry = common.model_registry

    # Preprocessor
    preprocessor = providers.Singleton(
        Preprocessor,
        config=config,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler
    )

    # Postprocessor - Probability calibrator
    probability_calibrator = providers.Factory(
        ProbabilityCalibrator,
        app_logger=app_logger,
        error_handler=error_handler
    )

    # Chart orchestrator
    chart_orchestrator = providers.Singleton(
        ChartOrchestrator,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler,
        app_file_handler=app_file_handler
    )

    # Feature auditor
    feature_auditor = providers.Singleton(
        FeatureAuditor,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler
    )

    # Feature pruner
    feature_pruner = providers.Singleton(
        FeaturePruner,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler
    )

    # Pruning comparison
    pruning_comparison = providers.Singleton(
        PruningComparison,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler
    )

    # Hyperparameter management with proper injection
    hyperparameter_manager = providers.Singleton(
        HyperparamsManager,
        config=config,
        app_logger=app_logger,
        app_file_handler=app_file_handler,
        error_handler=error_handler
    )

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
        preprocessor=preprocessor,
        chart_orchestrator=chart_orchestrator,
        model_registry=model_registry
    )

    # Experiment logger with proper injection
    experiment_logger = providers.Factory(
        ExperimentLoggerFactory.create_logger,
        logger_type=LoggerType.MLFLOW,
        config=config,
        app_logger=app_logger,
        error_handler=error_handler,
        app_file_handler=app_file_handler,
        chart_orchestrator=chart_orchestrator
    )

    # Hyperparameter optimizer with proper injection
    optimizer = providers.Factory(
        OptimizerFactory.create_optimizer,
        optimizer_type=OptimizerType.OPTUNA,
        config=config,
        hyperparameter_manager=hyperparameter_manager,
        app_logger=app_logger,
        error_handler=error_handler,
        app_file_handler=app_file_handler
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
                error_handler=cls.error_handler,
                app_file_handler=cls.app_file_handler
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
                app_file_handler=cls.app_file_handler,
                chart_orchestrator=cls.chart_orchestrator
            )
        )