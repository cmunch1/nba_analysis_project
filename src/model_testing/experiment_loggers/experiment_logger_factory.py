from enum import Enum, auto
from typing import Type, Dict
from .base_experiment_logger import BaseExperimentLogger
from .mlflow_logger import MLFlowLogger
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from ...common.app_file_handling.base_app_file_handler import BaseAppFileHandler
from ...visualization.orchestration.base_chart_orchestrator import BaseChartOrchestrator

class LoggerType(Enum):
    """Enum for supported experiment logger types."""
    MLFLOW = auto()
    # Add more logger types as needed
    # WANDB = auto()
    # TENSORBOARD = auto()

class ExperimentLoggerFactory:
    """Factory for creating experiment loggers with proper dependency injection."""
    
    # Registry of available loggers
    _loggers: Dict[LoggerType, Type[BaseExperimentLogger]] = {
        LoggerType.MLFLOW: MLFlowLogger,
        # Add more mappings as you create new loggers
        # LoggerType.WANDB: WandBLogger,
        # LoggerType.TENSORBOARD: TensorBoardLogger,
    }

    @classmethod
    def create_logger(cls, 
                     logger_type: LoggerType,
                     config: BaseConfigManager,
                     app_logger: BaseAppLogger,
                     error_handler: BaseErrorHandler,
                     chart_orchestrator: BaseChartOrchestrator,
                     app_file_handler: BaseAppFileHandler) -> BaseExperimentLogger:
        """
        Create an experiment logger instance with proper dependency injection.
        
        Args:
            logger_type: Type of logger to create
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            chart_orchestrator: Chart orchestrator for visualization
            
        Returns:
            Configured experiment logger instance
            
        Raises:
            ValueError: If logger_type is not supported
        """
        try:
            logger_class = cls._loggers[logger_type]
            
            app_logger.structured_log(
                "info",
                "Creating experiment logger",
                logger_type=logger_type.name
            )
            
            return logger_class(
                config=config,
                app_logger=app_logger,
                error_handler=error_handler,
                chart_orchestrator=chart_orchestrator,
                app_file_handler=app_file_handler
            )
            
        except KeyError:
            error_msg = f"Unsupported logger type: {logger_type}"
            app_logger.structured_log(
                "error",
                error_msg,
                logger_type=logger_type.name
            )
            raise ValueError(error_msg)

    @classmethod
    def register_logger(cls, 
                       logger_type: LoggerType, 
                       logger_class: Type[BaseExperimentLogger]) -> None:
        """
        Register a new logger type.
        
        Args:
            logger_type: Enum value for the logger type
            logger_class: Logger class to register
        """
        cls._loggers[logger_type] = logger_class