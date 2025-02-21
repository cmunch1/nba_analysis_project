from enum import Enum
from typing import Type, Dict
from .base_experiment_logger import BaseExperimentLogger
from .mlflow_logger import MLFlowLogger
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler

class LoggerType(Enum):
    """Supported experiment logger types."""
    MLFLOW = "mlflow"
    # Add more logger types as needed
    # WANDB = "wandb"
    # TENSORBOARD = "tensorboard"

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
                     error_handler: BaseErrorHandler) -> BaseExperimentLogger:
        """
        Create an experiment logger instance with dependencies.
        
        Args:
            logger_type: Type of logger to create
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            
        Returns:
            Configured experiment logger instance
            
        Raises:
            ValueError: If logger_type is unknown
        """
        logger_class = cls._loggers.get(logger_type)
        if logger_class is None:
            raise ValueError(f"Unknown logger type: {logger_type}")
            
        return logger_class(
            config=config,
            app_logger=app_logger,
            error_handler=error_handler
        )

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