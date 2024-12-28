from enum import Enum
from typing import Type
from ..config.config import AbstractConfig
from .abstract_model_testing import AbstractExperimentLogger
from .loggers.mlflow_logger import MLFlowLogger

class LoggerType(Enum):
    MLFLOW = "mlflow"
    # Add more logger types as needed
    # WANDB = "wandb"
    # TENSORBOARD = "tensorboard"

class ExperimentLoggerFactory:
    _loggers = {
        LoggerType.MLFLOW: MLFlowLogger,
        # Add more mappings as you create new loggers
        # LoggerType.WANDB: WandBLogger,
        # LoggerType.TENSORBOARD: TensorBoardLogger,
    }

    @classmethod
    def create_logger(cls, logger_type: LoggerType, config: AbstractConfig) -> AbstractExperimentLogger:
        logger_class = cls._loggers.get(logger_type)
        if logger_class is None:
            raise ValueError(f"Unknown logger type: {logger_type}")
        return logger_class(config)

    @classmethod
    def register_logger(cls, logger_type: LoggerType, logger_class: Type[AbstractExperimentLogger]):
        cls._loggers[logger_type] = logger_class 