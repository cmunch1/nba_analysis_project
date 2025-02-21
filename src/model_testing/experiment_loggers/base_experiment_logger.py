from abc import ABC, abstractmethod
from typing import Dict, Any
from ...common.data_classes import ModelTrainingResults
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler

class BaseExperimentLogger(ABC):
    """Abstract base class for experiment logging."""
    
    @abstractmethod
    def __init__(self, 
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize experiment logger with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger for structured logging
            error_handler: Error handler for standardized error management
        """
        pass

    @property
    @abstractmethod
    def log_performance(self):
        """Get the performance logging decorator from app_logger."""
        pass

    @abstractmethod
    def log_experiment(self, results: ModelTrainingResults) -> None:
        """
        Log an experiment's results to the tracking system.
        
        Args:
            results: ModelTrainingResults containing all experiment data
        """
        pass

    @abstractmethod
    def log_model(self, 
                  model: Any,
                  model_name: str,
                  model_params: Dict[str, Any]) -> None:
        """
        Log a trained model with its parameters.
        
        Args:
            model: The trained model object
            model_name: Name of the model
            model_params: Dictionary of model parameters
        """
        pass