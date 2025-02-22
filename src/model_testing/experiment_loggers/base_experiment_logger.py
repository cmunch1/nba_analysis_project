from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

from ...common.data_classes import ModelTrainingResults
from ...common.app_file_handling.base_app_file_handler import BaseAppFileHandler
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from ...visualization.orchestration.base_chart_orchestrator import BaseChartOrchestrator

class BaseExperimentLogger(ABC):
    """Base class for experiment loggers."""
    
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 chart_orchestrator: BaseChartOrchestrator,
                 app_file_handler: BaseAppFileHandler):
        """
        Initialize base experiment logger with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            chart_orchestrator: Chart orchestrator for visualization
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.chart_orchestrator = chart_orchestrator
        self.app_file_handler = app_file_handler
    @abstractmethod
    def log_experiment(self, results: ModelTrainingResults) -> None:
        """
        Log an experiment's results.
        
        Args:
            results: Model training results containing all experiment data
        """
        pass

    @property
    def log_performance(self):
        """Get the performance logging decorator."""
        return self.app_logger.log_performance

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