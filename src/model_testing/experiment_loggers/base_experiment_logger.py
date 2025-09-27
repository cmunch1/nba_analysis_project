from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

from src.common.data_classes import ModelTrainingResults
from src.common.app_file_handling.base_app_file_handler import BaseAppFileHandler
from src.common.config_management.base_config_manager import BaseConfigManager
from src.common.app_logging.base_app_logger import BaseAppLogger
from src.common.error_handling.base_error_handler import BaseErrorHandler
from src.visualization.orchestration.base_chart_orchestrator import BaseChartOrchestrator

class BaseExperimentLogger(ABC):
    """Base class for experiment loggers."""
    
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 app_file_handler: BaseAppFileHandler,
                 chart_orchestrator: BaseChartOrchestrator
                 ):
        """
        Initialize base experiment logger with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            app_file_handler: Application file handler for managing files
            chart_orchestrator: Chart orchestrator for visualization
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.app_file_handler = app_file_handler
        self.chart_orchestrator = chart_orchestrator

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