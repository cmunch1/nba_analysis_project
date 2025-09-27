from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
from typing import Any, Tuple, Optional
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from ...common.config_management.base_config_manager import BaseConfigManager

class BaseChart(ABC):
    @abstractmethod
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize chart with required dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger for structured logging
            error_handler: Error handler for standardized error management
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        
        self.app_logger.structured_log(
            logging.INFO,
            f"{self.__class__.__name__} initialized"
        )

    @property
    @abstractmethod
    def log_performance(self):
        """Get the performance logging decorator from app_logger."""
        return self.app_logger.log_performance

    @abstractmethod
    def create_figure(self, **kwargs) -> plt.Figure:
        """
        Create and return a figure.
        
        Args:
            **kwargs: Chart-specific parameters
            
        Returns:
            A matplotlib Figure object
        """
        pass

 