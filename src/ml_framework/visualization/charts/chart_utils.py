import logging
import matplotlib.pyplot as plt
from typing import Tuple
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler

class ChartUtils:
    """Utility class providing common chart methods."""
    
    def __init__(self, app_logger: BaseAppLogger, error_handler: BaseErrorHandler):
        """
        Initialize chart utilities.
        
        Args:
            app_logger: Application logger
            error_handler: Error handler
        """
        self.app_logger = app_logger
        self.error_handler = error_handler

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def create_figure(self, figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a new figure and axis with the specified size.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Tuple containing Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    @log_performance
    def finalize_plot(self, fig: plt.Figure, title: str) -> plt.Figure:
        """
        Apply final touches to the plot.
        
        Args:
            fig: The figure to finalize
            title: Plot title
            
        Returns:
            The finalized figure
        """
        plt.title(title)
        plt.tight_layout()
        return fig

    @log_performance
    def handle_chart_error(self, e: Exception, chart_type: str, **kwargs) -> None:
        """
        Handle chart creation errors consistently.
        
        Args:
            e: The exception that occurred
            chart_type: Type of chart being created
            **kwargs: Additional context to log
        """
        self.app_logger.structured_log(
            logging.ERROR,
            f"Error creating {chart_type}",
            error_message=str(e),
            error_type=type(e).__name__,
            **kwargs
        )
        raise self.error_handler.create_error_handler(
            'chart_creation',
            f"Error creating {chart_type}",
            original_error=str(e),
            **kwargs
        )