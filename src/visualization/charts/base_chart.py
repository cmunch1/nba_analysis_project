import logging
import matplotlib.pyplot as plt
from typing import Any
from ...logging.logging_utils import log_performance, structured_log
from ...error_handling.custom_exceptions import ChartCreationError

logger = logging.getLogger(__name__)

class BaseChart:
    @log_performance
    def __init__(self):
        """Initialize base chart class."""
        structured_log(logger, logging.INFO, "BaseChart initialized")

    def _create_figure(self, figsize=(12, 8)) -> tuple[plt.Figure, plt.Axes]:
        """
        Create a new figure and axis with the specified size.
        
        Args:
            figsize (tuple): Figure size (width, height)
            
        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    def _handle_error(self, e: Exception, chart_type: str, **kwargs) -> None:
        """
        Handle chart creation errors consistently.
        
        Args:
            e (Exception): The exception that occurred
            chart_type (str): Type of chart being created
            **kwargs: Additional context to log
        """
        structured_log(logger, logging.ERROR, 
                      f"Error creating {chart_type}",
                      error_message=str(e),
                      error_type=type(e).__name__,
                      **kwargs)
        raise ChartCreationError(f"Error creating {chart_type}",
                               error_message=str(e),
                               **kwargs)

    def _finalize_plot(self, fig: plt.Figure, title: str) -> plt.Figure:
        """
        Apply final touches to the plot.
        
        Args:
            fig (plt.Figure): The figure to finalize
            title (str): Plot title
            
        Returns:
            plt.Figure: The finalized figure
        """
        plt.title(title)
        plt.tight_layout()
        return fig 