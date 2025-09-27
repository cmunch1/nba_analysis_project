from abc import ABC, abstractmethod
from typing import Dict, Optional
import matplotlib.pyplot as plt
from ..charts.chart_types import ChartType
from ..charts.base_chart import BaseChart
from ...common.core.app_logging.base_app_logger import BaseAppLogger
from ...common.core.error_handling.base_error_handler import BaseErrorHandler
from ...common.core.config_management.base_config_manager import BaseConfigManager
from ...common.framework.data_classes import ModelTrainingResults

class BaseChartOrchestrator(ABC):
    """Abstract base class for chart orchestration."""

    @abstractmethod
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize the chart orchestrator with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
        """
        pass

    @property
    @abstractmethod
    def log_performance(self):
        """Get the performance logging decorator."""
        pass

    @abstractmethod
    def create_model_evaluation_charts(self, results: ModelTrainingResults) -> Dict[str, plt.Figure]:
        """
        Create all enabled charts for model evaluation.
        
        Args:
            results: Model training results containing data for charts
            
        Returns:
            Dictionary mapping chart names to figure objects
        """
        pass

    @abstractmethod
    def save_charts(self, charts: Dict[str, plt.Figure], output_dir: str) -> None:
        """
        Save generated charts to files.
        
        Args:
            charts: Dictionary of chart names and their corresponding matplotlib figures
            output_dir: Directory to save the charts
        """
        pass