import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from typing import Dict, Any, Optional
from .base_chart import BaseChart
from .chart_utils import ChartUtils
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from ...common.config_management.base_config_manager import BaseConfigManager
import logging

class MetricsCharts(BaseChart):
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize metrics charts with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
        """
        super().__init__(config, app_logger, error_handler)
        self.chart_utils = ChartUtils(app_logger, error_handler)
        self.chart_config = config.get('chart_options', {}).get('metrics', {})

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def create_figure(self, 
                     confusion_matrix_data: Dict[str, Any],
                     **kwargs) -> plt.Figure:
        """
        Create confusion matrix visualization.
        
        Args:
            confusion_matrix_data: Dictionary containing confusion matrix data
            **kwargs: Additional chart parameters
            
        Returns:
            Confusion matrix figure
        """
        return self.create_confusion_matrix(confusion_matrix_data)

    @log_performance
    def create_confusion_matrix(self, 
                              confusion_matrix_data: Dict[str, Any]) -> plt.Figure:
        """
        Create a confusion matrix visualization.
        
        Args:
            confusion_matrix_data: Dictionary containing:
                - matrix: The confusion matrix array
                - labels: Optional class labels
                
        Returns:
            Confusion matrix figure
        """
        try:
            matrix = confusion_matrix_data['matrix']
            labels = confusion_matrix_data.get('labels')
            
            # Get configuration values
            cm_config = self.chart_config.get('confusion_matrix', {})
            figure_size = cm_config.get('figure_size', [10, 8])
            color_map = cm_config.get('color_map', 'Blues')
            
            fig, ax = self.chart_utils.create_figure(figsize=figure_size)
            sns.heatmap(
                matrix,
                annot=True,
                fmt='d',
                cmap=color_map,
                ax=ax,
                xticklabels=labels,
                yticklabels=labels
            )
            
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            
            self.app_logger.structured_log(
                logging.INFO,
                "Confusion matrix created",
                matrix_shape=matrix.shape
            )
            
            return self.chart_utils.finalize_plot(fig, 'Confusion Matrix')
            
        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "confusion matrix",
                matrix_shape=matrix.shape if 'matrix' in locals() else None
            )

    @log_performance
    def create_roc_curve(self, 
                        roc_data: Dict[str, Any]) -> plt.Figure:
        """
        Create ROC curve visualization.
        
        Args:
            roc_data: Dictionary containing:
                - fpr: False positive rates
                - tpr: True positive rates
                - auc: Area under curve score
                
        Returns:
            ROC curve figure
        """
        try:
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            auc_score = roc_data['auc']
            
            fig, ax = self.chart_utils.create_figure()
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='lower right')
            
            self.app_logger.structured_log(
                logging.INFO,
                "ROC curve created",
                auc_score=auc_score
            )
            
            return self.chart_utils.finalize_plot(fig, 'ROC Curve')
            
        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "ROC curve",
                data_points=len(fpr) if 'fpr' in locals() else None
            )

    def create_figure(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> plt.Figure:
        """Default to creating ROC curve."""
        return self.create_roc_curve(y_true, y_score) 