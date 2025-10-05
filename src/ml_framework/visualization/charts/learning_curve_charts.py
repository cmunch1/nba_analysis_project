import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from .base_chart import BaseChart
from .chart_utils import ChartUtils
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.framework.data_classes import ModelTrainingResults
import logging

class LearningCurveCharts(BaseChart):
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize learning curve charts with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
        """
        super().__init__(config, app_logger, error_handler)
        self.chart_utils = ChartUtils(app_logger, error_handler)
        
        # Get chart configuration with appropriate fallbacks
        if hasattr(config, 'chart_options') and hasattr(config.chart_options, 'learning_curve'):
            self.chart_config = config.chart_options.learning_curve
        else:
            # Create empty config if not available
            self.chart_config = type('EmptyConfig', (), {})()

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    @log_performance
    def create_figure(self, results: ModelTrainingResults, **kwargs) -> plt.Figure:
        """
        Create learning curve visualization.
        
        Args:
            results: Model training results containing learning curve data
            **kwargs: Additional chart parameters
            
        Returns:
            Learning curve figure
        """
        try:
            plot_data = results.learning_curve_data.get_plot_data()
            if not plot_data:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "No learning curve data available"
                )
                return None

            # Get configuration values
            figure_size = getattr(self.chart_config, 'figure_size', [10, 6])
            
            fig, ax = self.chart_utils.create_figure(figsize=figure_size)
            
            # Plot averaged training and validation scores
            ax.plot(
                plot_data['iterations'],
                plot_data['train_scores'],
                label='Training score',
                color='blue',
                alpha=0.8
            )
            ax.plot(
                plot_data['iterations'],
                plot_data['val_scores'],
                label='Validation score',
                color='orange',
                alpha=0.8
            )
            
            # Add confidence intervals if available
            if 'train_std' in plot_data and 'val_std' in plot_data:
                train_std = plot_data['train_std']
                val_std = plot_data['val_std']
                iterations = plot_data['iterations']
                
                ax.fill_between(
                    iterations,
                    plot_data['train_scores'] - train_std,
                    plot_data['train_scores'] + train_std,
                    color='blue',
                    alpha=0.2
                )
                ax.fill_between(
                    iterations,
                    plot_data['val_scores'] - val_std,
                    plot_data['val_scores'] + val_std,
                    color='orange',
                    alpha=0.2
                )
            
            # Customize the plot
            ax.set_xlabel('Training examples')
            ax.set_ylabel(f'Score ({results.learning_curve_data.metric_name})')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
            
            self.app_logger.structured_log(
                logging.INFO,
                "Learning curve created",
                model_name=results.model_name,
                metric_name=results.learning_curve_data.metric_name
            )
            
            title = f'Learning Curve - {results.model_name}'
            return self.chart_utils.finalize_plot(fig, title)
            
        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "learning curve",
                model_name=results.model_name if results else None
            )