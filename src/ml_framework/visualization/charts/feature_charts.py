import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional
from .base_chart import BaseChart
from .chart_utils import ChartUtils
from platform_core.core.app_logging.base_app_logger import BaseAppLogger
from platform_core.core.error_handling.base_error_handler import BaseErrorHandler
from platform_core.core.config_management.base_config_manager import BaseConfigManager
import logging

class FeatureCharts(BaseChart):
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize feature charts with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
        """
        super().__init__(config, app_logger, error_handler)
        self.chart_utils = ChartUtils(app_logger, error_handler)
        
        # Get chart configuration with appropriate fallbacks
        if hasattr(config, 'chart_options') and hasattr(config.chart_options, 'feature_importance'):
            self.chart_config = config.chart_options.feature_importance
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
    def create_figure(self, 
                     feature_importance: np.ndarray,
                     feature_names: List[str],
                     top_n: Optional[int] = None,
                     **kwargs) -> plt.Figure:
        """
        Create a feature importance chart.

        Args:
            feature_importance: Array of feature importance scores
            feature_names: List of feature names
            top_n: Number of top features to display (overrides config)
            **kwargs: Additional chart parameters

        Returns:
            Feature importance chart figure
        """
        try:
            # Input validation
            if len(feature_importance) != len(feature_names):
                raise ValueError("Feature importance and names must have same length")
            
            # Get configuration values with defaults
            top_n = top_n or getattr(self.chart_config, 'top_n', 20)
            figure_size = getattr(self.chart_config, 'figure_size', [12, 8])
            
            # Sort features by importance and get top_n features
            indices = np.argsort(np.abs(feature_importance))[-top_n:]
            top_importance = feature_importance[indices]
            top_names = [feature_names[i] for i in indices]

            # Create figure and plot
            fig, ax = self.chart_utils.create_figure(figsize=figure_size)
            y_pos = np.arange(len(top_importance))
            ax.barh(y_pos, np.abs(top_importance))
            
            # Customize the plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance (absolute value)')
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            self.app_logger.structured_log(
                logging.INFO,
                "Feature importance chart created",
                feature_count=len(feature_names),
                top_n=top_n
            )
            
            return self.chart_utils.finalize_plot(fig, f'Top {top_n} Most Important Features')
            
        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "feature importance chart",
                feature_importance_shape=feature_importance.shape,
                feature_names_length=len(feature_names),
                top_n=top_n
            )

    @log_performance
    def create_feature_distribution(self, 
                                  feature_data: pd.Series,
                                  **kwargs) -> plt.Figure:
        """
        Create a distribution plot for a single feature.

        Args:
            feature_data: Series containing feature values
            **kwargs: Additional chart parameters

        Returns:
            Feature distribution chart figure
        """
        try:
            fig, ax = self.chart_utils.create_figure()
            
            # Create histogram with density plot
            ax.hist(feature_data.dropna(), bins='auto', density=True, alpha=0.7)
            feature_data.dropna().plot(kind='kde', ax=ax)
            
            # Customize the plot
            ax.set_xlabel(feature_data.name or 'Feature Value')
            ax.set_ylabel('Density')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            self.app_logger.structured_log(
                logging.INFO,
                "Feature distribution chart created",
                feature_name=feature_data.name,
                value_count=len(feature_data),
                nan_count=feature_data.isna().sum()
            )
            
            return self.chart_utils.finalize_plot(
                fig,
                f'Distribution of {feature_data.name or "Feature"}'
            )
            
        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "feature distribution chart",
                feature_name=feature_data.name,
                value_count=len(feature_data)
            )