import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Optional, List
import shap
from .base_chart import BaseChart
from .chart_utils import ChartUtils
from ...common.core.app_logging.base_app_logger import BaseAppLogger
from ...common.core.error_handling.base_error_handler import BaseErrorHandler
from ...common.core.config_management.base_config_manager import BaseConfigManager
import logging

class SHAPCharts(BaseChart):
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize SHAP charts with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
        """
        super().__init__(config, app_logger, error_handler)
        self.chart_utils = ChartUtils(app_logger, error_handler)
        
        # Get chart configuration with appropriate fallbacks
        if hasattr(config, 'chart_options') and hasattr(config.chart_options, 'shap'):
            self.chart_config = config.chart_options.shap
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
                     shap_values: np.ndarray,
                     feature_names: List[str],
                     **kwargs) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            **kwargs: Additional chart parameters
            
        Returns:
            SHAP summary figure
        """
        return self.create_summary_plot(shap_values, feature_names, **kwargs)

    @log_performance
    def create_summary_plot(self,
                          shap_values: np.ndarray,
                          feature_names: List[str],
                          max_display: Optional[int] = None) -> plt.Figure:
        """
        Create a SHAP summary plot.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            max_display: Maximum number of features to display
            
        Returns:
            SHAP summary figure
        """
        try:
            # Get configuration values
            summary_config = getattr(self.chart_config, 'summary_plot', type('EmptyConfig', (), {})())
            figure_size = getattr(summary_config, 'figure_size', [12, 8])
            max_display = max_display or getattr(summary_config, 'max_display', 20)
            
            fig = plt.figure(figsize=figure_size)
            shap.summary_plot(
                shap_values,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            
            self.app_logger.structured_log(
                logging.INFO,
                "SHAP summary plot created",
                shap_values_shape=shap_values.shape,
                feature_count=len(feature_names),
                max_display=max_display
            )
            
            return self.chart_utils.finalize_plot(fig, 'SHAP Summary Plot')
            
        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "SHAP summary plot",
                shap_values_shape=shap_values.shape if 'shap_values' in locals() else None,
                feature_count=len(feature_names) if 'feature_names' in locals() else None
            )

    @log_performance
    def create_dependence_plot(self,
                             shap_values: np.ndarray,
                             features: pd.DataFrame,
                             feature_name: str,
                             interaction_feature: Optional[str] = None) -> plt.Figure:
        """
        Create a SHAP dependence plot for a specific feature.
        
        Args:
            shap_values: SHAP values array
            features: Feature dataframe
            feature_name: Name of the main feature to plot
            interaction_feature: Optional name of feature to use for coloring
            
        Returns:
            SHAP dependence figure
        """
        try:
            # Get configuration values
            dependence_config = getattr(self.chart_config, 'dependence_plots', type('EmptyConfig', (), {})())
            figure_size = getattr(dependence_config, 'figure_size', [10, 7])
            
            fig = plt.figure(figsize=figure_size)
            shap.dependence_plot(
                feature_name,
                shap_values,
                features,
                interaction_index=interaction_feature,
                ax=plt.gca(),
                show=False
            )
            
            self.app_logger.structured_log(
                logging.INFO,
                "SHAP dependence plot created",
                feature_name=feature_name,
                interaction_feature=interaction_feature,
                sample_count=len(features)
            )
            
            title = f'SHAP Dependence Plot for {feature_name}'
            if interaction_feature:
                title += f'\nColored by {interaction_feature}'
            return self.chart_utils.finalize_plot(fig, title)
            
        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "SHAP dependence plot",
                feature_name=feature_name,
                interaction_feature=interaction_feature
            )

    def create_shap_waterfall_plot(self, model: Any, X: pd.DataFrame, 
                                 index: int = 0, 
                                 shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Create a SHAP waterfall plot for a single observation.

        Args:
            model: Trained model object
            X: Feature dataframe
            index: Index of the observation to explain
            shap_values: Pre-calculated SHAP values

        Returns:
            plt.Figure: SHAP waterfall plot
        """
        try:
            # Calculate SHAP values if not provided
            if shap_values is not None:
                values = shap_values[index]
                expected_value = values.sum() / 2
            else:
                explainer = shap.TreeExplainer(model)
                X_sample = X.iloc[[index]]
                shap_values = explainer.shap_values(X_sample)
                values = shap_values[1] if isinstance(shap_values, list) else shap_values
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

            fig = plt.figure(figsize=(16, 3))
            X_sample = X.iloc[[index]] if isinstance(X.iloc[index], pd.Series) else X.iloc[index]
            shap.waterfall_plot(
                shap.Explanation(
                    values=values.reshape(-1),
                    base_values=expected_value,
                    data=X_sample.values,
                    feature_names=X.columns
                ),
                show=False
            )
            
            return self.chart_utils.finalize_plot(fig, f'SHAP Waterfall Plot for Observation {index}')
            
        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "SHAP waterfall plot",
                model_type=type(model).__name__,
                dataframe_shape=X.shape,
                index=index
            )