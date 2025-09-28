import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Optional
import shap
from .base_chart import BaseChart
from .chart_utils import ChartUtils
from platform_core.core.app_logging.base_app_logger import BaseAppLogger
from platform_core.core.error_handling.base_error_handler import BaseErrorHandler
from platform_core.core.config_management.base_config_manager import BaseConfigManager
import logging

class ModelInterpretationCharts(BaseChart):
    """Charts for model interpretation and explanation."""
    
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler):
        """
        Initialize model interpretation charts with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
        """
        super().__init__(config, app_logger, error_handler)
        self.chart_utils = ChartUtils(app_logger, error_handler)
        
        # Get chart configuration with appropriate fallbacks
        if hasattr(config, 'chart_options') and hasattr(config.chart_options, 'model_interpretation'):
            self.chart_config = config.chart_options.model_interpretation
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
                     model: Any,
                     X: pd.DataFrame,
                     index: int = 0,
                     **kwargs) -> plt.Figure:
        """Default to creating force plot."""
        return self.create_shap_force_plot(model, X, index, **kwargs)

    @log_performance
    def create_shap_force_plot(self,
                              model: Any,
                              X: pd.DataFrame,
                              index: int = 0,
                              shap_values: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Create a SHAP force plot for a single prediction.
        
        Args:
            model: The trained model
            X: Feature dataframe
            index: Index of the instance to explain
            shap_values: Pre-computed SHAP values (optional)
            
        Returns:
            SHAP force plot figure
        """
        try:
            # Get configuration values
            force_plot_config = getattr(self.chart_config, 'force_plot', type('EmptyConfig', (), {})())
            figure_size = getattr(force_plot_config, 'figure_size', [12, 6])
            background_samples = getattr(self.chart_config, 'background_samples', 100)
            
            # Calculate SHAP values if not provided
            if shap_values is None:
                if hasattr(model, 'predict_proba'):
                    background = shap.sample(X, min(background_samples, len(X)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X.iloc[index:index+1])
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                else:
                    background = shap.sample(X, min(background_samples, len(X)))
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(X.iloc[index:index+1])

            # Create force plot
            fig = plt.figure(figsize=figure_size)
            shap.force_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value, list) 
                else explainer.expected_value,
                shap_values,
                X.iloc[index:index+1],
                show=False,
                matplotlib=True
            )
            
            self.app_logger.structured_log(
                logging.INFO,
                "SHAP force plot created",
                model_type=type(model).__name__,
                instance_index=index,
                feature_count=X.shape[1]
            )
            
            return self.chart_utils.finalize_plot(
                fig,
                f'SHAP Force Plot for Instance {index}'
            )
            
        except Exception as e:
            self.chart_utils.handle_chart_error(
                e,
                "SHAP force plot",
                model_type=type(model).__name__,
                instance_index=index,
                dataframe_shape=X.shape if 'X' in locals() else None
            )

    # Add other model interpretation charts as needed...