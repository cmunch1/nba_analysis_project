import logging
from typing import Dict, Optional
import matplotlib.pyplot as plt
from ..charts.chart_factory import ChartFactory
from ..charts.chart_types import ChartType
from ..charts.base_chart import BaseChart
from ...common.app_file_handling.base_app_file_handler import BaseAppFileHandler
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.data_classes import ModelTrainingResults
from .base_chart_orchestrator import BaseChartOrchestrator


class ChartOrchestrator(BaseChartOrchestrator):
    """Concrete implementation of chart orchestration."""
    
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 app_file_handler: BaseAppFileHandler):
        """
        Initialize the ChartOrchestrator with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            app_file_handler: Application file handler
        """
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.app_file_handler = app_file_handler
        
        # Initialize chart instances based on configuration
        self.charts: Dict[ChartType, BaseChart] = {}
        self._initialize_charts()
        
        self.app_logger.structured_log(
            logging.INFO,
            "ChartOrchestrator initialized",
            active_charts=list(self.charts.keys())
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

    def _initialize_charts(self) -> None:
        """Initialize enabled chart types based on configuration."""
        try:
            # Get chart options with fallbacks
            if hasattr(self.config, 'chart_options'):
                chart_options = self.config.chart_options
            else:
                chart_options = type('EmptyConfig', (), {})()  # Empty object
            
            # Check for feature importance chart
            if hasattr(chart_options, 'feature_importance') and getattr(chart_options.feature_importance, 'enabled', False):
                self.charts[ChartType.FEATURE] = ChartFactory.create_chart(
                    ChartType.FEATURE, self.config, self.app_logger, self.error_handler
                )
                
            # Check for metrics chart
            if hasattr(chart_options, 'metrics') and getattr(chart_options.metrics, 'enabled', False):
                self.charts[ChartType.METRICS] = ChartFactory.create_chart(
                    ChartType.METRICS, self.config, self.app_logger, self.error_handler
                )
                
            # Check for learning curve chart
            if hasattr(chart_options, 'learning_curve') and getattr(chart_options.learning_curve, 'enabled', False):
                self.charts[ChartType.LEARNING_CURVE] = ChartFactory.create_chart(
                    ChartType.LEARNING_CURVE, self.config, self.app_logger, self.error_handler
                )
                
            # Check for SHAP chart
            if hasattr(chart_options, 'shap') and getattr(chart_options.shap, 'enabled', False):
                self.charts[ChartType.SHAP] = ChartFactory.create_chart(
                    ChartType.SHAP, self.config, self.app_logger, self.error_handler
                )

            # Check for model interpretation chart
            if hasattr(chart_options, 'model_interpretation') and getattr(chart_options.model_interpretation, 'enabled', False):
                self.charts[ChartType.MODEL_INTERPRETATION] = ChartFactory.create_chart(
                    ChartType.MODEL_INTERPRETATION, self.config, self.app_logger, self.error_handler
                )
                
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'chart_creation',
                "Error initializing charts",
                original_error=str(e)
            )

    @log_performance
    def create_model_evaluation_charts(self, results: ModelTrainingResults) -> Dict[str, plt.Figure]:
        """
        Create all enabled charts for model evaluation.
        
        Args:
            results: Model training results containing data for charts
            
        Returns:
            Dictionary mapping chart names to figure objects
        """
        try:
            charts_dict = {}
            
            # Feature importance chart
            if ChartType.FEATURE in self.charts and hasattr(results, 'feature_importance') and results.feature_importance is not None:
                # Get top_n from config if available
                top_n = None
                if hasattr(self.config, 'chart_options') and hasattr(self.config.chart_options, 'feature_importance'):
                    top_n = getattr(self.config.chart_options.feature_importance, 'top_n', None)
                
                charts_dict['feature_importance'] = self.charts[ChartType.FEATURE].create_figure(
                    feature_importance=results.feature_importance,
                    feature_names=results.feature_names,
                    top_n=top_n
                )
            
            # Metrics charts (confusion matrix, ROC curve, etc.)
            if ChartType.METRICS in self.charts:
                metrics_charts = self._create_metrics_charts(results)
                charts_dict.update(metrics_charts)
            
            # Learning curve
            if ChartType.LEARNING_CURVE in self.charts and hasattr(results, 'learning_curve_data') and results.learning_curve_data:
                charts_dict['learning_curve'] = self.charts[ChartType.LEARNING_CURVE].create_figure(
                    results=results
                )
            
            # SHAP charts
            if ChartType.SHAP in self.charts and hasattr(results, 'shap_values') and results.shap_values is not None:
                shap_charts = self._create_shap_charts(results)
                charts_dict.update(shap_charts)

            # Model interpretation charts
            if ChartType.MODEL_INTERPRETATION in self.charts:
                interpretation_charts = self._create_interpretation_charts(results)
                charts_dict.update(interpretation_charts)
            
            self.app_logger.structured_log(
                logging.INFO,
                "Model evaluation charts created",
                chart_types=list(charts_dict.keys())
            )
            
            return charts_dict
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'chart_creation',
                "Error creating model evaluation charts",
                original_error=str(e)
            )

    def _create_metrics_charts(self, results: ModelTrainingResults) -> Dict[str, plt.Figure]:
        """Create all metrics-related charts."""
        metrics_charts = {}
        metrics = self.charts[ChartType.METRICS]
        
        if hasattr(results, 'confusion_matrix_data'):
            metrics_charts['confusion_matrix'] = metrics.create_confusion_matrix(
                results.confusion_matrix_data
            )
            
        if hasattr(results, 'roc_curve_data'):
            metrics_charts['roc_curve'] = metrics.create_roc_curve(
                results.roc_curve_data
            )
            
        return metrics_charts

    def _create_shap_charts(self, results: ModelTrainingResults) -> Dict[str, plt.Figure]:
        """Create all SHAP-related charts."""
        shap_charts = {}
        shap = self.charts[ChartType.SHAP]
        
        if hasattr(results, 'shap_values') and results.shap_values is not None:
            shap_charts['shap_summary'] = shap.create_summary_plot(
                shap_values=results.shap_values,
                feature_names=results.feature_names
            )
            
            # Check for dependence plots config
            if (hasattr(self.config, 'chart_options') and 
                hasattr(self.config.chart_options, 'shap') and 
                hasattr(self.config.chart_options.shap, 'dependence_plots') and
                self.config.chart_options.shap.dependence_plots):
                
                # Get dependence features
                dependence_features = []
                if hasattr(self.config.chart_options.shap, 'dependence_features'):
                    dependence_features = self.config.chart_options.shap.dependence_features
                
                for feature in dependence_features:
                    shap_charts[f'shap_dependence_{feature}'] = shap.create_dependence_plot(
                        shap_values=results.shap_values,
                        features=results.X_val,
                        feature_name=feature
                    )
                    
        return shap_charts

    def _create_interpretation_charts(self, results: ModelTrainingResults) -> Dict[str, plt.Figure]:
        """Create model interpretation charts."""
        interpretation_charts = {}
        interpreter = self.charts[ChartType.MODEL_INTERPRETATION]
        
        # Check if model and configuration are available
        if (hasattr(results, 'model') and 
            hasattr(self.config, 'chart_options') and 
            hasattr(self.config.chart_options, 'model_interpretation') and
            hasattr(self.config.chart_options.model_interpretation, 'force_plot_indices')):
            
            # Get indices for force plots
            force_plot_indices = self.config.chart_options.model_interpretation.force_plot_indices
            
            for idx in force_plot_indices:
                interpretation_charts[f'shap_force_plot_{idx}'] = interpreter.create_shap_force_plot(
                    model=results.model,
                    X=results.X_val,
                    index=idx,
                    shap_values=results.shap_values if hasattr(results, 'shap_values') else None
                )
                    
        return interpretation_charts

    @log_performance
    def save_charts(self, charts: Dict[str, plt.Figure], output_dir: str) -> None:
        """
        Save generated charts to files using the application's file handler.
        
        Args:
            charts: Dictionary of chart names and their corresponding matplotlib figures
            output_dir: Directory to save the charts
        """
        self.app_logger.structured_log(
            logging.INFO,
            "Saving charts",
            output_dir=output_dir
        )
        
        try:
            self.app_file_handler.ensure_directory(output_dir)
            
            for name, fig in charts.items():
                output_path = self.app_file_handler.join_paths(output_dir, f"{name}.png")
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close(fig)  # Clean up memory
                
            self.app_logger.structured_log(
                logging.INFO,
                "Charts saved successfully",
                chart_count=len(charts),
                output_dir=output_dir
            )
                         
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'chart_creation',
                "Error saving charts",
                error_message=str(e),
                output_dir=output_dir
            )