import logging
from typing import Dict, Optional
import matplotlib.pyplot as plt
from ..charts.chart_factory import ChartFactory
from ..charts.chart_types import ChartType
from ..charts.base_chart import BaseChart
from platform_core.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
from platform_core.core.app_logging.base_app_logger import BaseAppLogger
from platform_core.core.error_handling.base_error_handler import BaseErrorHandler
from platform_core.core.config_management.base_config_manager import BaseConfigManager
from platform_core.framework.data_classes import ModelTrainingResults
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
            if ChartType.FEATURE in self.charts and hasattr(results, 'feature_importance_scores') and results.feature_importance_scores is not None:
                # Get top_n from config if available
                top_n = None
                if hasattr(self.config, 'chart_options') and hasattr(self.config.chart_options, 'feature_importance'):
                    top_n = getattr(self.config.chart_options.feature_importance, 'top_n', None)
                
                charts_dict['feature_importance'] = self.charts[ChartType.FEATURE].create_figure(
                    feature_importance=results.feature_importance_scores,
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
        
        if hasattr(results, 'metrics') and results.metrics:
            try:
                # Create confusion matrix data
                from sklearn.metrics import confusion_matrix
                if hasattr(results, 'binary_predictions') and hasattr(results, 'target_data'):
                    cm = confusion_matrix(results.target_data, results.binary_predictions)
                    confusion_matrix_data = {
                        'matrix': cm,
                        'labels': ['0', '1']
                    }
                    metrics_charts['confusion_matrix'] = metrics.create_confusion_matrix(
                        confusion_matrix_data
                    )
            except Exception as e:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Failed to create confusion matrix",
                    error=str(e)
                )
            
            try:
                # Create ROC curve data
                from sklearn.metrics import roc_curve, auc
                if hasattr(results, 'probability_predictions') and hasattr(results, 'target_data'):
                    fpr, tpr, _ = roc_curve(results.target_data, results.probability_predictions)
                    roc_auc = auc(fpr, tpr)
                    roc_data = {
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': roc_auc
                    }
                    metrics_charts['roc_curve'] = metrics.create_roc_curve(
                        roc_data
                    )
            except Exception as e:
                self.app_logger.structured_log(
                    logging.WARNING,
                    "Failed to create ROC curve",
                    error=str(e)
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
                    if feature in results.feature_names:
                        shap_charts[f'shap_dependence_{feature}'] = shap.create_dependence_plot(
                            shap_values=results.shap_values,
                            features=results.feature_data,
                            feature_name=feature
                        )
                    
        return shap_charts

    def _create_interpretation_charts(self, results: ModelTrainingResults) -> Dict[str, plt.Figure]:
        """Create model interpretation charts."""
        interpretation_charts = {}
        
        if ChartType.MODEL_INTERPRETATION in self.charts:
            interpreter = self.charts[ChartType.MODEL_INTERPRETATION]
            
            # Check if model and configuration are available
            if (hasattr(results, 'model') and 
                hasattr(self.config, 'chart_options') and 
                hasattr(self.config.chart_options, 'model_interpretation') and
                hasattr(self.config.chart_options.model_interpretation, 'force_plot_indices')):
                
                # Get indices for force plots
                force_plot_indices = self.config.chart_options.model_interpretation.force_plot_indices
                
                for idx in force_plot_indices:
                    if hasattr(results, 'feature_data') and idx < len(results.feature_data):
                        interpretation_charts[f'shap_force_plot_{idx}'] = interpreter.create_shap_force_plot(
                            model=results.model,
                            X=results.feature_data,
                            index=idx,
                            shap_values=results.shap_values if hasattr(results, 'shap_values') else None
                        )
                    
        return interpretation_charts

    def create_model_comparison_charts(self, model_results: list) -> Dict[str, plt.Figure]:
        """
        Create charts comparing multiple models.
        
        Args:
            model_results: List of model results to compare
            
        Returns:
            Dictionary mapping chart names to figure objects
        """
        comparison_charts = {}
        
        try:
            # Create metrics comparison chart
            if len(model_results) > 1 and ChartType.METRICS in self.charts:
                # Extract metrics from each model
                metrics_data = {
                    result.model_name: vars(result.metrics) 
                    for result in model_results 
                    if hasattr(result, 'metrics') and result.metrics
                }
                
                if metrics_data:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc']
                    model_names = list(metrics_data.keys())
                    
                    # Create bar chart for each metric
                    bar_width = 0.15
                    index = np.arange(len(metrics_to_compare))
                    
                    for i, model_name in enumerate(model_names):
                        metric_values = [
                            metrics_data[model_name].get(metric, 0) 
                            for metric in metrics_to_compare
                        ]
                        ax.bar(
                            index + i * bar_width, 
                            metric_values,
                            bar_width,
                            label=model_name
                        )
                    
                    ax.set_xlabel('Metric')
                    ax.set_ylabel('Score')
                    ax.set_title('Model Performance Comparison')
                    ax.set_xticks(index + bar_width * (len(model_names) - 1) / 2)
                    ax.set_xticklabels(metrics_to_compare)
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    comparison_charts['metrics_comparison'] = fig
            
            return comparison_charts
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'chart_creation',
                "Error creating model comparison charts",
                original_error=str(e)
            )

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