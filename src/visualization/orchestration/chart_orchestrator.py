import logging
from typing import Dict
import matplotlib.pyplot as plt
from ..charts.feature_charts import FeatureCharts
from ..charts.metrics_charts import MetricsCharts
from ..charts.learning_curve_charts import LearningCurveCharts
from ..charts.shap_charts import SHAPCharts
from ...logging.logging_utils import log_performance, structured_log
from ...error_handling.custom_exceptions import ChartCreationError
from ...common.data_classes import ModelTrainingResults

logger = logging.getLogger(__name__)

class ChartOrchestrator:
    @log_performance
    def __init__(self, config):
        """
        Initialize the ChartOrchestrator with configuration.
        
        Args:
            config: Configuration object containing chart flags
        """
        self.config = config
        self.feature_charts = FeatureCharts()
        self.metrics_charts = MetricsCharts()
        self.learning_curve_charts = LearningCurveCharts()
        self.shap_charts = SHAPCharts()
        structured_log(logger, logging.INFO, "ChartOrchestrator initialized")

    @log_performance
    def generate_charts(self, results: ModelTrainingResults) -> Dict[str, plt.Figure]:
        """
        Generate all configured charts based on ModelTrainingResults.
        
        Args:
            results: ModelTrainingResults object containing all necessary data
            
        Returns:
            Dict[str, plt.Figure]: Dictionary mapping chart names to matplotlib figures
        """
        structured_log(logger, logging.INFO, "Starting chart generation")
        charts = {}

        try:
            chart_data = results.prepare_for_charting()

            # Feature Importance Chart
            chart_config = self.config.chart_options.feature_importance_chart
            if isinstance(chart_config, dict) and chart_config.get('enabled'):
                try:    
                    if chart_data["feature_importance"] is not None:
                        charts['feature_importance'] = self.feature_charts.create_feature_importance_chart(
                            feature_importance=chart_data["feature_importance"],
                            feature_names=chart_data["feature_names"],
                            top_n=chart_config.get('n_features', 20)
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create feature importance chart",
                                 error=str(e))

            # Confusion Matrix
            if self.config.chart_options.confusion_matrix:
                try:
                    if chart_data["y_true"] is not None and chart_data["y_pred"] is not None:
                        charts['confusion_matrix'] = self.metrics_charts.create_confusion_matrix(
                            y_true=chart_data["y_true"],
                            y_pred=chart_data["y_pred"]
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create confusion matrix",
                                 error=str(e))

            # ROC Curve
            if self.config.chart_options.roc_curve:
                try:
                    if chart_data["y_true"] is not None and chart_data["y_prob"] is not None:
                        charts['roc_curve'] = self.metrics_charts.create_roc_curve(
                            y_true=chart_data["y_true"],
                            y_score=chart_data["y_prob"]
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create ROC curve",
                                 error=str(e))

            # Learning Curve
            if self.config.chart_options.learning_curve:
                try:
                    charts['learning_curve'] = self.learning_curve_charts.create_learning_curve(
                        results=results
                    )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create learning curve",
                                 error=str(e))

            # SHAP Summary Plot
            chart_config = self.config.chart_options.shap_summary_plot
            if isinstance(chart_config, dict) and chart_config.get('enabled'):
                try:
                    if chart_data["model"] is not None and chart_data["X"] is not None:
                        charts['shap_summary'] = self.shap_charts.create_shap_summary_plot(
                            model=chart_data["model"],
                            X=chart_data["X"],
                            shap_values=results.shap_values,
                            n_features=chart_config.get('n_features')
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create SHAP summary plot",
                                 error=str(e))

            # SHAP Beeswarm Plot
            chart_config = self.config.chart_options.shap_beeswarm_plot
            if isinstance(chart_config, dict) and chart_config.get('enabled'):
                try:
                    if chart_data["model"] is not None and chart_data["X"] is not None:
                        charts['shap_beeswarm'] = self.shap_charts.create_shap_beeswarm_plot(
                            model=chart_data["model"],
                            X=chart_data["X"],
                            shap_values=results.shap_values,
                            n_features=chart_config.get('n_features')
                        )
                except Exception as e:
                    structured_log(logger, logging.WARNING, 
                                 "Failed to create SHAP beeswarm plot",
                                 error=str(e))

            structured_log(logger, logging.INFO, 
                         f"Chart generation completed. Generated {len(charts)} charts")
            return charts

        except Exception as e:
            raise ChartCreationError("Error in chart generation",
                                   error_message=str(e),
                                   charts_generated=list(charts.keys()))

    @log_performance
    def save_charts(self, charts: Dict[str, plt.Figure], output_dir: str) -> None:
        """
        Save generated charts to files.
        
        Args:
            charts: Dictionary of chart names and their corresponding matplotlib figures
            output_dir: Directory to save the charts
        """
        import os
        structured_log(logger, logging.INFO, "Saving charts", output_dir=output_dir)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for name, fig in charts.items():
                output_path = os.path.join(output_dir, f"{name}.png")
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close(fig)  # Clean up memory
                
            structured_log(logger, logging.INFO, "Charts saved successfully", 
                         chart_count=len(charts), 
                         output_dir=output_dir)
                         
        except Exception as e:
            raise ChartCreationError("Error saving charts",
                                   error_message=str(e),
                                   output_dir=output_dir) 