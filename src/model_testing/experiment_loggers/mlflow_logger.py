import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import tempfile
import os
import json
import numpy as np
from mlflow.models.signature import infer_signature
from typing import Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt

from .base_experiment_logger import BaseExperimentLogger
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from ...visualization.orchestration.base_chart_orchestrator import BaseChartOrchestrator
from ...common.app_file_handling.base_app_file_handler import BaseAppFileHandler
from ...common.data_classes import ModelTrainingResults


class MLflowChartLogger:
    """Helper class for generating and logging charts to MLflow."""
    
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, error_handler: BaseErrorHandler):
        """Initialize MLflow chart logger with dependencies."""
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
               
        self.app_logger.structured_log(
            logging.INFO, 
            "MLflowChartLogger initialized"
        )

    def log_model_charts(self, results: Any) -> None:
        """Generate and log model evaluation charts to MLflow."""
        try:
            chart_data = results.prepare_for_charting()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Feature Importance Chart
                if self._should_create_feature_importance_chart():
                    self._create_feature_importance_chart(chart_data, temp_dir)

                # Confusion Matrix
                if self.config.chart_options.confusion_matrix:
                    self._create_confusion_matrix(chart_data, temp_dir)

                # ROC Curve
                if self.config.chart_options.roc_curve:
                    self._create_roc_curve(chart_data, temp_dir)

                # SHAP Summary Plot
                if self._should_create_shap_summary():
                    self._create_shap_summary(chart_data, temp_dir)

                # Learning Curve
                if getattr(self.config.chart_options, 'learning_curve', False):
                    self._create_learning_curve(results, temp_dir)
                
                self.app_logger.structured_log(
                    logging.INFO, 
                    "All charts logged successfully"
                )

        except Exception as e:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Error in chart logging",
                original_error=str(e)
            )

    def _should_create_feature_importance_chart(self) -> bool:
        return (hasattr(self.config.chart_options, 'feature_importance') and 
                getattr(self.config.chart_options.feature_importance, 'enabled', False))

    def _should_create_shap_summary(self) -> bool:
        return (hasattr(self.config.chart_options, 'shap_summary') and 
                getattr(self.config.chart_options.shap_summary, 'enabled', False))

    # ... Additional chart creation methods ...

class MLFlowLogger(BaseExperimentLogger):
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 chart_orchestrator: BaseChartOrchestrator,
                 app_file_handler: BaseAppFileHandler):
        """
        Initialize MLflow logger with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            chart_orchestrator: Chart orchestrator for visualization
            app_file_handler: Application file handler for managing files
        """
        super().__init__(config, app_logger, error_handler, chart_orchestrator)
        self.app_file_handler = app_file_handler
        
        # Configure MLflow
        mlflow.set_tracking_uri(self.config.get('mlflow', {}).get('tracking_uri'))
        mlflow.set_experiment(self.config.get('mlflow', {}).get('experiment_name', 'default'))
        
        self.app_logger.structured_log(
            logging.INFO,
            "MLflow logger initialized",
            tracking_uri=mlflow.get_tracking_uri(),
            experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name if mlflow.active_run() else None
        )

    @property
    def log_performance(self):
        return self.app_logger.log_performance

    @log_performance
    def log_experiment(self, results: ModelTrainingResults) -> None:
        """
        Log an experiment's results to MLflow.
        
        Args:
            results: Model training results containing all experiment data
        """
        try:
            with mlflow.start_run(run_name=results.model_name):
                # Log parameters
                self._log_parameters(results)
                
                # Log metrics
                self._log_metrics(results)
                
                # Log model
                self._log_model(results)
                
                # Log charts using the orchestrator
                self._log_charts(results)
                
                self.app_logger.structured_log(
                    logging.INFO,
                    "Experiment logged successfully",
                    model_name=results.model_name,
                    run_id=mlflow.active_run().info.run_id
                )
                
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Error logging experiment to MLflow",
                original_error=str(e),
                model_name=results.model_name
            )

    def _log_charts(self, results: ModelTrainingResults) -> None:
        """
        Generate and log visualization charts to MLflow.
        
        Args:
            results: Model training results containing data for charts
        """
        try:
            # Get all charts from orchestrator
            charts = self.chart_orchestrator.create_model_evaluation_charts(results)
            
            # Use app_file_handler to create and manage temporary directory
            with self.app_file_handler.create_temp_directory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                for chart_name, fig in charts.items():
                    if fig is not None:
                        chart_path = temp_dir / f"{chart_name}.png"
                        # Use app_file_handler to save the figure
                        self.app_file_handler.save_figure(fig, chart_path)
                        plt.close(fig)
                        
                        mlflow.log_artifact(
                            str(chart_path),
                            artifact_path="charts"
                        )
                
                self.app_logger.structured_log(
                    logging.INFO,
                    "Charts logged successfully",
                    chart_types=list(charts.keys())
                )
                
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Error logging charts to MLflow",
                original_error=str(e)
            )

    def _log_parameters(self, results: ModelTrainingResults) -> None:
        """Log model parameters to MLflow."""
        try:
            # Log model parameters
            mlflow.log_params(results.model_params)
            
            # Log training parameters
            mlflow.log_params({
                "n_folds": results.n_folds,
                "feature_count": len(results.feature_names),
                "sample_count": results.X_train.shape[0]
            })
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Error logging parameters to MLflow",
                original_error=str(e)
            )

    def _log_metrics(self, results: ModelTrainingResults) -> None:
        """Log model metrics to MLflow."""
        try:
            # Log average metrics
            for metric_name, value in results.metrics.items():
                mlflow.log_metric(f"avg_{metric_name}", value)
            
            # Log fold-specific metrics
            for fold, fold_metrics in results.fold_metrics.items():
                for metric_name, value in fold_metrics.items():
                    mlflow.log_metric(f"{metric_name}_fold_{fold}", value)
                    
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Error logging metrics to MLflow",
                original_error=str(e)
            )

    def _log_model(self, results: ModelTrainingResults) -> None:
        """Log the trained model to MLflow."""
        try:
            mlflow.sklearn.log_model(
                results.best_model,
                "model",
                registered_model_name=results.model_name
            )
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Error logging model to MLflow",
                original_error=str(e),
                model_name=results.model_name
            )