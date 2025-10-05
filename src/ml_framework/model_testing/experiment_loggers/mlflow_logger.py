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
from ml_framework.core.config_management.base_config_manager import BaseConfigManager
from ml_framework.core.app_logging.base_app_logger import BaseAppLogger
from ml_framework.core.error_handling.base_error_handler import BaseErrorHandler
from ml_framework.visualization.orchestration.base_chart_orchestrator import BaseChartOrchestrator
from ml_framework.core.app_file_handling.base_app_file_handler import BaseAppFileHandler
from ml_framework.framework.data_classes import ModelTrainingResults


class MLFlowLogger(BaseExperimentLogger):
    def __init__(self,
                 config: BaseConfigManager,
                 app_logger: BaseAppLogger,
                 error_handler: BaseErrorHandler,
                 app_file_handler: BaseAppFileHandler,
                 chart_orchestrator: BaseChartOrchestrator
                 ):
        """
        Initialize MLflow logger with dependencies.
        
        Args:
            config: Configuration manager
            app_logger: Application logger
            error_handler: Error handler
            app_file_handler: Application file handler for managing files
            chart_orchestrator: Chart orchestrator for visualization
            
        """
        super().__init__(config, app_logger, error_handler, app_file_handler, chart_orchestrator)
        
        # default values if not provided in config
        tracking_uri = None
        experiment_name = "default"
        

        if hasattr(self.config, 'tracking_uri'):
            tracking_uri = self.config.tracking_uri
        if hasattr(self.config, 'experiment_name'):
            experiment_name = self.config.experiment_name
        
        # Set tracking URI if provided, otherwise use local file system
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Ensure we use a Linux-compatible local path
            local_mlruns_path = f"file://{os.path.abspath('mlruns')}"
            mlflow.set_tracking_uri(local_mlruns_path)


        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Safely check if there's an active run before trying to access experiment_id
        active_run = mlflow.active_run()
        experiment_name_log = None
        if active_run:
            try:
                experiment_name_log = mlflow.get_experiment(active_run.info.experiment_id).name
            except Exception:
                # If there's any issue getting the experiment name, just log None
                pass
        
        self.app_logger.structured_log(
            logging.INFO,
            "MLflow logger initialized",
            tracking_uri=mlflow.get_tracking_uri(),
            experiment_name=experiment_name_log
        )

    @staticmethod
    def log_performance(func):
        """Decorator factory for performance logging"""
        def wrapper(*args, **kwargs):
            # Get the self instance from args since this is now a static method
            instance = args[0]
            return instance.app_logger.log_performance(func)(*args, **kwargs)
        return wrapper

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
                "sample_count": results.feature_data.shape[0] if hasattr(results, 'feature_data') else 0
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
            # Log metrics
            if hasattr(results, 'metrics') and results.metrics:
                prefix = "val_" if results.is_validation else "oof_"
                for metric_name, value in vars(results.metrics).items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{prefix}{metric_name}", value)
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Error logging metrics to MLflow",
                original_error=str(e)
            )

    def _log_model(self, results: ModelTrainingResults) -> None:
        """Log the trained model to MLflow."""
        try:
            if hasattr(results, 'model') and results.model:
                if hasattr(results.model, "__module__") and "xgboost" in results.model.__module__:
                    mlflow.xgboost.log_model(results.model, "model")
                elif hasattr(results.model, "__module__") and "lightgbm" in results.model.__module__:
                    mlflow.lightgbm.log_model(results.model, "model")
                elif hasattr(results.model, "__module__") and "catboost" in results.model.__module__:
                    mlflow.catboost.log_model(results.model, "model")
                else:
                    mlflow.sklearn.log_model(results.model, "model")
            
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Error logging model to MLflow",
                original_error=str(e),
                model_name=results.model_name
            )