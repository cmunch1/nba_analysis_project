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
from typing import Dict, Any

from .base_experiment_logger import BaseExperimentLogger
from ...common.config_management.base_config_manager import BaseConfigManager
from ...common.app_logging.base_app_logger import BaseAppLogger
from ...common.error_handling.base_error_handler import BaseErrorHandler
from ...visualization.chart_functions import ChartFunctions

class MLflowChartLogger:
    """Helper class for generating and logging charts to MLflow."""
    
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, error_handler: BaseErrorHandler):
        """Initialize MLflow chart logger with dependencies."""
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self.chart_functions = ChartFunctions()
        
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
    def __init__(self, config: BaseConfigManager, app_logger: BaseAppLogger, error_handler: BaseErrorHandler):
        """Initialize MLflow logger with dependencies."""
        self.config = config
        self.app_logger = app_logger
        self.error_handler = error_handler
        self._chart_logger = MLflowChartLogger(config, app_logger, error_handler)
        
        self.app_logger.structured_log(
            logging.INFO, 
            "MLFlowLogger initialized"
        )

    @property
    def log_performance(self):
        """Get the performance logging decorator from app_logger."""
        return self.app_logger.log_performance

    @log_performance
    def log_experiment(self, results: Any) -> None:
        """Log an experiment to MLflow with preprocessing tracking."""
        if results.model is None:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Cannot log experiment: model object is None"
            )

        # Set experiment
        mlflow.set_experiment(self.config.experiment_name)
        
        # Create run name
        run_name = f"{results.model_name}_{results.evaluation_type}"
        
        with mlflow.start_run(run_name=run_name):
            try:
                self._log_experiment_metadata(results)
                self._log_model_parameters(results)
                self._log_metrics(results)
                self._log_preprocessing_info(results)
                self._log_data_info(results)
                self._log_model(results)
                self._chart_logger.log_model_charts(results)

                self.app_logger.structured_log(
                    logging.INFO, 
                    "Experiment logged successfully", 
                    model_name=results.model_name
                )

            except Exception as e:
                self.app_logger.structured_log(
                    logging.ERROR, 
                    "Failed to log experiment",
                    error=str(e)
                )
                raise

    def _log_experiment_metadata(self, results: Any) -> None:
        """Log experiment metadata to MLflow."""
        mlflow.set_tag("description", self.config.experiment_description)
        mlflow.set_tag("run_type", results.evaluation_type)

    def _log_model_parameters(self, results: Any) -> None:
        """Log model parameters to MLflow."""
        mlflow.log_params(results.model_params)
        mlflow.log_params({
            "num_boost_round": results.num_boost_round,
            "early_stopping": results.early_stopping,
            "enable_categorical": results.enable_categorical,
            "categorical_features": results.categorical_features
        })

    def _log_metrics(self, results: Any) -> None:
        """Log metrics to MLflow."""
        if results.metrics:
            prefix = "val_" if results.is_validation else "oof_"
            mlflow.log_metrics({
                f"{prefix}accuracy": results.metrics.accuracy,
                f"{prefix}precision": results.metrics.precision,
                f"{prefix}recall": results.metrics.recall,
                f"{prefix}f1": results.metrics.f1,
                f"{prefix}auc": results.metrics.auc,
                f"{prefix}optimal_threshold": results.metrics.optimal_threshold
            })

    def _log_preprocessing_info(self, results: Any) -> None:
        """Log preprocessing information to MLflow."""
        if results.preprocessing_results:
            preprocessing_summary = results.preprocessing_results.summarize()
            mlflow.log_params({
                "n_original_features": preprocessing_summary["n_original_features"],
                "n_final_features": preprocessing_summary["n_final_features"],
                "n_preprocessing_steps": preprocessing_summary["n_preprocessing_steps"]
            })

            with tempfile.TemporaryDirectory() as temp_dir:
                self._save_preprocessing_artifacts(results, temp_dir)

    def _save_preprocessing_artifacts(self, results: Any, temp_dir: str) -> None:
        """Save preprocessing artifacts to MLflow."""
        preproc_path = os.path.join(temp_dir, "preprocessing_info.json")
        with open(preproc_path, "w") as f:
            json.dump(results.preprocessing_results.to_dict(), f, indent=2)
        mlflow.log_artifact(preproc_path, "preprocessing")

        transformations = results.preprocessing_results.feature_transformations
        trans_path = os.path.join(temp_dir, "feature_transformations.json")
        with open(trans_path, "w") as f:
            json.dump(transformations, f, indent=2)
        mlflow.log_artifact(trans_path, "preprocessing")

    @log_performance
    def log_model(self, model: Any, model_name: str, model_params: Dict[str, Any]) -> None:
        """Log a model to MLflow with its parameters."""
        try:
            with mlflow.start_run():
                mlflow.set_experiment(self.config.experiment_name)
                mlflow.log_params(model_params)
                
                # Log model using appropriate MLflow flavor
                if 'xgboost' in model_name.lower():
                    mlflow.xgboost.log_model(model, model_name)
                elif 'lgbm' in model_name.lower():
                    mlflow.lightgbm.log_model(model, model_name)
                else:
                    mlflow.sklearn.log_model(model, model_name)
                    
                self.app_logger.structured_log(
                    logging.INFO, 
                    "Model logged successfully", 
                    model_name=model_name
                )
                             
        except Exception as e:
            raise self.error_handler.create_error_handler(
                'experiment_logging',
                "Error logging model to MLflow",
                error_message=str(e),
                model_name=model_name
            )